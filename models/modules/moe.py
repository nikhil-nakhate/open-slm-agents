"""MLP module with Mixture of Experts (MoE).

This module implements the MLP layer with MoE, featuring:
- Mixture of Experts with configurable number of experts
- Top-k expert routing per token
- SwiGLU activation with clamping
- Support for quantized expert weights (MXFP4)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .rms_norm import RMSNorm


class MLPBlock(nn.Module):
    """Mixture-of-Experts MLP with SwiGLU activation.

    This implements the MoE MLP with:
    - Configurable number of experts
    - Top-k routing per token
    - SwiGLU activation (clamped to prevent overflow)
    - Support for tensor parallelism (sharded across ranks)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        experts_per_token: int,
        swiglu_limit: float,
        rms_norm_eps: float,
        dtype: torch.dtype,
    ) -> None:
        """Initialize MLP block.

        Args:
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate dimension size (before sharding)
            num_experts: Total number of experts
            experts_per_token: Number of experts to route each token to
            swiglu_limit: Clamping limit for SwiGLU activation
            rms_norm_eps: RMSNorm epsilon
            dtype: Data type for parameters
        """
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        assert self.intermediate_size % self.world_size == 0
        hidden_per_rank = self.intermediate_size // self.world_size

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate = nn.Linear(hidden_size, num_experts, bias=True, dtype=dtype)

        # Expert weights (sharded across ranks for tensor parallelism)
        self.mlp1_weight = nn.Parameter(
            torch.empty(self.num_experts, hidden_per_rank * 2, hidden_size, dtype=torch.bfloat16)
        )
        self.mlp1_bias = nn.Parameter(torch.empty(self.num_experts, hidden_per_rank * 2, dtype=torch.bfloat16))
        self.mlp2_weight = nn.Parameter(
            torch.empty(self.num_experts, hidden_size, hidden_per_rank, dtype=torch.bfloat16)
        )
        self.mlp2_bias = nn.Parameter(torch.empty(self.num_experts, hidden_size, dtype=torch.bfloat16))
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        residual = x
        orig_shape = x.shape
        t = self.norm(x)

        # Flatten batch and sequence dimensions for MoE processing
        t = t.view(-1, self.hidden_size)  # (B*T, hidden)

        # Expert routing
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        # MLP #1 - Use einsum to keep token-expert structure
        mlp1_weight = self.mlp1_weight[expert_indices, ...]  # (B*T, experts_per_token, 2*intermediate, hidden)
        mlp1_bias = self.mlp1_bias[expert_indices, ...]      # (B*T, experts_per_token, 2*intermediate)
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias

        # SwiGLU activation (interleaved)
        t_glu, t_linear = t[..., ::2], t[..., 1::2]
        t_glu = t_glu.clamp(max=self.swiglu_limit)
        t_linear = t_linear.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        t = t_glu * torch.sigmoid(1.702 * t_glu) * (t_linear + 1)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]  # (B*T, experts_per_token, hidden, intermediate)
        mlp2_bias = self.mlp2_bias[expert_indices, ...]      # (B*T, experts_per_token, hidden)
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)

        if self.world_size > 1:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

        t += mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        # Reshape back to original batch/sequence dimensions
        t = t.view(*orig_shape)

        return residual + t
