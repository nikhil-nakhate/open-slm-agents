from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .activations import SwiGLU
from .rms_norm import RMSNorm


class MoEMLP(nn.Module):
    """Mixture-of-experts feed-forward layer with SwiGLU experts."""

    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        num_experts: int,
        experts_per_token: int,
        swiglu_limit: float = 7.0,
        dropout: float = 0.0,
        normalized_input: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.normalize_input = normalized_input
        if normalized_input:
            self.input_norm = RMSNorm(dim, device=device)
        else:
            self.input_norm = None

        proj_dtype = dtype or torch.float32
        self.gate = nn.Linear(dim, num_experts, device=device, dtype=proj_dtype)
        assert intermediate_size % self.world_size == 0, "intermediate_size must divide world_size"
        hidden_per_rank = intermediate_size // self.world_size

        self.expert_ff1 = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_per_rank * 2,
                dim,
                device=device,
                dtype=proj_dtype,
            )
        )
        self.expert_ff1_bias = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_per_rank * 2,
                device=device,
                dtype=proj_dtype,
            )
        )
        self.expert_ff2 = nn.Parameter(
            torch.empty(
                num_experts,
                dim,
                hidden_per_rank,
                device=device,
                dtype=proj_dtype,
            )
        )
        self.expert_ff2_bias = nn.Parameter(
            torch.empty(
                num_experts,
                dim,
                device=device,
                dtype=proj_dtype,
            )
        )
        self.activation = SwiGLU(limit=swiglu_limit)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.expert_ff1)
        nn.init.zeros_(self.expert_ff1_bias)
        nn.init.xavier_uniform_(self.expert_ff2)
        nn.init.zeros_(self.expert_ff2_bias)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        if self.input_norm is not None:
            nn.init.ones_(self.input_norm.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        tokens = x.view(-1, self.dim)
        if self.input_norm is not None:
            tokens = self.input_norm(tokens)

        gate_scores = self.gate(tokens)
        topk = torch.topk(gate_scores, k=self.experts_per_token, dim=-1, sorted=True)
        weights = torch.softmax(topk.values, dim=-1)
        indices = topk.indices

        ff1_weight = self.expert_ff1[indices, ...]
        ff1_bias = self.expert_ff1_bias[indices, ...]
        hidden = torch.einsum("behd,bd->beh", ff1_weight, tokens) + ff1_bias
        hidden = self.activation(hidden)

        ff2_weight = self.expert_ff2[indices, ...]
        ff2_bias = self.expert_ff2_bias[indices, ...]
        updates = torch.einsum("bedh,beh->bed", ff2_weight, hidden)
        if self.world_size > 1:
            dist.all_reduce(updates, op=dist.ReduceOp.SUM)
        updates = updates + ff2_bias
        updates = torch.einsum("bed,be->bd", updates, weights)
        updates = updates.view(*orig_shape)
        return self.dropout(updates)
