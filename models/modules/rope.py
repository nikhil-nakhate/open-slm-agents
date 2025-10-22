"""Rotary Position Embedding (RoPE) with YaRN scaling.

This implementation supports:
- Standard RoPE for positional encoding
- YaRN scaling for extended context length
- NTK-by-parts interpolation/extrapolation
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding.

    Args:
        x: Input tensor of shape [n_tokens, n_heads, head_dim] or [n_tokens, n_heads, q_mult, head_dim]
        cos: Cosine values [n_tokens, head_dim//2]
        sin: Sine values [n_tokens, head_dim//2]

    Returns:
        Rotated tensor
    """
    # Add dimension for broadcasting
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    # Split into first and second half
    x1, x2 = x.chunk(2, dim=-1)

    # Apply rotation
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) with YaRN scaling."""

    def __init__(
        self,
        head_dim: int,
        base: float,
        scaling_factor: float = 1.0,
        initial_context_length: int = 4096,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.initial_context_length = initial_context_length
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> Tuple[float, torch.Tensor]:
        """Compute concentration factor and inverse frequencies for YaRN scaling.

        See YaRN paper: https://arxiv.org/abs/2309.00071
        """
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) / self.head_dim
        )

        if self.scaling_factor > 1.0:
            # YaRN concentration
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.head_dim / 2
            # NTK by parts boundaries
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )

            # Interpolation and extrapolation
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            # Smooth transition ramp
            ramp = (torch.arange(d_half, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor [n_tokens, n_heads, (q_mult), head_dim]
            k: Key tensor [n_tokens, n_heads, head_dim]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        num_tokens = q.shape[0]
        concentration, inv_freq = self._compute_concentration_and_inv_freq()

        # Build position encodings
        t = torch.arange(num_tokens, dtype=torch.float32, device=q.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration

        # Apply to q and k
        q_shape = q.shape
        q = q.view(num_tokens, -1, self.head_dim)
        q = apply_rotary_emb(q, cos, sin)
        q = q.reshape(q_shape)

        k_shape = k.shape
        k = k.view(num_tokens, -1, self.head_dim)
        k = apply_rotary_emb(k, cos, sin)
        k = k.reshape(k_shape)

        return q, k
