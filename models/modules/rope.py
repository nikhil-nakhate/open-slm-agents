import math
from typing import Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) with optional YaRN scaling."""

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        *,
        scaling_factor: float = 1.0,
        initial_context_length: int = 4096,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even head_dim")
        self.head_dim = head_dim
        self.base = float(base)
        self.scaling_factor = float(scaling_factor)
        self.initial_context_length = int(initial_context_length)
        self.ntk_alpha = float(ntk_alpha)
        self.ntk_beta = float(ntk_beta)
        self.dtype = dtype
        self.register_buffer(
            "inv_freq_base",
            torch.arange(0, head_dim, 2, dtype=torch.float32),
            persistent=False,
        )

    def _compute_concentration_and_inv_freq(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        freq = self.base ** (self.inv_freq_base.to(device) / self.head_dim)
        if self.scaling_factor > 1.0:
            concentration = torch.tensor(
                0.1 * math.log(self.scaling_factor) + 1.0,
                device=device,
                dtype=torch.float32,
            )

            d_half = self.head_dim / 2
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
            if not (0 < low < high < d_half - 1):
                raise ValueError("Invalid YaRN configuration for rotary embedding scaling.")

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, device=device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = torch.tensor(1.0, device=device, dtype=torch.float32)
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        concentration, inv_freq = self._compute_concentration_and_inv_freq(device)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = (freqs.cos() * concentration).to(dtype)
        sin = (freqs.sin() * concentration).to(dtype)
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos, sin = self._build_cache(seq_len, q.device, q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q1, q2 = torch.chunk(q, 2, dim=-1)
        k1, k2 = torch.chunk(k, 2, dim=-1)
        q = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
        k = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
        return q, k
