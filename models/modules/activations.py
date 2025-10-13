import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


class SwiGLU(nn.Module):
    """SwiGLU activation used in GPT-OSS feed-forward and MoE experts."""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = x.chunk(2, dim=-1)
        u = u.clamp(max=self.limit)
        v = v.clamp(min=-self.limit, max=self.limit)
        gated = u * torch.sigmoid(self.alpha * u)
        return gated * (v + 1.0)


def build_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return GELU()
    if name in {"silu", "swish"}:
        return SiLU()
    if name == "swiglu":
        return SwiGLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")
