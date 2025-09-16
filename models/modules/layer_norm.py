import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Manual LayerNorm per provided reference, with type hints and cfg support.

    Computes: y = (x - mean) / sqrt(var + eps); out = y * weight + bias.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))
        self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is None:
            return norm * self.weight
        return norm * self.weight + self.bias
