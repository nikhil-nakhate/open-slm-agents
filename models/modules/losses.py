import torch
import torch.nn as nn
from typing import Dict, Any


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V], targets: [B, T]
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(B * T, V), targets.view(B * T))
        return loss


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    kind = (cfg or {}).get("kind", "cross_entropy").lower()
    params = (cfg or {}).get("params", {})
    if kind in {"cross_entropy", "ce"}:
        return CrossEntropyLossWrapper(**params)
    raise ValueError(f"Unknown loss kind: {kind}")
