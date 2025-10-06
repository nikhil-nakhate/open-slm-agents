import math
import torch
import torch.nn as nn


class OutputProjection(nn.Module):
    def __init__(self, dim: int, vocab_size: int, tie_weights: bool = False, tie_to: nn.Embedding = None):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            if tie_to is None:
                raise ValueError("tie_weights=True requires tie_to Embedding")
            self.proj.weight = tie_to.weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.token_emb(idx)


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(0, seq_len, device=device).unsqueeze(0)
        return self.pos_emb(pos)
