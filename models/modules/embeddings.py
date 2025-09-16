import math
import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)

        # Initialize pos embeddings with normal init similar to GPT2
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(0, T, device=device).unsqueeze(0)  # [1, T]
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        return x


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
