import torch
import torch.nn as nn
from typing import Dict, Any

from .layer_norm import LayerNorm
from .mha import MultiHeadAttention
from .activations import build_activation


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = build_activation(activation)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        context_length: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
        qkv_bias: bool = False,
        prenorm: bool = True,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(
            d_in=dim,
            d_out=dim,
            context_length=context_length,
            dropout=dropout,
            num_heads=n_heads,
            qkv_bias=qkv_bias,
        )
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim, hidden_mult=mlp_mult, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlp(x))
        return x
