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


class Transformer(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        dim = cfg.get("dim")
        n_layers = cfg.get("n_layers")
        n_heads = cfg.get("n_heads")
        dropout = cfg.get("dropout", 0.0)
        mlp_mult = cfg.get("mlp_mult", 4)
        activation = cfg.get("activation", "gelu")
        qkv_bias = cfg.get("qkv_bias", False)
        context_length = cfg.get("context_length")
        prenorm = cfg.get("prenorm", True)
        use_final_ln = bool(cfg.get("final_ln", True))

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                context_length=context_length,
                mlp_mult=mlp_mult,
                dropout=dropout,
                activation=activation,
                qkv_bias=qkv_bias,
                prenorm=prenorm,
            )
            for _ in range(n_layers)
        ])
        self.final_ln = LayerNorm(dim) if use_final_ln else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x)
        if self.final_ln is not None:
            x = self.final_ln(x)
        return x
