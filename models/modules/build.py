from typing import Any, Dict, Optional
import torch.nn as nn

from .embeddings import TokenPositionalEmbedding, OutputProjection, TokenEmbedding, PositionEmbedding
from .transformer import Transformer, TransformerBlock
from .layer_norm import LayerNorm
from .losses import build_loss as _build_loss
from ops.tokenizer import build_tokenizer as _build_tokenizer


def _maybe_freeze(module, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    if bool(cfg.get("freeze", False)) and hasattr(module, "parameters"):
        for p in module.parameters():
            p.requires_grad = False
    return module


def build_embedding(vocab_size: int, dim: int, max_seq_len: int, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    dropout = cfg.get("dropout", 0.0)
    mod = TokenPositionalEmbedding(vocab_size, dim, max_seq_len, dropout=dropout)
    return _maybe_freeze(mod, cfg)



def build_output_projection(dim: int, vocab_size: int, cfg: Optional[Dict[str, Any]] = None, tie_to=None):
    cfg = cfg or {}
    tie = cfg.get("tie_weights", True)
    mod = OutputProjection(dim, vocab_size, tie_weights=tie, tie_to=tie_to if tie else None)
    return _maybe_freeze(mod, cfg)


def build_loss(cfg: Optional[Dict[str, Any]] = None):
    # Loss is usually stateless; freeze flag is a no-op here.
    return _build_loss(cfg or {"kind": "cross_entropy"})


def build_tokenizer(cfg: Optional[Dict[str, Any]] = None):
    """Builds the tokenizer from a tokenizer config.

    This is placed under modules/build to keep all model component builders together,
    while delegating actual implementations to ops.tokenizer.
    """
    return _build_tokenizer(cfg or {"kind": "simple_char"})


def build_layer_norm(dim: int, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    eps = cfg.get("eps", 1e-5)
    bias = cfg.get("bias", True)
    mod = LayerNorm(dim, eps=eps, bias=bias)
    return _maybe_freeze(mod, cfg)


def build_token_embedding(vocab_size: int, dim: int, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    mod = TokenEmbedding(vocab_size, dim)
    return _maybe_freeze(mod, cfg)


def build_position_embedding(max_seq_len: int, dim: int, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    mod = PositionEmbedding(max_seq_len, dim)
    return _maybe_freeze(mod, cfg)


def build_emb_dropout(default_p: float, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    p = float(cfg.get("p", default_p))
    return nn.Dropout(p)


def build_transformer_blocks(dim: int, n_layers: int, n_heads: int, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    blocks = [
        TransformerBlock(
            dim=dim,
            n_heads=n_heads,
            context_length=cfg.get("context_length"),
            mlp_mult=cfg.get("mlp_mult", 4),
            dropout=cfg.get("dropout", 0.0),
            activation=cfg.get("activation", "gelu"),
            qkv_bias=cfg.get("qkv_bias", False),
            prenorm=cfg.get("prenorm", True),
        )
        for _ in range(n_layers)
    ]
    mod = nn.Sequential(*blocks)
    return _maybe_freeze(mod, cfg)
