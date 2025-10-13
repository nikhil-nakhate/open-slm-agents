from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .. import register_model
from ..modules.activations import SwiGLU
from ..modules.build import (
    build_emb_dropout,
    build_loss,
    build_output_projection,
    build_token_embedding,
    build_tokenizer,
)
from ..modules.rms_norm import RMSNorm
# reuse modular components
from ..modules.gqa import GroupedQueryAttention
from ..modules.moe import MoEMLP
from ..modules.rope import RotaryEmbedding


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        dropout: float = 0.0,
        swiglu_limit: float = 7.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate_size * 2)
        self.act = SwiGLU(limit=swiglu_limit)
        self.fc2 = nn.Linear(intermediate_size, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = self.act(hidden)
        hidden = self.fc2(hidden)
        return self.dropout(hidden)


class GPTOSSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        context_length: int,
        attn_cfg: Dict[str, Any],
        mlp_cfg: Dict[str, Any],
        norm_cfg: Dict[str, Any],
        dropout: float,
        resid_dropout: float,
        layer_idx: int,
    ) -> None:
        super().__init__()
        eps = norm_cfg.get("eps", 1e-5)
        attn_dropout = attn_cfg.get("dropout", dropout)
        attn_resid_dropout = attn_cfg.get("resid_dropout", resid_dropout)
        rope_cfg = attn_cfg.get("rope", {})
        rope = None
        if rope_cfg is not None and rope_cfg.get("kind", "rope") != "none":
            rope_dtype_cfg = rope_cfg.get("dtype", None)
            if isinstance(rope_dtype_cfg, str):
                dtype = getattr(torch, rope_dtype_cfg)
            elif isinstance(rope_dtype_cfg, torch.dtype):
                dtype = rope_dtype_cfg
            else:
                dtype = torch.float32
            rope = RotaryEmbedding(
                head_dim=head_dim,
                base=rope_cfg.get("theta", rope_cfg.get("base", 10000.0)),
                scaling_factor=rope_cfg.get("scaling_factor", 1.0),
                initial_context_length=rope_cfg.get("initial_context_length", context_length),
                ntk_alpha=rope_cfg.get("ntk_alpha", 1.0),
                ntk_beta=rope_cfg.get("ntk_beta", 32.0),
                dtype=dtype,
            )
        sliding_window = attn_cfg.get("sliding_window", None)
        sink_init = attn_cfg.get("sink_init", 0.0)
        self.attn_norm = RMSNorm(dim, eps=eps)
        self.attn = GroupedQueryAttention(
            dim=dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            context_length=context_length,
            dropout=attn_dropout,
            qkv_bias=attn_cfg.get("qkv_bias", False),
            rope=rope,
            sliding_window=sliding_window,
            sink_init=sink_init,
        )
        self.attn_dropout = nn.Dropout(attn_resid_dropout)

        mlp_kind = mlp_cfg.get("kind", "moe")
        mlp_dropout = mlp_cfg.get("dropout", dropout)
        mlp_resid_dropout = mlp_cfg.get("resid_dropout", resid_dropout)
        if mlp_kind == "moe":
            intermediate = mlp_cfg.get("intermediate_size", mlp_cfg.get("hidden_size", dim * 4))
            num_experts = mlp_cfg.get("num_experts", 8)
            experts_per_token = mlp_cfg.get("experts_per_token", 2)
            swiglu_limit = mlp_cfg.get("swiglu_limit", 7.0)
            normalized_input = mlp_cfg.get("normalized_input", False)
            self.mlp = MoEMLP(
                dim=dim,
                intermediate_size=intermediate,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                swiglu_limit=swiglu_limit,
                dropout=mlp_dropout,
                normalized_input=normalized_input,
            )
            self.mlp_norm = RMSNorm(dim, eps=eps)
        else:
            intermediate = mlp_cfg.get("intermediate_size", mlp_cfg.get("mlp_mult", 4) * dim)
            swiglu_limit = mlp_cfg.get("swiglu_limit", 7.0)
            self.mlp = SwiGLUFFN(dim, intermediate, dropout=mlp_dropout, swiglu_limit=swiglu_limit)
            self.mlp_norm = RMSNorm(dim, eps=eps)
        self.mlp_dropout = nn.Dropout(mlp_resid_dropout)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_dropout(attn_out)
        mlp_in = self.mlp_norm(x)
        mlp_out = self.mlp(mlp_in)
        x = x + self.mlp_dropout(mlp_out)
        return x


def build_final_norm(dim: int, cfg: Optional[Dict[str, Any]] = None) -> RMSNorm:
    cfg = cfg or {}
    eps = cfg.get("eps", 1e-5)
    return RMSNorm(dim, eps=eps)


def build_gpt_oss_blocks(dim: int, n_layers: int, n_heads: int, cfg: Dict[str, Any]) -> nn.ModuleList:
    cfg = dict(cfg or {})
    n_kv_heads = cfg.get("n_kv_heads", cfg.get("num_key_value_heads", n_heads))
    head_dim = cfg.get("head_dim", dim // n_heads)
    context_length = cfg.get("context_length")
    if context_length is None:
        raise KeyError("transformer config must include context_length for gpt_oss")
    dropout = cfg.get("dropout", 0.0)
    resid_dropout = cfg.get("resid_dropout", dropout)
    attn_cfg_base = deepcopy(cfg.get("attention") or {})
    mlp_cfg_base = deepcopy(cfg.get("mlp") or {})
    norm_cfg = dict(cfg.get("norm", {}))
    sliding_window_base = attn_cfg_base.get("sliding_window", None)

    blocks = nn.ModuleList()
    for layer_idx in range(n_layers):
        attn_cfg = deepcopy(attn_cfg_base)
        if sliding_window_base is not None:
            attn_cfg["sliding_window"] = sliding_window_base if (layer_idx % 2 == 0) else 0
        mlp_cfg = deepcopy(mlp_cfg_base)
        block = GPTOSSBlock(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            context_length=context_length,
            attn_cfg=attn_cfg,
            mlp_cfg=mlp_cfg,
            norm_cfg=norm_cfg,
            dropout=dropout,
            resid_dropout=resid_dropout,
            layer_idx=layer_idx,
        )
        blocks.append(block)
    return blocks


@register_model("gpt_oss")
class GPTOSS(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        head_dim: Optional[int],
        max_seq_len: int,
        dropout: float = 0.0,
        modules_cfg: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        modules_cfg = modules_cfg or {}
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.head_dim = head_dim or (dim // n_heads)
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        tok_emb_cfg = modules_cfg.get("token_embedding", {})
        self.tok_emb = build_token_embedding(vocab_size, dim, tok_emb_cfg)

        pos_emb_cfg = modules_cfg.get("position_embedding")
        self.pos_emb = None
        if pos_emb_cfg:
            from ..modules.build import build_position_embedding  # local import to avoid cycle

            self.pos_emb = build_position_embedding(max_seq_len, dim, pos_emb_cfg)

        self.drop_emb = build_emb_dropout(dropout, modules_cfg.get("emb_dropout"))

        tf_cfg = dict(modules_cfg.get("transformer", {}))
        tf_cfg.setdefault("context_length", max_seq_len)
        tf_cfg.setdefault("dropout", dropout)
        tf_cfg.setdefault("n_kv_heads", n_kv_heads)
        tf_cfg.setdefault("head_dim", self.head_dim)
        self.blocks = build_gpt_oss_blocks(dim, n_layers, n_heads, tf_cfg)

        self.final_norm = build_final_norm(dim, modules_cfg.get("final_norm"))

        out_cfg = modules_cfg.get("output_projection", {})
        tie_to = getattr(self.tok_emb, "token_emb", None)
        self.out_head = build_output_projection(dim, vocab_size, out_cfg, tie_to=tie_to)

        loss_cfg = modules_cfg.get("loss", {"kind": "cross_entropy"})
        self.loss_fn = build_loss(loss_cfg)

        if weights is not None:
            self.load_state_dict(weights, strict=False)

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GPTOSS":
        model_cfg = cfg.get("model", {})
        params = dict(model_cfg.get("params", {}))
        modules_cfg = dict(model_cfg.get("modules", {}))
        weights_path = model_cfg.get("weights")
        weights = None
        if weights_path is not None:
            weights = torch.load(str(weights_path), weights_only=False)

        tok_cfg = modules_cfg.get("tokenizer", {"kind": "simple_char"})
        tokenizer = build_tokenizer(tok_cfg)
        params.setdefault("vocab_size", tokenizer.vocab_size)

        tf_cfg = modules_cfg.get("transformer", {})
        dim = tf_cfg.get("dim", params.get("dim"))
        n_layers = tf_cfg.get("n_layers", params.get("n_layers"))
        n_heads = tf_cfg.get("n_heads", params.get("n_heads"))
        n_kv_heads = tf_cfg.get(
            "n_kv_heads",
            tf_cfg.get("num_key_value_heads", params.get("n_kv_heads", params.get("num_key_value_heads"))),
        )
        head_dim = tf_cfg.get("head_dim", params.get("head_dim"))
        max_seq_len = params.get("max_seq_len")
        dropout = params.get("dropout", model_cfg.get("dropout", 0.0))

        required = {"dim": dim, "n_layers": n_layers, "n_heads": n_heads, "max_seq_len": max_seq_len}
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise KeyError(f"Missing required transformer parameters for gpt_oss: {missing}")

        init_args = dict(
            vocab_size=params["vocab_size"],
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            modules_cfg=modules_cfg,
            weights=weights,
        )
        model = cls(**init_args)
        model.tokenizer = tokenizer
        return model

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.ndim != 2:
            raise ValueError("Expected input shape [batch, seq_len]")
        batch_size, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model max_seq_len={self.max_seq_len}")

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = self.pos_emb(seq_len, idx.device)
            x = x + pos
        x = self.drop_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        if targets is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, targets)
            return logits, loss
        return logits
