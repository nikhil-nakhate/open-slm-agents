from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from .. import register_model
from ..modules.build import (
    build_token_embedding,
    build_position_embedding,
    build_emb_dropout,
    build_transformer_blocks,
    build_output_projection,
    build_loss,
    build_tokenizer,
    build_layer_norm,
)


@register_model("gpt")
class GPT(nn.Module):
    """GPT-style Transformer language model.

    Expected config structure (model section):
    model:
      name: gpt
      params:
        vocab_size: int
        dim: int
        n_layers: int
        n_heads: int
        max_seq_len: int
        dropout: float
      modules:
        tokenizer: {...}
        embedding: {dropout: float, tie_weights: bool}
        transformer: {dropout: float, ...}
        output_projection: {tie_weights: bool}
        loss: {kind: cross_entropy}
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        modules_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        modules_cfg = modules_cfg or {}

        # Embeddings (separate token and position) + dropout
        tok_emb_cfg = modules_cfg.get("token_embedding", {})
        pos_emb_cfg = modules_cfg.get("position_embedding", {})
        emb_drop_cfg = modules_cfg.get("emb_dropout", {})
        self.tok_emb = build_token_embedding(vocab_size, dim, tok_emb_cfg)
        self.pos_emb = build_position_embedding(max_seq_len, dim, pos_emb_cfg)
        self.drop_emb = build_emb_dropout(dropout, emb_drop_cfg)

        # Transformer blocks (sequential)
        tf_cfg = modules_cfg.get("transformer", {})
        if "dropout" not in tf_cfg:
            tf_cfg["dropout"] = dropout
        tf_cfg.setdefault("context_length", max_seq_len)
        self.trf_blocks = build_transformer_blocks(dim, n_layers, n_heads, tf_cfg)

        # Output projection (optionally tie weights) ties to token embedding
        out_cfg = modules_cfg.get("output_projection", {})
        tie_to = getattr(self.tok_emb, "token_emb", None)
        self.out_head = build_output_projection(dim, vocab_size, out_cfg, tie_to=tie_to)

        # Final norm as in many GPT implementations
        self.final_norm = build_layer_norm(dim, modules_cfg.get("final_norm", {}))

        # Buffers for causal LM head convenience
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Loss
        loss_cfg = modules_cfg.get("loss", {"kind": "cross_entropy"})
        self.loss_fn = build_loss(loss_cfg)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GPT":
        model_cfg = cfg.get("model", {})
        params = model_cfg.get("params", {})
        print(params)
        modules_cfg = model_cfg.get("modules", {})

        # Build tokenizer first and infer vocab if needed
        tok_cfg = modules_cfg.get("tokenizer", {"kind": "simple_char"})
        tokenizer = build_tokenizer(tok_cfg)
        if "vocab_size" not in params:
            params["vocab_size"] = tokenizer.vocab_size
        # Prefer transformer-local config for these
        tf_cfg_for_init = modules_cfg.get("transformer", {})
        dim = tf_cfg_for_init.get("dim", params.get("dim"))
        n_layers = tf_cfg_for_init.get("n_layers", params.get("n_layers"))
        n_heads = tf_cfg_for_init.get("n_heads", params.get("n_heads"))
        if dim is None or n_layers is None or n_heads is None:
            raise KeyError("Transformer config must include dim, n_layers and n_heads (under model.modules.transformer)")
        init_args = dict(
            vocab_size=params["vocab_size"],
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=params["max_seq_len"],
            dropout=params.get("dropout", 0.0),
            modules_cfg=modules_cfg,
        )
        model = cls(**init_args)
        # Attach tokenizer to model for downstream use (e.g., datasets/eval)
        model.tokenizer = tokenizer
        # Validate attachments / perform any extra setup
        model._post_init()
        return model

    def _post_init(self):
        # Ensure tokenizer is present when built via from_config
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError(
                "Tokenizer must be attached to the model. Ensure model.modules.tokenizer is configured and built."
            )

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # idx: [B, T]
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(T, idx.device)
        x = tok + pos
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        if targets is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, targets)
            return logits, loss
        return logits
