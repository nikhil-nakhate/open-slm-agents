from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import numpy as np

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
        weights: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        modules_cfg = modules_cfg or {}

        # Embeddings + dropout
        tok_emb_cfg = modules_cfg.get("token_embedding", {})
        pos_emb_cfg = modules_cfg.get("position_embedding", {})
        emb_drop_cfg = modules_cfg.get("emb_dropout", {})
        self.tok_emb = build_token_embedding(vocab_size, dim, tok_emb_cfg)
        self.pos_emb = build_position_embedding(max_seq_len, dim, pos_emb_cfg)
        self.drop_emb = build_emb_dropout(dropout, emb_drop_cfg)

        # Transformer blocks
        tf_cfg = modules_cfg.get("transformer", {})
        if "dropout" not in tf_cfg:
            tf_cfg["dropout"] = dropout
        tf_cfg.setdefault("context_length", max_seq_len)
        self.trf_blocks = build_transformer_blocks(dim, n_layers, n_heads, tf_cfg)

        out_cfg = modules_cfg.get("output_projection", {})
        tie_to = getattr(self.tok_emb, "token_emb", None)
        self.out_head = build_output_projection(dim, vocab_size, out_cfg, tie_to=tie_to)

        self.final_norm = build_layer_norm(dim, modules_cfg.get("final_norm", {}))

        if weights is not None:
            self.load_weights_into_gpt(weights)

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Loss
        loss_cfg = modules_cfg.get("loss", {"kind": "cross_entropy"})
        self.loss_fn = build_loss(loss_cfg)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GPT":
        model_cfg = cfg.get("model", {})
        params = model_cfg.get("params", {})
        modules_cfg = model_cfg.get("modules", {})
        weights_path = model_cfg.get("weights", None)
        weights = None
        if weights_path is not None:
            weights = torch.load(f"{weights_path}", weights_only=False)
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
            weights=weights,
        )
        model = cls(**init_args)
        # Attach tokenizer to model for downstream use
        model.tokenizer = tokenizer

        return model

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

    def assign(self, left: torch.nn.Parameter, right: np.ndarray) -> torch.nn.Parameter:
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError(f"Shape mismatch. Left: {tuple(left.shape)}, Right: {tuple(right.shape)}")
        # Create a Parameter with the same dtype/device as left
        tensor = torch.as_tensor(right, dtype=left.dtype, device=left.device)
        return torch.nn.Parameter(tensor)

    def load_weights_into_gpt(self, params: Dict[str, Any]):
        """Loads converted GPT-2 weights into our GPT model structure.

        This function adapts to our module naming:
        - gpt.tok_emb.token_emb.weight
        - gpt.pos_emb.pos_emb.weight
        - gpt.trf_blocks[b].attn.(W_query/W_key/W_value/out_proj)
        - gpt.trf_blocks[b].mlp.(fc1/fc2)
        - gpt.trf_blocks[b].ln1/ln2.(weight/bias)
        - gpt.final_norm.(weight/bias)
        - gpt.out_head.weight
        """

        # Embeddings
        self.pos_emb.pos_emb.weight = self.assign(self.pos_emb.pos_emb.weight, params["wpe"])
        self.tok_emb.token_emb.weight = self.assign(self.tok_emb.token_emb.weight, params["wte"])

        # Transformer blocks
        for b in range(len(params["blocks"])):
            block = self.trf_blocks[b]

            # Attention qkv split (transpose for PyTorch linear layout)
            q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
            block.attn.W_query.weight = self.assign(block.attn.W_query.weight, q_w.T)
            block.attn.W_key.weight = self.assign(block.attn.W_key.weight, k_w.T)
            block.attn.W_value.weight = self.assign(block.attn.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
            block.attn.W_query.bias = self.assign(block.attn.W_query.bias, q_b)
            block.attn.W_key.bias = self.assign(block.attn.W_key.bias, k_b)
            block.attn.W_value.bias = self.assign(block.attn.W_value.bias, v_b)

            block.attn.out_proj.weight = self.assign(
                block.attn.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
            )
            block.attn.out_proj.bias = self.assign(
                block.attn.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
            )

            # MLP
            block.mlp.fc1.weight = self.assign(block.mlp.fc1.weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            block.mlp.fc1.bias = self.assign(block.mlp.fc1.bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
            block.mlp.fc2.weight = self.assign(block.mlp.fc2.weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            block.mlp.fc2.bias = self.assign(block.mlp.fc2.bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

            # LayerNorms
            block.ln1.weight = self.assign(block.ln1.weight, params["blocks"][b]["ln_1"]["g"])
            block.ln1.bias = self.assign(block.ln1.bias, params["blocks"][b]["ln_1"]["b"])
            block.ln2.weight = self.assign(block.ln2.weight, params["blocks"][b]["ln_2"]["g"])
            block.ln2.bias = self.assign(block.ln2.bias, params["blocks"][b]["ln_2"]["b"])

        # Final norm and output head
        ln_f = params.get("ln_f", {})
        if ln_f:
            self.final_norm.weight = self.assign(self.final_norm.weight, ln_f.get("g"))
            self.final_norm.bias = self.assign(self.final_norm.bias, ln_f.get("b"))
        else:
            # Fallback if already flattened (unlikely)
            self.final_norm.weight = self.assign(self.final_norm.weight, params["g"])
            self.final_norm.bias = self.assign(self.final_norm.bias, params["b"])
        # Output head (may be weight-tied to token embeddings)
        if hasattr(self.out_head, "proj"):
            self.out_head.proj.weight = self.assign(self.out_head.proj.weight, params["wte"])
        else:
            self.out_head.weight = self.assign(self.out_head.weight, params["wte"])
