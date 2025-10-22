"""Clean, modular GPT-OSS implementation following the official torch reference.

This implementation matches the official GPT-OSS repository structure exactly:
https://github.com/openai/gpt-oss/tree/main/gpt_oss/torch

Key features:
- Exact weight compatibility with HuggingFace checkpoints
- Support for sharded safetensors files
- Proper MXFP4 quantized weight dequantization
- Harmony format integration for chat
- Modular architecture with separate component files
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .. import register_model
from ..modules.build import build_tokenizer
from ..modules.gqa import AttentionBlock
from ..modules.moe import MLPBlock
from ..modules.gpt_oss_utils import load_safetensors_sharded, remap_hf_to_official
from ..modules.rms_norm import RMSNorm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GPTOSSConfig:
    """Configuration matching official GPT-OSS parameters."""

    vocab_size: int = 201088
    hidden_size: int = 2880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2880
    num_experts: int = 32
    experts_per_token: int = 4
    swiglu_limit: float = 7.0
    max_position_embeddings: int = 131072
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    sliding_window: int = 128
    rms_norm_eps: float = 1e-5
    dtype: str = "bfloat16"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GPTOSSConfig":
        """Create config from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


# ============================================================================
# Model Layers
# ============================================================================

class GPTOSSLayer(nn.Module):
    """Transformer layer with attention and MLP."""

    def __init__(self, config: GPTOSSConfig, layer_idx: int, dtype: torch.dtype) -> None:
        """Initialize transformer layer.

        Args:
            config: Model configuration
            layer_idx: Layer index
            dtype: Data type for parameters
        """
        super().__init__()
        self.attn = AttentionBlock(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_idx=layer_idx,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_ntk_alpha=config.rope_ntk_alpha,
            rope_ntk_beta=config.rope_ntk_beta,
            sliding_window=config.sliding_window,
            rms_norm_eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.mlp = MLPBlock(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            experts_per_token=config.experts_per_token,
            swiglu_limit=config.swiglu_limit,
            rms_norm_eps=config.rms_norm_eps,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        x = self.attn(x)
        x = self.mlp(x)
        return x


class GPTOSSBackbone(nn.Module):
    """GPT-OSS transformer backbone."""

    def __init__(self, config: GPTOSSConfig, dtype: torch.dtype) -> None:
        """Initialize backbone.

        Args:
            config: Model configuration
            dtype: Data type for parameters
        """
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.block = nn.ModuleList(
            [GPTOSSLayer(config, layer_idx, dtype) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)
        for layer in self.block:
            x = layer(x)
        x = self.norm(x)
        return self.unembedding(x)


# ============================================================================
# Weight Loading
# ============================================================================

def load_gpt_oss_weights(model: GPTOSSBackbone, weights_path: Path, device: torch.device) -> None:
    """Load GPT-OSS weights from HuggingFace checkpoint.

    Args:
        model: Model to load weights into
        weights_path: Path to weights directory
        device: Device to load weights on
    """
    print(f"Loading GPT-OSS weights from {weights_path}")

    # Find checkpoint file
    if weights_path.is_dir():
        # Look for sharded safetensors
        index_file = weights_path / "model.safetensors.index.json"
        if index_file.exists():
            state_dict = load_safetensors_sharded(index_file, device)
        else:
            raise FileNotFoundError(f"No checkpoint found in {weights_path}")
    else:
        raise ValueError(f"Expected directory, got {weights_path}")

    # Remap keys from HF format to official format
    state_dict = remap_hf_to_official(state_dict)

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Warning: Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}...")

    print(f"✓ Loaded {len(state_dict)} tensors")


# ============================================================================
# Model Wrapper
# ============================================================================

@register_model("gpt_oss")
class GPTOSS(nn.Module):
    """GPT-OSS model wrapper with config-driven initialization."""

    def __init__(self, config: GPTOSSConfig, tokenizer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GPT-OSS model.

        Args:
            config: Model configuration
            tokenizer_cfg: Tokenizer configuration (optional)
        """
        super().__init__()
        self.config = config

        # Resolve dtype
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # Build model
        self.backbone = GPTOSSBackbone(config, dtype)

        # Build tokenizer
        tokenizer_cfg = tokenizer_cfg or {"kind": "o200k_harmony"}
        self.tokenizer = build_tokenizer(tokenizer_cfg)
        self.max_seq_len = config.max_position_embeddings

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            targets: Target token IDs for loss computation (optional)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        logits = self.backbone(input_ids)

        if targets is not None:
            # Cross entropy loss (not implemented here for brevity)
            raise NotImplementedError("Loss computation not implemented")

        return logits

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GPTOSS":
        """Create model from config dictionary.

        Args:
            cfg: Configuration dictionary from YAML

        Returns:
            Initialized GPT-OSS model with loaded weights
        """
        model_cfg = cfg.get("model", {})
        params = model_cfg.get("params", {})
        modules_cfg = model_cfg.get("modules", {})

        # Build config
        config = GPTOSSConfig.from_dict(params)

        # Create model
        tokenizer_cfg = modules_cfg.get("tokenizer")
        model = cls(config, tokenizer_cfg)

        # Load weights if specified
        weights_path = model_cfg.get("weights")
        if weights_path:
            weights_path = Path(weights_path)
            device = next(model.parameters()).device
            load_gpt_oss_weights(model.backbone, weights_path, device)

        return model
