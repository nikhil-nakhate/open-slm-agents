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

def load_gpt_oss_weights(model: GPTOSSBackbone, weights_path: Path, device: torch.device, low_memory: bool = True) -> None:
    """Load GPT-OSS weights from HuggingFace checkpoint.

    Args:
        model: Model to load weights into (should already be on target device)
        weights_path: Path to weights directory
        device: Device to load weights on (deprecated, inferred from model)
        low_memory: If True, loads weights directly into model to save memory
    """
    print(f"Loading GPT-OSS weights from {weights_path}")

    # Find checkpoint file
    if weights_path.is_dir():
        # Look for sharded safetensors
        index_file = weights_path / "model.safetensors.index.json"
        if index_file.exists():
            if low_memory:
                # Load weights directly into model to save memory
                _load_weights_low_memory(model, index_file)
            else:
                # Traditional loading (uses more memory)
                # Infer device from model parameters
                model_device = next(model.parameters()).device
                state_dict = load_safetensors_sharded(index_file, model_device)
                state_dict = remap_hf_to_official(state_dict)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"Warning: Missing keys: {missing[:5]}...")
                if unexpected:
                    print(f"Warning: Unexpected keys: {unexpected[:5]}...")
                print(f"✓ Loaded {len(state_dict)} tensors")
        else:
            raise FileNotFoundError(f"No checkpoint found in {weights_path}")
    else:
        raise ValueError(f"Expected directory, got {weights_path}")


def _load_weights_low_memory(model: GPTOSSBackbone, index_file: Path) -> None:
    """Load weights directly into model parameters to minimize memory usage.

    Args:
        model: Model to load weights into (should already be on target device)
        index_file: Path to the safetensors index JSON file
    """
    import json
    import torch
    from safetensors import safe_open
    from ..modules.gpt_oss_utils import dequantize_mxfp4

    with open(index_file, "r") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    shards: Dict[str, set] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, set()).add(name)

    print(f"Loading weights from {len(shards)} shard(s) (low-memory mode)...")

    # Build model parameter map
    model_params = {name: param for name, param in model.named_parameters()}
    print(f"Model has {len(model_params)} parameters")

    loaded_count = 0
    missing_mappings = set()
    processed_keys = set()

    for shard_idx, (shard_name, names) in enumerate(shards.items()):
        print(f"  Loading shard {shard_idx + 1}/{len(shards)}: {shard_name}")
        shard_path = index_file.parent / shard_name

        # Load to CPU first to avoid GPU OOM during dequantization
        # (safer for large models with limited VRAM)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_key in names:
                if hf_key in processed_keys:
                    continue

                # Skip _scales keys - they're handled together with _blocks
                if hf_key.endswith("_scales"):
                    processed_keys.add(hf_key)
                    continue

                # Determine the base key and whether it's quantized
                is_quantized = hf_key.endswith("_blocks")
                base_hf_key = hf_key[:-7] if is_quantized else hf_key

                # Map HF key to official model parameter name
                official_key = _map_hf_key_to_official(base_hf_key)

                if not official_key:
                    processed_keys.add(hf_key)
                    continue

                if official_key not in model_params:
                    missing_mappings.add(f"{base_hf_key} -> {official_key}")
                    processed_keys.add(hf_key)
                    if is_quantized:
                        processed_keys.add(base_hf_key + "_scales")
                    continue

                # Load the tensor
                if is_quantized:
                    # Load and dequantize quantized weights
                    scales_key = base_hf_key + "_scales"
                    blocks = f.get_tensor(hf_key)
                    scales = f.get_tensor(scales_key)
                    tensor = dequantize_mxfp4(blocks, scales, dtype=torch.bfloat16)
                    del blocks, scales
                    processed_keys.add(hf_key)
                    processed_keys.add(scales_key)
                else:
                    # Load regular unquantized weights
                    tensor = f.get_tensor(hf_key)
                    processed_keys.add(hf_key)

                # Copy tensor into model parameter
                param = model_params[official_key]
                if tensor.shape == param.shape:
                    # Move tensor to the same device and dtype as the parameter
                    # This is critical for CUDA compatibility
                    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                    loaded_count += 1
                else:
                    print(f"  Warning: Shape mismatch for {official_key}: {tensor.shape} vs {param.shape}")

                del tensor

    print(f"✓ Loaded {loaded_count} tensors into model")

    if missing_mappings:
        print(f"⚠ Warning: {len(missing_mappings)} keys could not be mapped to model parameters")
        if len(missing_mappings) <= 10:
            for mapping in sorted(missing_mappings)[:10]:
                print(f"  - {mapping}")

    # Check what wasn't loaded
    loaded_params = set()
    for hf_key in processed_keys:
        if hf_key.endswith("_scales"):
            continue
        # Strip _blocks suffix if present before mapping
        base_key = hf_key[:-7] if hf_key.endswith("_blocks") else hf_key
        official_key = _map_hf_key_to_official(base_key)
        if official_key:
            loaded_params.add(official_key)

    unloaded = set(model_params.keys()) - loaded_params
    if unloaded:
        print(f"⚠ Warning: {len(unloaded)} model parameters were not loaded")
        for param_name in sorted(unloaded)[:5]:
            print(f"  - {param_name}")


def _map_hf_key_to_official(hf_key: str) -> Optional[str]:
    """Map a single HF key to official format."""
    # Embedding and output layers
    if hf_key == "model.embed_tokens.weight":
        return "embedding.weight"
    if hf_key == "model.norm.weight":
        return "norm.scale"
    if hf_key == "lm_head.weight":
        return "unembedding.weight"

    # Transformer layers
    if not hf_key.startswith("model.layers."):
        return None

    remainder = hf_key[len("model.layers."):]
    parts = remainder.split(".", 1)
    if len(parts) < 2:
        return None

    layer_idx, sub = parts
    base = f"block.{layer_idx}."

    # Map attention weights
    mapping = {
        "input_layernorm.weight": "attn.norm.scale",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.q_proj.bias": "attn.q_proj.bias",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.k_proj.bias": "attn.k_proj.bias",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.v_proj.bias": "attn.v_proj.bias",
        "self_attn.o_proj.weight": "attn.out.weight",
        "self_attn.o_proj.bias": "attn.out.bias",
        "self_attn.sinks": "attn.sink_logits",
        "post_attention_layernorm.weight": "mlp.norm.scale",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.bias": "mlp.gate.bias",
        "mlp.experts.gate_up_proj": "mlp.mlp1_weight",
        "mlp.experts.gate_up_proj_bias": "mlp.mlp1_bias",
        "mlp.experts.down_proj": "mlp.mlp2_weight",
        "mlp.experts.down_proj_bias": "mlp.mlp2_bias",
    }

    if sub in mapping:
        return base + mapping[sub]

    return None


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

    def reset_cache(self):
        """Reset KV cache in all attention layers.

        Call this before starting a new generation sequence.
        """
        for layer in self.backbone.block:
            layer.attn.reset_cache()

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
            # Make path absolute if it's relative
            if not weights_path.is_absolute():
                # Try to resolve relative to current working directory
                if not weights_path.exists():
                    # If not found, try relative to project root (parent of models/)
                    project_root = Path(__file__).parent.parent.parent
                    alt_path = project_root / weights_path
                    if alt_path.exists():
                        weights_path = alt_path
            device = next(model.parameters()).device
            load_gpt_oss_weights(model.backbone, weights_path, device)

        return model
