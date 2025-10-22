"""Utility functions for GPT-OSS model.

This module contains helper functions for:
- MXFP4 weight dequantization
- Rotary position embeddings
- Weight loading and remapping
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 quantized weights to float32.

    Args:
        blocks: Quantized 4-bit blocks [..., num_blocks, 16] (uint8)
        scales: Scaling factors [..., num_blocks] (uint8)

    Returns:
        Dequantized weights in float32 with shape [..., num_blocks * 32]
    """
    # Unpack 4-bit values from uint8 blocks
    # Each uint8 contains 2 4-bit values
    low = blocks & 0x0F
    high = (blocks >> 4) & 0x0F
    packed = torch.stack([low, high], dim=-1).flatten(-2, -1)  # [..., num_blocks, 32]

    # Decode MXFP4 format
    sign = (packed >> 3) & 0x1
    exponent = (packed >> 1) & 0x3
    mantissa = packed & 0x1

    is_zero = (exponent == 0) & (mantissa == 0)
    is_subnormal = (exponent == 0) & (mantissa == 1)

    subnormal = torch.full_like(packed, 0.25, dtype=torch.float32)
    normal = torch.pow(2.0, exponent.to(torch.float32) - 1.0) * (1.0 + mantissa.to(torch.float32) * 0.5)
    values = torch.where(is_zero, torch.zeros_like(normal), torch.where(is_subnormal, subnormal, normal))
    values = values * (1 - 2 * sign.to(torch.float32))

    # Apply scale (one scale per block of 32 elements)
    scale = torch.where(
        scales == 0,
        torch.zeros_like(scales, dtype=torch.float32),
        torch.pow(2.0, scales.to(torch.float32) - 127.0),
    )
    values = values * scale.unsqueeze(-1)  # Broadcast scale across the 32 elements

    # Flatten the last two dimensions (num_blocks, 32) -> (num_blocks * 32)
    new_shape = list(values.shape[:-2]) + [-1]
    return values.reshape(new_shape)


def load_safetensors_sharded(index_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load sharded safetensors checkpoint with MXFP4 dequantization.

    Args:
        index_path: Path to model.safetensors.index.json
        device: Device to load tensors on

    Returns:
        Dictionary of parameter name -> tensor
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors package required. Install with: pip install safetensors")

    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    shards: Dict[str, set] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, set()).add(name)

    tensors: Dict[str, torch.Tensor] = {}
    processed_keys = set()  # Track which keys we've already processed

    for shard_name, names in shards.items():
        shard_path = index_path.parent / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard {shard_name} not found at {shard_path}")

        with safe_open(shard_path, framework="pt", device=str(device)) as f:
            for key in names:
                if key in processed_keys:
                    continue

                if key.endswith("_blocks"):
                    # Quantized weight - dequantize it
                    base = key[:-7]  # Remove "_blocks"
                    scales_key = base + "_scales"

                    blocks = f.get_tensor(key)
                    scales = f.get_tensor(scales_key)
                    tensors[base] = dequantize_mxfp4(blocks, scales).to(device)

                    # Mark both _blocks and _scales as processed
                    processed_keys.add(key)
                    processed_keys.add(scales_key)
                elif not key.endswith("_scales"):
                    # Regular weight (not quantized)
                    tensors[key] = f.get_tensor(key).to(device)
                    processed_keys.add(key)

    return tensors


def remap_hf_to_official(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap HuggingFace checkpoint keys to official GPT-OSS keys.

    Args:
        state_dict: Dictionary with HuggingFace format keys

    Returns:
        Dictionary with official GPT-OSS format keys
    """
    mapped: Dict[str, torch.Tensor] = {}

    # Embedding and output layers
    if "model.embed_tokens.weight" in state_dict:
        mapped["embedding.weight"] = state_dict["model.embed_tokens.weight"]
    if "model.norm.weight" in state_dict:
        mapped["norm.scale"] = state_dict["model.norm.weight"]
    if "lm_head.weight" in state_dict:
        mapped["unembedding.weight"] = state_dict["lm_head.weight"]

    # Transformer layers
    layer_prefix = "model.layers."
    for key, tensor in state_dict.items():
        if not key.startswith(layer_prefix):
            continue

        parts = key[len(layer_prefix):].split(".", 1)
        if len(parts) < 2:
            continue

        layer_idx, remainder = parts
        base = f"block.{layer_idx}."

        # Map attention weights
        if remainder == "input_layernorm.weight":
            mapped[base + "attn.norm.scale"] = tensor
        elif remainder.startswith("self_attn."):
            sub = remainder[len("self_attn."):]
            mapping = {
                "q_proj.weight": "attn.q_proj.weight",
                "q_proj.bias": "attn.q_proj.bias",
                "k_proj.weight": "attn.k_proj.weight",
                "k_proj.bias": "attn.k_proj.bias",
                "v_proj.weight": "attn.v_proj.weight",
                "v_proj.bias": "attn.v_proj.bias",
                "o_proj.weight": "attn.out.weight",
                "o_proj.bias": "attn.out.bias",
                "sinks": "attn.sink_logits",
            }
            if sub in mapping:
                mapped[base + mapping[sub]] = tensor

        # Map MLP weights
        elif remainder == "post_attention_layernorm.weight":
            mapped[base + "mlp.norm.scale"] = tensor
        elif remainder == "mlp.router.weight":
            mapped[base + "mlp.gate.weight"] = tensor
        elif remainder == "mlp.router.bias":
            mapped[base + "mlp.gate.bias"] = tensor
        elif remainder == "mlp.experts.gate_up_proj":
            mapped[base + "mlp.mlp1_weight"] = tensor
        elif remainder == "mlp.experts.gate_up_proj_bias":
            mapped[base + "mlp.mlp1_bias"] = tensor
        elif remainder == "mlp.experts.down_proj":
            mapped[base + "mlp.mlp2_weight"] = tensor
        elif remainder == "mlp.experts.down_proj_bias":
            mapped[base + "mlp.mlp2_bias"] = tensor

    return mapped
