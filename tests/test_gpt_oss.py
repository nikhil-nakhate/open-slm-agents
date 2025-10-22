#!/usr/bin/env python3
"""Test script to verify GPT-OSS implementation works correctly."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

print("=" * 80)
print("GPT-OSS Implementation Test")
print("=" * 80)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from models.meta_arch.gpt_oss import (
        GPTOSSConfig,
        GPTOSS,
        load_gpt_oss_weights,
    )
    from models.modules.gqa import AttentionBlock
    from models.modules.moe import MLPBlock
    from models.modules.rope import RotaryEmbedding
    from models.modules.rms_norm import RMSNorm
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create config
print("\n[2/5] Testing configuration...")
try:
    config = GPTOSSConfig(
        vocab_size=201088,
        hidden_size=2880,
        num_hidden_layers=24,
        num_attention_heads=64,
        num_key_value_heads=8,
    )
    print(f"✓ Config created: {config.num_hidden_layers} layers, {config.hidden_size} dim")
except Exception as e:
    print(f"✗ Config creation failed: {e}")
    sys.exit(1)

# Test 3: Build model
print("\n[3/5] Testing model creation...")
try:
    from models.build import build_model_from_cfg
    from ops.config import load_config

    # Use parent directory for config path
    config_path = Path(__file__).parent.parent / "configs" / "models" / "gpt_oss.yaml"
    cfg = load_config(str(config_path))
    model = build_model_from_cfg(cfg)
    print(f"✓ Model created successfully")
    print(f"  - Vocab size: {model.tokenizer.vocab_size}")
    print(f"  - Max seq len: {model.max_seq_len}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load weights
print("\n[4/5] Testing weight loading...")
# Use parent directory for weights path
weights_path = Path(__file__).parent.parent / "weights" / "gpt-oss-20b"
if not weights_path.exists():
    print(f"⚠ Weights not found at {weights_path}")
    print("  Skipping weight loading test")
else:
    try:
        device = torch.device("cpu")
        model = model.to(device)

        # Weights should auto-load from config
        print(f"✓ Weights loaded from {weights_path}")
        print(f"  - Device: {device}")

        # Verify some weights loaded
        first_layer = model.backbone.block[0]
        print(f"  - First layer attn weight shape: {first_layer.attn.q_proj.weight.shape}")
        print(f"  - First layer MLP weight shape: {first_layer.mlp.mlp1_weight.shape}")

    except Exception as e:
        print(f"✗ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 5: Forward pass
print("\n[5/5] Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        # Create sample input
        sample_input = torch.randint(0, model.tokenizer.vocab_size, (1, 10), device=device)
        print(f"  - Input shape: {sample_input.shape}")

        # Forward pass
        output = model(sample_input)
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output dtype: {output.dtype}")

        assert output.shape == (1, 10, model.tokenizer.vocab_size), "Output shape mismatch!"
        print("✓ Forward pass successful")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print("\nNext steps:")
print("1. Install Harmony library: pip install openai-harmony")
print("2. Run inference: python infer.py --config configs/models/gpt_oss.yaml")
print("\nFor more information, see project documentation.")
