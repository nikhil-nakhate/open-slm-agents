#!/usr/bin/env python3
"""Quick test to verify weight loading works correctly."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.build import build_model_from_cfg
from ops.config import load_config

print("Loading config...")
config_path = Path(__file__).parent.parent / "configs" / "models" / "gpt_oss.yaml"
cfg = load_config(str(config_path))

print("Building model and loading weights...")
model = build_model_from_cfg(cfg)

print(f"\n✓ Model loaded successfully!")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"  Device: {next(model.parameters()).device}")
print(f"  Dtype: {next(model.parameters()).dtype}")

# Quick forward pass test
print("\nTesting forward pass...")
model.eval()
with torch.no_grad():
    test_input = torch.randint(0, model.tokenizer.vocab_size, (1, 5))
    output = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")

print("\n✓ All tests passed!")
