"""Unit tests for models.modules.transformer module."""

import pytest
import torch
import torch.nn as nn

from models.modules.transformer import MLP, TransformerBlock, Transformer
from models.modules.activations import build_activation


class TestMLP:
    """Test the MLP class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        mlp = MLP(dim=512)
        assert mlp.fc1.in_features == 512
        assert mlp.fc1.out_features == 2048  # 4 * 512
        assert mlp.fc2.in_features == 2048
        assert mlp.fc2.out_features == 512
        assert isinstance(mlp.act, nn.Module)
        assert isinstance(mlp.drop, nn.Dropout)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        mlp = MLP(dim=256, hidden_mult=2, dropout=0.1, activation="relu")
        assert mlp.fc1.in_features == 256
        assert mlp.fc1.out_features == 512  # 2 * 256
        assert mlp.fc2.in_features == 512
        assert mlp.fc2.out_features == 256
        assert mlp.drop.p == 0.1

    def test_forward(self):
        """Test forward pass."""
        mlp = MLP(dim=512)
        x = torch.randn(2, 10, 512)
        result = mlp(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        mlp = MLP(dim=512)
        x = torch.randn(2, 10, 512, requires_grad=True)
        result = mlp(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_different_activations(self):
        """Test forward pass with different activations."""
        for activation in ["gelu", "relu", "silu"]:
            mlp = MLP(dim=256, activation=activation)
            x = torch.randn(2, 5, 256)
            result = mlp(x)
            assert result.shape == x.shape

    def test_forward_dropout_training_mode(self):
        """Test that dropout is applied in training mode."""
        mlp = MLP(dim=512, dropout=0.5)
        mlp.train()
        x = torch.randn(2, 10, 512)
        result1 = mlp(x)
        result2 = mlp(x)
        # Results should be different due to dropout
        assert not torch.allclose(result1, result2, atol=1e-6)

    def test_forward_dropout_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        mlp = MLP(dim=512, dropout=0.5)
        mlp.eval()
        x = torch.randn(2, 10, 512)
        result1 = mlp(x)
        result2 = mlp(x)
        # Results should be identical in eval mode
        assert torch.allclose(result1, result2, atol=1e-6)


class TestTransformerBlock:
    """Test the TransformerBlock class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        block = TransformerBlock(
            dim=512, n_heads=8, context_length=1024
        )
        assert block.prenorm is True
        assert isinstance(block.ln1, nn.Module)
        assert isinstance(block.attn, nn.Module)
        assert isinstance(block.ln2, nn.Module)
        assert isinstance(block.mlp, nn.Module)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        block = TransformerBlock(
            dim=256, n_heads=4, context_length=512,
            mlp_mult=2, dropout=0.1, activation="relu",
            qkv_bias=True, prenorm=False
        )
        assert block.prenorm is False

    def test_forward_prenorm(self):
        """Test forward pass with prenorm=True."""
        block = TransformerBlock(
            dim=512, n_heads=8, context_length=1024, prenorm=True
        )
        x = torch.randn(2, 10, 512)
        result = block(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_forward_postnorm(self):
        """Test forward pass with prenorm=False."""
        block = TransformerBlock(
            dim=512, n_heads=8, context_length=1024, prenorm=False
        )
        x = torch.randn(2, 10, 512)
        result = block(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        block = TransformerBlock(
            dim=512, n_heads=8, context_length=1024
        )
        x = torch.randn(2, 10, 512, requires_grad=True)
        result = block(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_residual_connection(self):
        """Test that residual connections work correctly."""
        block = TransformerBlock(
            dim=512, n_heads=8, context_length=1024, dropout=0.0
        )
        block.eval()  # Disable dropout for deterministic test
        
        x = torch.randn(2, 10, 512)
        result = block(x)
        
        # The result should be different from input due to transformations
        assert not torch.allclose(result, x, atol=1e-6)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        block = TransformerBlock(
            dim=256, n_heads=4, context_length=512
        )
        
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, 256)
            result = block(x)
            assert result.shape == (2, seq_len, 256)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        block = TransformerBlock(
            dim=256, n_heads=4, context_length=512
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 10, 256)
            result = block(x)
            assert result.shape == (batch_size, 10, 256)


class TestTransformer:
    """Test the Transformer class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        cfg = {
            "dim": 512,
            "n_layers": 6,
            "n_heads": 8,
            "context_length": 1024
        }
        transformer = Transformer(cfg)
        
        assert len(transformer.layers) == 6
        assert transformer.final_ln is not None
        assert isinstance(transformer.final_ln, nn.Module)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        cfg = {
            "dim": 256,
            "n_layers": 4,
            "n_heads": 4,
            "context_length": 512,
            "dropout": 0.1,
            "mlp_mult": 2,
            "activation": "relu",
            "qkv_bias": True,
            "prenorm": False,
            "final_ln": False
        }
        transformer = Transformer(cfg)
        
        assert len(transformer.layers) == 4
        assert transformer.final_ln is None

    def test_forward(self):
        """Test forward pass."""
        cfg = {
            "dim": 512,
            "n_layers": 6,
            "n_heads": 8,
            "context_length": 1024
        }
        transformer = Transformer(cfg)
        x = torch.randn(2, 10, 512)
        result = transformer(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        cfg = {
            "dim": 512,
            "n_layers": 6,
            "n_heads": 8,
            "context_length": 1024
        }
        transformer = Transformer(cfg)
        x = torch.randn(2, 10, 512, requires_grad=True)
        result = transformer(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_with_final_ln(self):
        """Test forward pass with final layer norm."""
        cfg = {
            "dim": 256,
            "n_layers": 4,
            "n_heads": 4,
            "context_length": 512,
            "final_ln": True
        }
        transformer = Transformer(cfg)
        x = torch.randn(2, 10, 256)
        result = transformer(x)
        
        assert result.shape == x.shape
        assert transformer.final_ln is not None

    def test_forward_without_final_ln(self):
        """Test forward pass without final layer norm."""
        cfg = {
            "dim": 256,
            "n_layers": 4,
            "n_heads": 4,
            "context_length": 512,
            "final_ln": False
        }
        transformer = Transformer(cfg)
        x = torch.randn(2, 10, 256)
        result = transformer(x)
        
        assert result.shape == x.shape
        assert transformer.final_ln is None

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        cfg = {
            "dim": 256,
            "n_layers": 4,
            "n_heads": 4,
            "context_length": 512
        }
        transformer = Transformer(cfg)
        
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, 256)
            result = transformer(x)
            assert result.shape == (2, seq_len, 256)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        cfg = {
            "dim": 256,
            "n_layers": 4,
            "n_heads": 4,
            "context_length": 512
        }
        transformer = Transformer(cfg)
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 10, 256)
            result = transformer(x)
            assert result.shape == (batch_size, 10, 256)

    def test_forward_zero_layers(self):
        """Test forward pass with zero layers."""
        cfg = {
            "dim": 256,
            "n_layers": 0,
            "n_heads": 4,
            "context_length": 512
        }
        transformer = Transformer(cfg)
        x = torch.randn(2, 10, 256)
        result = transformer(x)
        
        assert result.shape == x.shape
        assert len(transformer.layers) == 0

    def test_forward_single_layer(self):
        """Test forward pass with single layer."""
        cfg = {
            "dim": 256,
            "n_layers": 1,
            "n_heads": 4,
            "context_length": 512
        }
        transformer = Transformer(cfg)
        x = torch.randn(2, 10, 256)
        result = transformer(x)
        
        assert result.shape == x.shape
        assert len(transformer.layers) == 1

    def test_forward_different_activations(self):
        """Test forward pass with different activations."""
        for activation in ["gelu", "relu", "silu"]:
            cfg = {
                "dim": 256,
                "n_layers": 2,
                "n_heads": 4,
                "context_length": 512,
                "activation": activation
            }
            transformer = Transformer(cfg)
            x = torch.randn(2, 10, 256)
            result = transformer(x)
            assert result.shape == x.shape

    def test_forward_dropout_training_mode(self):
        """Test that dropout is applied in training mode."""
        cfg = {
            "dim": 256,
            "n_layers": 2,
            "n_heads": 4,
            "context_length": 512,
            "dropout": 0.5
        }
        transformer = Transformer(cfg)
        transformer.train()
        x = torch.randn(2, 10, 256)
        result1 = transformer(x)
        result2 = transformer(x)
        # Results should be different due to dropout
        assert not torch.allclose(result1, result2, atol=1e-6)

    def test_forward_dropout_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        cfg = {
            "dim": 256,
            "n_layers": 2,
            "n_heads": 4,
            "context_length": 512,
            "dropout": 0.5
        }
        transformer = Transformer(cfg)
        transformer.eval()
        x = torch.randn(2, 10, 256)
        result1 = transformer(x)
        result2 = transformer(x)
        # Results should be identical in eval mode
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_forward_gradient_flow(self):
        """Test that gradients flow properly through all layers."""
        cfg = {
            "dim": 256,
            "n_layers": 3,
            "n_heads": 4,
            "context_length": 512
        }
        transformer = Transformer(cfg)
        x = torch.randn(2, 10, 256, requires_grad=True)
        result = transformer(x)
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist for input
        assert x.grad is not None
        
        # Check that gradients exist for all layer parameters
        for layer in transformer.layers:
            for param in layer.parameters():
                if param.requires_grad:
                    assert param.grad is not None

    def test_forward_missing_required_params(self):
        """Test that missing required parameters raise appropriate errors."""
        # Missing dim
        with pytest.raises(TypeError):
            Transformer({"n_layers": 6, "n_heads": 8, "context_length": 1024})
        
        # Missing n_layers
        with pytest.raises(TypeError):
            Transformer({"dim": 512, "n_heads": 8, "context_length": 1024})
        
        # Missing n_heads
        with pytest.raises(TypeError):
            Transformer({"dim": 512, "n_layers": 6, "context_length": 1024})
        
        # Missing context_length
        with pytest.raises(TypeError):
            Transformer({"dim": 512, "n_layers": 6, "n_heads": 8})
