"""Unit tests for models.modules.layer_norm module."""

import pytest
import torch
import torch.nn as nn

from models.modules.layer_norm import LayerNorm


class TestLayerNorm:
    """Test the LayerNorm class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        layer_norm = LayerNorm(dim=512)
        assert layer_norm.eps == 1e-5
        assert layer_norm.weight.shape == (512,)
        assert layer_norm.bias.shape == (512,)
        assert torch.allclose(layer_norm.weight, torch.ones(512))
        assert torch.allclose(layer_norm.bias, torch.zeros(512))

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        layer_norm = LayerNorm(dim=256, eps=1e-6, bias=False)
        assert layer_norm.eps == 1e-6
        assert layer_norm.weight.shape == (256,)
        assert layer_norm.bias is None
        assert torch.allclose(layer_norm.weight, torch.ones(256))

    def test_init_with_bias(self):
        """Test initialization with bias enabled."""
        layer_norm = LayerNorm(dim=128, bias=True)
        assert layer_norm.bias is not None
        assert layer_norm.bias.shape == (128,)
        assert torch.allclose(layer_norm.bias, torch.zeros(128))

    def test_forward_2d_input(self):
        """Test forward pass with 2D input."""
        layer_norm = LayerNorm(dim=512)
        x = torch.randn(32, 512)
        result = layer_norm(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_forward_3d_input(self):
        """Test forward pass with 3D input."""
        layer_norm = LayerNorm(dim=512)
        x = torch.randn(2, 10, 512)
        result = layer_norm(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_forward_4d_input(self):
        """Test forward pass with 4D input."""
        layer_norm = LayerNorm(dim=512)
        x = torch.randn(2, 3, 10, 512)
        result = layer_norm(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        layer_norm = LayerNorm(dim=512)
        x = torch.randn(32, 512, requires_grad=True)
        result = layer_norm(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_normalization_property(self):
        """Test that LayerNorm normalizes the last dimension."""
        layer_norm = LayerNorm(dim=512)
        x = torch.randn(32, 512)
        result = layer_norm(x)
        
        # Check that the mean is approximately 0
        mean = result.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        
        # Check that the variance is approximately 1
        var = result.var(dim=-1, unbiased=False)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

    def test_forward_with_bias(self):
        """Test forward pass with bias enabled."""
        layer_norm = LayerNorm(dim=512, bias=True)
        x = torch.randn(32, 512)
        result = layer_norm(x)
        
        assert result.shape == x.shape
        # With bias, the result should not be exactly normalized
        # but should still be close to normalized

    def test_forward_without_bias(self):
        """Test forward pass without bias."""
        layer_norm = LayerNorm(dim=512, bias=False)
        x = torch.randn(32, 512)
        result = layer_norm(x)
        
        assert result.shape == x.shape
        # Without bias, the result should be normalized
        mean = result.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_forward_eps_parameter(self):
        """Test that eps parameter affects numerical stability."""
        x = torch.tensor([[1.0, 2.0, 3.0]])  # Different values to avoid constant input
        
        # With default eps
        layer_norm1 = LayerNorm(dim=3, eps=1e-5)
        result1 = layer_norm1(x)
        
        # With larger eps
        layer_norm2 = LayerNorm(dim=3, eps=1e-1)
        result2 = layer_norm2(x)
        
        # Results should be different due to different eps values
        assert not torch.allclose(result1, result2)

    def test_forward_constant_input(self):
        """Test forward pass with constant input."""
        layer_norm = LayerNorm(dim=512)
        x = torch.ones(32, 512) * 5.0  # Constant input
        result = layer_norm(x)
        
        # With constant input, result should be close to zero
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_forward_single_element(self):
        """Test forward pass with single element."""
        layer_norm = LayerNorm(dim=1)
        x = torch.randn(32, 1)
        result = layer_norm(x)
        
        assert result.shape == x.shape
        # With single element, result should be close to zero
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_forward_gradient_flow(self):
        """Test that gradients flow properly through LayerNorm."""
        layer_norm = LayerNorm(dim=512)
        x = torch.randn(32, 512, requires_grad=True)
        result = layer_norm(x)
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert layer_norm.weight.grad is not None
        if layer_norm.bias is not None:
            assert layer_norm.bias.grad is not None

    def test_forward_batch_independence(self):
        """Test that LayerNorm operates independently on each batch element."""
        layer_norm = LayerNorm(dim=512)
        x = torch.randn(2, 512)
        result = layer_norm(x)
        
        # Each batch element should be normalized independently
        for i in range(x.shape[0]):
            mean = result[i].mean()
            var = result[i].var(unbiased=False)
            assert torch.allclose(mean, torch.tensor(0.0), atol=1e-5)
            assert torch.allclose(var, torch.tensor(1.0), atol=1e-5)

    def test_forward_against_pytorch_layernorm(self):
        """Test that our LayerNorm matches PyTorch's LayerNorm."""
        dim = 512
        x = torch.randn(32, dim)
        
        our_layer_norm = LayerNorm(dim=dim)
        pytorch_layer_norm = nn.LayerNorm(dim)
        
        # Initialize PyTorch LayerNorm with same parameters
        pytorch_layer_norm.weight.data = our_layer_norm.weight.data.clone()
        pytorch_layer_norm.bias.data = our_layer_norm.bias.data.clone()
        
        our_result = our_layer_norm(x)
        pytorch_result = pytorch_layer_norm(x)
        
        # Results should be very close
        assert torch.allclose(our_result, pytorch_result, atol=1e-6)

    def test_forward_with_different_dims(self):
        """Test forward pass with different dimension sizes."""
        for dim in [1, 10, 100, 1000]:
            layer_norm = LayerNorm(dim=dim)
            x = torch.randn(16, dim)
            result = layer_norm(x)
            assert result.shape == x.shape
