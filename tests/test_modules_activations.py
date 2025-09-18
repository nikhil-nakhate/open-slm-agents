"""Unit tests for models.modules.activations module."""

import pytest
import torch
import torch.nn as nn

from models.modules.activations import GELU, SiLU, build_activation


class TestGELU:
    """Test the GELU activation function."""

    def test_forward(self):
        """Test GELU forward pass."""
        gelu = GELU()
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        result = gelu(x)
        
        # GELU should be approximately 0 at x=0
        assert torch.isclose(result[0], torch.tensor(0.0), atol=1e-6)
        
        # GELU should be positive for positive inputs (indices 1 and 3)
        assert result[1] > 0  # x=1.0
        assert result[3] > 0  # x=2.0
        
        # GELU can be negative for negative inputs (index 2)
        # This is expected behavior for GELU
        
        # GELU should be approximately 0 for large negative inputs
        large_negative = torch.tensor([-10.0])
        result_neg = gelu(large_negative)
        assert torch.isclose(result_neg, torch.tensor(0.0), atol=1e-6)

    def test_forward_shape_preservation(self):
        """Test that GELU preserves input shape."""
        gelu = GELU()
        x = torch.randn(2, 3, 4)
        result = gelu(x)
        assert result.shape == x.shape

    def test_forward_differentiable(self):
        """Test that GELU is differentiable."""
        gelu = GELU()
        x = torch.tensor([1.0], requires_grad=True)
        result = gelu(x)
        result.backward()
        assert x.grad is not None

    def test_forward_approximation(self):
        """Test GELU approximation against known values."""
        gelu = GELU()
        x = torch.tensor([0.0])
        result = gelu(x)
        # GELU(0) should be approximately 0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestSiLU:
    """Test the SiLU activation function."""

    def test_forward(self):
        """Test SiLU forward pass."""
        silu = SiLU()
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        result = silu(x)
        
        # SiLU should be 0 at x=0
        assert torch.isclose(result[0], torch.tensor(0.0), atol=1e-6)
        
        # SiLU should be positive for positive inputs (indices 1 and 3)
        assert result[1] > 0  # x=1.0
        assert result[3] > 0  # x=2.0
        
        # SiLU can be negative for negative inputs (index 2)
        # This is expected behavior for SiLU

    def test_forward_shape_preservation(self):
        """Test that SiLU preserves input shape."""
        silu = SiLU()
        x = torch.randn(2, 3, 4)
        result = silu(x)
        assert result.shape == x.shape

    def test_forward_differentiable(self):
        """Test that SiLU is differentiable."""
        silu = SiLU()
        x = torch.tensor([1.0], requires_grad=True)
        result = silu(x)
        result.backward()
        assert x.grad is not None

    def test_forward_against_torch_silu(self):
        """Test SiLU against PyTorch's built-in SiLU."""
        silu = SiLU()
        x = torch.randn(10)
        result = silu(x)
        expected = torch.nn.functional.silu(x)
        assert torch.allclose(result, expected, atol=1e-6)


class TestBuildActivation:
    """Test the build_activation function."""

    def test_build_gelu(self):
        """Test building GELU activation."""
        activation = build_activation("gelu")
        assert isinstance(activation, GELU)

    def test_build_gelu_case_insensitive(self):
        """Test building GELU activation with different cases."""
        activation = build_activation("GELU")
        assert isinstance(activation, GELU)

    def test_build_silu(self):
        """Test building SiLU activation."""
        activation = build_activation("silu")
        assert isinstance(activation, SiLU)

    def test_build_swish(self):
        """Test building SiLU activation with 'swish' alias."""
        activation = build_activation("swish")
        assert isinstance(activation, SiLU)

    def test_build_relu(self):
        """Test building ReLU activation."""
        activation = build_activation("relu")
        assert isinstance(activation, nn.ReLU)

    def test_build_activation_default(self):
        """Test building activation with None input (default to GELU)."""
        activation = build_activation(None)
        assert isinstance(activation, GELU)

    def test_build_activation_empty_string(self):
        """Test building activation with empty string (default to GELU)."""
        activation = build_activation("")
        assert isinstance(activation, GELU)

    def test_build_activation_unknown(self):
        """Test building activation with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation: unknown"):
            build_activation("unknown")

    def test_build_activation_case_insensitive_silu(self):
        """Test building SiLU activation with different cases."""
        activation = build_activation("SILU")
        assert isinstance(activation, SiLU)

    def test_build_activation_case_insensitive_relu(self):
        """Test building ReLU activation with different cases."""
        activation = build_activation("RELU")
        assert isinstance(activation, nn.ReLU)

    def test_build_activation_functionality(self):
        """Test that built activations work correctly."""
        # Test GELU
        gelu = build_activation("gelu")
        x = torch.tensor([1.0])
        result = gelu(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

        # Test SiLU
        silu = build_activation("silu")
        x = torch.tensor([1.0])
        result = silu(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

        # Test ReLU
        relu = build_activation("relu")
        x = torch.tensor([-1.0, 1.0])
        result = relu(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert torch.allclose(result, torch.tensor([0.0, 1.0]))
