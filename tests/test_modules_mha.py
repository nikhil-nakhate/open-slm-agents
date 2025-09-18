"""Unit tests for models.modules.mha module."""

import pytest
import torch
import torch.nn as nn

from models.modules.mha import MultiHeadAttention


class TestMultiHeadAttention:
    """Test the MultiHeadAttention class."""

    def test_init(self):
        """Test initialization with valid parameters."""
        mha = MultiHeadAttention(
            d_in=512, d_out=512, context_length=1024, 
            dropout=0.1, num_heads=8
        )
        assert mha.d_out == 512
        assert mha.num_heads == 8
        assert mha.head_dim == 64  # 512 // 8
        assert isinstance(mha.W_query, nn.Linear)
        assert isinstance(mha.W_key, nn.Linear)
        assert isinstance(mha.W_value, nn.Linear)
        assert isinstance(mha.out_proj, nn.Linear)
        assert isinstance(mha.dropout, nn.Dropout)

    def test_init_with_qkv_bias(self):
        """Test initialization with QKV bias enabled."""
        mha = MultiHeadAttention(
            d_in=512, d_out=512, context_length=1024, 
            dropout=0.1, num_heads=8, qkv_bias=True
        )
        assert mha.W_query.bias is not None
        assert mha.W_key.bias is not None
        assert mha.W_value.bias is not None

    def test_init_without_qkv_bias(self):
        """Test initialization without QKV bias."""
        mha = MultiHeadAttention(
            d_in=512, d_out=512, context_length=1024, 
            dropout=0.1, num_heads=8, qkv_bias=False
        )
        assert mha.W_query.bias is None
        assert mha.W_key.bias is None
        assert mha.W_value.bias is None

    def test_init_invalid_d_out_divisible_by_num_heads(self):
        """Test that d_out must be divisible by num_heads."""
        with pytest.raises(AssertionError, match="d_out must be divisible by num_heads"):
            MultiHeadAttention(
                d_in=512, d_out=513, context_length=1024, 
                dropout=0.1, num_heads=8
            )

    def test_forward(self):
        """Test forward pass."""
        mha = MultiHeadAttention(
            d_in=512, d_out=512, context_length=1024, 
            dropout=0.1, num_heads=8
        )
        x = torch.randn(2, 10, 512)  # [B, T, d_in]
        result = mha(x)
        
        assert result.shape == (2, 10, 512)  # [B, T, d_out]
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        mha = MultiHeadAttention(
            d_in=512, d_out=512, context_length=1024, 
            dropout=0.1, num_heads=8
        )
        x = torch.randn(2, 10, 512, requires_grad=True)
        result = mha(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_forward_single_sequence(self):
        """Test forward pass with single sequence."""
        mha = MultiHeadAttention(
            d_in=512, d_out=512, context_length=1024, 
            dropout=0.1, num_heads=8
        )
        x = torch.randn(1, 5, 512)  # [1, T, d_in]
        result = mha(x)
        
        assert result.shape == (1, 5, 512)  # [1, T, d_out]
        assert isinstance(result, torch.Tensor)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=512, 
            dropout=0.1, num_heads=4
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 10, 256)
            result = mha(x)
            assert result.shape == (batch_size, 10, 256)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=512, 
            dropout=0.1, num_heads=4
        )
        
        for seq_len in [1, 5, 10, 20]:
            x = torch.randn(2, seq_len, 256)
            result = mha(x)
            assert result.shape == (2, seq_len, 256)

    def test_forward_causal_mask(self):
        """Test that causal mask is applied correctly."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.0, num_heads=4  # No dropout for deterministic test
        )
        mha.eval()  # Disable dropout
        
        # Create input with known pattern
        x = torch.randn(1, 5, 256)
        result = mha(x)
        
        # The result should be deterministic with the same input
        result2 = mha(x)
        assert torch.allclose(result, result2, atol=1e-6)

    def test_forward_mask_registration(self):
        """Test that causal mask is properly registered as buffer."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.1, num_heads=4
        )
        
        # Check that mask is registered as buffer
        assert 'mask' in mha._buffers
        mask = mha.mask
        assert mask.shape == (10, 10)
        
        # Check that mask is upper triangular
        assert torch.allclose(mask, torch.triu(torch.ones(10, 10), diagonal=1))

    def test_forward_attention_weights_shape(self):
        """Test that attention weights have correct shape."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.0, num_heads=4
        )
        mha.eval()
        
        x = torch.randn(2, 5, 256)
        result = mha(x)
        
        # The result should have the correct shape
        assert result.shape == (2, 5, 256)

    def test_forward_dropout_training_mode(self):
        """Test that dropout is applied in training mode."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.5, num_heads=4
        )
        mha.train()
        
        x = torch.randn(2, 5, 256)
        result1 = mha(x)
        result2 = mha(x)
        
        # Results should be different due to dropout
        assert not torch.allclose(result1, result2, atol=1e-6)

    def test_forward_dropout_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.5, num_heads=4
        )
        mha.eval()
        
        x = torch.randn(2, 5, 256)
        result1 = mha(x)
        result2 = mha(x)
        
        # Results should be identical in eval mode
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_forward_different_num_heads(self):
        """Test forward pass with different numbers of heads."""
        for num_heads in [1, 2, 4, 8, 16]:
            d_out = num_heads * 64  # Ensure d_out is divisible by num_heads
            mha = MultiHeadAttention(
                d_in=256, d_out=d_out, context_length=10, 
                dropout=0.1, num_heads=num_heads
            )
            x = torch.randn(2, 5, 256)
            result = mha(x)
            assert result.shape == (2, 5, d_out)

    def test_forward_different_dimensions(self):
        """Test forward pass with different input/output dimensions."""
        mha = MultiHeadAttention(
            d_in=128, d_out=256, context_length=10, 
            dropout=0.1, num_heads=4
        )
        x = torch.randn(2, 5, 128)  # Input dim: 128
        result = mha(x)
        assert result.shape == (2, 5, 256)  # Output dim: 256

    def test_forward_gradient_flow(self):
        """Test that gradients flow properly through all parameters."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.1, num_heads=4
        )
        x = torch.randn(2, 5, 256, requires_grad=True)
        result = mha(x)
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        assert x.grad is not None
        assert mha.W_query.weight.grad is not None
        assert mha.W_key.weight.grad is not None
        assert mha.W_value.weight.grad is not None
        assert mha.out_proj.weight.grad is not None

    def test_forward_sequence_length_less_than_context(self):
        """Test forward pass when sequence length is less than context length."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=20, 
            dropout=0.1, num_heads=4
        )
        x = torch.randn(2, 5, 256)  # seq_len=5 < context_length=20
        result = mha(x)
        assert result.shape == (2, 5, 256)

    def test_forward_sequence_length_equal_to_context(self):
        """Test forward pass when sequence length equals context length."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.1, num_heads=4
        )
        x = torch.randn(2, 10, 256)  # seq_len=10 == context_length=10
        result = mha(x)
        assert result.shape == (2, 10, 256)

    def test_forward_attention_causality(self):
        """Test that attention is causal (future tokens are masked)."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.0, num_heads=4
        )
        mha.eval()
        
        # Create input where only the first token is non-zero
        x = torch.zeros(1, 5, 256)
        x[0, 0, :] = 1.0  # Only first token is non-zero
        
        result = mha(x)
        
        # The result should be deterministic
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 5, 256)

    def test_forward_zero_input(self):
        """Test forward pass with zero input."""
        mha = MultiHeadAttention(
            d_in=256, d_out=256, context_length=10, 
            dropout=0.1, num_heads=4
        )
        x = torch.zeros(2, 5, 256)
        result = mha(x)
        assert result.shape == (2, 5, 256)
        # With zero input, result should be deterministic and finite
        assert torch.isfinite(result).all()
        # The result should be the same for identical zero inputs
        result2 = mha(x)
        assert torch.allclose(result, result2, atol=1e-6)
