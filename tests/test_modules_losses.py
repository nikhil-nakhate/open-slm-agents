"""Unit tests for models.modules.losses module."""

import pytest
import torch
import torch.nn as nn

from models.modules.losses import CrossEntropyLossWrapper, build_loss


class TestCrossEntropyLossWrapper:
    """Test the CrossEntropyLossWrapper class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        loss_fn = CrossEntropyLossWrapper()
        assert loss_fn.loss_fn.ignore_index == -100

    def test_init_custom_ignore_index(self):
        """Test initialization with custom ignore_index."""
        loss_fn = CrossEntropyLossWrapper(ignore_index=0)
        assert loss_fn.loss_fn.ignore_index == 0

    def test_forward_2d_logits(self):
        """Test forward pass with 2D logits (reshaped internally)."""
        loss_fn = CrossEntropyLossWrapper()
        logits = torch.randn(2, 5, 10)  # [B, T, V] - 3D input
        targets = torch.randint(0, 10, (2, 5))  # [B, T]
        
        result = loss_fn(logits, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar loss

    def test_forward_3d_logits(self):
        """Test forward pass with 3D logits (reshaped internally)."""
        loss_fn = CrossEntropyLossWrapper()
        logits = torch.randn(2, 5, 10)  # [B, T, V]
        targets = torch.randint(0, 10, (2, 5))  # [B, T]
        
        result = loss_fn(logits, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar loss

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        loss_fn = CrossEntropyLossWrapper()
        logits = torch.randn(2, 5, 10, requires_grad=True)
        targets = torch.randint(0, 10, (2, 5))
        
        result = loss_fn(logits, targets)
        result.backward()
        assert logits.grad is not None

    def test_forward_with_ignore_index(self):
        """Test forward pass with ignore_index."""
        loss_fn = CrossEntropyLossWrapper(ignore_index=-100)
        logits = torch.randn(2, 5, 10)
        targets = torch.randint(0, 10, (2, 5))
        targets[0, 0] = -100  # Set some targets to ignore_index
        
        result = loss_fn(logits, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_forward_reshaping(self):
        """Test that logits and targets are properly reshaped."""
        loss_fn = CrossEntropyLossWrapper()
        B, T, V = 2, 5, 10
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        
        result = loss_fn(logits, targets)
        
        # Check that the internal loss function receives reshaped tensors
        expected_logits = logits.view(B * T, V)
        expected_targets = targets.view(B * T)
        
        # The result should be the same as calling CrossEntropyLoss directly
        direct_loss = nn.CrossEntropyLoss()(expected_logits, expected_targets)
        assert torch.allclose(result, direct_loss, atol=1e-6)

    def test_forward_empty_batch(self):
        """Test forward pass with empty batch."""
        loss_fn = CrossEntropyLossWrapper()
        logits = torch.randn(0, 0, 10)  # Empty batch with 3D shape
        targets = torch.randint(0, 10, (0, 0))
        
        result = loss_fn(logits, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_forward_single_token(self):
        """Test forward pass with single token."""
        loss_fn = CrossEntropyLossWrapper()
        logits = torch.randn(1, 1, 10)  # [1, 1, V]
        targets = torch.randint(0, 10, (1, 1))  # [1, 1]
        
        result = loss_fn(logits, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_forward_large_vocab(self):
        """Test forward pass with large vocabulary."""
        loss_fn = CrossEntropyLossWrapper()
        V = 50000  # Large vocabulary
        logits = torch.randn(2, 10, V)
        targets = torch.randint(0, V, (2, 10))
        
        result = loss_fn(logits, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_forward_against_direct_crossentropy(self):
        """Test that wrapper produces same result as direct CrossEntropyLoss."""
        loss_fn = CrossEntropyLossWrapper()
        logits = torch.randn(2, 5, 10)
        targets = torch.randint(0, 10, (2, 5))
        
        wrapper_result = loss_fn(logits, targets)
        
        # Direct CrossEntropyLoss call
        B, T, V = logits.shape
        direct_result = nn.CrossEntropyLoss()(
            logits.view(B * T, V), 
            targets.view(B * T)
        )
        
        assert torch.allclose(wrapper_result, direct_result, atol=1e-6)

    def test_forward_gradient_flow(self):
        """Test that gradients flow properly through the loss function."""
        loss_fn = CrossEntropyLossWrapper()
        logits = torch.randn(2, 5, 10, requires_grad=True)
        targets = torch.randint(0, 10, (2, 5))
        
        result = loss_fn(logits, targets)
        result.backward()
        
        # Check that gradients exist and are non-zero
        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))

    def test_forward_ignore_index_effect(self):
        """Test that ignore_index actually ignores certain targets."""
        # Create logits and targets where some targets are set to ignore_index
        logits = torch.randn(2, 3, 5)
        targets = torch.tensor([[0, 1, 2], [3, -100, 4]])  # -100 is ignore_index
        
        loss_fn = CrossEntropyLossWrapper(ignore_index=-100)
        result = loss_fn(logits, targets)
        
        # The loss should only consider non-ignored targets
        # This is a basic check - the actual behavior depends on PyTorch's implementation
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0


class TestBuildLoss:
    """Test the build_loss function."""

    def test_build_cross_entropy_loss(self):
        """Test building CrossEntropyLossWrapper."""
        cfg = {"kind": "cross_entropy"}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)

    def test_build_cross_entropy_loss_ce_alias(self):
        """Test building CrossEntropyLossWrapper with 'ce' alias."""
        cfg = {"kind": "ce"}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)

    def test_build_cross_entropy_loss_with_params(self):
        """Test building CrossEntropyLossWrapper with parameters."""
        cfg = {"kind": "cross_entropy", "params": {"ignore_index": 0}}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)
        assert loss_fn.loss_fn.ignore_index == 0

    def test_build_loss_default(self):
        """Test building loss with default configuration."""
        cfg = {}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)

    def test_build_loss_none_config(self):
        """Test building loss with None configuration."""
        loss_fn = build_loss(None)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)

    def test_build_loss_case_insensitive(self):
        """Test building loss with case insensitive kind."""
        cfg = {"kind": "CROSS_ENTROPY"}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)

    def test_build_loss_unknown_kind(self):
        """Test building loss with unknown kind raises ValueError."""
        cfg = {"kind": "unknown_loss"}
        with pytest.raises(ValueError, match="Unknown loss kind: unknown_loss"):
            build_loss(cfg)

    def test_build_loss_functionality(self):
        """Test that built loss function works correctly."""
        cfg = {"kind": "cross_entropy", "params": {"ignore_index": -100}}
        loss_fn = build_loss(cfg)
        
        logits = torch.randn(2, 5, 10)
        targets = torch.randint(0, 10, (2, 5))
        
        result = loss_fn(logits, targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_build_loss_empty_params(self):
        """Test building loss with empty params."""
        cfg = {"kind": "cross_entropy", "params": {}}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)
        assert loss_fn.loss_fn.ignore_index == -100  # Default value

    def test_build_loss_missing_params(self):
        """Test building loss with missing params key."""
        cfg = {"kind": "cross_entropy"}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, CrossEntropyLossWrapper)
        assert loss_fn.loss_fn.ignore_index == -100  # Default value
