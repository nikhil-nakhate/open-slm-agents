"""Unit tests for models.modules.embeddings module."""

import pytest
import torch
import torch.nn as nn

from models.modules.embeddings import (
    TokenPositionalEmbedding, OutputProjection, TokenEmbedding, PositionEmbedding
)


class TestTokenPositionalEmbedding:
    """Test the TokenPositionalEmbedding class."""

    def test_init(self):
        """Test initialization with default parameters."""
        embedding = TokenPositionalEmbedding(vocab_size=1000, dim=512, max_seq_len=1024)
        assert embedding.token_emb.num_embeddings == 1000
        assert embedding.token_emb.embedding_dim == 512
        assert embedding.pos_emb.num_embeddings == 1024
        assert embedding.pos_emb.embedding_dim == 512
        assert isinstance(embedding.drop, nn.Dropout)

    def test_init_with_dropout(self):
        """Test initialization with custom dropout."""
        embedding = TokenPositionalEmbedding(
            vocab_size=1000, dim=512, max_seq_len=1024, dropout=0.1
        )
        assert embedding.drop.p == 0.1

    def test_forward(self):
        """Test forward pass."""
        embedding = TokenPositionalEmbedding(vocab_size=1000, dim=512, max_seq_len=1024)
        idx = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [B, T]
        result = embedding(idx)
        
        assert result.shape == (2, 3, 512)  # [B, T, dim]
        assert isinstance(result, torch.Tensor)

    def test_forward_single_sequence(self):
        """Test forward pass with single sequence."""
        embedding = TokenPositionalEmbedding(vocab_size=1000, dim=512, max_seq_len=1024)
        idx = torch.tensor([[1, 2, 3]])  # [1, T]
        result = embedding(idx)
        
        assert result.shape == (1, 3, 512)
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        embedding = TokenPositionalEmbedding(vocab_size=1000, dim=512, max_seq_len=1024)
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        result = embedding(idx)
        loss = result.sum()
        loss.backward()
        # Check that embedding parameters have gradients
        assert embedding.token_emb.weight.grad is not None

    def test_position_embedding_initialization(self):
        """Test that position embeddings are initialized with normal distribution."""
        embedding = TokenPositionalEmbedding(vocab_size=1000, dim=512, max_seq_len=1024)
        pos_weights = embedding.pos_emb.weight
        assert pos_weights.shape == (1024, 512)
        # Check that weights are initialized (not all zeros)
        assert not torch.allclose(pos_weights, torch.zeros_like(pos_weights))

    def test_dropout_training_mode(self):
        """Test that dropout is applied in training mode."""
        embedding = TokenPositionalEmbedding(vocab_size=1000, dim=512, max_seq_len=1024, dropout=0.5)
        embedding.train()
        idx = torch.tensor([[1, 2, 3]])
        result1 = embedding(idx)
        result2 = embedding(idx)
        # Results should be different due to dropout
        assert not torch.allclose(result1, result2)

    def test_dropout_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        embedding = TokenPositionalEmbedding(vocab_size=1000, dim=512, max_seq_len=1024, dropout=0.5)
        embedding.eval()
        idx = torch.tensor([[1, 2, 3]])
        result1 = embedding(idx)
        result2 = embedding(idx)
        # Results should be identical in eval mode
        assert torch.allclose(result1, result2)


class TestOutputProjection:
    """Test the OutputProjection class."""

    def test_init_without_tie_weights(self):
        """Test initialization without weight tying."""
        projection = OutputProjection(dim=512, vocab_size=1000)
        assert projection.proj.in_features == 512
        assert projection.proj.out_features == 1000
        assert projection.proj.bias is None

    def test_init_with_tie_weights(self):
        """Test initialization with weight tying."""
        embedding = nn.Embedding(1000, 512)
        projection = OutputProjection(dim=512, vocab_size=1000, tie_weights=True, tie_to=embedding)
        assert projection.proj.weight is embedding.weight

    def test_init_tie_weights_without_tie_to(self):
        """Test that tie_weights=True without tie_to raises ValueError."""
        with pytest.raises(ValueError, match="tie_weights=True requires tie_to Embedding"):
            OutputProjection(dim=512, vocab_size=1000, tie_weights=True)

    def test_forward(self):
        """Test forward pass."""
        projection = OutputProjection(dim=512, vocab_size=1000)
        x = torch.randn(2, 10, 512)  # [B, T, dim]
        result = projection(x)
        
        assert result.shape == (2, 10, 1000)  # [B, T, vocab_size]
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        projection = OutputProjection(dim=512, vocab_size=1000)
        x = torch.randn(2, 10, 512, requires_grad=True)
        result = projection(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None

    def test_weight_tying_shared_parameters(self):
        """Test that weight tying creates shared parameters."""
        embedding = nn.Embedding(1000, 512)
        projection = OutputProjection(dim=512, vocab_size=1000, tie_weights=True, tie_to=embedding)
        
        # Modify embedding weights
        embedding.weight.data.fill_(1.0)
        
        # Check that projection weights are also modified
        assert torch.allclose(projection.proj.weight, torch.ones_like(projection.proj.weight))


class TestTokenEmbedding:
    """Test the TokenEmbedding class."""

    def test_init(self):
        """Test initialization."""
        embedding = TokenEmbedding(vocab_size=1000, dim=512)
        assert embedding.token_emb.num_embeddings == 1000
        assert embedding.token_emb.embedding_dim == 512

    def test_forward(self):
        """Test forward pass."""
        embedding = TokenEmbedding(vocab_size=1000, dim=512)
        idx = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [B, T]
        result = embedding(idx)
        
        assert result.shape == (2, 3, 512)  # [B, T, dim]
        assert isinstance(result, torch.Tensor)

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        embedding = TokenEmbedding(vocab_size=1000, dim=512)
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        result = embedding(idx)
        loss = result.sum()
        loss.backward()
        # Check that embedding parameters have gradients
        assert embedding.token_emb.weight.grad is not None

    def test_forward_single_sequence(self):
        """Test forward pass with single sequence."""
        embedding = TokenEmbedding(vocab_size=1000, dim=512)
        idx = torch.tensor([[1, 2, 3]])  # [1, T]
        result = embedding(idx)
        
        assert result.shape == (1, 3, 512)
        assert isinstance(result, torch.Tensor)


class TestPositionEmbedding:
    """Test the PositionEmbedding class."""

    def test_init(self):
        """Test initialization."""
        embedding = PositionEmbedding(max_seq_len=1024, dim=512)
        assert embedding.pos_emb.num_embeddings == 1024
        assert embedding.pos_emb.embedding_dim == 512

    def test_forward(self):
        """Test forward pass."""
        embedding = PositionEmbedding(max_seq_len=1024, dim=512)
        device = torch.device('cpu')
        result = embedding(seq_len=10, device=device)
        
        assert result.shape == (1, 10, 512)  # [1, seq_len, dim]
        assert isinstance(result, torch.Tensor)

    def test_forward_different_device(self):
        """Test forward pass with different device."""
        if torch.cuda.is_available():
            embedding = PositionEmbedding(max_seq_len=1024, dim=512)
            device = torch.device('cuda')
            result = embedding(seq_len=10, device=device)
            
            assert result.shape == (1, 10, 512)
            assert result.device.type == 'cuda'

    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        embedding = PositionEmbedding(max_seq_len=1024, dim=512)
        device = torch.device('cpu')
        result = embedding(seq_len=10, device=device)
        loss = result.sum()
        loss.backward()
        # Position embeddings should have gradients
        assert embedding.pos_emb.weight.grad is not None

    def test_position_embedding_initialization(self):
        """Test that position embeddings are initialized with normal distribution."""
        embedding = PositionEmbedding(max_seq_len=1024, dim=512)
        pos_weights = embedding.pos_emb.weight
        assert pos_weights.shape == (1024, 512)
        # Check that weights are initialized (not all zeros)
        assert not torch.allclose(pos_weights, torch.zeros_like(pos_weights))

    def test_forward_max_seq_len(self):
        """Test forward pass with maximum sequence length."""
        embedding = PositionEmbedding(max_seq_len=1024, dim=512)
        device = torch.device('cpu')
        result = embedding(seq_len=1024, device=device)
        
        assert result.shape == (1, 1024, 512)
        assert isinstance(result, torch.Tensor)

    def test_forward_zero_length(self):
        """Test forward pass with zero length sequence."""
        embedding = PositionEmbedding(max_seq_len=1024, dim=512)
        device = torch.device('cpu')
        result = embedding(seq_len=0, device=device)
        
        assert result.shape == (1, 0, 512)
        assert isinstance(result, torch.Tensor)
