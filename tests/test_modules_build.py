"""Unit tests for models.modules.build module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from models.modules.build import (
    _maybe_freeze, build_embedding, build_output_projection,
    build_loss, build_tokenizer, build_layer_norm,
    build_token_embedding, build_position_embedding,
    build_emb_dropout, build_transformer_blocks
)


class TestMaybeFreeze:
    """Test the _maybe_freeze function."""

    def test_freeze_enabled(self):
        """Test freezing when freeze=True."""
        module = nn.Linear(10, 5)
        cfg = {"freeze": True}
        result = _maybe_freeze(module, cfg)
        
        assert result is module
        for param in module.parameters():
            assert not param.requires_grad

    def test_freeze_disabled(self):
        """Test not freezing when freeze=False."""
        module = nn.Linear(10, 5)
        cfg = {"freeze": False}
        result = _maybe_freeze(module, cfg)
        
        assert result is module
        for param in module.parameters():
            assert param.requires_grad

    def test_freeze_missing_key(self):
        """Test not freezing when freeze key is missing."""
        module = nn.Linear(10, 5)
        cfg = {}
        result = _maybe_freeze(module, cfg)
        
        assert result is module
        for param in module.parameters():
            assert param.requires_grad

    def test_freeze_none_config(self):
        """Test not freezing when config is None."""
        module = nn.Linear(10, 5)
        result = _maybe_freeze(module, None)
        
        assert result is module
        for param in module.parameters():
            assert param.requires_grad

    def test_freeze_truthy_values(self):
        """Test freezing with truthy values."""
        module = nn.Linear(10, 5)
        
        for truthy_value in [1, "yes", True, [1]]:
            cfg = {"freeze": truthy_value}
            result = _maybe_freeze(module, cfg)
            assert result is module
            for param in module.parameters():
                assert not param.requires_grad

    def test_freeze_falsy_values(self):
        """Test not freezing with falsy values."""
        module = nn.Linear(10, 5)
        
        for falsy_value in [0, "", False, None, []]:
            cfg = {"freeze": falsy_value}
            result = _maybe_freeze(module, cfg)
            assert result is module
            for param in module.parameters():
                assert param.requires_grad


class TestBuildEmbedding:
    """Test the build_embedding function."""

    def test_build_embedding_default(self):
        """Test building embedding with default parameters."""
        cfg = {}
        embedding = build_embedding(vocab_size=1000, dim=512, max_seq_len=1024, cfg=cfg)
        
        assert isinstance(embedding, nn.Module)
        assert hasattr(embedding, 'token_emb')
        assert hasattr(embedding, 'pos_emb')

    def test_build_embedding_with_dropout(self):
        """Test building embedding with custom dropout."""
        cfg = {"dropout": 0.1}
        embedding = build_embedding(vocab_size=1000, dim=512, max_seq_len=1024, cfg=cfg)
        
        assert isinstance(embedding, nn.Module)
        assert hasattr(embedding, 'drop')
        assert embedding.drop.p == 0.1

    def test_build_embedding_with_freeze(self):
        """Test building embedding with freeze enabled."""
        cfg = {"freeze": True}
        embedding = build_embedding(vocab_size=1000, dim=512, max_seq_len=1024, cfg=cfg)
        
        assert isinstance(embedding, nn.Module)
        for param in embedding.parameters():
            assert not param.requires_grad

    def test_build_embedding_none_config(self):
        """Test building embedding with None config."""
        embedding = build_embedding(vocab_size=1000, dim=512, max_seq_len=1024, cfg=None)
        
        assert isinstance(embedding, nn.Module)
        assert hasattr(embedding, 'token_emb')
        assert hasattr(embedding, 'pos_emb')


class TestBuildOutputProjection:
    """Test the build_output_projection function."""

    def test_build_output_projection_default(self):
        """Test building output projection with default parameters."""
        cfg = {"tie_weights": False}  # Disable weight tying for default test
        projection = build_output_projection(dim=512, vocab_size=1000, cfg=cfg)
        
        assert isinstance(projection, nn.Module)
        assert hasattr(projection, 'proj')

    def test_build_output_projection_with_tie_weights(self):
        """Test building output projection with weight tying."""
        embedding = nn.Embedding(1000, 512)
        cfg = {"tie_weights": True}
        projection = build_output_projection(dim=512, vocab_size=1000, cfg=cfg, tie_to=embedding)
        
        assert isinstance(projection, nn.Module)
        assert projection.proj.weight is embedding.weight

    def test_build_output_projection_with_freeze(self):
        """Test building output projection with freeze enabled."""
        cfg = {"freeze": True, "tie_weights": False}  # Disable weight tying
        projection = build_output_projection(dim=512, vocab_size=1000, cfg=cfg)
        
        assert isinstance(projection, nn.Module)
        for param in projection.parameters():
            assert not param.requires_grad

    def test_build_output_projection_none_config(self):
        """Test building output projection with None config."""
        # Need to provide tie_to when cfg is None since default tie_weights=True
        embedding = nn.Embedding(1000, 512)
        projection = build_output_projection(dim=512, vocab_size=1000, cfg=None, tie_to=embedding)
        
        assert isinstance(projection, nn.Module)
        assert hasattr(projection, 'proj')


class TestBuildLoss:
    """Test the build_loss function."""

    @patch('models.modules.build._build_loss')
    def test_build_loss_default(self, mock_build_loss):
        """Test building loss with default configuration."""
        mock_loss = nn.CrossEntropyLoss()
        mock_build_loss.return_value = mock_loss
        
        result = build_loss()
        
        assert result is mock_loss
        mock_build_loss.assert_called_once_with({"kind": "cross_entropy"})

    @patch('models.modules.build._build_loss')
    def test_build_loss_with_config(self, mock_build_loss):
        """Test building loss with custom configuration."""
        mock_loss = nn.CrossEntropyLoss()
        mock_build_loss.return_value = mock_loss
        
        cfg = {"kind": "custom_loss"}
        result = build_loss(cfg)
        
        assert result is mock_loss
        mock_build_loss.assert_called_once_with(cfg)

    @patch('models.modules.build._build_loss')
    def test_build_loss_none_config(self, mock_build_loss):
        """Test building loss with None config."""
        mock_loss = nn.CrossEntropyLoss()
        mock_build_loss.return_value = mock_loss
        
        result = build_loss(None)
        
        assert result is mock_loss
        mock_build_loss.assert_called_once_with({"kind": "cross_entropy"})


class TestBuildTokenizer:
    """Test the build_tokenizer function."""

    @patch('models.modules.build._build_tokenizer')
    def test_build_tokenizer_default(self, mock_build_tokenizer):
        """Test building tokenizer with default configuration."""
        mock_tokenizer = MagicMock()
        mock_build_tokenizer.return_value = mock_tokenizer
        
        result = build_tokenizer()
        
        assert result is mock_tokenizer
        mock_build_tokenizer.assert_called_once_with({"kind": "simple_char"})

    @patch('models.modules.build._build_tokenizer')
    def test_build_tokenizer_with_config(self, mock_build_tokenizer):
        """Test building tokenizer with custom configuration."""
        mock_tokenizer = MagicMock()
        mock_build_tokenizer.return_value = mock_tokenizer
        
        cfg = {"kind": "tiktoken", "params": {"name": "gpt2"}}
        result = build_tokenizer(cfg)
        
        assert result is mock_tokenizer
        mock_build_tokenizer.assert_called_once_with(cfg)

    @patch('models.modules.build._build_tokenizer')
    def test_build_tokenizer_none_config(self, mock_build_tokenizer):
        """Test building tokenizer with None config."""
        mock_tokenizer = MagicMock()
        mock_build_tokenizer.return_value = mock_tokenizer
        
        result = build_tokenizer(None)
        
        assert result is mock_tokenizer
        mock_build_tokenizer.assert_called_once_with({"kind": "simple_char"})


class TestBuildLayerNorm:
    """Test the build_layer_norm function."""

    def test_build_layer_norm_default(self):
        """Test building layer norm with default parameters."""
        cfg = {}
        layer_norm = build_layer_norm(dim=512, cfg=cfg)
        
        assert isinstance(layer_norm, nn.Module)
        assert hasattr(layer_norm, 'weight')
        assert hasattr(layer_norm, 'bias')

    def test_build_layer_norm_with_custom_params(self):
        """Test building layer norm with custom parameters."""
        cfg = {"eps": 1e-6, "bias": False}
        layer_norm = build_layer_norm(dim=512, cfg=cfg)
        
        assert isinstance(layer_norm, nn.Module)
        assert layer_norm.eps == 1e-6
        assert layer_norm.bias is None

    def test_build_layer_norm_with_freeze(self):
        """Test building layer norm with freeze enabled."""
        cfg = {"freeze": True}
        layer_norm = build_layer_norm(dim=512, cfg=cfg)
        
        assert isinstance(layer_norm, nn.Module)
        for param in layer_norm.parameters():
            assert not param.requires_grad

    def test_build_layer_norm_none_config(self):
        """Test building layer norm with None config."""
        layer_norm = build_layer_norm(dim=512, cfg=None)
        
        assert isinstance(layer_norm, nn.Module)
        assert hasattr(layer_norm, 'weight')
        assert hasattr(layer_norm, 'bias')


class TestBuildTokenEmbedding:
    """Test the build_token_embedding function."""

    def test_build_token_embedding_default(self):
        """Test building token embedding with default parameters."""
        cfg = {}
        embedding = build_token_embedding(vocab_size=1000, dim=512, cfg=cfg)
        
        assert isinstance(embedding, nn.Module)
        assert hasattr(embedding, 'token_emb')

    def test_build_token_embedding_with_freeze(self):
        """Test building token embedding with freeze enabled."""
        cfg = {"freeze": True}
        embedding = build_token_embedding(vocab_size=1000, dim=512, cfg=cfg)
        
        assert isinstance(embedding, nn.Module)
        for param in embedding.parameters():
            assert not param.requires_grad

    def test_build_token_embedding_none_config(self):
        """Test building token embedding with None config."""
        embedding = build_token_embedding(vocab_size=1000, dim=512, cfg=None)
        
        assert isinstance(embedding, nn.Module)
        assert hasattr(embedding, 'token_emb')


class TestBuildPositionEmbedding:
    """Test the build_position_embedding function."""

    def test_build_position_embedding_default(self):
        """Test building position embedding with default parameters."""
        cfg = {}
        embedding = build_position_embedding(max_seq_len=1024, dim=512, cfg=cfg)
        
        assert isinstance(embedding, nn.Module)
        assert hasattr(embedding, 'pos_emb')

    def test_build_position_embedding_with_freeze(self):
        """Test building position embedding with freeze enabled."""
        cfg = {"freeze": True}
        embedding = build_position_embedding(max_seq_len=1024, dim=512, cfg=cfg)
        
        assert isinstance(embedding, nn.Module)
        for param in embedding.parameters():
            assert not param.requires_grad

    def test_build_position_embedding_none_config(self):
        """Test building position embedding with None config."""
        embedding = build_position_embedding(max_seq_len=1024, dim=512, cfg=None)
        
        assert isinstance(embedding, nn.Module)
        assert hasattr(embedding, 'pos_emb')


class TestBuildEmbDropout:
    """Test the build_emb_dropout function."""

    def test_build_emb_dropout_default(self):
        """Test building embedding dropout with default parameters."""
        cfg = {}
        dropout = build_emb_dropout(default_p=0.1, cfg=cfg)
        
        assert isinstance(dropout, nn.Dropout)
        assert dropout.p == 0.1

    def test_build_emb_dropout_with_custom_p(self):
        """Test building embedding dropout with custom p."""
        cfg = {"p": 0.2}
        dropout = build_emb_dropout(default_p=0.1, cfg=cfg)
        
        assert isinstance(dropout, nn.Dropout)
        assert dropout.p == 0.2

    def test_build_emb_dropout_none_config(self):
        """Test building embedding dropout with None config."""
        dropout = build_emb_dropout(default_p=0.1, cfg=None)
        
        assert isinstance(dropout, nn.Dropout)
        assert dropout.p == 0.1


class TestBuildTransformerBlocks:
    """Test the build_transformer_blocks function."""

    def test_build_transformer_blocks_default(self):
        """Test building transformer blocks with default parameters."""
        cfg = {"context_length": 1024}  # Provide required context_length
        blocks = build_transformer_blocks(dim=512, n_layers=6, n_heads=8, cfg=cfg)
        
        assert isinstance(blocks, nn.Sequential)
        assert len(blocks) == 6

    def test_build_transformer_blocks_with_custom_params(self):
        """Test building transformer blocks with custom parameters."""
        cfg = {
            "context_length": 1024,
            "mlp_mult": 2,
            "dropout": 0.1,
            "activation": "relu",
            "qkv_bias": True,
            "prenorm": False
        }
        blocks = build_transformer_blocks(dim=256, n_layers=4, n_heads=4, cfg=cfg)
        
        assert isinstance(blocks, nn.Sequential)
        assert len(blocks) == 4

    def test_build_transformer_blocks_with_freeze(self):
        """Test building transformer blocks with freeze enabled."""
        cfg = {"freeze": True, "context_length": 512}  # Provide required context_length
        blocks = build_transformer_blocks(dim=256, n_layers=2, n_heads=4, cfg=cfg)
        
        assert isinstance(blocks, nn.Sequential)
        assert len(blocks) == 2
        for param in blocks.parameters():
            assert not param.requires_grad

    def test_build_transformer_blocks_none_config(self):
        """Test building transformer blocks with None config."""
        # This will fail because context_length is required but not provided
        with pytest.raises(TypeError):
            blocks = build_transformer_blocks(dim=256, n_layers=2, n_heads=4, cfg=None)

    def test_build_transformer_blocks_zero_layers(self):
        """Test building transformer blocks with zero layers."""
        cfg = {}
        blocks = build_transformer_blocks(dim=256, n_layers=0, n_heads=4, cfg=cfg)
        
        assert isinstance(blocks, nn.Sequential)
        assert len(blocks) == 0

    def test_build_transformer_blocks_functionality(self):
        """Test that built transformer blocks work correctly."""
        cfg = {"context_length": 512}
        blocks = build_transformer_blocks(dim=256, n_layers=2, n_heads=4, cfg=cfg)
        
        x = torch.randn(2, 10, 256)
        result = blocks(x)
        
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)
