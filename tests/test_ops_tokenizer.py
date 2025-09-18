"""Unit tests for ops.tokenizer module."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from ops.tokenizer import (
    BaseTokenizer, SimpleCharTokenizer, RegexWordTokenizer, 
    TiktokenTokenizer, HFTokenizer, build_tokenizer
)


class TestBaseTokenizer:
    """Test the BaseTokenizer abstract base class."""

    def test_pad_id_property(self):
        """Test that pad_id property returns vocab_size - 1."""
        class TestTokenizer(BaseTokenizer):
            def encode(self, text: str):
                return []
            
            def decode(self, ids):
                return ""
            
            @property
            def vocab_size(self):
                return 100
        
        tokenizer = TestTokenizer()
        assert tokenizer.pad_id == 99

    def test_eos_id_property(self):
        """Test that eos_id property returns None by default."""
        class TestTokenizer(BaseTokenizer):
            def encode(self, text: str):
                return []
            
            def decode(self, ids):
                return ""
            
            @property
            def vocab_size(self):
                return 100
        
        tokenizer = TestTokenizer()
        assert tokenizer.eos_id is None


class TestSimpleCharTokenizer:
    """Test the SimpleCharTokenizer class."""

    def test_init_default_vocab(self):
        """Test initialization with default vocabulary."""
        tokenizer = SimpleCharTokenizer()
        assert len(tokenizer.itos) > 0
        assert len(tokenizer.stoi) == len(tokenizer.itos)
        assert tokenizer.vocab_size == len(tokenizer.itos) + 1  # +1 for PAD

    def test_init_custom_vocab(self):
        """Test initialization with custom vocabulary."""
        vocab = "abc"
        tokenizer = SimpleCharTokenizer(vocab)
        assert tokenizer.itos == ['a', 'b', 'c']
        assert tokenizer.stoi == {'a': 0, 'b': 1, 'c': 2}
        assert tokenizer.vocab_size == 4  # 3 chars + 1 PAD

    def test_encode_decode_roundtrip(self):
        """Test that encode and decode are inverse operations."""
        tokenizer = SimpleCharTokenizer("abc")
        text = "abc"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_encode_unknown_char(self):
        """Test encoding with unknown characters."""
        tokenizer = SimpleCharTokenizer("ab")
        ids = tokenizer.encode("abc")
        # 'c' should be mapped to unknown token (index 0)
        assert ids == [0, 1, 0]  # 'a'->0, 'b'->1, 'c'->0 (unknown)

    def test_decode_with_padding(self):
        """Test decoding with padding tokens."""
        tokenizer = SimpleCharTokenizer("ab")
        # Pad token is vocab_size - 1 = 2
        ids = [0, 1, 2, 2]  # 'a', 'b', PAD, PAD
        decoded = tokenizer.decode(ids)
        assert decoded == "ab"  # PAD tokens should be removed

    def test_decode_invalid_ids(self):
        """Test decoding with invalid token IDs."""
        tokenizer = SimpleCharTokenizer("ab")
        ids = [0, 1, 5]  # 5 is out of range
        decoded = tokenizer.decode(ids)
        assert decoded == "ab?"  # Invalid ID should become '?'

    def test_vocab_size_property(self):
        """Test vocab_size property."""
        tokenizer = SimpleCharTokenizer("abc")
        assert tokenizer.vocab_size == 4  # 3 chars + 1 PAD

    def test_eos_id_property(self):
        """Test eos_id property returns None."""
        tokenizer = SimpleCharTokenizer()
        assert tokenizer.eos_id is None


class TestRegexWordTokenizer:
    """Test the RegexWordTokenizer class."""

    def test_init(self):
        """Test tokenizer initialization."""
        tokenizer = RegexWordTokenizer()
        assert tokenizer.stoi == {"<unk>": 0}
        assert tokenizer.itos == ["<unk>"]
        assert tokenizer.vocab_size == 2  # 1 word + 1 PAD

    def test_encode_new_words(self):
        """Test encoding with new words."""
        tokenizer = RegexWordTokenizer()
        ids = tokenizer.encode("hello world")
        assert len(ids) == 2
        assert tokenizer.vocab_size == 4  # 2 new words + 1 <unk> + 1 PAD

    def test_encode_decode_roundtrip(self):
        """Test that encode and decode are inverse operations."""
        tokenizer = RegexWordTokenizer()
        text = "hello world"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_encode_punctuation(self):
        """Test encoding with punctuation."""
        tokenizer = RegexWordTokenizer()
        ids = tokenizer.encode("hello, world!")
        assert len(ids) == 4  # "hello", ",", "world", "!"

    def test_decode_with_padding(self):
        """Test decoding with padding tokens."""
        tokenizer = RegexWordTokenizer()
        tokenizer.encode("hello world")  # Build vocab: hello=1, world=2
        pad_id = tokenizer.pad_id
        ids = [1, 2, pad_id, pad_id]  # "hello", "world", PAD, PAD
        decoded = tokenizer.decode(ids)
        assert decoded == "hello world"  # PAD tokens should be removed

    def test_decode_invalid_ids(self):
        """Test decoding with invalid token IDs."""
        tokenizer = RegexWordTokenizer()
        ids = [0, 5]  # 5 is out of range
        decoded = tokenizer.decode(ids)
        assert decoded == "<unk> <unk>"

    def test_vocab_size_property(self):
        """Test vocab_size property."""
        tokenizer = RegexWordTokenizer()
        tokenizer.encode("hello world")
        assert tokenizer.vocab_size == 4  # 2 words + 1 <unk> + 1 PAD


class TestTiktokenTokenizer:
    """Test the TiktokenTokenizer class."""

    @patch('ops.tokenizer.tiktoken.get_encoding')
    def test_init(self, mock_get_encoding):
        """Test tokenizer initialization."""
        mock_encoding = MagicMock()
        mock_encoding.n_vocab = 50257
        mock_encoding.encode.return_value = [50256]  # <|endoftext|> token
        mock_get_encoding.return_value = mock_encoding
        
        tokenizer = TiktokenTokenizer(mock_encoding)
        assert tokenizer.encoding == mock_encoding
        assert tokenizer.vocab_size == 50257
        assert tokenizer.eos_id == 50256

    @patch('ops.tokenizer.tiktoken.get_encoding')
    def test_encode(self, mock_get_encoding):
        """Test encoding text."""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_encoding.n_vocab = 50257
        mock_get_encoding.return_value = mock_encoding
        
        tokenizer = TiktokenTokenizer(mock_encoding)
        result = tokenizer.encode("hello")
        assert result == [1, 2, 3]
        # The encode method is called twice: once during init for eos_id, once for actual encode
        assert mock_encoding.encode.call_count == 2

    @patch('ops.tokenizer.tiktoken.get_encoding')
    def test_decode(self, mock_get_encoding):
        """Test decoding token IDs."""
        mock_encoding = MagicMock()
        mock_encoding.decode.return_value = "hello"
        mock_encoding.n_vocab = 50257
        mock_get_encoding.return_value = mock_encoding
        
        tokenizer = TiktokenTokenizer(mock_encoding)
        result = tokenizer.decode([1, 2, 3])
        assert result == "hello"
        mock_encoding.decode.assert_called_once_with([1, 2, 3])

    @patch('ops.tokenizer.tiktoken.get_encoding')
    def test_decode_with_padding(self, mock_get_encoding):
        """Test decoding with padding tokens."""
        mock_encoding = MagicMock()
        mock_encoding.decode.return_value = "hello"
        mock_encoding.n_vocab = 50257
        mock_get_encoding.return_value = mock_encoding
        
        tokenizer = TiktokenTokenizer(mock_encoding)
        pad_id = tokenizer.pad_id  # Should be 50256
        result = tokenizer.decode([1, 2, pad_id, 3])
        assert result == "hello"
        mock_encoding.decode.assert_called_once_with([1, 2, 3])  # PAD token filtered out


class TestHFTokenizer:
    """Test the HFTokenizer class."""

    def test_init(self):
        """Test tokenizer initialization."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        
        tokenizer = HFTokenizer(mock_tok)
        assert tokenizer.tok == mock_tok

    def test_encode(self):
        """Test encoding text."""
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3]
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        
        tokenizer = HFTokenizer(mock_tok)
        result = tokenizer.encode("hello")
        assert result == [1, 2, 3]
        mock_tok.encode.assert_called_once_with("hello")

    def test_decode(self):
        """Test decoding token IDs."""
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello"
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        
        tokenizer = HFTokenizer(mock_tok)
        result = tokenizer.decode([1, 2, 3])
        assert result == "hello"
        mock_tok.decode.assert_called_once_with([1, 2, 3])

    def test_vocab_size(self):
        """Test vocab_size property."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        
        tokenizer = HFTokenizer(mock_tok)
        assert tokenizer.vocab_size == 50257

    def test_pad_id(self):
        """Test pad_id property."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        
        tokenizer = HFTokenizer(mock_tok)
        assert tokenizer.pad_id == 50256  # vocab_size - 1

    def test_eos_id(self):
        """Test eos_id property."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        
        tokenizer = HFTokenizer(mock_tok)
        assert tokenizer.eos_id == 50256

    def test_eos_id_none(self):
        """Test eos_id property when eos_token_id is None."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = None
        
        tokenizer = HFTokenizer(mock_tok)
        assert tokenizer.eos_id is None


class TestBuildTokenizer:
    """Test the build_tokenizer function."""

    def test_build_simple_char_tokenizer(self):
        """Test building SimpleCharTokenizer."""
        cfg = {"kind": "simple_char", "params": {"vocab_chars": "abc"}}
        tokenizer = build_tokenizer(cfg)
        assert isinstance(tokenizer, SimpleCharTokenizer)
        assert tokenizer.vocab_size == 4  # 3 chars + 1 PAD

    def test_build_regex_word_tokenizer(self):
        """Test building RegexWordTokenizer."""
        cfg = {"kind": "regex_word"}
        tokenizer = build_tokenizer(cfg)
        assert isinstance(tokenizer, RegexWordTokenizer)

    @patch('ops.tokenizer.tiktoken.get_encoding')
    def test_build_tiktoken_tokenizer(self, mock_get_encoding):
        """Test building TiktokenTokenizer."""
        mock_encoding = MagicMock()
        mock_encoding.n_vocab = 50257
        mock_encoding.encode.return_value = [50256]
        mock_get_encoding.return_value = mock_encoding
        
        cfg = {"kind": "tiktoken", "params": {"name": "gpt2"}}
        tokenizer = build_tokenizer(cfg)
        assert isinstance(tokenizer, TiktokenTokenizer)
        mock_get_encoding.assert_called_once_with("gpt2")

    @patch('ops.tokenizer.tiktoken.get_encoding')
    def test_build_tiktoken_gpt2_tokenizer(self, mock_get_encoding):
        """Test building TiktokenTokenizer with tiktoken_gpt2 kind."""
        mock_encoding = MagicMock()
        mock_encoding.n_vocab = 50257
        mock_encoding.encode.return_value = [50256]
        mock_get_encoding.return_value = mock_encoding
        
        cfg = {"kind": "tiktoken_gpt2", "params": {"name": "gpt2"}}
        tokenizer = build_tokenizer(cfg)
        assert isinstance(tokenizer, TiktokenTokenizer)

    @pytest.mark.skipif(True, reason="transformers not available")
    @patch('transformers.GPT2TokenizerFast')
    def test_build_hf_tokenizer(self, mock_gpt2_tokenizer):
        """Test building HFTokenizer."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        mock_gpt2_tokenizer.from_pretrained.return_value = mock_tok
        
        cfg = {"kind": "huggingface", "params": {"name": "gpt2"}}
        tokenizer = build_tokenizer(cfg)
        assert isinstance(tokenizer, HFTokenizer)
        mock_gpt2_tokenizer.from_pretrained.assert_called_once_with("gpt2", add_prefix_space=True)

    @pytest.mark.skipif(True, reason="transformers not available")
    @patch('transformers.GPT2TokenizerFast')
    def test_build_hf_gpt2_tokenizer(self, mock_gpt2_tokenizer):
        """Test building HFTokenizer with hf_gpt2 kind."""
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50257
        mock_tok.eos_token_id = 50256
        mock_gpt2_tokenizer.from_pretrained.return_value = mock_tok
        
        cfg = {"kind": "hf_gpt2", "params": {"name": "gpt2"}}
        tokenizer = build_tokenizer(cfg)
        assert isinstance(tokenizer, HFTokenizer)

    def test_build_tokenizer_default_kind(self):
        """Test building tokenizer with default kind (tiktoken)."""
        with patch('ops.tokenizer.tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = MagicMock()
            mock_encoding.n_vocab = 50257
            mock_encoding.encode.return_value = [50256]
            mock_get_encoding.return_value = mock_encoding
            
            cfg = {}  # No kind specified
            tokenizer = build_tokenizer(cfg)
            assert isinstance(tokenizer, TiktokenTokenizer)

    def test_build_tokenizer_unknown_kind(self):
        """Test building tokenizer with unknown kind raises ValueError."""
        cfg = {"kind": "unknown"}
        with pytest.raises(ValueError, match="Unknown tokenizer kind: unknown"):
            build_tokenizer(cfg)

    def test_build_tokenizer_empty_params(self):
        """Test building tokenizer with empty params."""
        with patch('ops.tokenizer.tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = MagicMock()
            mock_encoding.n_vocab = 50257
            mock_encoding.encode.return_value = [50256]
            mock_get_encoding.return_value = mock_encoding
            
            cfg = {"kind": "tiktoken", "params": {}}
            tokenizer = build_tokenizer(cfg)
            assert isinstance(tokenizer, TiktokenTokenizer)
            mock_get_encoding.assert_called_once_with("gpt2")  # Default name
