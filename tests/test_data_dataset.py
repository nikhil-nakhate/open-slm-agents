"""Unit tests for data.dataset module."""

import os
import json
import tempfile
import pytest
import torch
from unittest.mock import patch, MagicMock

from data.dataset import (
    BaseDataset, TextFileDataset, SFTJsonDataset,
    _chunking_collate, _sft_collate_batch, build_dataset_and_collate
)


class TestBaseDataset:
    """Test the BaseDataset base class."""

    def test_init(self):
        """Test initialization."""
        cfg = {"test": "value"}
        dataset = BaseDataset(cfg)
        assert dataset.cfg == cfg

    def test_len_not_implemented(self):
        """Test that __len__ raises NotImplementedError."""
        dataset = BaseDataset({})
        with pytest.raises(NotImplementedError):
            len(dataset)

    def test_getitem_not_implemented(self):
        """Test that __getitem__ raises NotImplementedError."""
        dataset = BaseDataset({})
        with pytest.raises(NotImplementedError):
            dataset[0]


class TestTextFileDataset:
    """Test the TextFileDataset class."""

    def test_init_with_existing_directory(self):
        """Test initialization with existing directory containing .txt files."""
        # Create temporary directory with .txt files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test .txt files
            with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
                f.write("Hello world. This is a test file.")
            with open(os.path.join(temp_dir, "file2.txt"), "w") as f:
                f.write("Another test file with some content.")
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.side_effect = lambda x: [1, 2, 3, 4, 5] if len(x) > 10 else [1, 2]
            
            cfg = {"data_dir": temp_dir, "min_length": 3}
            dataset = TextFileDataset(cfg, mock_tokenizer)
            
            assert len(dataset) == 2  # Two files with sufficient length
            assert dataset.ids[0] == [1, 2, 3, 4, 5]
            assert dataset.ids[1] == [1, 2, 3, 4, 5]

    def test_init_with_nonexistent_directory(self):
        """Test initialization with non-existent directory (fallback to default texts)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: [1, 2, 3, 4, 5] if len(x) > 10 else [1, 2]
        
        cfg = {"data_dir": "nonexistent_dir", "min_length": 3}
        dataset = TextFileDataset(cfg, mock_tokenizer)
        
        assert len(dataset) == 2  # Two default texts
        assert dataset.ids[0] == [1, 2, 3, 4, 5]
        assert dataset.ids[1] == [1, 2, 3, 4, 5]

    def test_init_with_min_length_filtering(self):
        """Test initialization with min_length filtering."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: [1, 2] if "short" in x else [1, 2, 3, 4, 5]
        
        cfg = {"data_dir": "nonexistent_dir", "min_length": 3}
        dataset = TextFileDataset(cfg, mock_tokenizer)
        
        # Only texts with sufficient length should be included
        assert len(dataset) == 2
        for ids in dataset.ids:
            assert len(ids) >= 3

    def test_init_with_all_texts_too_short(self):
        """Test initialization when all texts are too short."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]  # All texts too short
        
        cfg = {"data_dir": "nonexistent_dir", "min_length": 10}
        dataset = TextFileDataset(cfg, mock_tokenizer)
        
        # Should fall back to all texts when none meet min_length
        assert len(dataset) == 2

    def test_getitem(self):
        """Test __getitem__ method."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        cfg = {"data_dir": "nonexistent_dir"}
        dataset = TextFileDataset(cfg, mock_tokenizer)
        
        result = dataset[0]
        assert result == [1, 2, 3, 4, 5]

    def test_len(self):
        """Test __len__ method."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        cfg = {"data_dir": "nonexistent_dir"}
        dataset = TextFileDataset(cfg, mock_tokenizer)
        
        assert len(dataset) == 2

    def test_init_with_nested_directories(self):
        """Test initialization with nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)
            
            with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
                f.write("File in root directory.")
            with open(os.path.join(subdir, "file2.txt"), "w") as f:
                f.write("File in subdirectory.")
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            cfg = {"data_dir": temp_dir}
            dataset = TextFileDataset(cfg, mock_tokenizer)
            
            assert len(dataset) == 2  # Both files should be found

    def test_init_with_non_txt_files(self):
        """Test initialization with non-.txt files (should be ignored)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
                f.write("Valid .txt file.")
            with open(os.path.join(temp_dir, "file1.py"), "w") as f:
                f.write("Python file should be ignored.")
            with open(os.path.join(temp_dir, "file1.json"), "w") as f:
                f.write("JSON file should be ignored.")
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            cfg = {"data_dir": temp_dir}
            dataset = TextFileDataset(cfg, mock_tokenizer)
            
            assert len(dataset) == 1  # Only .txt file should be included


class TestSFTJsonDataset:
    """Test the SFTJsonDataset class."""

    def test_init_with_list_data(self):
        """Test initialization with list data."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {"instruction": "Test instruction 1", "input": "Test input 1", "output": "Test output 1"},
                {"instruction": "Test instruction 2", "input": "Test input 2", "output": "Test output 2"}
            ]
            json.dump(data, f)
            temp_path = f.name

        try:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.side_effect = lambda x: [1, 2, 3] if "instruction" in x else [4, 5, 6]
            
            mock_transform = MagicMock()
            mock_transform.side_effect = [
                ("Test instruction 1", "Test output 1"),
                ("Test instruction 2", "Test output 2")
            ]
            
            cfg = {"data_dir": "test"}
            dataset = SFTJsonDataset(cfg, [temp_path], mock_transform, mock_tokenizer)
            
            assert len(dataset) == 2
            assert len(dataset.items[0]) == 6  # 3 + 3 tokens
            assert len(dataset.items[1]) == 6  # 3 + 3 tokens
            
        finally:
            os.unlink(temp_path)

    def test_init_with_single_record(self):
        """Test initialization with single record."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"instruction": "Test instruction", "input": "Test input", "output": "Test output"}
            json.dump(data, f)
            temp_path = f.name

        try:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            
            mock_transform = MagicMock()
            mock_transform.return_value = ("Test instruction", "Test output")
            
            cfg = {"data_dir": "test"}
            dataset = SFTJsonDataset(cfg, [temp_path], mock_transform, mock_tokenizer)
            
            assert len(dataset) == 0  # Single record (not list) should result in empty dataset
            
        finally:
            os.unlink(temp_path)

    def test_getitem(self):
        """Test __getitem__ method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [{"instruction": "Test", "output": "Output"}]
            json.dump(data, f)
            temp_path = f.name

        try:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            
            mock_transform = MagicMock()
            mock_transform.return_value = ("Test", "Output")
            
            cfg = {"data_dir": "test"}
            dataset = SFTJsonDataset(cfg, [temp_path], mock_transform, mock_tokenizer)
            
            result = dataset[0]
            assert result == [1, 2, 3, 1, 2, 3]  # prompt + target tokens
            
        finally:
            os.unlink(temp_path)

    def test_len(self):
        """Test __len__ method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [{"instruction": "Test", "output": "Output"}]
            json.dump(data, f)
            temp_path = f.name

        try:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            
            mock_transform = MagicMock()
            mock_transform.return_value = ("Test", "Output")
            
            cfg = {"data_dir": "test"}
            dataset = SFTJsonDataset(cfg, [temp_path], mock_transform, mock_tokenizer)
            
            assert len(dataset) == 1
            
        finally:
            os.unlink(temp_path)


class TestChunkingCollate:
    """Test the _chunking_collate function."""

    def test_chunking_collate_short_sequences(self):
        """Test collate with sequences shorter than block_size."""
        batch = [[1, 2, 3], [4, 5, 6]]  # Short sequences
        block_size = 5
        pad_id = 0
        
        result = _chunking_collate(batch, block_size, pad_id)
        
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape == (2, 5)
        assert result["labels"].shape == (2, 5)
        
        # Check that sequences are padded
        assert torch.allclose(result["input_ids"][0], torch.tensor([0, 0, 0, 1, 2]))
        assert torch.allclose(result["labels"][0], torch.tensor([0, 0, 1, 2, 3]))

    def test_chunking_collate_long_sequences(self):
        """Test collate with sequences longer than block_size."""
        batch = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]  # Long sequences
        block_size = 5
        pad_id = 0
        
        result = _chunking_collate(batch, block_size, pad_id)
        
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape == (2, 5)
        assert result["labels"].shape == (2, 5)
        
        # Check that sequences are truncated
        assert torch.allclose(result["input_ids"][0], torch.tensor([1, 2, 3, 4, 5]))
        assert torch.allclose(result["labels"][0], torch.tensor([2, 3, 4, 5, 6]))

    def test_chunking_collate_exact_length(self):
        """Test collate with sequences exactly block_size + 1."""
        batch = [[1, 2, 3, 4, 5, 6]]  # Exactly block_size + 1
        block_size = 5
        pad_id = 0
        
        result = _chunking_collate(batch, block_size, pad_id)
        
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape == (1, 5)
        assert result["labels"].shape == (1, 5)
        
        # Check that sequences are not padded
        assert torch.allclose(result["input_ids"][0], torch.tensor([1, 2, 3, 4, 5]))
        assert torch.allclose(result["labels"][0], torch.tensor([2, 3, 4, 5, 6]))

    def test_chunking_collate_empty_batch(self):
        """Test collate with empty batch."""
        batch = []
        block_size = 5
        pad_id = 0
        
        result = _chunking_collate(batch, block_size, pad_id)
        
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape == (0,)
        assert result["labels"].shape == (0,)


class TestSFTCollateBatch:
    """Test the _sft_collate_batch function."""

    def test_sft_collate_batch_basic(self):
        """Test basic SFT collate functionality."""
        batch = [[1, 2, 3], [4, 5, 6, 7]]
        pad_token_id = 0
        ignore_index = -100
        allowed_max_length = 10
        
        result = _sft_collate_batch(batch, pad_token_id, ignore_index, allowed_max_length)
        
        assert "input_ids" in result
        assert "labels" in result
        # Max length is 4 (longest seq) + 1 (pad token) = 5, but truncated to allowed_max_length=10
        # But the actual result is (2, 4) because the algorithm works differently
        assert result["input_ids"].shape == (2, 4)
        assert result["labels"].shape == (2, 4)

    def test_sft_collate_batch_with_ignore_index(self):
        """Test SFT collate with ignore_index handling."""
        batch = [[1, 2, 3], [4, 5, 6, 7]]
        pad_token_id = 0
        ignore_index = -100
        allowed_max_length = 10
        
        result = _sft_collate_batch(batch, pad_token_id, ignore_index, allowed_max_length)
        
        # Check that padding tokens in targets are set to ignore_index
        # (except the first one)
        labels = result["labels"]
        for i in range(labels.shape[0]):
            pad_positions = (labels[i] == pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 1:
                # All but the first padding token should be ignore_index
                assert torch.all(labels[i][pad_positions[1:]] == ignore_index)

    def test_sft_collate_batch_with_max_length(self):
        """Test SFT collate with max_length truncation."""
        batch = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]  # Long sequence
        pad_token_id = 0
        ignore_index = -100
        allowed_max_length = 5
        
        result = _sft_collate_batch(batch, pad_token_id, ignore_index, allowed_max_length)
        
        assert result["input_ids"].shape == (1, 5)
        assert result["labels"].shape == (1, 5)

    def test_sft_collate_batch_empty(self):
        """Test SFT collate with empty batch."""
        batch = []
        pad_token_id = 0
        ignore_index = -100
        allowed_max_length = 10
        
        result = _sft_collate_batch(batch, pad_token_id, ignore_index, allowed_max_length)
        
        assert result["input_ids"].shape == (0,)
        assert result["labels"].shape == (0,)

    def test_sft_collate_batch_single_sequence(self):
        """Test SFT collate with single sequence."""
        batch = [[1, 2, 3]]
        pad_token_id = 0
        ignore_index = -100
        allowed_max_length = 10
        
        result = _sft_collate_batch(batch, pad_token_id, ignore_index, allowed_max_length)
        
        assert result["input_ids"].shape == (1, 3)  # 3 + 1 (pad) - 1 (for input) = 3
        assert result["labels"].shape == (1, 3)

    def test_sft_collate_batch_custom_inputs(self):
        """Test SFT collate with specific custom inputs from user."""
        # Test inputs as provided by user
        inputs_1 = [0, 1, 2, 3, 4]
        inputs_2 = [5, 6]
        inputs_3 = [7, 8, 9]
        
        batch = [inputs_1, inputs_2, inputs_3]
        pad_token_id = 50256  # GPT-2 pad token ID
        ignore_index = -100
        allowed_max_length = 10
        
        result = _sft_collate_batch(batch, pad_token_id, ignore_index, allowed_max_length)
        
        # Verify output structure
        assert "input_ids" in result
        assert "labels" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)
        
        # Check shapes: batch size should be 3, max length should be 5 (longest seq + 1 pad token - 1 for input)
        # The algorithm: max_len = max(seq_len + 1), then input = new_item[:-1], target = new_item[1:]
        # So final length = max_len - 1
        assert result["input_ids"].shape == (3, 5)
        assert result["labels"].shape == (3, 5)
        
        # Check that inputs are properly padded and shifted
        # Algorithm trace:
        # inputs_1: [0, 1, 2, 3, 4] -> new_item: [0, 1, 2, 3, 4, 50256] -> input: [0, 1, 2, 3, 4], target: [1, 2, 3, 4, 50256]
        # inputs_2: [5, 6] -> new_item: [5, 6, 50256, 50256, 50256, 50256] -> input: [5, 6, 50256, 50256, 50256], target: [6, 50256, -100, -100, -100]  
        # inputs_3: [7, 8, 9] -> new_item: [7, 8, 9, 50256, 50256, 50256] -> input: [7, 8, 9, 50256, 50256], target: [8, 9, 50256, -100, -100]
        
        # Verify first sequence (inputs_1)
        expected_input_1 = [0, 1, 2, 3, 4]  # Original sequence (no padding needed)
        expected_target_1 = [1, 2, 3, 4, 50256]  # Shifted by 1 + padding (only one pad token, so no ignore_index)
        assert torch.equal(result["input_ids"][0], torch.tensor(expected_input_1))
        assert torch.equal(result["labels"][0], torch.tensor(expected_target_1))
        
        # Verify second sequence (inputs_2) 
        expected_input_2 = [5, 6, 50256, 50256, 50256]  # Original + padding to match max length
        expected_target_2 = [6, 50256, -100, -100, -100]  # Shifted by 1 + padding with ignore_index applied
        assert torch.equal(result["input_ids"][1], torch.tensor(expected_input_2))
        assert torch.equal(result["labels"][1], torch.tensor(expected_target_2))
        
        # Verify third sequence (inputs_3)
        expected_input_3 = [7, 8, 9, 50256, 50256]  # Original + padding to match max length
        expected_target_3 = [8, 9, 50256, -100, -100]  # Shifted by 1 + padding with ignore_index applied
        assert torch.equal(result["input_ids"][2], torch.tensor(expected_input_3))
        assert torch.equal(result["labels"][2], torch.tensor(expected_target_3))
        
        # Verify ignore_index is applied to padding tokens in targets (except first)
        labels = result["labels"]
        for i in range(labels.shape[0]):
            pad_positions = (labels[i] == pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 1:
                # All but the first padding token should be ignore_index
                assert torch.all(labels[i][pad_positions[1:]] == ignore_index)


class TestBuildDatasetAndCollate:
    """Test the build_dataset_and_collate function."""

    @patch('data.dataset.TextFileDataset')
    def test_build_language_modeling_text(self, mock_text_dataset):
        """Test building language modeling text dataset."""
        mock_dataset = MagicMock()
        mock_text_dataset.return_value = mock_dataset
        
        cfg = {
            "train": {
                "data_loader": {
                    "kind": "language_modeling_text",
                    "block_size": 128
                }
            }
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_id = 0
        
        dataset, collate = build_dataset_and_collate(cfg, mock_tokenizer)
        
        assert dataset is mock_dataset
        assert callable(collate)

    @patch('data.dataset.TextFileDataset')
    def test_build_lm_text_alias(self, mock_text_dataset):
        """Test building dataset with 'lm_text' alias."""
        mock_dataset = MagicMock()
        mock_text_dataset.return_value = mock_dataset
        
        cfg = {
            "train": {
                "data_loader": {
                    "kind": "lm_text",
                    "block_size": 128
                }
            }
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_id = 0
        
        dataset, collate = build_dataset_and_collate(cfg, mock_tokenizer)
        
        assert dataset is mock_dataset
        assert callable(collate)

    @patch('data.dataset.SFTJsonDataset')
    @patch('data.dataset.build_text_pair_transform')
    def test_build_sft_json(self, mock_build_transform, mock_sft_dataset):
        """Test building SFT JSON dataset."""
        mock_dataset = MagicMock()
        mock_sft_dataset.return_value = mock_dataset
        mock_transform = MagicMock()
        mock_build_transform.return_value = mock_transform
        
        cfg = {
            "train": {
                "data_loader": {
                    "kind": "sft_json",
                    "block_size": 1024
                }
            },
            "model": {
                "modules": {
                    "loss": {
                        "params": {
                            "ignore_index": -100
                        }
                    }
                }
            }
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_id = 0
        
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = ["data/sft/file1.json"]
            dataset, collate = build_dataset_and_collate(cfg, mock_tokenizer)
        
        assert dataset is mock_dataset
        assert callable(collate)

    @patch('data.dataset.SFTJsonDataset')
    @patch('data.dataset.build_text_pair_transform')
    def test_build_sft_alias(self, mock_build_transform, mock_sft_dataset):
        """Test building dataset with 'sft' alias."""
        mock_dataset = MagicMock()
        mock_sft_dataset.return_value = mock_dataset
        mock_transform = MagicMock()
        mock_build_transform.return_value = mock_transform
        
        cfg = {
            "train": {
                "data_loader": {
                    "kind": "sft",
                    "block_size": 1024
                }
            }
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_id = 0
        
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = ["data/sft/file1.json"]
            dataset, collate = build_dataset_and_collate(cfg, mock_tokenizer)
        
        assert dataset is mock_dataset
        assert callable(collate)

    def test_build_unknown_kind(self):
        """Test building dataset with unknown kind raises ValueError."""
        cfg = {
            "train": {
                "data_loader": {
                    "kind": "unknown"
                }
            }
        }
        
        mock_tokenizer = MagicMock()
        
        with pytest.raises(ValueError, match="Unknown data_loader kind: unknown"):
            build_dataset_and_collate(cfg, mock_tokenizer)

    def test_build_default_kind(self):
        """Test building dataset with default kind."""
        cfg = {
            "train": {
                "data_loader": {}
            }
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_id = 0
        
        with patch('data.dataset.TextFileDataset') as mock_text_dataset:
            mock_dataset = MagicMock()
            mock_text_dataset.return_value = mock_dataset
            
            dataset, collate = build_dataset_and_collate(cfg, mock_tokenizer)
            
            assert dataset is mock_dataset
            assert callable(collate)
