"""Unit tests for data.transforms module."""

import pytest
from unittest.mock import patch, MagicMock

from data.transforms import build_text_pair_transform
from data.transforms.alpaca_transform import (
    format_input, AlpacaTransform, build_alpaca_transform
)


class TestFormatInput:
    """Test the format_input function."""

    def test_format_input_with_instruction_only(self):
        """Test formatting with instruction only."""
        entry = {"instruction": "Test instruction"}
        result = format_input(entry)
        
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert result == expected

    def test_format_input_with_instruction_and_input(self):
        """Test formatting with instruction and input."""
        entry = {
            "instruction": "Test instruction",
            "input": "Test input"
        }
        result = format_input(entry)
        
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
            "\n\n### Input:\nTest input"
        )
        assert result == expected

    def test_format_input_with_empty_input(self):
        """Test formatting with empty input."""
        entry = {
            "instruction": "Test instruction",
            "input": ""
        }
        result = format_input(entry)
        
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert result == expected

    def test_format_input_with_none_input(self):
        """Test formatting with None input."""
        entry = {
            "instruction": "Test instruction",
            "input": None
        }
        result = format_input(entry)
        
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert result == expected

    def test_format_input_with_missing_instruction(self):
        """Test formatting with missing instruction."""
        entry = {"input": "Test input"}
        result = format_input(entry)
        
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\n\n\n### Input:\nTest input"
        )
        assert result == expected

    def test_format_input_with_missing_input_key(self):
        """Test formatting with missing input key."""
        entry = {"instruction": "Test instruction"}
        result = format_input(entry)
        
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert result == expected

    def test_format_input_with_empty_instruction(self):
        """Test formatting with empty instruction."""
        entry = {"instruction": ""}
        result = format_input(entry)
        
        expected = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\n"
        )
        assert result == expected


class TestAlpacaTransform:
    """Test the AlpacaTransform class."""

    def test_call_with_instruction_only(self):
        """Test __call__ with instruction only."""
        transform = AlpacaTransform()
        entry = {"instruction": "Test instruction", "output": "Test output"}
        
        prompt, target = transform(entry)
        
        expected_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert prompt == expected_prompt
        assert target == "Test output"

    def test_call_with_instruction_and_input(self):
        """Test __call__ with instruction and input."""
        transform = AlpacaTransform()
        entry = {
            "instruction": "Test instruction",
            "input": "Test input",
            "output": "Test output"
        }
        
        prompt, target = transform(entry)
        
        expected_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
            "\n\n### Input:\nTest input"
        )
        assert prompt == expected_prompt
        assert target == "Test output"

    def test_call_with_missing_output(self):
        """Test __call__ with missing output."""
        transform = AlpacaTransform()
        entry = {"instruction": "Test instruction"}
        
        prompt, target = transform(entry)
        
        expected_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert prompt == expected_prompt
        assert target == ""

    def test_call_with_empty_output(self):
        """Test __call__ with empty output."""
        transform = AlpacaTransform()
        entry = {"instruction": "Test instruction", "output": ""}
        
        prompt, target = transform(entry)
        
        expected_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert prompt == expected_prompt
        assert target == ""

    def test_call_with_none_output(self):
        """Test __call__ with None output."""
        transform = AlpacaTransform()
        entry = {"instruction": "Test instruction", "output": None}
        
        prompt, target = transform(entry)
        
        expected_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
        )
        assert prompt == expected_prompt
        assert target is None

    def test_call_with_complex_entry(self):
        """Test __call__ with complex entry containing additional fields."""
        transform = AlpacaTransform()
        entry = {
            "instruction": "Test instruction",
            "input": "Test input",
            "output": "Test output",
            "extra_field": "This should be ignored"
        }
        
        prompt, target = transform(entry)
        
        expected_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n### Instruction:\nTest instruction"
            "\n\n### Input:\nTest input"
        )
        assert prompt == expected_prompt
        assert target == "Test output"


class TestBuildAlpacaTransform:
    """Test the build_alpaca_transform function."""

    def test_build_alpaca_transform(self):
        """Test building AlpacaTransform."""
        transform = build_alpaca_transform()
        
        assert callable(transform)
        assert isinstance(transform, AlpacaTransform)

    def test_build_alpaca_transform_functionality(self):
        """Test that built transform works correctly."""
        transform = build_alpaca_transform()
        entry = {"instruction": "Test", "output": "Output"}
        
        prompt, target = transform(entry)
        
        assert "Test" in prompt
        assert target == "Output"


class TestBuildTextPairTransform:
    """Test the build_text_pair_transform function."""

    def test_build_none_transform(self):
        """Test building none/identity transform."""
        cfg = {"kind": "none"}
        transform = build_text_pair_transform(cfg)
        
        # Test with sample entry
        entry = {"prompt": "Test prompt", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert prompt == "Test prompt"
        assert target == "Test output"

    def test_build_identity_transform(self):
        """Test building identity transform."""
        cfg = {"kind": "identity"}
        transform = build_text_pair_transform(cfg)
        
        # Test with sample entry
        entry = {"prompt": "Test prompt", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert prompt == "Test prompt"
        assert target == "Test output"

    def test_build_off_transform(self):
        """Test building off transform."""
        cfg = {"kind": "off"}
        transform = build_text_pair_transform(cfg)
        
        # Test with sample entry
        entry = {"prompt": "Test prompt", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert prompt == "Test prompt"
        assert target == "Test output"

    def test_build_alpaca_transform(self):
        """Test building AlpacaTransform."""
        cfg = {"kind": "alpaca"}
        transform = build_text_pair_transform(cfg)
        
        # Test with sample entry
        entry = {"instruction": "Test instruction", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert "Test instruction" in prompt
        assert target == "Test output"

    def test_build_alpaca_transform_alias(self):
        """Test building AlpacaTransform with alias."""
        cfg = {"kind": "alpaca_transform"}
        transform = build_text_pair_transform(cfg)
        
        # Test with sample entry
        entry = {"instruction": "Test instruction", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert "Test instruction" in prompt
        assert target == "Test output"

    def test_build_transform_default_kind(self):
        """Test building transform with default kind."""
        cfg = {}
        transform = build_text_pair_transform(cfg)
        
        # Test with sample entry
        entry = {"prompt": "Test prompt", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert prompt == "Test prompt"
        assert target == "Test output"

    def test_build_transform_none_config(self):
        """Test building transform with None config."""
        transform = build_text_pair_transform(None)
        
        # Test with sample entry
        entry = {"prompt": "Test prompt", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert prompt == "Test prompt"
        assert target == "Test output"

    def test_build_transform_case_insensitive(self):
        """Test building transform with case insensitive kind."""
        cfg = {"kind": "NONE"}
        transform = build_text_pair_transform(cfg)
        
        # Test with sample entry
        entry = {"prompt": "Test prompt", "output": "Test output"}
        prompt, target = transform(entry)
        
        assert prompt == "Test prompt"
        assert target == "Test output"

    def test_build_transform_unknown_kind(self):
        """Test building transform with unknown kind raises ValueError."""
        cfg = {"kind": "unknown"}
        
        with pytest.raises(ValueError, match="Unknown text-pair transform kind: unknown"):
            build_text_pair_transform(cfg)

    def test_build_transform_missing_prompt_output(self):
        """Test building transform with missing prompt/output keys."""
        cfg = {"kind": "none"}
        transform = build_text_pair_transform(cfg)
        
        # Test with entry missing prompt and output
        entry = {"instruction": "Test instruction"}
        prompt, target = transform(entry)
        
        assert prompt == ""
        assert target == ""

    def test_build_transform_with_none_values(self):
        """Test building transform with None values."""
        cfg = {"kind": "none"}
        transform = build_text_pair_transform(cfg)
        
        # Test with entry containing None values
        entry = {"prompt": None, "output": None}
        prompt, target = transform(entry)
        
        assert prompt is None
        assert target is None

    def test_build_transform_functionality(self):
        """Test that built transforms work correctly."""
        # Test none transform
        none_transform = build_text_pair_transform({"kind": "none"})
        entry = {"prompt": "Hello", "output": "World"}
        prompt, target = none_transform(entry)
        assert prompt == "Hello"
        assert target == "World"
        
        # Test alpaca transform
        alpaca_transform = build_text_pair_transform({"kind": "alpaca"})
        entry = {"instruction": "Say hello", "output": "Hello!"}
        prompt, target = alpaca_transform(entry)
        assert "Say hello" in prompt
        assert target == "Hello!"
