"""Unit tests for ops.config module."""

import os
import tempfile
import pytest
import yaml
from unittest.mock import patch, mock_open

from ops.config import load_config, freeze_flags_from_cfg, _deep_update


class TestDeepUpdate:
    """Test the _deep_update function."""

    def test_simple_override(self):
        """Test simple key-value override."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_update(base, override)
        expected = {"a": 1, "b": 3, "c": 4}
        assert result == expected

    def test_nested_override(self):
        """Test nested dictionary override."""
        base = {"a": 1, "b": {"x": 10, "y": 20}}
        override = {"b": {"y": 30, "z": 40}}
        result = _deep_update(base, override)
        expected = {"a": 1, "b": {"x": 10, "y": 30, "z": 40}}
        assert result == expected

    def test_deeply_nested_override(self):
        """Test deeply nested dictionary override."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"d": 3, "e": 4}}}
        result = _deep_update(base, override)
        expected = {"a": {"b": {"c": 1, "d": 3, "e": 4}}}
        assert result == expected

    def test_override_with_none(self):
        """Test override with None values."""
        base = {"a": 1, "b": 2}
        override = {"b": None}
        result = _deep_update(base, override)
        expected = {"a": 1, "b": None}
        assert result == expected

    def test_empty_override(self):
        """Test with empty override dictionary."""
        base = {"a": 1, "b": 2}
        override = {}
        result = _deep_update(base, override)
        assert result == base

    def test_original_unchanged(self):
        """Test that original dictionaries are not modified."""
        base = {"a": 1, "b": {"x": 10}}
        override = {"b": {"y": 20}}
        _deep_update(base, override)
        assert base == {"a": 1, "b": {"x": 10}}
        assert override == {"b": {"y": 20}}


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_absolute_path(self):
        """Test loading config from absolute path."""
        config_data = {"model": {"dim": 512}, "train": {"lr": 0.001}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_config(temp_path)
            assert result == config_data
        finally:
            os.unlink(temp_path)

    def test_load_relative_path_with_yaml_extension(self):
        """Test loading config from relative path with .yaml extension."""
        config_data = {"model": {"dim": 512}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir='configs') as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_config(os.path.basename(temp_path))
            assert result == config_data
        finally:
            os.unlink(temp_path)

    def test_load_relative_path_without_extension(self):
        """Test loading config from relative path without extension."""
        config_data = {"model": {"dim": 512}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir='configs') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
            base_name = os.path.splitext(os.path.basename(temp_path))[0]

        try:
            result = load_config(base_name)
            assert result == config_data
        finally:
            os.unlink(temp_path)

    def test_load_with_extends(self):
        """Test loading config with hierarchical extends."""
        parent_config = {"model": {"dim": 512}, "train": {"lr": 0.001}}
        child_config = {"extends": "parent", "model": {"dim": 768}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir='configs') as f:
            yaml.dump(parent_config, f)
            parent_path = f.name
            parent_name = os.path.splitext(os.path.basename(parent_path))[0]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir='configs') as f:
            yaml.dump(child_config, f)
            child_path = f.name
            child_name = os.path.splitext(os.path.basename(child_path))[0]

        try:
            # Update the child config to reference the correct parent name
            child_config["extends"] = parent_name
            with open(child_path, 'w') as f:
                yaml.dump(child_config, f)
            
            result = load_config(child_name)
            expected = {"model": {"dim": 768}, "train": {"lr": 0.001}}
            # Remove the 'extends' key from result as it should not be in the final config
            result_without_extends = {k: v for k, v in result.items() if k != "extends"}
            assert result_without_extends == expected
        finally:
            os.unlink(parent_path)
            os.unlink(child_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_empty_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            result = load_config(temp_path)
            assert result == {}
        finally:
            os.unlink(temp_path)

    def test_load_yaml_with_comments(self):
        """Test loading YAML file with comments."""
        config_data = {"model": {"dim": 512}}  # Comments should be ignored
        yaml_content = """
# This is a comment
model:
  dim: 512  # Another comment
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = load_config(temp_path)
            assert result == config_data
        finally:
            os.unlink(temp_path)


class TestFreezeFlagsFromCfg:
    """Test the freeze_flags_from_cfg function."""

    def test_default_freeze_flags(self):
        """Test default freeze flags when no config provided."""
        cfg = {}
        result = freeze_flags_from_cfg(cfg)
        expected = {
            "tokenizer": False,
            "embedding": False,
            "transformer": False,
            "output_projection": False,
            "loss": False,
        }
        assert result == expected

    def test_freeze_flags_from_config(self):
        """Test freeze flags extracted from config."""
        cfg = {
            "model": {
                "modules": {
                    "tokenizer": {"freeze": True},
                    "embedding": {"freeze": False},
                    "transformer": {"freeze": 1},  # Truthy value
                    "output_projection": {"freeze": "yes"},  # Truthy value
                    "loss": {"freeze": 0},  # Falsy value
                }
            }
        }
        result = freeze_flags_from_cfg(cfg)
        expected = {
            "tokenizer": True,
            "embedding": False,
            "transformer": True,
            "output_projection": True,
            "loss": False,
        }
        assert result == expected

    def test_freeze_flags_missing_modules(self):
        """Test freeze flags when modules section is missing."""
        cfg = {"model": {}}
        result = freeze_flags_from_cfg(cfg)
        expected = {
            "tokenizer": False,
            "embedding": False,
            "transformer": False,
            "output_projection": False,
            "loss": False,
        }
        assert result == expected

    def test_freeze_flags_missing_model(self):
        """Test freeze flags when model section is missing."""
        cfg = {"other": "value"}
        result = freeze_flags_from_cfg(cfg)
        expected = {
            "tokenizer": False,
            "embedding": False,
            "transformer": False,
            "output_projection": False,
            "loss": False,
        }
        assert result == expected

    def test_freeze_flags_partial_config(self):
        """Test freeze flags with partial module configuration."""
        cfg = {
            "model": {
                "modules": {
                    "tokenizer": {"freeze": True},
                    "transformer": {"freeze": True},
                }
            }
        }
        result = freeze_flags_from_cfg(cfg)
        expected = {
            "tokenizer": True,
            "embedding": False,
            "transformer": True,
            "output_projection": False,
            "loss": False,
        }
        assert result == expected
