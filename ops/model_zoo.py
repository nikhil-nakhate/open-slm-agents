"""Model registry utilities for ingestion configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from ops.config import load_config


_MODEL_ZOO: Dict[str, Dict[str, Any]] = {
    "openai/text-embedding-3-small": {
        "provider": "openai",
        "name": "text-embedding-3-small",
        "type": "embedding",
        "dimensions": 1536,
    },
    "openai/gpt-4o-mini": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "type": "chat",
    },
}

_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_model_from_config(name_or_path: str) -> Dict[str, Any]:
    cfg = load_config(name_or_path, search_dir=str(_MODEL_CONFIG_DIR))
    model_section = cfg.get("model")
    if isinstance(model_section, dict):
        resolved = deepcopy(model_section)
    else:
        resolved = deepcopy(cfg)
    resolved.setdefault("_source", {})["config"] = str(name_or_path)
    return resolved


def get_model(model_id: str) -> Dict[str, Any]:
    """Resolve a model configuration.

    The lookup order is:
      1. Exact match in the in-memory model zoo registry.
      2. A YAML config under ``configs/models`` matching ``model_id``.
    """

    if not model_id:
        raise ValueError("model_id must be a non-empty string")

    entry = _MODEL_ZOO.get(model_id)
    if entry is None:
        return _load_model_from_config(model_id)

    entry = deepcopy(entry)
    config_ref = entry.pop("config", None) or entry.pop("config_name", None)
    resolved = {}
    if config_ref:
        resolved = _deep_merge(resolved, _load_model_from_config(config_ref))

    resolved = _deep_merge(resolved, entry)
    return resolved


def get_model_config(name_or_path: str) -> Dict[str, Any]:
    """Load a model definition directly from ``configs/models``."""

    if not name_or_path:
        raise ValueError("name_or_path must be a non-empty string")
    return _load_model_from_config(name_or_path)


def available_models() -> Dict[str, Dict[str, Any]]:
    return {key: deepcopy(value) for key, value in _MODEL_ZOO.items()}
