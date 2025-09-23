"""Shared utilities for working with agent configurations."""

from __future__ import annotations

import yaml
from copy import deepcopy
from typing import Any, Dict

from ops.chunking.chunker import Chunker, ChunkerFactory
from ops.model_zoo import get_model, get_model_config


def load_agent_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    agent_cfg = data.get("agent")
    if not agent_cfg:
        raise ValueError("Agent configuration must be nested under the 'agent' key")
    return agent_cfg


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_model_section(section: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(section or {})

    model_id = model_cfg.pop("model_zoo_id", None)
    config_name = model_cfg.pop("config_name", None)
    config_path = model_cfg.pop("config_path", None)

    resolved: Dict[str, Any] = {}
    if model_id:
        try:
            resolved = _merge_dicts(resolved, get_model(model_id))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Model zoo entry '{model_id}' not found and no matching config exists"
            ) from exc

    if config_name:
        resolved = _merge_dicts(resolved, get_model_config(config_name))

    if config_path:
        resolved = _merge_dicts(resolved, get_model_config(config_path))

    resolved = _merge_dicts(resolved, model_cfg)
    return resolved


def build_chunker(config: Dict[str, Any]) -> Chunker:
    strategy = (config or {}).get("strategy", "fixed")
    params = (config or {}).get("params", {})
    return ChunkerFactory.create_chunker(strategy, **params)


__all__ = [
    "load_agent_config",
    "resolve_model_section",
    "build_chunker",
]
