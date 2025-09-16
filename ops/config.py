import os
import yaml
from copy import deepcopy
from typing import Any, Dict


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path_or_name: str, search_dir: str = "configs") -> Dict[str, Any]:
    """
    Loads a YAML config. Supports hierarchical configs via an `extends` key that
    references another YAML file (relative to search_dir). Performs a deep merge
    where the child overrides the parent.
    
    Examples:
    - load_config("gpt2_base") -> configs/gpt2_base.yaml
    - load_config("/abs/path/model.yaml")
    - load_config("custom.yaml", search_dir="/path/to/configs")
    """
    path = config_path_or_name
    if not os.path.isabs(path):
        # Allow bare names without .yaml
        if not path.endswith(".yaml") and not path.endswith(".yml"):
            cand = os.path.join(search_dir, f"{path}.yaml")
            if os.path.exists(cand):
                path = cand
            else:
                cand_yml = os.path.join(search_dir, f"{path}.yml")
                if os.path.exists(cand_yml):
                    path = cand_yml
                else:
                    # If provided string has a slash, treat as relative file
                    path = os.path.join(search_dir, path)
        else:
            # file name provided without directory -> search in search_dir
            if not os.path.dirname(path):
                path = os.path.join(search_dir, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Handle hierarchical base config via `extends`
    extends = cfg.get("extends")
    if extends:
        parent_cfg = load_config(extends, search_dir=search_dir)
        cfg = _deep_update(parent_cfg, cfg)

    return cfg


def freeze_flags_from_cfg(cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Extracts freeze flags for modules from config, defaulting to False."""
    mods = cfg.get("model", {}).get("modules", {})
    return {
        "tokenizer": bool(mods.get("tokenizer", {}).get("freeze", False)),
        "embedding": bool(mods.get("embedding", {}).get("freeze", False)),
        "transformer": bool(mods.get("transformer", {}).get("freeze", False)),
        "output_projection": bool(mods.get("output_projection", {}).get("freeze", False)),
        "loss": bool(mods.get("loss", {}).get("freeze", False)),
    }

