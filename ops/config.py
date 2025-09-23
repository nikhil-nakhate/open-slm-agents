import os
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml


CONFIGS_ROOT = os.path.abspath("configs")
ALLOWED_DOMAINS = {
    "models": os.path.join(CONFIGS_ROOT, "models"),
    "agents": os.path.join(CONFIGS_ROOT, "agents"),
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(
    path_or_name: str,
    search_dir: str,
    domain_root: Optional[str],
) -> str:
    """Resolve a config reference to an absolute path within the permitted domain."""

    candidates = []
    if os.path.isabs(path_or_name):
        candidates.append(path_or_name)
    else:
        if os.path.exists(path_or_name):
            candidates.append(os.path.abspath(path_or_name))
        name = path_or_name
        # append extension if missing
        name_has_ext = os.path.splitext(name)[1] in {".yaml", ".yml"}

        base_searches = [search_dir]
        if domain_root:
            base_searches.insert(0, domain_root)
        elif search_dir == CONFIGS_ROOT:
            base_searches.extend(ALLOWED_DOMAINS.values())

        for base_dir in base_searches:
            if not name_has_ext:
                candidates.append(os.path.join(base_dir, f"{name}.yaml"))
                candidates.append(os.path.join(base_dir, f"{name}.yml"))
            candidates.append(os.path.join(base_dir, name))

    for cand in candidates:
        if os.path.exists(cand):
            resolved = os.path.abspath(cand)
            if domain_root and not os.path.commonpath([resolved, domain_root]) == domain_root:
                raise ValueError(
                    f"Config '{path_or_name}' resolves outside allowed domain '{domain_root}'."
                )
            return resolved

    raise FileNotFoundError(f"Config file not found: {path_or_name}")


def _infer_domain_root(resolved_path: str) -> Optional[str]:
    if not resolved_path.startswith(CONFIGS_ROOT):
        return None

    rel_parts = os.path.relpath(resolved_path, CONFIGS_ROOT).split(os.sep)
    if rel_parts:
        domain_key = rel_parts[0]
        if domain_key in ALLOWED_DOMAINS:
            return ALLOWED_DOMAINS[domain_key]
    return os.path.dirname(resolved_path)


def load_config(
    config_path_or_name: str,
    search_dir: str = "configs",
    *,
    domain_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads a YAML config. Supports hierarchical configs via an `extends` key that
    references another YAML file (relative to search_dir). Performs a deep merge
    where the child overrides the parent.

    Examples:
    - load_config("gpt2_base") -> configs/gpt2_base.yaml
    - load_config("/abs/path/model.yaml")
    - load_config("custom.yaml", search_dir="/path/to/configs")
    """
    search_dir_abs = os.path.abspath(search_dir)
    resolved_path = _resolve_path(config_path_or_name, search_dir_abs, domain_root)

    with open(resolved_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Handle hierarchical base config via `extends`
    extends = cfg.get("extends")
    if extends:
        inferred_domain = _infer_domain_root(resolved_path)
        if domain_root and inferred_domain and os.path.commonpath([inferred_domain, domain_root]) != domain_root:
            raise ValueError(
                f"Config '{config_path_or_name}' cannot extend outside its domain '{domain_root}'."
            )

        next_domain = domain_root or inferred_domain
        parent_cfg = load_config(extends, search_dir=next_domain or search_dir_abs, domain_root=next_domain)
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
