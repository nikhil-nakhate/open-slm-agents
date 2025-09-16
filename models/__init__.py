from typing import Dict, Type


_MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str):
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name]


def available_models() -> Dict[str, Type]:
    return dict(_MODEL_REGISTRY)

# Import built-in models to populate registry
from .meta_arch import gpt  # noqa: F401,E402
