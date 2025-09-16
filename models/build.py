from typing import Any, Dict

from . import get_model_class


def build_model_from_cfg(cfg: Dict[str, Any]):
    model_name = cfg.get("model", {}).get("name")
    if not model_name:
        raise KeyError("Config must include model.name")
    ModelCls = get_model_class(model_name)
    if not hasattr(ModelCls, "from_config"):
        raise AttributeError(f"Model class '{ModelCls.__name__}' missing 'from_config' method")
    return ModelCls.from_config(cfg)

