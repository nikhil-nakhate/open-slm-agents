from typing import Callable, Dict, Tuple
from .alpaca_transform import build_alpaca_transform


def build_text_pair_transform(cfg: Dict) -> Callable[[Dict], Tuple[str, str]]:
    kind = (cfg or {}).get("kind", "none").lower()
    if kind in {"none", "identity", "off"}:
        return lambda e: (e.get("prompt", ""), e.get("output", ""))
    if kind in {"alpaca", "alpaca_transform"}:
        return build_alpaca_transform()
    raise ValueError(f"Unknown text-pair transform kind: {kind}")

