from typing import Any, Dict


class BaseLogger:
    def __init__(self, project: str = None, run_name: str = None, config: Dict[str, Any] = None, **kwargs):
        self.project = project
        self.run_name = run_name
        self.config = config or {}

    def log(self, metrics: Dict[str, Any], step: int = None):
        raise NotImplementedError

    def watch(self, model: Any):
        pass

    def close(self):
        pass


class NoOpLogger(BaseLogger):
    def log(self, metrics: Dict[str, Any], step: int = None):
        # Simple print for visibility
        msg = {"step": step, **metrics} if step is not None else metrics
        print(f"[LOG] {msg}")

