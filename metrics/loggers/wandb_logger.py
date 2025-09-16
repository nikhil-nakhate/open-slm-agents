from typing import Any, Dict

from .base import BaseLogger


class WandBLogger(BaseLogger):
    def __init__(self, project: str = None, run_name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(project, run_name, config, **kwargs)
        try:
            import wandb  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("wandb is not installed. pip install wandb") from e

        self.wandb = wandb
        self.run = self.wandb.init(project=project, name=run_name, config=config, **kwargs)

    def log(self, metrics: Dict[str, Any], step: int = None):
        self.wandb.log(metrics, step=step)

    def watch(self, model: Any):
        self.wandb.watch(model)

    def close(self):
        if self.run is not None:
            self.run.finish()

