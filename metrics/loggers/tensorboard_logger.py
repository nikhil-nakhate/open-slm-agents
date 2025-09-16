from typing import Any, Dict
import os

from .base import BaseLogger


class TensorBoardLogger(BaseLogger):
    def __init__(self, project: str = None, run_name: str = None, config: Dict[str, Any] = None, log_dir: str = "runs", **kwargs):
        super().__init__(project, run_name, config, **kwargs)
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("TensorBoard not available. pip install tensorboard") from e
        run_dir = os.path.join(log_dir, run_name or "run")
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(run_dir)

    def log(self, metrics: Dict[str, Any], step: int = None):
        step = 0 if step is None else step
        for k, v in metrics.items():
            try:
                self.writer.add_scalar(k, v, step)
            except Exception:
                pass

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

