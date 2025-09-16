import os
import argparse
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ops.config import load_config
from models.build import build_model_from_cfg
from metrics import NoOpLogger
from data.dataset import build_dataset_and_collate


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], max_steps: int, start_step: int = 0):
    sch_cfg = cfg.get("train", {}).get("scheduler", {})
    kind = (sch_cfg.get("kind", "none") or "none").lower()
    warmup_steps = int(sch_cfg.get("warmup_steps", 0))
    min_lr = float(sch_cfg.get("min_lr", 0.0))

    if kind in {"none", "off", "disabled"}:
        return None

    def linear_warmup_decay(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # linear decay to min_lr
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr / cfg.get("train", {}).get("lr", 3e-4), 1.0 - progress)

    def cosine_warmup_decay(step: int):
        import math
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = min_lr / cfg.get("train", {}).get("lr", 3e-4)
        return max(min_factor, cosine)

    if kind in {"linear", "linear_warmup"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay, last_epoch=start_step)
    if kind in {"cosine", "cosine_warmup"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_warmup_decay, last_epoch=start_step)

    raise ValueError(f"Unknown scheduler kind: {kind}")


def create_logger(kind: str, run_name: str, cfg: Dict[str, Any]):
    kind = (kind or "none").lower()
    if kind in {"none", "noop", "print"}:
        return NoOpLogger(run_name=run_name, config=cfg)
    if kind in {"wandb", "weights_and_biases"}:
        from metrics.loggers.wandb_logger import WandBLogger
        project = cfg.get("train", {}).get("project", "open-slm")
        return WandBLogger(project=project, run_name=run_name, config=cfg)
    if kind in {"tb", "tensorboard"}:
        from metrics.loggers.tensorboard_logger import TensorBoardLogger
        log_dir = cfg.get("train", {}).get("log_dir", "runs")
        return TensorBoardLogger(run_name=run_name, config=cfg, log_dir=log_dir)
    raise ValueError(f"Unknown logger: {kind}")




def save_checkpoint(state: Dict[str, Any], out_dir: str, step: int):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"step_{step}.pt")
    torch.save(state, ckpt_path)
    return ckpt_path


def train_loop(cfg: Dict[str, Any], mode: str, logger_kind: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (model builds and holds the tokenizer)
    model = build_model_from_cfg(cfg).to(device)
    tokenizer = model.tokenizer
    # Optimizer
    train_cfg = cfg.get("train", {})
    lr = train_cfg.get("lr", 3e-4)
    betas = tuple(train_cfg.get("betas", (0.9, 0.95)))
    weight_decay = train_cfg.get("weight_decay", 0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # Dataset and DataLoader from config
    dataset, collate = build_dataset_and_collate(cfg, tokenizer)
    dl_cfg = train_cfg.get("data_loader", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    shuffle = bool(dl_cfg.get("shuffle", True))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    drop_last = bool(dl_cfg.get("drop_last", False))
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )

    # Logger and checkpointing
    run_name = train_cfg.get("run_name", f"{cfg.get('model',{}).get('name','model')}-{int(time.time())}")
    logger = create_logger(logger_kind, run_name, cfg)
    out_dir = train_cfg.get("output_dir", os.path.join("outputs", run_name))
    save_every = int(train_cfg.get("save_every", 500))
    max_steps = int(train_cfg.get("max_steps", 1000))
    log_every = int(train_cfg.get("log_every", 10))

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")
    # Optional resume
    resume_path = cfg.get("train", {}).get("resume", None)
    step = 0
    scheduler = None
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt.get("model", {}))
        opt.load_state_dict(ckpt.get("opt", {}))
        step = int(ckpt.get("step", 0))
        print(f"Resumed from {resume_path} at step {step}")
        # Build scheduler with correct starting step and load its state if available
        scheduler = _build_scheduler(opt, cfg, max_steps=max_steps, start_step=step)
        sch_state = ckpt.get("scheduler")
        if scheduler is not None and sch_state is not None:
            try:
                scheduler.load_state_dict(sch_state)
            except Exception:
                pass
    else:
        scheduler = _build_scheduler(opt, cfg, max_steps=max_steps)

    model.train()
    try:
        for epoch in range(10**9):  # run until max_steps
            for batch in dl:
                step += 1
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits, loss = model(x, targets=y)

                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                if scheduler is not None:
                    scheduler.step()

                if step % log_every == 0:
                    lr_cur = opt.param_groups[0]["lr"]
                    logger.log({"loss": float(loss.detach().cpu().item()), "lr": float(lr_cur)}, step=step)

                if step % save_every == 0:
                    ckpt_path = save_checkpoint({
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "cfg": cfg,
                        "step": step,
                    }, out_dir, step)
                    print(f"Saved checkpoint: {ckpt_path}")

                if step >= max_steps:
                    break
            if step >= max_steps:
                break
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train models from registry using YAML configs")
    parser.add_argument("--mode", type=str, default="pretraining", choices=["pretraining", "sft", "rl"], help="Training mode")
    parser.add_argument("--config", type=str, required=True, help="Config name or path (e.g., gpt2_base)")
    parser.add_argument("--logger", type=str, default="none", help="Logger: none|wandb|tensorboard")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Allow overriding resume via CLI; also keep in config under train.resume
    if args.resume:
        cfg.setdefault("train", {})["resume"] = args.resume

    if args.mode == "rl":
        # Skeleton: RL training requires reward fn and sampling; left as future work.
        raise NotImplementedError("RL mode is a placeholder in this template. Provide an RL trainer.")

    train_loop(cfg, args.mode, args.logger)


if __name__ == "__main__":
    main()
