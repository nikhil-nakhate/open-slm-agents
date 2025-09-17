import os
import argparse
import time
from typing import Dict, Any, Tuple, Optional

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


def _merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def build_dataloader_for_split(cfg: Dict[str, Any], tokenizer, split: str) -> DataLoader:
    """Builds a DataLoader for a specific split by merging split overrides.

    Supports optional per-split overrides under `train.splits.<split>`:
      - data_dir
      - batch_size, shuffle, num_workers, pin_memory, drop_last
      - data_loader: {...} (merged into base train.data_loader)
    """
    base_train = cfg.get("train", {})
    split_over = base_train.get("splits", {}).get(split, {})

    # Effective train section for this split
    eff_train: Dict[str, Any] = dict(base_train)
    if split_over.get("data_dir"):
        eff_train["data_dir"] = split_over["data_dir"]
    # Merge data_loader dicts (split overrides base)
    eff_train["data_loader"] = _merge_dict(base_train.get("data_loader", {}), split_over.get("data_loader", {}))

    # Build dataset + collate using an ephemeral cfg copy
    eff_cfg = dict(cfg)
    eff_cfg["train"] = eff_train
    dataset, collate = build_dataset_and_collate(eff_cfg, tokenizer)

    dl_cfg = eff_train.get("data_loader", {})
    # Defaults: train shuffles by default; eval/test don't
    default_shuffle = True if split == "train" else False
    batch_size = int(split_over.get("batch_size", eff_train.get("batch_size", 8)))
    shuffle = bool(split_over.get("shuffle", dl_cfg.get("shuffle", default_shuffle)))
    num_workers = int(split_over.get("num_workers", dl_cfg.get("num_workers", 0)))
    pin_memory = bool(split_over.get("pin_memory", dl_cfg.get("pin_memory", True)))
    drop_last = bool(split_over.get("drop_last", dl_cfg.get("drop_last", False)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )


def build_dataloaders(cfg: Dict[str, Any], tokenizer) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Returns (train_dl, eval_dl, test_dl) with support for ratio-based splitting.

    Two modes:
    1) Ratio mode: train.splits.ratios.{train,eval,test} define fractions. One dataset is
       built and split deterministically (seed) across these ratios.
    2) Per-split mode (fallback): each split can override data_dir and data_loader.
    """
    train_cfg = cfg.get("train", {})
    splits_cfg = train_cfg.get("splits", {})

    ratios = splits_cfg.get("ratios")
    if isinstance(ratios, dict):
        # Build the base dataset once
        dataset, collate = build_dataset_and_collate(cfg, tokenizer)
        n = len(dataset)
        r_train = float(ratios.get("train", 0.85))
        r_eval = float(ratios.get("eval", 0.10))
        r_test = float(ratios.get("test", 1.0 - (r_train + r_eval)))
        # Clamp and normalize minimal issues
        r_test = max(0.0, r_test)
        n_train = int(n * r_train)
        n_eval = int(n * r_eval)
        n_test = max(0, n - n_train - n_eval)
        g = torch.Generator()
        g.manual_seed(int(splits_cfg.get("seed", 42)))
        train_set, eval_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_eval, n_test], generator=g)

        dl_cfg = train_cfg.get("data_loader", {})
        batch_size = int(train_cfg.get("batch_size", 8))
        num_workers = int(dl_cfg.get("num_workers", 0))
        pin_memory = bool(dl_cfg.get("pin_memory", True))
        drop_last = bool(dl_cfg.get("drop_last", False))

        mk = lambda ds, shuffle: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate,
        )

        return mk(train_set, True), mk(eval_set, False), mk(test_set, False)

    # Fallback: per-split config
    train_dl = build_dataloader_for_split(cfg, tokenizer, split="train")
    eval_dl = build_dataloader_for_split(cfg, tokenizer, split="eval")
    test_dl = build_dataloader_for_split(cfg, tokenizer, split="test")
    return train_dl, eval_dl, test_dl


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

    # Build train/eval/test dataloaders
    train_dl, eval_dl, test_dl = build_dataloaders(cfg, tokenizer)

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
            for batch in train_dl:
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
