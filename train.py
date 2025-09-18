import os
import argparse
import time
import pickle
import math
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ops.config import load_config
from models.build import build_model_from_cfg
from metrics import NoOpLogger
from data.dataset import build_dataset_and_collate
from scripts.load_gpt_weights import load_weights_into_gpt


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], max_steps: int, start_step: int = 0):
    sch_cfg = cfg.get("train", {}).get("scheduler", {})
    
    # Handle case where scheduler is explicitly disabled
    if sch_cfg is None or sch_cfg == {}:
        return None
        
    kind = (sch_cfg.get("kind", "none") or "none").lower()
    warmup_steps = int(sch_cfg.get("warmup_steps", 0))
    min_lr = float(sch_cfg.get("min_lr", 0.0))

    if kind in {"none", "off", "disabled", "null"}:
        return None

    if kind in {"cosine", "cosine_warmup"}:
        if warmup_steps > 0:
            # Custom cosine with warmup - more robust implementation
            def cosine_warmup_decay(step: int):
                if step < warmup_steps:
                    # Linear warmup
                    return float(step) / float(max(1, warmup_steps))
                # Cosine decay after warmup
                progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_factor = min_lr / optimizer.param_groups[0]["lr"]
                return max(min_factor, cosine)
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_warmup_decay, last_epoch=-1)
        else:
            # Use built-in CosineAnnealingLR
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps, eta_min=min_lr, last_epoch=-1
            )
    elif kind in {"linear", "linear_warmup"}:
        if warmup_steps > 0:
            # Custom linear with warmup
            def linear_warmup_decay(step: int):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(min_lr / optimizer.param_groups[0]["lr"], 1.0 - progress)
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay, last_epoch=-1)
        else:
            # Use built-in LinearLR for decay
            sch = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=min_lr / optimizer.param_groups[0]["lr"], total_iters=max_steps, last_epoch=-1
            )
    elif kind == "constant":
        # Constant learning rate
        return None
    else:
        raise ValueError(f"Unknown scheduler kind: {kind}")

    if start_step > 0:
        sch.last_epoch = start_step
    return sch


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


def _load_weights_from_path(weights_dir: str, model, device):
    """Load weights from a directory or file path."""
    path = weights_dir
    params = None
    if os.path.isdir(path):
        pkl = os.path.join(path, "params.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                params = pickle.load(f)
        else:
            raise FileNotFoundError(f"No params.pt or params.pkl found in {path}")
    else:
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                params = pickle.load(f)
        else:
            raise ValueError("weights_dir must be a directory or a .pt/.pkl file")
    load_weights_into_gpt(model, params)
    model.to(device)


def _resume_from_checkpoint(resume_path: str, model, opt, device):
    """Resume training from a checkpoint."""
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", {}))
    opt.load_state_dict(ckpt.get("opt", {}))
    step = int(ckpt.get("step", 0))
    print(f"Resumed from {resume_path} at step {step}")
    return step, ckpt.get("scheduler")


def _build_optimizer(model, cfg: Dict[str, Any]):
    """Build optimizer from config."""
    opt_cfg = cfg.get("train", {}).get("optimizer", {})
    kind = opt_cfg.get("kind", "AdamW").lower()
    
    lr = float(opt_cfg.get("lr", 3e-4))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
    weight_decay = float(opt_cfg.get("weight_decay", 0.1))
    eps = float(opt_cfg.get("eps", 1e-8))
    
    if kind == "adamw":
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
            eps=eps
        )
    elif kind == "adam":
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
            eps=eps
        )
    elif kind == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer kind: {kind}")


def _setup_training_components(cfg: Dict[str, Any], device, logger_kind: str):
    """Setup model, optimizer, dataloaders, and other training components."""
    # Build model (model builds and holds the tokenizer)
    model = build_model_from_cfg(cfg).to(device)
    tokenizer = model.tokenizer
    
    # Build optimizer
    opt = _build_optimizer(model, cfg)

    # Build train/eval/test dataloaders
    train_dl, eval_dl, test_dl = build_dataloaders(cfg, tokenizer)

    # Logger and checkpointing
    train_cfg = cfg.get("train", {})
    run_name = train_cfg.get("run_name", f"{cfg.get('model',{}).get('name','model')}-{int(time.time())}")
    logger = create_logger(logger_kind, run_name, cfg)
    out_dir = train_cfg.get("output_dir", os.path.join("outputs", run_name))
    save_every = int(train_cfg.get("save_every", 50))
    max_steps = int(train_cfg.get("max_steps", 1000))
    log_every = int(train_cfg.get("log_every", 10))

    # scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")
    scaler = None
    
    return model, tokenizer, opt, train_dl, eval_dl, test_dl, logger, out_dir, save_every, max_steps, log_every, scaler


def _calc_loss_batch(logits, target_batch, model, device):
    """Calculate loss for a single batch using the model's loss function."""
    target_batch = target_batch.to(device)
    print("logits.shape, target_batch.shape", logits.shape, target_batch.shape)
    loss = model.loss_fn(logits, target_batch)
    return loss


def _training_step(model, batch, opt, scaler, scheduler, device, grad_clip=1.0, grad_accum_steps=1, step_count=0):
    """Execute a single training step with gradient clipping and accumulation."""
    x = batch["input_ids"].to(device)
    y = batch["labels"].to(device)

    # Forward pass
    logits = model(x)
    # Calculate loss using the model's loss function
    loss = _calc_loss_batch(logits, y, model, device)
    
    # Scale loss for gradient accumulation
    loss = loss / grad_accum_steps

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # Only step optimizer and scheduler after accumulating gradients
    should_step = step_count % grad_accum_steps == 0
    print("should_step", should_step)
    if should_step:
        # Gradient clipping
        if grad_clip > 0:
            print("grad_clip", grad_clip)
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Step optimizer
        if scaler is not None and scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        
        opt.zero_grad()

        if scheduler is not None:
            scheduler.step()
    
    return loss * grad_accum_steps  # Return unscaled loss


def _evaluate_model(model, eval_dl, device, max_eval_batches=100):
    """Evaluate model on validation set using notebook's approach."""
    if eval_dl is None:
        return None
        
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        if len(eval_dl) == 0:
            val_loss = float("nan")
        else:
            num_batches = min(max_eval_batches, len(eval_dl))
            for i, batch in enumerate(eval_dl):
                if i < num_batches:
                    # Handle both tuple format (notebook) and dict format (our system)
                    if isinstance(batch, dict):
                        input_batch = batch["input_ids"]
                        target_batch = batch["labels"]
                    else:
                        input_batch, target_batch = batch
                    
                    # Forward pass
                    logits = model(input_batch.to(device))
                    loss = _calc_loss_batch(logits, target_batch, model, device)
                    total_loss += loss.item()
                else:
                    break
            val_loss = total_loss / num_batches
    model.train()
    return val_loss


def _log_training_progress(step, loss, opt, logger, log_every, eval_loss=None):
    """Log training progress to console and logger."""
    loss_value = float(loss.detach().cpu().item())
    
    if eval_loss is not None:
        print(f"Step {step:6d} | Train Loss: {loss_value:.6f} | Eval Loss: {eval_loss:.6f}")
    else:
        print(f"Step {step:6d} | Loss: {loss_value:.6f}")
    
    if step % log_every == 0:
        lr_cur = opt.param_groups[0]["lr"]
        log_data = {"loss": loss_value, "lr": float(lr_cur)}
        if eval_loss is not None:
            log_data["eval_loss"] = eval_loss
        logger.log(log_data, step=step)


def _save_checkpoint_if_needed(step, save_every, model, opt, scheduler, cfg, out_dir, batch):
    """Save checkpoint if it's time to do so."""
    if step % save_every == 0:
        print(f"\n--- Saving checkpoint at step {step} ---")
        print("Sample input_ids:", model.tokenizer.decode(batch["input_ids"][0].tolist()[:50]) + "...")
        print("Sample labels:", model.tokenizer.decode(batch["labels"][0].tolist()[:50]) + "...")
        
        ckpt_path = save_checkpoint({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": cfg,
            "step": step,
        }, out_dir, step)
        print(f"Checkpoint saved: {ckpt_path}\n")


def train_loop(cfg: Dict[str, Any], mode: str, logger_kind: str, weights_dir: Optional[str] = None):
    """Main training loop function with modern improvements."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup training components
    model, tokenizer, opt, train_dl, eval_dl, test_dl, logger, out_dir, save_every, max_steps, log_every, scaler = _setup_training_components(cfg, device, logger_kind)
    
    # Training hyperparameters
    train_cfg = cfg.get("train", {})
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    eval_every = int(train_cfg.get("eval_every", 100))
    max_eval_batches = int(train_cfg.get("max_eval_batches", 100))
    
    # Optional initialize from converted GPT weights or resume from checkpoint
    resume_path = cfg.get("train", {}).get("resume", None)
    step = 0
    scheduler = None
    
    if weights_dir:
        _load_weights_from_path(weights_dir, model, device)
        scheduler = _build_scheduler(opt, cfg, max_steps=max_steps)
    elif resume_path:
        step, sch_state = _resume_from_checkpoint(resume_path, model, opt, device)
        scheduler = _build_scheduler(opt, cfg, max_steps=max_steps, start_step=step)
        if scheduler is not None and sch_state is not None:
            scheduler.load_state_dict(sch_state)
    else:
        scheduler = _build_scheduler(opt, cfg, max_steps=max_steps)

    model.train()
    print(f"Starting training for {max_steps} steps...")
    print(f"Gradient clipping: {grad_clip}, Accumulation steps: {grad_accum_steps}")
    print(f"Evaluation every: {eval_every} steps")
    print(f"Training dataset size: {len(train_dl.dataset)}")
    print(f"Eval dataset size: {len(eval_dl.dataset) if eval_dl else 0}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(10**9):  # run until max_steps
        for batch in train_dl:
            step += 1
            
            # Training step
            loss = _training_step(
                model, batch, opt, scaler, scheduler, device, 
                grad_clip=grad_clip, grad_accum_steps=grad_accum_steps, step_count=step
            )
            
            # Evaluation
            eval_loss = None
            if step % eval_every == 0 and eval_dl is not None:
                eval_loss = _evaluate_model(model, eval_dl, device, max_eval_batches)
            
            # Logging
            _log_training_progress(step, loss, opt, logger, log_every, eval_loss)
            
            # Checkpointing
            _save_checkpoint_if_needed(step, save_every, model, opt, scheduler, cfg, out_dir, batch)

            if step >= max_steps:
                break
        if step >= max_steps:
            break
    
    print(f"Training completed after {step} steps.")
    logger.close()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, 
                       eval_freq=5, eval_iter=5, start_context=None, tokenizer=None):
    """
    Simple training function based on the notebook's train_model_simple.
    No bells and whistles - just basic training loop.
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch in train_loader:
            # Handle both tuple format (notebook) and dict format (our system)
            if isinstance(batch, dict):
                input_batch = batch["input_ids"]
                target_batch = batch["labels"]
            else:
                input_batch, target_batch = batch
                
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            
            # Forward pass
            logits = model(input_batch.to(device))
            loss = model.loss_fn(logits, target_batch.to(device))
            
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            
            tokens_seen += input_batch.numel()  # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_simple(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch if context and tokenizer provided
        if start_context and tokenizer:
            generate_and_print_sample_simple(
                model, tokenizer, device, start_context
            )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model_simple(model, train_loader, val_loader, device, eval_iter):
    """Simple evaluation function matching the notebook."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_simple(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader_simple(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def calc_loss_loader_simple(data_loader, model, device, num_batches=None):
    """Simple loss calculation matching the notebook."""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            # Handle both tuple format (notebook) and dict format (our system)
            if isinstance(batch, dict):
                input_batch = batch["input_ids"]
                target_batch = batch["labels"]
            else:
                input_batch, target_batch = batch
                
            # Forward pass
            logits = model(input_batch.to(device))
            loss = model.loss_fn(logits, target_batch.to(device))
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def generate_and_print_sample_simple(model, tokenizer, device, start_context):
    """Simple text generation matching the notebook."""
    model.eval()
    # Get context size from model config - it's stored in the model's init args
    context_size = getattr(model, 'max_seq_len', 1024)
    
    # Encode the start context
    if isinstance(start_context, str):
        encoded = tokenizer.encode(start_context)
    else:
        encoded = start_context
    
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    
    # Decode the generated text
    if hasattr(tokenizer, 'decode'):
        decoded_text = tokenizer.decode(token_ids[0].cpu().tolist())
    else:
        # Fallback for simple tokenizers
        decoded_text = str(token_ids[0].cpu().tolist())
    
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Simple text generation matching the notebook."""
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)  # batch, n_tokens, vocab_size

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    parser = argparse.ArgumentParser(description="Train models from registry using YAML configs")
    parser.add_argument("--mode", type=str, default="pretraining", choices=["pretraining", "sft", "rl"], help="Training mode")
    parser.add_argument("--config", type=str, required=True, help="Config name or path (e.g., gpt2_base)")
    parser.add_argument("--logger", type=str, default="none", help="Logger: none|wandb|tensorboard")
    parser.add_argument("--weights_dir", type=str, default=None, help="Path to converted GPT weights (dir or .pt/.pkl)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--simple", action="store_true", help="Use simple training function (no bells and whistles)")
    args = parser.parse_args()

    if args.simple:
        # Simple training mode
        cfg = load_config(args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model
        model = build_model_from_cfg(cfg)
        if args.weights_dir:
            # Use the existing weight loading function
            from scripts.load_gpt_weights import load_weights_into_gpt
            params = torch.load(f"{args.weights_dir}/params.pt", map_location=device, weights_only=False)
            load_weights_into_gpt(model, params)
        model = model.to(device)
        
        # Build dataset and dataloaders
        tokenizer = model.tokenizer
        dataset, collate = build_dataset_and_collate(cfg, tokenizer)
        
        # Split dataset
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.get("train", {}).get("batch_size", 8),
            shuffle=True, collate_fn=collate, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.get("train", {}).get("batch_size", 8),
            shuffle=False, collate_fn=collate, num_workers=0
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.get("train", {}).get("optimizer", {}).get("lr", 0.00005),
            weight_decay=cfg.get("train", {}).get("optimizer", {}).get("weight_decay", 0.1)
        )
        
        # Get a sample context for generation
        if len(val_dataset) > 0:
            sample = val_dataset[0]
            if isinstance(sample, dict):
                start_context = sample["input_ids"][:20]  # First 20 tokens
            else:
                start_context = sample[:20]
        else:
            start_context = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        
        # Train
        num_epochs = 2
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context=start_context, tokenizer=tokenizer
        )
        
        print(f"Training completed. Final train loss: {train_losses[-1]:.3f}, Final val loss: {val_losses[-1]:.3f}")
    else:
        # Regular training mode
        cfg = load_config(args.config)
        # Allow overriding resume/weights via CLI; also keep in config under train.*
        if args.resume:
            cfg.setdefault("train", {})["resume"] = args.resume
        weights_dir = args.weights_dir

        if args.mode == "rl":
            # Skeleton: RL training requires reward fn and sampling; left as future work.
            raise NotImplementedError("RL mode is a placeholder in this template. Provide an RL trainer.")

        train_loop(cfg, args.mode, args.logger, weights_dir=weights_dir)


if __name__ == "__main__":
    main()
