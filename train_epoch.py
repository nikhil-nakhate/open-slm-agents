import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ops.config import load_config
from models.build import build_model_from_cfg
from metrics import NoOpLogger
from data.dataset import build_dataset_and_collate
from scripts.load_gpt_weights import load_weights_into_gpt


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
    args = parser.parse_args()

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

if __name__ == "__main__":
    main()