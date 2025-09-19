import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ops.config import load_config
from models.build import build_model_from_cfg
from metrics import NoOpLogger
from data.dataset import build_dataset_and_collate
from scripts.load_gpt_weights import load_weights_into_gpt
from tqdm import tqdm



def process_batch_samples(input_batch: torch.Tensor, target_batch: torch.Tensor, 
                         model: Any, device: Any, tokenizer: Any, 
                         max_new_tokens: int, context_size: int, batch_idx: int, 
                         max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Process all samples in a batch using multithreading.
    This function includes both generation and post-processing for each sample.
    """
    def process_single_sample(i: int) -> Dict[str, Any]:
        """Process a single sample within the batch - includes generation."""
        input_tokens = input_batch[i]
        target_tokens = target_batch[i]
        
        # Generate response for this single sample
        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=input_tokens.unsqueeze(0).to(device),  # Add batch dimension
                max_new_tokens=max_new_tokens,
                context_size=context_size
            )
            # Remove batch dimension
            token_ids = token_ids.squeeze(0)
        
        # Decode input and target to get the formatted text
        if hasattr(tokenizer, 'decode'):
            input_text = tokenizer.decode(input_tokens.cpu().tolist())
            # Filter out ignore index tokens for target decoding
            target_tokens_filtered = target_tokens[target_tokens != -100]
            if len(target_tokens_filtered) > 0:
                target_text = tokenizer.decode(target_tokens_filtered.cpu().tolist())
            else:
                target_text = ""
        else:
            input_text = str(input_tokens.cpu().tolist())
            target_text = str(target_tokens.cpu().tolist())
        
        # Extract original fields using reverse function
        original_fields = reverse_format_input(input_text)
        instruction = original_fields["instruction"]
        input_field = original_fields["input"]
        
        # Extract output from target text
        output = ""
        if "### Response:" in target_text:
            output = target_text.split("### Response:")[1].strip()
    
        # Decode generated text
        if hasattr(tokenizer, 'decode'):
            generated_text = tokenizer.decode(token_ids.cpu().tolist())
        else:
            generated_text = str(token_ids.cpu().tolist())
        
        # Extract response by finding the "### Response:" marker and taking everything after it
        response_marker = "### Response:"
        if response_marker in generated_text:
            model_response = generated_text.split(response_marker, 1)[1].strip()
        else:
            # If no response marker found, take everything after the input text
            model_response = generated_text[len(input_text):].strip()
        
        # Return result in original instruction-data.json format
        return {
            "instruction": instruction,
            "input": input_field,
            "output": output,
            "model_response": model_response
        }
    
    # Process samples in parallel using ThreadPoolExecutor
    batch_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_sample, i): i 
            for i in range(input_batch.size(0))
        }
        
        # Collect results as they complete
        for future in tqdm(future_to_index, desc=f"Processing batch {batch_idx}"):
            try:
                result = future.result()
                batch_results.append(result)
            except Exception as exc:
                print(f'Sample processing generated an exception: {exc}')
                # Add empty result to maintain order
                batch_results.append({
                    "instruction": "",
                    "input": "",
                    "output": "",
                    "model_response": ""
                })
    
    return batch_results


def reverse_format_input(formatted_text: str) -> dict:
    """
    Reverse the format_input function to extract original instruction and input fields.
    """
    instruction = ""
    input_field = ""
    
    # Remove the prefix
    prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    if formatted_text.startswith(prefix):
        remaining = formatted_text[len(prefix):]
        
        # Extract instruction
        if "### Instruction:" in remaining:
            instruction_part = remaining.split("### Instruction:")[1]
            if "### Input:" in instruction_part:
                instruction = instruction_part.split("### Input:")[0].strip()
            else:
                instruction = instruction_part.strip()
        
        # Extract input field
        if "### Input:" in remaining:
            input_part = remaining.split("### Input:")[1]
            input_field = input_part.strip()
    
    return {"instruction": instruction, "input": input_field}


def test_model_simple(model, test_loader, device, tokenizer, max_new_tokens=256, output_file="test-results.json", max_workers=4):
    """
    Test model on test loader and save results - extracting original fields from dataset.
    Uses multithreading to process samples in parallel for better performance.
    """
    print("Starting test_model_simple...")
    model.eval()
    
    # Get context size from model config
    context_size = getattr(model, 'max_seq_len', 1024)
    print(f"Context size: {context_size}")
    print(f"Test loader length: {len(test_loader)}")
    print(f"Using {max_workers} worker threads for parallel processing")
    print("WARNING: Individual sample generation may use more GPU memory. Consider reducing max_workers if you encounter memory issues.")
    
    # Process test loader and extract original fields by reversing the transform
    test_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Handle both tuple format and dict format
            if isinstance(batch, dict):
                input_batch = batch["input_ids"]
                target_batch = batch["labels"]
            else:
                input_batch, target_batch = batch
            
            print(f"Processing batch {batch_idx} with {input_batch.size(0)} samples")
            
            # Process samples in parallel using the batch processing function
            # This now includes both generation and post-processing
            batch_results = process_batch_samples(
                input_batch, target_batch, model, device, tokenizer, 
                max_new_tokens, context_size, batch_idx, max_workers
            )
            
            # Add batch results to test_data
            test_data.extend(batch_results)
    
    # Create output directory and save results (instruction-data.json format)
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", output_file)
    
    with open(output_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    
    print(f"Test completed. Results saved to {output_path}")
    print(f"Generated {len(test_data)} responses")
    
    model.train()
    return test_data

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


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


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
    parser.add_argument("--test", action="store_true", help="Run testing instead of training")
    parser.add_argument("--test_output", type=str, default="test-results.json", help="Output file for test results")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads for parallel processing during testing")
    args = parser.parse_args()

    # Simple training mode
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    model = build_model_from_cfg(cfg)

    model = model.to(device)
    
    # Build dataset and dataloaders
    tokenizer = model.tokenizer
    dataset, collate = build_dataset_and_collate(cfg, tokenizer)
    
    # Split dataset into train/val/test
    train_size = int(0.85 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    batch_size = cfg.get("train", {}).get("batch_size", 8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
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
    
    if args.test:
        # Test mode
        print("Running test mode...")
        print(f"Test dataset size: {len(test_dataset)}")
        results = test_model_simple(
            model, test_loader, device, tokenizer, 
            max_new_tokens=75, output_file=args.test_output, max_workers=args.max_workers
        )
        print(f"Test completed with {len(results)} results")
    else:
        # Training mode
        num_epochs = 2
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context=start_context, tokenizer=tokenizer
        )
        
        print(f"Training completed. Final train loss: {train_losses[-1]:.3f}, Final val loss: {val_losses[-1]:.3f}")
        
        # Optionally run test after training
        print("Running test on trained model...")
        results = test_model_simple(
            model, test_loader, device, tokenizer, 
            max_new_tokens=256, output_file="trained-model-test-results.json", max_workers=args.max_workers
        )

if __name__ == "__main__":
    main()