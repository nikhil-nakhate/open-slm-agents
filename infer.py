import argparse
import os
from typing import Optional
import torch

from ops.config import load_config
from models.build import build_model_from_cfg


def _sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, greedy: bool = False) -> int:
    # logits: [V]
    if greedy or temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(1e-5, temperature)
    probs = torch.softmax(logits, dim=-1)

    if top_k > 0:
        v, _ = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        thresh = v.min()
        probs = torch.where(probs >= thresh, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[idx].item())

    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def generate(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, greedy: bool = False) -> str:
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt)
        ids = list(input_ids)
        eos_id: Optional[int] = getattr(tokenizer, "eos_id", None)

        max_ctx = int(getattr(model, "max_seq_len", 1024))

        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            x_cond = x[:, -max_ctx:]
            logits = model(x_cond)  # [1, T, V]
            next_logits = logits[0, -1, :]
            next_id = _sample_next_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy)
            ids.append(next_id)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            if eos_id is not None and next_id == eos_id:
                break

        # Return only the generated continuation
        gen_ids = ids[len(input_ids):]
        return tokenizer.decode(gen_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a model checkpoint (interactive)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint .pt")
    # Generation parameters now sourced from config under `eval` section
    parser.add_argument("--weights_dir", type=str, default=None, help="Path to downloaded GPT weights dir or file (params.pt/pkl)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model_from_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = model.tokenizer

    # Load custom converted GPT-2 weights if provided
    if args.weights_dir:
        import pickle
        from scripts.load_gpt_weights import load_weights_into_gpt

        path = args.weights_dir
        params = None
        if os.path.isdir(path):
            pt = os.path.join(path, "params.pt")
            pkl = os.path.join(path, "params.pkl")
            if os.path.exists(pkl):
                with open(pkl, "rb") as f:
                    params = pickle.load(f)
            else:
                raise FileNotFoundError(f"No params.pt or params.pkl found in {path}")
        else:
            if path.endswith(".pt"):
                params = torch.load(path, map_location="cpu")
            elif path.endswith(".pkl"):
                with open(path, "rb") as f:
                    params = pickle.load(f)
            else:
                raise ValueError("weights_dir must be a directory or a .pt/.pkl file")
        load_weights_into_gpt(model, params)
        model.to(device)
    elif args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
    else:
        model.to(device)
    model.eval()
    # Load generation settings from config
    gen_cfg = cfg.get("eval", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 50))
    temperature = float(gen_cfg.get("temperature", 1.0))
    top_k = int(gen_cfg.get("top_k", 0))
    top_p = float(gen_cfg.get("top_p", 0.0))
    greedy = bool(gen_cfg.get("greedy", False))

    print("Interactive generation ready. Type /exit to quit.\n")
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit", ":q", "q"}:
            print("Bye.")
            break

        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        print(out, "\n")


if __name__ == "__main__":
    main()
