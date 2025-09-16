from typing import Dict, Any
import numpy as np
import torch


def load_weights_into_gpt(gpt, params: Dict[str, Any]):
    """Loads converted GPT-2 weights into our GPT model structure.

    This function adapts to our module naming:
    - gpt.tok_emb.token_emb.weight
    - gpt.pos_emb.pos_emb.weight
    - gpt.trf_blocks[b].attn.(W_query/W_key/W_value/out_proj)
    - gpt.trf_blocks[b].mlp.(fc1/fc2)
    - gpt.trf_blocks[b].ln1/ln2.(weight/bias)
    - gpt.final_norm.(weight/bias)
    - gpt.out_head.weight
    """

    # Embeddings
    print(params["wpe"])
    print("From config", gpt.pos_emb.pos_emb.weight)
    gpt.pos_emb.pos_emb.weight = assign(gpt.pos_emb.pos_emb.weight, params["wpe"])
    gpt.tok_emb.token_emb.weight = assign(gpt.tok_emb.token_emb.weight, params["wte"])

    # Transformer blocks
    for b in range(len(params["blocks"])):
        block = gpt.trf_blocks[b]
        print("block", block)

        # Attention qkv split (transpose for PyTorch linear layout)
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        block.attn.W_query.weight = assign(block.attn.W_query.weight, q_w.T)
        block.attn.W_key.weight = assign(block.attn.W_key.weight, k_w.T)
        block.attn.W_value.weight = assign(block.attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        block.attn.W_query.bias = assign(block.attn.W_query.bias, q_b)
        block.attn.W_key.bias = assign(block.attn.W_key.bias, k_b)
        block.attn.W_value.bias = assign(block.attn.W_value.bias, v_b)

        block.attn.out_proj.weight = assign(
            block.attn.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        block.attn.out_proj.bias = assign(
            block.attn.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # MLP
        block.mlp.fc1.weight = assign(block.mlp.fc1.weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        block.mlp.fc1.bias = assign(block.mlp.fc1.bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        block.mlp.fc2.weight = assign(block.mlp.fc2.weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        block.mlp.fc2.bias = assign(block.mlp.fc2.bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        # LayerNorms
        block.ln1.weight = assign(block.ln1.weight, params["blocks"][b]["ln_1"]["g"])
        block.ln1.bias = assign(block.ln1.bias, params["blocks"][b]["ln_1"]["b"])
        block.ln2.weight = assign(block.ln2.weight, params["blocks"][b]["ln_2"]["g"])
        block.ln2.bias = assign(block.ln2.bias, params["blocks"][b]["ln_2"]["b"])

    # Final norm and output head
    ln_f = params.get("ln_f", {})
    if ln_f:
        gpt.final_norm.weight = assign(gpt.final_norm.weight, ln_f.get("g"))
        gpt.final_norm.bias = assign(gpt.final_norm.bias, ln_f.get("b"))
    else:
        # Fallback if already flattened (unlikely)
        gpt.final_norm.weight = assign(gpt.final_norm.weight, params["g"])
        gpt.final_norm.bias = assign(gpt.final_norm.bias, params["b"])
    # Output head (may be weight-tied to token embeddings)
    if hasattr(gpt.out_head, "proj"):
        gpt.out_head.proj.weight = assign(gpt.out_head.proj.weight, params["wte"])
    else:
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def assign(left: torch.nn.Parameter, right: np.ndarray) -> torch.nn.Parameter:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Shape mismatch. Left: {tuple(left.shape)}, Right: {tuple(right.shape)}")
    # Create a Parameter with the same dtype/device as left
    tensor = torch.as_tensor(right, dtype=left.dtype, device=left.device)
    return torch.nn.Parameter(tensor)
