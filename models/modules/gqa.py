from typing import Optional

import torch
import torch.nn as nn

from .rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Multi-head attention that shares key/value heads across query heads (GQA)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        sliding_window: Optional[int] = None,
        sink_init: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % max(1, num_kv_heads) != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

        mask = torch.triu(torch.full((context_length, context_length), float("-inf")), diagonal=1)
        if sliding_window is not None and sliding_window > 0:
            mask += torch.tril(torch.full_like(mask, float("-inf")), diagonal=-sliding_window)
        self.register_buffer("attn_mask", mask, persistent=False)

        self.register_parameter(
            "sink_logits", nn.Parameter(torch.full((num_heads,), sink_init, dtype=torch.float32))
        )

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        return self.rope(q, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input tensor with shape [batch, seq_len, dim]")
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)
        q_t, k_t = self._apply_rope(q_t, k_t)
        q = q_t.permute(0, 2, 1, 3)
        k = k_t.permute(0, 2, 1, 3)

        q_mult = self.num_heads // self.num_kv_heads
        q = q.view(bsz, seq_len, self.num_kv_heads, q_mult, self.head_dim)

        mask = self.attn_mask[:seq_len, :seq_len].to(x.dtype).to(x.device)

        outputs = []
        for b in range(bsz):
            q_b = q[b].view(seq_len, self.num_kv_heads, q_mult, self.head_dim)
            k_b = k[b]
            v_b = v[b]
            context = self._sdpa(q_b, k_b, v_b, mask, q_mult)
            outputs.append(context)

        context = torch.stack(outputs, dim=0)  # [B, seq_len, num_heads * head_dim]
        context = context.view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(context)

    def _sdpa(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor,
        q_mult: int,
    ) -> torch.Tensor:
        n_tokens = Q.shape[0]
        K_exp = K.unsqueeze(2).expand(-1, -1, q_mult, -1)
        V_exp = V.unsqueeze(2).expand(-1, -1, q_mult, -1)
        sinks = (
            self.sink_logits.view(self.num_kv_heads, q_mult, 1, 1)
            .to(Q.dtype)
            .expand(-1, -1, n_tokens, -1)
        )
        scores = torch.einsum("qhmd,khmd->hmqk", Q, K_exp) * self.scale
        scores = scores + mask.unsqueeze(0).unsqueeze(0).to(Q.dtype)
        scores = torch.cat([scores, sinks], dim=-1)
        weights = torch.softmax(scores, dim=-1)[..., :-1]
        weights = self.dropout(weights)
        attn = torch.einsum("hmqk,khmd->qhmd", weights, V_exp)
        return attn.reshape(n_tokens, self.num_heads * self.head_dim)
