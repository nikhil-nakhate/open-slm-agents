"""Attention module with Grouped Query Attention and sink tokens.

This module implements the attention mechanism with:
- Grouped Query Attention (GQA) with configurable ratio
- YaRN scaled Rotary Position Embeddings (RoPE)
- Sink tokens for attention stability
- Optional sliding window attention
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .rope import RotaryEmbedding
from .rms_norm import RMSNorm


def causal_attention_with_sinks(
    use_cache: bool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink_logits: torch.Tensor,
    scale: float,
    sliding_window: int,
) -> torch.Tensor:
    """Causal attention with sink tokens and optional sliding window.

    Args:
        q: Query tensor [n_tokens, n_heads, q_mult, head_dim]
        k: Key tensor [n_tokens, n_heads, head_dim]
        v: Value tensor [n_tokens, n_heads, head_dim]
        sink_logits: Sink token logits [n_heads]
        scale: Attention scale factor (1/sqrt(head_dim))
        sliding_window: Sliding window size (0 = no window)

    Returns:
        Attention output [n_tokens, n_heads * q_mult * head_dim]
    """
    n_tokens, n_heads, q_mult, head_dim = q.shape
    assert k.shape == (n_tokens, n_heads, head_dim)
    assert v.shape == (n_tokens, n_heads, head_dim)

    # Expand k, v for grouped query attention
    k = k[:, :, None, :].expand(-1, -1, q_mult, -1)
    v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
    sinks = sink_logits.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, 1)

    # Causal mask (use q's dtype for consistency)
    mask = torch.triu(
        torch.full((n_tokens, n_tokens), float("-inf"), device=q.device, dtype=q.dtype),
        diagonal=1,
    )

    # Sliding window mask
    if sliding_window > 0:
        mask += torch.tril(torch.full_like(mask, float("-inf")), diagonal=-sliding_window)

    # Attention scores
    attn = torch.einsum("qhmd,khmd->hmqk", q, k) * scale
    attn = attn + mask.unsqueeze(0).unsqueeze(0)
    attn = torch.cat([attn, sinks], dim=-1)
    weights = torch.softmax(attn.to(torch.float32), dim=-1).to(q.dtype)[..., :-1]
    out = torch.einsum("hmqk,khmd->qhmd", weights, v)

    return out.reshape(n_tokens, -1)


class AttentionBlock(nn.Module):
    """Grouped query attention with RoPE and sink tokens.

    This implements the attention mechanism with:
    - Configurable GQA ratio (num_attention_heads / num_key_value_heads)
    - YaRN scaled rotary position embeddings
    - Sink tokens for stability
    - Optional sliding window
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float,
        rope_ntk_alpha: float,
        rope_ntk_beta: float,
        sliding_window: int,
        rms_norm_eps: float,
        dtype: torch.dtype,
        use_cache: bool = True,
    ) -> None:
        """Initialize attention block.

        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads (for GQA)
            head_dim: Dimension of each attention head
            layer_idx: Layer index (for sliding window)
            max_position_embeddings: Maximum sequence length
            rope_theta: RoPE base frequency
            rope_scaling_factor: YaRN scaling factor
            rope_ntk_alpha: YaRN NTK alpha parameter
            rope_ntk_beta: YaRN NTK beta parameter
            sliding_window: Sliding window size (applied to even layers)
            rms_norm_eps: RMSNorm epsilon
            dtype: Data type for parameters
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.q_per_kv = self.num_heads // self.num_kv_heads
        self.sliding_window = sliding_window if layer_idx % 2 == 0 else 0
        self.scale = 1 / (head_dim ** 0.5)

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True, dtype=dtype)
        self.out = nn.Linear(self.head_dim * self.num_heads, hidden_size, bias=True, dtype=dtype)
        self.sink_logits = nn.Parameter(torch.empty(self.num_heads, dtype=torch.bfloat16))
        nn.init.zeros_(self.sink_logits)

        self.use_cache = use_cache
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

        self.rotary = RotaryEmbedding(
            head_dim=head_dim,
            base=rope_theta,
            scaling_factor=rope_scaling_factor,
            initial_context_length=max_position_embeddings,
            ntk_alpha=rope_ntk_alpha,
            ntk_beta=rope_ntk_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        residual = x
        h = self.norm(x)
        q = self.q_proj(h)
        k_new = self.k_proj(h)
        v_new = self.v_proj(h)

        # Reshape to [n_tokens, n_heads, (q_mult), head_dim]
        num_tokens_q = q.shape[0]
        q = q.view(-1, self.num_kv_heads, self.q_per_kv, self.head_dim)
        k_new = k_new.view(-1, self.num_kv_heads, self.head_dim)
        v_new = v_new.view(-1, self.num_kv_heads, self.head_dim)

        # Handle KV cache
        if self.use_cache:
            if self.cache_k is None:
                # First forward pass - initialize cache with RoPE applied
                pos_offset = 0
                q, k_new = self.rotary(q, k_new, start_pos=pos_offset)
                self.cache_k, self.cache_v = k_new, v_new
                k, v = k_new, v_new
            else:
                # Subsequent passes - apply RoPE to new tokens, then concatenate to cache
                pos_offset = self.ptr_current_pos
                q, k_new = self.rotary(q, k_new, start_pos=pos_offset)
                self.cache_k = torch.cat([self.cache_k, k_new], dim=0)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=0)
                k, v = self.cache_k, self.cache_v
        else:
            # No cache - standard RoPE application
            k, v = k_new, v_new
            q, k = self.rotary(q, k, start_pos=0)

        # Compute attention
        n_tokens_q = q.shape[0]
        n_tokens_k = k.shape[0]
        n_heads, q_mult, head_dim = self.num_kv_heads, self.q_per_kv, self.head_dim

        # Expand k, v for grouped query attention
        k = k[:, :, None, :].expand(-1, -1, q_mult, -1)  # [n_tokens_k, n_heads, q_mult, head_dim]
        v = v[:, :, None, :].expand(-1, -1, q_mult, -1)  # [n_tokens_k, n_heads, q_mult, head_dim]
        sinks = self.sink_logits.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens_q, 1)

        # Create causal mask
        # Note: With KV cache, n_tokens_k > n_tokens_q during incremental generation
        # Example: first pass has 37 tokens, second pass has 1 new Q but 38 total K (cached + new)
        if self.use_cache and pos_offset > 0:
            # KV cache mode: shape [n_tokens_q, n_tokens_k]
            # New queries can attend to all cached keys (no masking for cached portion)
            # But new queries must use causal masking among themselves
            mask = torch.zeros((n_tokens_q, n_tokens_k), device=q.device, dtype=q.dtype)

            # Apply causal mask only within the new tokens (last n_tokens_q positions)
            # This prevents new token i from attending to new token j where j > i
            if n_tokens_q > 1:
                causal_mask = torch.triu(
                    torch.full((n_tokens_q, n_tokens_q), float("-inf"), device=q.device, dtype=q.dtype),
                    diagonal=1,
                )
                # Place causal mask in the rightmost n_tokens_q columns (new tokens)
                mask[:, -n_tokens_q:] = causal_mask
        else:
            # Standard mode (no cache): n_tokens_q == n_tokens_k
            # Standard causal mask: token i cannot attend to token j where j > i
            mask = torch.triu(
                torch.full((n_tokens_q, n_tokens_k), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1,
            )

        # Sliding window mask (only on even layers)
        if self.sliding_window > 0:
            mask += torch.tril(torch.full_like(mask, float("-inf")), diagonal=-self.sliding_window)

        # Attention scores
        attn = torch.einsum("qhmd,khmd->hmqk", q, k) * self.scale
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = torch.cat([attn, sinks], dim=-1)
        weights = torch.softmax(attn.to(torch.float32), dim=-1).to(q.dtype)[..., :-1]
        ctx = torch.einsum("hmqk,khmd->qhmd", weights, v)

        out = ctx.reshape(n_tokens_q, -1)

        # Update position tracker
        if self.use_cache:
            self.ptr_current_pos += num_tokens_q

        return residual + self.out(out)

    def reset_cache(self):
        """Reset the KV cache and position pointer.

        Call this before starting a new generation sequence.
        """
        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0
