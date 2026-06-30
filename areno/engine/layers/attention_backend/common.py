"""Shared utilities for attention backends.

Defines the `AttentionCall` value object, which packages a set of Q/K/V
tensors together with the attention-call parameters (normalized window,
optional softmax scale, padded V head dim) so train, prefill, and native
fallback paths can share one shape contract. Also exposes the small helpers
used to pad value heads and expand grouped-query KV heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

AttnBackend = Literal["flash", "native"]


@dataclass(frozen=True)
class AttentionCall:
    """Normalized arguments shared by train and prefill FlashAttention calls.

    The model code may pass a logical sliding-window value and Q/K/V tensors
    whose value head dim is smaller than the QK head dim. This object turns
    those inputs into one consistent FlashAttention contract: normalized
    window, explicit softmax scale, QK head dim, original V dim, and padded V.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    value_dim: int
    qk_head_dim: int
    window_size: tuple[int, int]
    softmax_scale: float | None

    @property
    def flash_supported(self) -> bool:
        """flash-attn currently caps the QK head dim at 256."""

        return self.qk_head_dim <= 256

    def trim_value_dim(self, out: torch.Tensor) -> torch.Tensor:
        """Drop the padding columns added to V so callers see the original head dim."""

        return out[..., : self.value_dim] if out.shape[-1] != self.value_dim else out


def build_attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int] | None,
    softmax_scale: float | None,
) -> AttentionCall:
    """Prepare shared FlashAttention parameters for train and prefill."""

    qk_head_dim = int(q.shape[-1])
    value_dim = int(v.shape[-1])
    return AttentionCall(
        q=q,
        k=k,
        # flash-attn requires V's head dim to match QK; pad with zeros when
        # the model uses a smaller value head and trim back on output.
        v=pad_last_dim(v, qk_head_dim),
        value_dim=value_dim,
        qk_head_dim=qk_head_dim,
        window_size=flash_window_size(window_size),
        softmax_scale=softmax_scale,
    )


def flash_window_size(window_size: tuple[int, int] | None) -> tuple[int, int]:
    """Normalize optional logical window size to FlashAttention's sentinel."""

    # flash-attn uses (-1, -1) to mean "full attention" (no window).
    return window_size or (-1, -1)


def pad_last_dim(x: torch.Tensor, size: int) -> torch.Tensor:
    """Pad the value/cache head dim to the attention kernel head dim."""

    if x.shape[-1] > size:
        raise ValueError(f"cannot fit last dim {x.shape[-1]} into target dim {size}")
    if x.shape[-1] == size:
        return x
    out = x.new_zeros(*x.shape[:-1], size)
    out[..., : x.shape[-1]] = x
    return out


def expand_kv_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention kernels."""

    # Native attention paths need physical KV head replication because they
    # operate after grouped-query heads have been expanded.
    num_kv_heads = x.shape[-2]
    if num_kv_heads == num_q_heads:
        return x
    if num_q_heads < num_kv_heads or num_q_heads % num_kv_heads != 0:
        raise ValueError(f"cannot expand {num_kv_heads} KV heads to {num_q_heads} query heads")
    repeat = num_q_heads // num_kv_heads
    return (
        x.unsqueeze(-2)
        .expand(*x.shape[:-2], num_kv_heads, repeat, x.shape[-1])
        .reshape(*x.shape[:-2], num_q_heads, x.shape[-1])
        .contiguous()
    )


def use_native_attention(attn_backend: AttnBackend) -> bool:
    """Return whether attention should bypass flash-attn and use native attention."""

    return attn_backend == "native"


def require_flash_attention_supported(call: AttentionCall, *, mode: str) -> None:
    """Fail with an actionable message when flash-attn cannot handle the shape."""

    if call.flash_supported:
        return
    raise RuntimeError(
        f"flash attention does not support qk head dim {call.qk_head_dim} for {mode}. "
        "Use --attn-backend native to run the native consistency attention backend instead; "
        "this is slower and intended for compatibility or logprob diagnostics."
    )
