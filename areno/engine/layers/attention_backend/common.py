"""Shared utilities for FlashAttention backends.

Defines the `AttentionCall` value object, which packages a set of Q/K/V
tensors together with the FlashAttention-shaped parameters (normalized
window, optional softmax scale, padded V head dim) so train and infer paths
present identical kernel arguments. Also exposes the small helpers used to
pad value heads, expand grouped-query KV heads, and translate window-size
conventions between FlashAttention and SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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


def sdpa_window_size(window_size: tuple[int, int]) -> tuple[int, int] | None:
    """Convert FlashAttention's full-window sentinel back to SDPA semantics."""

    # SDPA represents "no window" as None; flash-attn uses the (-1, -1) tuple.
    return None if window_size == (-1, -1) else window_size


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

    # SDPA fallback paths need physical KV head replication because they do
    # not implement GQA natively; flash-attn handles GQA without expansion.
    num_kv_heads = x.shape[-2]
    if num_kv_heads == num_q_heads:
        return x
    repeat = num_q_heads // num_kv_heads
    return x.repeat_interleave(repeat, dim=-2)
