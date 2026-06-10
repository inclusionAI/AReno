"""Training-time FlashAttention backend.

Supports two activation layouts:

- padded (default): tensors are ``(batch, seqlen, heads, head_dim)`` and
  use `flash_attn_func` directly.
- varlen packed: when `TrainMeta.cu_seqlens` is supplied the tensors are
  flattened to ``(total_tokens, heads, head_dim)`` and routed through
  `flash_attn_varlen_func` so packed batches without padding can be
  trained efficiently.

When flash-attn cannot serve the QK head dim (>256) the backend falls back
to a manual SDPA path that supports both layouts and sliding windows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from flash_attn import flash_attn_func, flash_attn_varlen_func

from areno.engine.layers.attention_backend.common import build_attention_call, expand_kv_heads, sdpa_window_size
from areno.engine.runtime.metadata import TrainMeta


class TrainAttentionBackend(nn.Module, ABC):
    """Abstract training attention backend."""

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: TrainMeta | None,
        window_size: tuple[int, int] | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class FlashAttnTrainAttentionBackend(TrainAttentionBackend):
    """FlashAttention backend shared by padded and varlen packed training."""

    def __init__(self):
        super().__init__()
        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        meta: TrainMeta | None,
        window_size: tuple[int, int] | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        call = build_attention_call(q, k, v, window_size, softmax_scale)
        if not call.flash_supported:
            # Unsupported QK head dim: drop to SDPA fallback (slower).
            return _sdpa_train(q, k, v, meta, sdpa_window_size(call.window_size), call.softmax_scale)
        if meta is not None and meta.cu_seqlens is not None:
            # Varlen packed path: tensors must be flattened to (T, H, D) so
            # cu_seqlens can carve out the per-sequence boundaries.
            if meta.max_seqlen is None:
                raise ValueError("TrainMeta.max_seqlen is required with cu_seqlens")
            batch, seqlen = q.shape[:2]
            del batch, seqlen
            q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
            k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
            v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
            # Rebuild the AttentionCall on the flat layout so V padding is
            # applied to the tensor we actually hand to the kernel.
            flat_call = build_attention_call(q_flat, k_flat, v_flat, window_size, softmax_scale)
            out = _flash_attn_varlen_train_no_compile(
                flat_call.q,
                flat_call.k,
                flat_call.v,
                cu_seqlens_q=meta.cu_seqlens,
                cu_seqlens_k=meta.cu_seqlens,
                max_seqlen_q=meta.max_seqlen,
                max_seqlen_k=meta.max_seqlen,
                causal=True,
                window_size=flat_call.window_size,
                softmax_scale=flat_call.softmax_scale,
            )
            out = flat_call.trim_value_dim(out)
            # Restore the original (B, S, H, D) layout for downstream code.
            return out.view(q.shape[0], q.shape[1], q.shape[2], flat_call.value_dim)

        # Padded path: directly call flash-attn over the 4D batch tensor.
        out = _flash_attn_train_no_compile(
            call.q,
            call.k,
            call.v,
            causal=True,
            window_size=call.window_size,
            softmax_scale=call.softmax_scale,
        )
        return call.trim_value_dim(out)


def build_train_attention_backend() -> TrainAttentionBackend:
    """Build the default FlashAttention training backend."""

    return FlashAttnTrainAttentionBackend()


@torch._dynamo.disable
def _flash_attn_train_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper for the dense flash-attn training kernel."""

    return flash_attn_func(*args, **kwargs)


@torch._dynamo.disable
def _flash_attn_varlen_train_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper for the packed varlen flash-attn training kernel."""

    return flash_attn_varlen_func(*args, **kwargs)


@torch._dynamo.disable
def _sdpa_train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: TrainMeta | None,
    window_size: tuple[int, int] | None,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Training fallback for QK head dimensions unsupported by flash-attn."""

    if meta is not None and meta.cu_seqlens is not None:
        # Packed-layout SDPA: iterate sequences and concatenate the outputs.
        outs = []
        cu = meta.cu_seqlens.tolist()
        q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
        k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
        v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
        for start, end in zip(cu[:-1], cu[1:], strict=True):
            outs.append(_sdpa_sequence(q_flat[start:end], k_flat[start:end], v_flat[start:end], window_size, softmax_scale))
        return torch.cat(outs, dim=0).view(q.shape)
    # Padded SDPA: expand KV heads for GQA and reshape to (B, H, S, D).
    k = expand_kv_heads(k, q.shape[-2])
    v = expand_kv_heads(v, q.shape[-2])
    q_seq = q.transpose(1, 2)
    k_seq = k.transpose(1, 2)
    v_seq = v.transpose(1, 2)
    if window_size is None:
        return F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True, scale=softmax_scale).transpose(1, 2)
    # Build a banded causal mask covering only positions inside the window.
    seqlen = q.shape[1]
    rows = torch.arange(seqlen, device=q.device).view(seqlen, 1)
    cols = torch.arange(seqlen, device=q.device).view(1, seqlen)
    mask = cols <= rows
    if window_size[0] >= 0:
        mask = mask & (cols >= rows - int(window_size[0]))
    return F.scaled_dot_product_attention(
        q_seq,
        k_seq,
        v_seq,
        attn_mask=mask.view(1, 1, seqlen, seqlen),
        scale=softmax_scale,
    ).transpose(1, 2)


def _sdpa_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int] | None,
    softmax_scale: float | None,
) -> torch.Tensor:
    """SDPA over a single varlen-packed sequence slice."""

    # Expand KV heads to match Q heads (SDPA has no GQA support).
    k = expand_kv_heads(k, q.shape[1])
    v = expand_kv_heads(v, q.shape[1])
    q_seq = q.transpose(0, 1).unsqueeze(0)
    k_seq = k.transpose(0, 1).unsqueeze(0)
    v_seq = v.transpose(0, 1).unsqueeze(0)
    if window_size is None:
        out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True, scale=softmax_scale)
    else:
        seqlen = q.shape[0]
        rows = torch.arange(seqlen, device=q.device).view(seqlen, 1)
        cols = torch.arange(seqlen, device=q.device).view(1, seqlen)
        mask = cols <= rows
        if window_size[0] >= 0:
            mask = mask & (cols >= rows - int(window_size[0]))
        out = F.scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            attn_mask=mask.view(1, 1, seqlen, seqlen),
            scale=softmax_scale,
        )
    return out.squeeze(0).transpose(0, 1)
