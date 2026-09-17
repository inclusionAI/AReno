"""Adapt AReno attention layouts to flash-attn-npu; kernels and autograd stay upstream."""

import torch

from areno.accel.flash_attention import flash_attention


def _check(q, k, v):
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("flash-attn-npu requires matching FP16/BF16 q/k/v dtype")
    if not 0 < q.shape[-1] <= 256:
        raise ValueError("flash-attn-npu supports head dimensions from 1 through 256")


def _window(left):
    if left < -1:
        raise ValueError("window_left must be -1 or nonnegative")
    return (-1, -1) if left == -1 else (left, 0)


def _empty(q, k, v):
    # Preserve zero gradients for all inputs without invoking a zero-size kernel.
    return q + k.reshape(-1)[:0].sum() + v.reshape(-1)[:0].sum()


def causal_attention(q, k, v, query_start, window_left, softmax_scale):
    _check(q, k, v)
    end = query_start + q.shape[2]
    if k.shape != v.shape or query_start < 0 or end > k.shape[2]:
        raise ValueError("invalid causal attention shape or query positions")
    if not q.numel():
        return _empty(q, k, v)
    # FlashAttention aligns causal masks at the bottom right. Truncating the
    # invisible K/V suffix maps AReno's explicit query offset to that contract.
    out = flash_attention(q.device).flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k[:, :, :end].transpose(1, 2).contiguous(),
        v[:, :, :end].transpose(1, 2).contiguous(),
        causal=True,
        window_size=_window(window_left),
        softmax_scale=softmax_scale,
    )
    return out.transpose(1, 2)


def varlen_causal_attention(q, k, v, cu_seqlens, window_left, softmax_scale):
    _check(q, k, v)
    if cu_seqlens.numel() < 2:
        raise ValueError("packed attention requires at least two sequence boundaries")
    if not q.numel():
        return _empty(q, k, v)
    cu = cu_seqlens.contiguous()
    return flash_attention(q.device).flash_attn_varlen_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        # A shape-only upper bound avoids copying device lengths to the host.
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        causal=True,
        window_size=_window(window_left),
        softmax_scale=softmax_scale,
    )


def paged_causal_attention_decode(
    q,
    k_update,
    v_update,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    window_left,
    num_splits,
    softmax_scale,
):
    _check(q, k_cache, v_cache)
    _check(q, k_update, v_update)
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("paged attention needs contiguous caller-owned KV caches for in-place updates")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k_update, v_update, k_cache, v_cache)):
        raise RuntimeError(
            "flash-attn-npu KV-cache attention is inference-only; use dense/varlen attention for training"
        )
    if q.shape[0] == 0:
        return torch.empty_like(q)
    out = flash_attention(q.device).flash_attn_with_kvcache(
        q.unsqueeze(1).contiguous(),
        k_cache,
        v_cache,
        k=k_update.unsqueeze(1).contiguous(),
        v=v_update.unsqueeze(1).contiguous(),
        block_table=block_table.contiguous(),
        cache_seqlens=cache_seqlens.contiguous(),
        causal=True,
        window_size=_window(window_left),
        softmax_scale=softmax_scale,
        num_splits=num_splits,
    )
    return out.squeeze(1)
