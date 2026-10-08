"""Select NPU attention by library availability, tensor layout and explicit choice."""

import warnings

import torch

from areno.accel.flash_attention import FlashAttentionUnavailable, flash_attention

_UNAVAILABLE_FLASH_DEVICES: set[str] = set()


def _flash_library(device):
    key = str(device)
    if key in _UNAVAILABLE_FLASH_DEVICES:
        return None
    try:
        return flash_attention(device)
    except FlashAttentionUnavailable as exc:
        _UNAVAILABLE_FLASH_DEVICES.add(key)
        warnings.warn(
            f"NPU FlashAttention unavailable ({exc}); falling back to attn_backend='native', which may be slower.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def native_attention_required(q, *, block_size=None):
    """Resolve NPU flash compatibility lazily on the worker's selected device."""
    if block_size is not None and (q.shape[-1] % 8 or block_size % 256):
        return True
    return not _flash_supported(q) or _flash_library(q.device) is None


def _check(q, k, v):
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("NPU attention requires matching FP32/FP16/BF16 q/k/v dtype")
    if q.shape[-1] <= 0:
        raise ValueError("NPU attention requires a positive head dimension")


def _flash_supported(q):
    return q.dtype in (torch.float16, torch.bfloat16) and q.shape[-1] <= 256


def _window(left):
    if left < -1:
        raise ValueError("window_left must be -1 or nonnegative")
    return (-1, -1) if left == -1 else (left, 0)


def _empty(q, k, v):
    # Preserve zero gradients for all inputs without invoking a zero-size kernel.
    return q + k.reshape(-1)[:0].sum() + v.reshape(-1)[:0].sum()


def causal_attention(q, k, v, query_start, window_left, softmax_scale, *, force_native=False):
    _check(q, k, v)
    window = _window(window_left)
    end = query_start + q.shape[2]
    if k.shape != v.shape or query_start < 0 or end > k.shape[2]:
        raise ValueError("invalid causal attention shape or query positions")
    if not q.numel():
        return _empty(q, k, v)
    if force_native or native_attention_required(q):
        from areno.accel.attention import _ArenoCausalAttention

        return _ArenoCausalAttention.apply(q, k, v, query_start, window_left, softmax_scale)
    # FlashAttention aligns causal masks at the bottom right. Truncating the
    # invisible K/V suffix maps AReno's explicit query offset to that contract.
    out = flash_attention(q.device).flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k[:, :, :end].transpose(1, 2).contiguous(),
        v[:, :, :end].transpose(1, 2).contiguous(),
        causal=True,
        window_size=window,
        softmax_scale=softmax_scale,
    )
    return out.transpose(1, 2)


def varlen_causal_attention(q, k, v, cu_seqlens, window_left, softmax_scale, *, force_native=False):
    _check(q, k, v)
    window = _window(window_left)
    if cu_seqlens.numel() < 2:
        raise ValueError("packed attention requires at least two sequence boundaries")
    if not q.numel():
        return _empty(q, k, v)
    if force_native or native_attention_required(q):
        from areno.accel.attention import _ArenoVarlenCausalAttention

        return _ArenoVarlenCausalAttention.apply(q, k, v, cu_seqlens, window_left, softmax_scale)
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
        window_size=window,
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
    *,
    force_native=False,
):
    _check(q, k_cache, v_cache)
    _check(q, k_update, v_update)
    window = _window(window_left)
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("paged attention needs contiguous caller-owned KV caches for in-place updates")
    if force_native or native_attention_required(q, block_size=k_cache.shape[1]):
        from areno.accel.attention import _ArenoPagedCausalAttentionDecode

        return _ArenoPagedCausalAttentionDecode.apply(
            q,
            k_update,
            v_update,
            k_cache,
            v_cache,
            block_table,
            cache_seqlens,
            window_left,
            num_splits or 8,
            softmax_scale,
        )
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
        window_size=window,
        softmax_scale=softmax_scale,
        num_splits=num_splits,
    )
    return out.squeeze(1)
