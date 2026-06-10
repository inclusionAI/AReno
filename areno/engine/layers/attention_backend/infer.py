"""Inference-time FlashAttention backend with paged KV cache.

The backend serves two distinct modes signalled by `InferMeta.mode`:

- prefill: a batch of variable-length prompts packed together. New K/V
  rows are scattered into the paged KV cache by ``(block_id, offset)``
  index puts, then `flash_attn_varlen_func` runs causal attention over the
  packed segments using `cu_seqlens` boundaries.
- decode: each active sequence contributes one new token. The new K/V row
  is appended at the next slot inside the last block of each sequence's
  block table, and `flash_attn_with_kvcache` reads the full history for
  each sequence directly from the paged cache.

An SDPA fallback handles QK head dims that flash-attn does not support.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from areno.engine.layers.attention_backend.common import (
    build_attention_call,
    expand_kv_heads,
    pad_last_dim,
    sdpa_window_size,
)
from areno.engine.runtime.metadata import InferMeta


class FlashAttnInferBackend(nn.Module):
    """Inference attention backend for prefill and single-token decode.

    Prefill writes all prompt K/V into the paged cache and runs varlen fused
    attention. Decode updates one cache slot per active sequence and uses the
    fused KV-cache attention path.
    """

    def __init__(self):
        """Bind flash-attn entrypoints once for the module instance."""

        super().__init__()
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: InferMeta,
        window_size: tuple[int, int] | None = None,
        softmax_scale: float | None = None,
        update_cache: bool = True,
    ) -> torch.Tensor:
        # flash-attn varlen/with_kvcache expect 3D (tokens, heads, head_dim).
        q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
        k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
        v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
        v_cache_dim = v_cache.shape[-1]
        call = build_attention_call(q_flat, k_flat, v_flat, window_size, softmax_scale)

        if meta.mode == "prefill":
            if meta.cu_seqlens is None or meta.max_seqlen is None or meta.block_table is None:
                raise ValueError("prefill inference requires cu_seqlens, max_seqlen, block_table")
            # Persist freshly computed K/V for the prompt into paged cache.
            if update_cache:
                _store_prefill_cache(k_flat, v_flat, k_cache, v_cache, meta)
            if not call.flash_supported:
                # SDPA fallback when flash-attn cannot serve this head dim.
                out = _sdpa_prefill(
                    q_flat,
                    k_flat,
                    v_flat,
                    meta,
                    sdpa_window_size(call.window_size),
                    call.softmax_scale,
                )
                return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)
            # cu_seqlens tells flash-attn where each packed sequence ends so
            # causal attention does not bleed across sequence boundaries.
            out = _flash_attn_varlen_no_compile(
                call.q,
                call.k,
                call.v,
                cu_seqlens_q=meta.cu_seqlens,
                cu_seqlens_k=meta.cu_seqlens,
                max_seqlen_q=meta.max_seqlen,
                max_seqlen_k=meta.max_seqlen,
                causal=True,
                window_size=call.window_size,
                softmax_scale=call.softmax_scale,
            )
            out = call.trim_value_dim(out)
            return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)

        if meta.mode == "decode":
            if meta.cache_seqlens is None or meta.block_table is None:
                raise ValueError("decode inference requires cache_seqlens and block_table")
            # flash-attn's num_splits=0 heuristic enables split-KV for small
            # decode batches. For local/sliding attention, many splits can be
            # fully outside the window and some flash-attn builds produce NaNs
            # when combining those masked splits. Keep local decode on the
            # single-split kvcache kernel; full attention can keep the heuristic.
            num_splits = 1 if call.window_size != (-1, -1) else 0
            if not call.flash_supported:
                # SDPA decode fallback: write the new token then materialize
                # the full key/value matrices from the paged cache.
                if update_cache:
                    _store_decode_cache(k_flat, v_flat, k_cache, v_cache, meta)
                out = _sdpa_decode(
                    q=q_flat,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    meta=meta,
                    window_size=sdpa_window_size(call.window_size),
                    softmax_scale=call.softmax_scale,
                )
                return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)
            # When value head dim < cache head dim we pad to match the cache
            # layout that was sized to the QK head dim at prefill time.
            v_update = pad_last_dim(v_flat, v_cache_dim).unsqueeze(1) if v_cache_dim != call.value_dim else v_flat.unsqueeze(1)
            cache_seqlens = meta.cache_seqlens if update_cache else meta.cache_seqlens + 1
            k_update = k_flat.unsqueeze(1) if update_cache else None
            v_update = v_update if update_cache else None
            # flash-attn appends the new token in-place inside the paged cache
            # using cache_seqlens (current length) and block_table mapping.
            out = _flash_attn_with_kvcache_no_compile(
                q_flat.unsqueeze(1),
                k_cache,
                v_cache,
                k=k_update,
                v=v_update,
                cache_seqlens=cache_seqlens,
                block_table=meta.block_table,
                causal=True,
                window_size=call.window_size,
                softmax_scale=call.softmax_scale,
                num_splits=num_splits,
            )
            out = call.trim_value_dim(out)
            return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)

        raise ValueError(f"unsupported inference mode: {meta.mode}")


def build_infer_attention_backend() -> FlashAttnInferBackend:
    """Build the default inference attention backend."""

    return FlashAttnInferBackend()


@torch._dynamo.disable
def _flash_attn_varlen_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper so torch.compile does not specialize flash-attn."""

    return flash_attn_varlen_func(*args, **kwargs)


@torch._dynamo.disable
def _flash_attn_with_kvcache_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper for the kvcache-aware flash-attn entrypoint."""

    return flash_attn_with_kvcache(*args, **kwargs)


@torch._dynamo.disable
def _sdpa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: InferMeta,
    window_size: tuple[int, int] | None,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Per-sequence SDPA prefill fallback used for unsupported QK head dims."""

    if meta.cu_seqlens is None:
        raise ValueError("prefill inference requires cu_seqlens")
    outs = []
    # Iterate packed sequences using their start/end offsets.
    cu = meta.cu_seqlens.tolist()
    for start, end in zip(cu[:-1], cu[1:], strict=True):
        q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
        # SDPA does not understand GQA; replicate KV heads up to Q head count.
        k_seq = expand_kv_heads(k[start:end], q.shape[1]).transpose(0, 1).unsqueeze(0)
        v_seq = expand_kv_heads(v[start:end], q.shape[1]).transpose(0, 1).unsqueeze(0)
        if window_size is None:
            out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True, scale=softmax_scale)
        else:
            # Build a banded causal mask covering only positions inside the
            # sliding window when one is requested.
            seqlen = end - start
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
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _sdpa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: InferMeta,
    window_size: tuple[int, int] | None,
    softmax_scale: float | None,
) -> torch.Tensor:
    """SDPA-based single-token decode fallback over the paged KV cache."""

    if meta.cache_seqlens is None or meta.block_table is None:
        raise ValueError("decode inference requires cache_seqlens and block_table")
    out_dtype = q.dtype
    # Materialize each sequence's K/V history by gathering the paged blocks
    # listed in its block_table (block_size becomes the second axis).
    k = k_cache[meta.block_table.long()].flatten(1, 2).float()
    v = v_cache[meta.block_table.long()].flatten(1, 2).float()
    q = q.float()
    q_heads = q.shape[1]
    kv_heads = k.shape[2]
    # Reshape Q into (batch, kv_heads, groups, head_dim) to match GQA layout.
    groups = q_heads // kv_heads
    q = q.view(q.shape[0], kv_heads, groups, q.shape[-1])
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    # Mask out positions beyond each sequence's current length (and outside
    # the sliding window when one was requested).
    positions = torch.arange(k.shape[1], device=q.device).view(1, 1, 1, -1)
    total_lens = (meta.cache_seqlens + 1).view(-1, 1, 1, 1)
    mask = positions < total_lens
    if window_size is not None and window_size[0] >= 0:
        start = (total_lens - int(window_size[0]) - 1).clamp_min(0)
        mask = mask & (positions >= start)
    kv_mask = mask.view(mask.shape[0], mask.shape[-1], 1, 1)
    k = k.masked_fill(~kv_mask, 0.0)
    v = v.masked_fill(~kv_mask, 0.0)
    k_t = k.permute(0, 2, 3, 1)
    scores = torch.matmul(q, k_t) * scale
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    v_t = v.permute(0, 2, 1, 3)
    out = torch.matmul(probs, v_t)
    return out.reshape(out.shape[0], q_heads, out.shape[-1]).to(dtype=out_dtype)


def _store_prefill_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: InferMeta,
) -> None:
    """Scatter prompt K/V into paged KV cache slots given by the prefill plan."""

    if meta.cache_block_ids is None or meta.cache_block_offsets is None:
        raise ValueError("prefill inference requires cache_block_ids and cache_block_offsets")
    # Each token gets explicit (block_id, offset) coordinates assigned by
    # the scheduler, so a single vectorized index_put places the prompt.
    k_cache.index_put_((meta.cache_block_ids, meta.cache_block_offsets), key)
    if v_cache.shape[-1] != value.shape[-1]:
        # Cache rows were sized to the QK head dim; pad smaller V before store.
        v_cache.index_put_((meta.cache_block_ids, meta.cache_block_offsets), pad_last_dim(value, v_cache.shape[-1]))
    else:
        v_cache.index_put_((meta.cache_block_ids, meta.cache_block_offsets), value)


def _store_decode_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: InferMeta,
) -> None:
    """Append the per-sequence single token to the paged KV cache."""

    if meta.cache_seqlens is None or meta.block_table is None:
        raise ValueError("decode inference requires cache_seqlens and block_table")
    # Derive (block, offset) for the next slot from each sequence's current
    # length: floor div picks the column in block_table, mod picks the slot
    # inside the block.
    block_size = k_cache.shape[1]
    block_cols = torch.div(meta.cache_seqlens, block_size, rounding_mode="floor").long()
    block_offsets = (meta.cache_seqlens % block_size).long()
    block_ids = meta.block_table[torch.arange(key.shape[0], device=key.device), block_cols].long()
    k_cache.index_put_((block_ids, block_offsets), key)
    if v_cache.shape[-1] != value.shape[-1]:
        v_cache.index_put_((block_ids, block_offsets), pad_last_dim(value, v_cache.shape[-1]))
    else:
        v_cache.index_put_((block_ids, block_offsets), value)
