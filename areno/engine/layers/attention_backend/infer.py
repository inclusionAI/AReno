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

The ``native`` backend selects the areno_accel native attention path that
shares forward math with training attention for logprob diagnostics.
Unsupported flash-attn shapes fail with an actionable message instead of
silently falling back.
"""

from __future__ import annotations

import torch
from torch import nn

from areno.accel.attention import areno_paged_causal_attention_decode, areno_varlen_causal_attention
from areno.engine.layers.attention_backend.common import (
    AttnBackend,
    build_attention_call,
    pad_last_dim,
    require_flash_attention_supported,
    use_native_attention,
)
from areno.engine.runtime.metadata import InferMeta


class FlashAttnInferBackend(nn.Module):
    """Inference attention backend for prefill and single-token decode.

    Prefill writes all prompt K/V into the paged cache and runs varlen fused
    attention. Decode updates one cache slot per active sequence and uses the
    fused KV-cache attention path.
    """

    def __init__(self, attn_backend: AttnBackend = "flash"):
        """Bind flash-attn entrypoints once for the module instance."""

        super().__init__()
        self.attn_backend = attn_backend

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
            if use_native_attention(self.attn_backend):
                out = _native_prefill(
                    call.q,
                    call.k,
                    call.v,
                    meta,
                    call.window_size,
                    call.softmax_scale,
                )
                out = call.trim_value_dim(out)
                return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)
            require_flash_attention_supported(call, mode="prefill attention")
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
            if use_native_attention(self.attn_backend):
                if not update_cache:
                    raise ValueError("native decode requires update_cache=True")
                out = _native_decode(
                    q=q_flat,
                    k_update=k_flat,
                    v_update=pad_last_dim(v_flat, v_cache_dim),
                    k_cache=k_cache,
                    v_cache=v_cache,
                    meta=meta,
                    window_size=call.window_size,
                    softmax_scale=call.softmax_scale,
                )
                return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)
            require_flash_attention_supported(call, mode="decode attention")
            # When value head dim < cache head dim we pad to match the cache
            # layout that was sized to the QK head dim at prefill time.
            v_update = (
                pad_last_dim(v_flat, v_cache_dim).unsqueeze(1) if v_cache_dim != call.value_dim else v_flat.unsqueeze(1)
            )
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


def build_infer_attention_backend(attn_backend: AttnBackend = "flash") -> FlashAttnInferBackend:
    """Build the default inference attention backend."""

    return FlashAttnInferBackend(attn_backend=attn_backend)


@torch._dynamo.disable
def _flash_attn_varlen_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper so torch.compile does not specialize flash-attn."""

    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(*args, **kwargs)


@torch._dynamo.disable
def _flash_attn_with_kvcache_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper for the kvcache-aware flash-attn entrypoint."""

    from flash_attn import flash_attn_with_kvcache

    return flash_attn_with_kvcache(*args, **kwargs)


def _window_left(window_size: tuple[int, int]) -> int | None:
    if window_size == (-1, -1):
        return None
    if window_size[1] != 0:
        raise ValueError("native attention backend only supports causal right window 0")
    return int(window_size[0])


@torch._dynamo.disable
def _native_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: InferMeta,
    window_size: tuple[int, int],
    softmax_scale: float | None,
) -> torch.Tensor:
    """Per-sequence native prefill path shared with training attention."""

    if meta.cu_seqlens is None:
        raise ValueError("prefill inference requires cu_seqlens")
    return areno_varlen_causal_attention(
        q,
        k,
        v,
        meta.cu_seqlens,
        window_left=_window_left(window_size),
        softmax_scale=softmax_scale,
    )


def _native_decode(
    q: torch.Tensor,
    k_update: torch.Tensor,
    v_update: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: InferMeta,
    window_size: tuple[int, int],
    softmax_scale: float | None,
) -> torch.Tensor:
    """Native single-token decode path over the paged KV cache."""

    if meta.cache_seqlens is None or meta.block_table is None:
        raise ValueError("decode inference requires cache_seqlens and block_table")
    out_dtype = q.dtype
    if meta.cache_seqlens.dtype != torch.int32:
        raise ValueError("native decode requires int32 cache_seqlens")
    if meta.block_table.dtype != torch.int32:
        raise ValueError("native decode requires int32 block_table")
    return areno_paged_causal_attention_decode(
        q.contiguous(),
        k_update.contiguous(),
        v_update.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        meta.block_table.contiguous(),
        meta.cache_seqlens.contiguous(),
        window_left=_window_left(window_size),
        num_splits=8,
        softmax_scale=softmax_scale,
    ).to(dtype=out_dtype)


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
