"""Inference-time FlashAttention backend with paged KV cache.

The backend serves two distinct modes signalled by `InferMeta.mode`:

- prefill: a batch of variable-length prompts packed together. New K/V
  rows are scattered into the paged KV cache by ``(block_id, offset)``
  index puts, then `flash_attn_varlen_func` runs causal attention over the
  packed segments using `cu_seqlens` boundaries.
- decode: each active sequence contributes ``tokens_per_seq`` new tokens
  (one for plain decode, ``k + 1`` for speculative verify). The new K/V rows
  are appended after the sequence's current length, and
  `flash_attn_with_kvcache` reads the full history for each sequence
  directly from the paged cache with a causal mask over the new rows.

The ``native`` backend selects the areno_accel native attention path that
shares forward math with training attention for logprob diagnostics.
NPU workers also select native attention when the optional flash library is
unavailable or the tensor layout is unsupported.
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

    def __init__(self, attn_backend: AttnBackend = "flash", *, decode_num_splits: int | None = None):
        """Bind flash-attn entrypoints once for the module instance."""

        super().__init__()
        self.attn_backend = attn_backend
        self.decode_num_splits = decode_num_splits

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
            if meta.cache_seqlens is not None and call.q.device.type == "npu":
                raise ValueError("chunked prefill with a cached prefix is not supported on NPU")
            # Persist freshly computed K/V for the prompt into paged cache.
            if update_cache:
                _store_prefill_cache(k_flat, v_flat, k_cache, v_cache, meta)
            if meta.cache_seqlens is not None:
                out = _prefill_with_cached_prefix(
                    call.q,
                    call.k,
                    call.v,
                    k_cache,
                    v_cache,
                    meta,
                    self.attn_backend,
                    call.window_size,
                    call.softmax_scale,
                )
                out = call.trim_value_dim(out)
                return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)
            if use_native_attention(self.attn_backend, call.q):
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
            steps = getattr(meta, "tokens_per_seq", 1)
            # flash-attn's num_splits=0 heuristic enables split-KV for small
            # decode batches. For local/sliding attention, many splits can be
            # fully outside the window and some flash-attn builds produce NaNs
            # when combining those masked splits. Keep local decode on the
            # single-split kvcache kernel; full attention can keep the heuristic.
            num_splits = (
                self.decode_num_splits
                if self.decode_num_splits is not None
                else (1 if call.window_size != (-1, -1) else 0)
            )
            if use_native_attention(self.attn_backend, call.q, block_size=k_cache.shape[1]):
                if not update_cache:
                    raise ValueError("native decode requires update_cache=True")
                if steps != 1:
                    raise ValueError("native decode attention supports one token per sequence")
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
                out = call.trim_value_dim(out)
                return out.view(q.shape[0], q.shape[1], q.shape[2], call.value_dim)
            require_flash_attention_supported(call, mode="decode attention")
            # flash-attn's kvcache entry takes (rows, new_tokens, heads, dim):
            # regroup the flat sequence-major token axis by active row.
            rows = int(meta.cache_seqlens.shape[0])
            if q_flat.shape[0] != rows * steps:
                raise ValueError(
                    f"decode attention got {q_flat.shape[0]} tokens for {rows} rows x {steps} tokens per sequence"
                )
            # When value head dim < cache head dim we pad to match the cache
            # layout that was sized to the QK head dim at prefill time.
            v_update = pad_last_dim(v_flat, v_cache_dim) if v_cache_dim != call.value_dim else v_flat
            cache_seqlens = meta.cache_seqlens if update_cache else meta.cache_seqlens + steps
            k_update = k_flat.view(rows, steps, *k_flat.shape[1:]) if update_cache else None
            v_update = v_update.view(rows, steps, *v_update.shape[1:]) if update_cache else None
            # flash-attn appends the new tokens in-place inside the paged cache
            # using cache_seqlens (current length) and block_table mapping.
            out = _flash_attn_with_kvcache_no_compile(
                q_flat.view(rows, steps, *q_flat.shape[1:]),
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


def build_infer_attention_backend(
    attn_backend: AttnBackend = "flash", *, decode_num_splits: int | None = None
) -> FlashAttnInferBackend:
    """Build the default inference attention backend."""

    return FlashAttnInferBackend(attn_backend=attn_backend, decode_num_splits=decode_num_splits)


@torch._dynamo.disable
def _flash_attn_varlen_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper so torch.compile does not specialize flash-attn."""

    from areno.accel.flash_attention import flash_attention

    q = args[0] if args else kwargs["q"]
    return flash_attention(q.device).flash_attn_varlen_func(*args, **kwargs)


@torch._dynamo.disable
def _flash_attn_with_kvcache_no_compile(*args, **kwargs) -> torch.Tensor:
    """Dynamo-opaque wrapper for the kvcache-aware flash-attn entrypoint."""

    from areno.accel.flash_attention import flash_attention

    q = args[0] if args else kwargs["q"]
    return flash_attention(q.device).flash_attn_with_kvcache(*args, **kwargs)


def _window_left(window_size: tuple[int, int]) -> int | None:
    if window_size == (-1, -1):
        return None
    if window_size[1] != 0:
        raise ValueError("native attention backend only supports causal right window 0")
    return int(window_size[0])


@torch._dynamo.disable
def _prefill_with_cached_prefix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: InferMeta,
    attn_backend: AttnBackend,
    window_size: tuple[int, int],
    softmax_scale: float | None,
) -> torch.Tensor:
    """Attend each new query to its entire cached causal prefix.

    A chunk's cu_seqlens only describes its new queries, not all cached keys.
    Reuse paged attention instead of treating the chunk as a fresh sequence.
    """
    lengths = meta.cu_seqlens[1:] - meta.cu_seqlens[:-1]
    if use_native_attention(attn_backend, q, block_size=k_cache.shape[1]):
        rows = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=q.device), lengths.long(), output_size=q.shape[0]
        )
        positions = torch.arange(q.shape[0], device=q.device) - meta.cu_seqlens[rows]
        decode_meta = InferMeta(
            mode="decode",
            cache_seqlens=(meta.cache_seqlens[rows] + positions).to(dtype=torch.int32),
            block_table=meta.block_table.index_select(0, rows),
        )
        block_size = k_cache.shape[1]
        cache_positions = decode_meta.cache_seqlens.long()
        block_ids = meta.block_table[rows, cache_positions // block_size].long()
        block_offsets = cache_positions % block_size
        return _native_decode(
            q,
            k_cache[block_ids, block_offsets],
            v_cache[block_ids, block_offsets],
            k_cache,
            v_cache,
            decode_meta,
            window_size,
            softmax_scale,
        )
    require_flash_attention_supported(
        build_attention_call(q, k, v, window_size, softmax_scale), mode="prefill attention"
    )
    # Build this once on the scheduler's CPU metadata path. A manual caller
    # may omit it; only that compatibility path needs a device synchronization.
    cu_k, max_k = meta.cu_seqlens_k, meta.max_seqlen_k
    if cu_k is None or max_k is None:
        kv_lengths = meta.cache_seqlens + lengths
        cu_k = torch.cat((meta.cu_seqlens.new_zeros(1), kv_lengths.cumsum(0, dtype=torch.int32)))
        max_k = int(kv_lengths.max().item())
    # Paged varlen attention accepts different Q and KV lengths. Its causal
    # mask is bottom-right aligned: query i sees prefix_length + i keys.
    # One call handles mixed prefix/chunk lengths without copying cached KV
    # or a CPU synchronization and separate launch for every sequence/layer.
    return _flash_attn_varlen_no_compile(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=meta.cu_seqlens,
        cu_seqlens_k=cu_k,
        max_seqlen_q=meta.max_seqlen,
        max_seqlen_k=max_k,
        block_table=meta.block_table,
        causal=True,
        window_size=window_size,
        softmax_scale=softmax_scale,
    )


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
        force_native=True,
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
        force_native=True,
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
