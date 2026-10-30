"""KIVI asymmetric low-bit KV cache for inference attention.

KIVI (https://arxiv.org/abs/2402.02750) quantizes keys per channel and values
per token. This backend keeps the paged block table owned by the scheduler but
stores each block as packed 2/4-bit codes plus fp16 scale/zero metadata (see
``areno/accel/kernels/kivi.py`` for the layout). Recent tokens stay in full
precision in a per-sequence residual ring indexed by ``InferMeta.recurrent_slots``.

Residual semantics, with ``R = residual_length`` and ``G = group_size``: once a
sequence holds ``n`` tokens, positions ``[0, q(n))`` are quantized, where
``q(n) = max(0, floor((n - 1 - R) / G) * G)``, and positions ``[q(n), n)`` live
in the ring at ``position % (R + G)``. The newest token therefore always sees at
least ``R`` preceding tokens in full precision. Whenever ``n - R`` reaches a
positive multiple of ``G`` the oldest ``G`` ring tokens are quantized into
their paged block before the new token reuses their ring entries.

Prefill attention over fresh prompt tokens stays full precision (the regular
flash/native backend). A prefill chunk that extends cached tokens reads the
prefix through a bounded per-row dequantization. Decode runs a fused split-K
kernel that dequantizes paged tiles in registers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from areno.engine.layers.attention_backend.common import AttnBackend, expand_kv_heads, use_native_attention
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend
from areno.engine.runtime.metadata import InferMeta

KIVI_BITS = {"kivi2": 2, "kivi4": 4}


@dataclass(frozen=True, slots=True)
class KiviSpec:
    """Quantization geometry shared by every layer's KIVI cache."""

    bits: int
    group_size: int
    residual_length: int

    @property
    def ring_size(self) -> int:
        """Ring entries per slot: ``R`` residual tokens plus one group awaiting quantization."""

        return self.residual_length + self.group_size

    def quantized_length(self, tokens: int) -> int:
        """Return ``q(tokens)``: how many leading positions are stored quantized."""

        past = tokens - 1 - self.residual_length
        return past // self.group_size * self.group_size if past >= 0 else 0


@dataclass(slots=True)
class KiviKVCache:
    """One attention layer's KIVI storage.

    Paged tensors are head-major inside each block so kernels read contiguous
    per-head tiles: codes ``[blocks, kv_heads, block_size, head_dim * bits // 8]``
    (uint8), key scale/zero ``[blocks, kv_heads, block_size // G, head_dim]`` and
    value scale/zero ``[blocks, kv_heads, block_size, head_dim // G]`` (fp16).
    Rings are ``[slots, kv_heads, R + G, head_dim]`` in the model dtype.
    """

    spec: KiviSpec
    block_size: int
    k_codes: torch.Tensor
    k_scale: torch.Tensor
    k_zero: torch.Tensor
    v_codes: torch.Tensor
    v_scale: torch.Tensor
    v_zero: torch.Tensor
    k_ring: torch.Tensor
    v_ring: torch.Tensor

    @classmethod
    def allocate(
        cls,
        spec: KiviSpec,
        *,
        num_blocks: int,
        block_size: int,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> KiviKVCache:
        validate_kivi_geometry(spec, block_size=block_size, head_dim=head_dim)
        packed = head_dim * spec.bits // 8
        group = spec.group_size

        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(*shape, dtype=dtype, device=device)

        return cls(
            spec=spec,
            block_size=block_size,
            k_codes=empty(num_blocks, num_kv_heads, block_size, packed, dtype=torch.uint8),
            k_scale=empty(num_blocks, num_kv_heads, block_size // group, head_dim, dtype=torch.float16),
            k_zero=empty(num_blocks, num_kv_heads, block_size // group, head_dim, dtype=torch.float16),
            v_codes=empty(num_blocks, num_kv_heads, block_size, packed, dtype=torch.uint8),
            v_scale=empty(num_blocks, num_kv_heads, block_size, head_dim // group, dtype=torch.float16),
            v_zero=empty(num_blocks, num_kv_heads, block_size, head_dim // group, dtype=torch.float16),
            k_ring=empty(num_slots, num_kv_heads, spec.ring_size, head_dim, dtype=dtype),
            v_ring=empty(num_slots, num_kv_heads, spec.ring_size, head_dim, dtype=dtype),
        )

    @property
    def quantized_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.k_codes, self.k_scale, self.k_zero, self.v_codes, self.v_scale, self.v_zero

    @property
    def num_kv_heads(self) -> int:
        return int(self.k_ring.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.k_ring.shape[3])

    def nbytes(self) -> int:
        """Total device bytes, including scale/zero metadata and the residual rings."""

        return sum(
            tensor.numel() * tensor.element_size() for tensor in (*self.quantized_tensors, self.k_ring, self.v_ring)
        )


def validate_kivi_geometry(spec: KiviSpec, *, block_size: int, head_dim: int) -> None:
    """Reject shapes the packed layout or the Triton kernels cannot express."""

    if spec.bits not in KIVI_BITS.values():
        raise ValueError(f"KIVI supports 2 or 4 bits, got {spec.bits}")
    group = spec.group_size
    if group < 32 or group & (group - 1):
        raise ValueError(f"KIVI group size must be a power of two >= 32, got {group}")
    if spec.residual_length < 0:
        raise ValueError(f"KIVI residual length must be non-negative, got {spec.residual_length}")
    if head_dim & (head_dim - 1) or head_dim % group:
        raise ValueError(f"KIVI needs a power-of-two head dim divisible by group size {group}, got {head_dim}")
    if block_size % group:
        raise ValueError(f"KIVI group size {group} must divide the KV block size {block_size}")


def attach_kivi_kv_caches(
    model: nn.Module,
    spec: KiviSpec,
    *,
    num_blocks: int,
    block_size: int,
    num_slots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[KiviKVCache]:
    """Allocate one KIVI cache per decoder attention layer and bind it."""

    caches = []
    for index, layer in enumerate(model.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None or not hasattr(attention, "set_kivi_cache"):
            raise RuntimeError(f"decoder layer {index} has no attention module that supports the KIVI KV cache")
        cache = KiviKVCache.allocate(
            spec,
            num_blocks=num_blocks,
            block_size=block_size,
            num_slots=num_slots,
            num_kv_heads=attention.local_kv_heads,
            head_dim=attention.head_dim,
            dtype=dtype,
            device=device,
        )
        attention.set_kivi_cache(cache)
        caches.append(cache)
    return caches


@dataclass(slots=True)
class _PrefillPlan:
    """Index plan shared by every layer of one prefill forward.

    Computed once from host-side lengths; layers reuse the device tensors to
    gather their own K/V without further host synchronization.
    """

    rows: list[tuple[int, int, int, int]]  # (token_start, new_tokens, cached_tokens, slot) per row
    has_prefix: bool
    # Optional staging gather: stage[stage_fresh_dst] = fresh[stage_fresh_src],
    # stage[stage_ring_dst] = ring[stage_ring_slot, :, stage_ring_pos].
    stage_tokens: int
    stage_fresh_src: torch.Tensor | None
    stage_fresh_dst: torch.Tensor | None
    stage_ring_slot: torch.Tensor | None
    stage_ring_pos: torch.Tensor | None
    stage_ring_dst: torch.Tensor | None
    # Groups to quantize: source position in the staged token axis and target.
    group_src: torch.Tensor
    group_block: torch.Tensor
    group_offset: torch.Tensor
    # Fresh tokens that end inside the residual window.
    ring_src: torch.Tensor
    ring_slot: torch.Tensor
    ring_pos: torch.Tensor


_PLAN_CACHE: tuple[InferMeta, KiviSpec, int, _PrefillPlan] | None = None


def _prefill_plan(meta: InferMeta, spec: KiviSpec, block_size: int, device: torch.device) -> _PrefillPlan:
    global _PLAN_CACHE
    if _PLAN_CACHE is not None:
        cached_meta, cached_spec, cached_block, plan = _PLAN_CACHE
        if cached_meta is meta and cached_spec == spec and cached_block == block_size:
            return plan
    plan = _build_prefill_plan(meta, spec, block_size, device)
    _PLAN_CACHE = (meta, spec, block_size, plan)
    return plan


def _build_prefill_plan(meta: InferMeta, spec: KiviSpec, block_size: int, device: torch.device) -> _PrefillPlan:
    if meta.cu_seqlens is None or meta.block_table is None or meta.recurrent_slots is None:
        raise ValueError("KIVI prefill requires cu_seqlens, block_table and recurrent_slots")
    offsets = meta.cu_seqlens.tolist()
    slots = meta.recurrent_slots.tolist()
    cached = meta.cache_seqlens.tolist() if meta.cache_seqlens is not None else [0] * len(slots)
    ring_size = spec.ring_size
    group = spec.group_size
    rows = []
    stage_fresh_src: list[int] = []
    stage_fresh_dst: list[int] = []
    stage_ring_slot: list[int] = []
    stage_ring_pos: list[int] = []
    stage_ring_dst: list[int] = []
    group_src: list[int] = []
    group_row: list[int] = []
    group_pos: list[int] = []
    ring_src: list[int] = []
    ring_slot: list[int] = []
    ring_pos: list[int] = []
    stage_cursor = 0
    for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        new_tokens, old, slot = end - start, int(cached[row]), int(slots[row])
        rows.append((start, new_tokens, old, slot))
        old_q, new_q = spec.quantized_length(old), spec.quantized_length(old + new_tokens)
        # Staged positions [old_q, old + new_tokens): ring residue then fresh tokens.
        base = stage_cursor - old_q
        for position in range(old_q, old):
            stage_ring_slot.append(slot)
            stage_ring_pos.append(position % ring_size)
            stage_ring_dst.append(base + position)
        stage_fresh_src.extend(range(start, end))
        stage_fresh_dst.extend(range(base + old, base + old + new_tokens))
        stage_cursor += old + new_tokens - old_q
        for position in range(old_q, new_q, group):
            group_src.append(base + position)
            group_row.append(row)
            group_pos.append(position)
        # Fresh positions that stay in full precision go to the ring. Older
        # ring entries in [new_q, old) are already in place.
        for position in range(max(new_q, old), old + new_tokens):
            ring_src.append(start + position - old)
            ring_slot.append(slot)
            ring_pos.append(position % ring_size)

    def tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device)

    # Without ring residue (every row fresh) the staged axis is exactly the
    # fresh token axis, so groups read the prefill K/V directly.
    has_ring_stage = bool(stage_ring_dst)
    group_row_t = tensor(group_row)
    group_pos_t = tensor(group_pos)
    return _PrefillPlan(
        rows=rows,
        has_prefix=any(old > 0 for _, _, old, _ in rows),
        stage_tokens=stage_cursor,
        stage_fresh_src=tensor(stage_fresh_src) if has_ring_stage else None,
        stage_fresh_dst=tensor(stage_fresh_dst) if has_ring_stage else None,
        stage_ring_slot=tensor(stage_ring_slot) if has_ring_stage else None,
        stage_ring_pos=tensor(stage_ring_pos) if has_ring_stage else None,
        stage_ring_dst=tensor(stage_ring_dst) if has_ring_stage else None,
        group_src=tensor(group_src),
        group_block=meta.block_table.to(dtype=torch.int64)[group_row_t, group_pos_t // block_size],
        group_offset=group_pos_t % block_size,
        ring_src=tensor(ring_src),
        ring_slot=tensor(ring_slot),
        ring_pos=tensor(ring_pos),
    )


def unpack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Expand packed ``(..., head_dim * bits // 8)`` uint8 codes to ``(..., head_dim)`` floats.

    Byte ``j`` holds channels ``j * (8 // bits) + i`` at bit offset ``i * bits``.
    """

    pack = 8 // bits
    shifts = torch.arange(pack, device=codes.device, dtype=torch.int32) * bits
    fields = (codes.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return fields.reshape(*codes.shape[:-1], codes.shape[-1] * pack).to(torch.float32)


def dequantize_blocks(cache: KiviKVCache, block_ids: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the first ``length`` positions stored in ``block_ids``.

    Returns token-major ``(length, kv_heads, head_dim)`` K/V in the ring dtype.
    The workspace is bounded by one sequence's quantized prefix.
    """

    group = cache.spec.group_size
    k = unpack_codes(cache.k_codes[block_ids], cache.spec.bits)
    k = k * cache.k_scale[block_ids].float().repeat_interleave(group, dim=2)
    k = k + cache.k_zero[block_ids].float().repeat_interleave(group, dim=2)
    v = unpack_codes(cache.v_codes[block_ids], cache.spec.bits)
    v = v * cache.v_scale[block_ids].float().repeat_interleave(group, dim=3)
    v = v + cache.v_zero[block_ids].float().repeat_interleave(group, dim=3)

    def token_major(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])[:length].to(cache.k_ring.dtype)

    return token_major(k), token_major(v)


class KiviInferBackend(nn.Module):
    """Prefill/decode attention over a :class:`KiviKVCache`."""

    def __init__(self, attn_backend: AttnBackend = "flash"):
        super().__init__()
        self.attn_backend = attn_backend
        # Fresh prompt attention is unchanged: reuse the dense backend without
        # letting it touch any cache tensor.
        self.dense = FlashAttnInferBackend(attn_backend=attn_backend)

    @torch._dynamo.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: KiviKVCache,
        meta: InferMeta,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        # Dynamo-opaque: the prefill plan synchronizes with the host and the
        # decode path launches Triton kernels on graph-stable buffers.
        if q.shape[-1] != v.shape[-1]:
            raise ValueError("KIVI KV cache requires matching QK and V head dims")
        if meta.recurrent_slots is None:
            raise ValueError("KIVI KV cache requires per-sequence recurrent_slots")
        scale = float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5)
        q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
        k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
        v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
        if meta.mode == "prefill":
            out = self._prefill(q, k, v, q_flat, k_flat, v_flat, cache, meta, scale)
        elif meta.mode == "decode":
            out = self._decode(q_flat, k_flat, v_flat, cache, meta, scale)
        else:
            raise ValueError(f"unsupported inference mode: {meta.mode}")
        return out.view(q.shape)

    def _prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_flat: torch.Tensor,
        cache: KiviKVCache,
        meta: InferMeta,
        scale: float,
    ) -> torch.Tensor:
        plan = _prefill_plan(meta, cache.spec, cache.block_size, q.device)
        if plan.has_prefix:
            out = self._prefix_attention(q_flat, k_flat, v_flat, cache, meta, plan, scale)
        else:
            empty = q.new_empty(0)
            fresh = dataclasses.replace(meta, cache_seqlens=None)
            out = self.dense(q, k, v, empty, empty, fresh, softmax_scale=scale, update_cache=False)
        _store_prefill(k_flat, v_flat, cache, plan)
        return out

    def _prefix_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: KiviKVCache,
        meta: InferMeta,
        plan: _PrefillPlan,
        scale: float,
    ) -> torch.Tensor:
        """Attend each chunk row to its cached prefix plus its own causal tokens."""

        spec = cache.spec
        outputs = []
        for row, (start, new_tokens, old, slot) in enumerate(plan.rows):
            keys, values = [k[start : start + new_tokens]], [v[start : start + new_tokens]]
            if old:
                old_q = spec.quantized_length(old)
                ring = torch.arange(old_q, old, device=q.device) % spec.ring_size
                keys.insert(0, cache.k_ring[slot, :, ring].transpose(0, 1))
                values.insert(0, cache.v_ring[slot, :, ring].transpose(0, 1))
                if old_q:
                    blocks = meta.block_table[row, : -(-old_q // cache.block_size)].long()
                    prefix_k, prefix_v = dequantize_blocks(cache, blocks, old_q)
                    keys.insert(0, prefix_k)
                    values.insert(0, prefix_v)
            outputs.append(
                self._causal_tail_attention(
                    q[start : start + new_tokens], torch.cat(keys), torch.cat(values), old, scale
                )
            )
        return torch.cat(outputs)

    @torch._dynamo.disable
    def _causal_tail_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int, scale: float
    ) -> torch.Tensor:
        """Causal attention for queries at positions ``[query_start, query_start + len(q))``."""

        if use_native_attention(self.attn_backend, q):
            from areno.accel.attention import areno_causal_attention

            heads = q.shape[1]
            out = areno_causal_attention(
                q.transpose(0, 1).unsqueeze(0),
                expand_kv_heads(k, heads).transpose(0, 1).unsqueeze(0),
                expand_kv_heads(v, heads).transpose(0, 1).unsqueeze(0),
                query_start=query_start,
                softmax_scale=scale,
                force_native=True,
            )
            return out.squeeze(0).transpose(0, 1)
        from areno.accel.flash_attention import flash_attention

        cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=q.device)
        # flash-attn aligns causal masks bottom-right when seqlen_k > seqlen_q.
        return flash_attention(q.device).flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=k.shape[0],
            causal=True,
            softmax_scale=scale,
        )

    def _decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: KiviKVCache,
        meta: InferMeta,
        scale: float,
    ) -> torch.Tensor:
        from areno.accel.kernels.kivi import append_tokens, decode_attention

        if getattr(meta, "tokens_per_seq", 1) != 1:
            raise NotImplementedError("KIVI KV cache decodes one token per sequence; disable speculative decoding")
        if meta.cache_seqlens is None or meta.block_table is None:
            raise ValueError("KIVI decode requires cache_seqlens and block_table")
        spec = cache.spec
        cached = meta.cache_seqlens
        slots = meta.recurrent_slots
        # 1) Rows whose residual window is full flush their oldest group into
        #    the paged cache, then every row appends its token to the ring.
        #    One device-side kernel decides per row (CUDA-graph safe).
        append_tokens(
            k.to(cache.k_ring.dtype).contiguous(),
            v.to(cache.v_ring.dtype).contiguous(),
            cached,
            slots,
            meta.block_table,
            cache.k_ring,
            cache.v_ring,
            cache.quantized_tensors,
            residual_length=spec.residual_length,
            group_size=spec.group_size,
            bits=spec.bits,
            block_size=cache.block_size,
        )
        # 2) Fused dequantize + split-K attention over prefix and ring.
        return decode_attention(
            q.contiguous(),
            cache.quantized_tensors,
            cache.k_ring,
            cache.v_ring,
            meta.block_table,
            cached,
            slots,
            softmax_scale=scale,
            residual_length=spec.residual_length,
            group_size=spec.group_size,
            bits=spec.bits,
            block_size=cache.block_size,
            num_splits=decode_num_splits(q.shape[0], cache.num_kv_heads),
        )


def decode_num_splits(rows: int, num_kv_heads: int) -> int:
    """Power-of-two split count giving about 16K (row, KV head, split) programs.

    Tuned on RTX PRO 5000 (sm_120) under CUDA graphs: many more programs than
    SMs keep enough dequantization loads in flight at long context, and the
    empty splits of short rows exit immediately. It depends only on static
    shapes, so a captured decode graph keeps one kernel specialization. The
    split workspace stays near 16K * queries_per_kv * head_dim fp32 values.
    """

    splits = 1
    while splits < 64 and rows * num_kv_heads * splits < 16384:
        splits *= 2
    return splits


def _store_prefill(k: torch.Tensor, v: torch.Tensor, cache: KiviKVCache, plan: _PrefillPlan) -> None:
    """Quantize completed groups and refresh residual rings after a prefill forward."""

    from areno.accel.kernels.kivi import store_groups

    spec = cache.spec
    if plan.group_src.numel():
        if plan.stage_ring_dst is not None:
            # Some groups straddle old ring residue and fresh tokens: stage
            # them contiguously before quantizing.
            stage_k = k.new_empty(plan.stage_tokens, *k.shape[1:])
            stage_v = v.new_empty(plan.stage_tokens, *v.shape[1:])
            stage_k[plan.stage_fresh_dst] = k[plan.stage_fresh_src]
            stage_v[plan.stage_fresh_dst] = v[plan.stage_fresh_src]
            stage_k[plan.stage_ring_dst] = cache.k_ring[plan.stage_ring_slot, :, plan.stage_ring_pos].to(k.dtype)
            stage_v[plan.stage_ring_dst] = cache.v_ring[plan.stage_ring_slot, :, plan.stage_ring_pos].to(v.dtype)
        else:
            stage_k, stage_v = k.contiguous(), v.contiguous()
        store_groups(
            stage_k,
            stage_v,
            plan.group_src,
            plan.group_block,
            plan.group_offset,
            cache.quantized_tensors,
            group_size=spec.group_size,
            bits=spec.bits,
        )
    if plan.ring_src.numel():
        cache.k_ring[plan.ring_slot, :, plan.ring_pos] = k[plan.ring_src].to(cache.k_ring.dtype)
        cache.v_ring[plan.ring_slot, :, plan.ring_pos] = v[plan.ring_src].to(cache.v_ring.dtype)
