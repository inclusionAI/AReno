"""KIVI low-bit KV-cache kernels (asymmetric 2/4-bit storage).

KIVI (Liu et al., 2024, https://arxiv.org/abs/2402.02750) stores keys with
per-channel quantization (one scale/zero per channel for a group of ``GROUP``
consecutive tokens) and values with per-token quantization (one scale/zero per
token for a group of ``GROUP`` consecutive channels). Scales and zeros are
fp16; codes are unsigned ``BITS``-bit integers produced by round-to-nearest
asymmetric min/max quantization.

The paged runtime needs three device operations:

* ``store_groups`` quantizes complete ``GROUP``-token groups of K/V (prefill)
  and writes packed codes plus scale/zero metadata into paged cache blocks.
* ``append_tokens`` is the decode-time cache update: when a sequence's
  residual ring is full it quantizes the oldest ring group into its paged
  block, then writes the new token into the ring.
* ``decode_attention`` runs split-K single-token attention. Each program walks
  ``BLOCK_N``-token tiles of one sequence, dequantizes the paged prefix in
  registers (no full-cache materialization), reads the recent full-precision
  tokens from the ring, and either writes the output or split partials that a
  combine kernel merges.

Packing layout: ``8 // BITS`` codes share one byte. Byte ``j`` of a token row
holds channels ``j * (8 // BITS) + i`` at bit offset ``i * BITS``, so unpacking
is a join of the shifted fields followed by a reshape of the trailing axes.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# Largest finite fp16 value; scales/zeros saturate here instead of overflowing.
_FP16_MAX = tl.constexpr(65504.0)


@triton.jit
def _quantize(x, zero, scale, QMAX: tl.constexpr):
    # Round-to-nearest asymmetric quantization against the stored fp16
    # scale/zero, so dequantization reproduces exactly what was rounded. A zero
    # scale (constant group) maps every element to code 0, i.e. back to `zero`.
    inv = tl.where(scale > 0, 1.0 / scale, 0.0)
    codes = tl.floor((x - zero) * inv + 0.5)
    return tl.minimum(tl.maximum(codes, 0.0), QMAX).to(tl.int32)


@triton.jit
def _pack(codes, ROWS: tl.constexpr, HEAD_DIM: tl.constexpr, BITS: tl.constexpr):
    # (ROWS, HEAD_DIM) int codes -> (ROWS, HEAD_DIM * BITS // 8) uint8. Channel
    # c = j * PACK + i lands in byte j at bit offset i * BITS; the fields are
    # disjoint, so summing the shifted codes is a bitwise OR.
    PACK: tl.constexpr = 8 // BITS
    shifts = tl.arange(0, PACK) * BITS
    grouped = tl.reshape(codes, (ROWS, HEAD_DIM // PACK, PACK))
    return tl.sum(grouped << shifts[None, None, :], axis=2).to(tl.uint8)


@triton.jit
def _unpack(packed, ROWS: tl.constexpr, HEAD_DIM: tl.constexpr, BITS: tl.constexpr):
    # Inverse of `_pack`: (ROWS, packed_dim) uint8 -> (ROWS, HEAD_DIM) fp32 codes.
    # Joining the shifted fields adds trailing axes held by the same thread, so
    # the reshape back to channel order needs no data movement. `tl.join`
    # appends its new axis last, so the outer join pairs fields (0, 2) with
    # (1, 3): element [j, a, b] holds field 2 * a + b, which the reshape places
    # at channel 4 * j + 2 * a + b.
    x = packed.to(tl.int32)
    if BITS == 4:
        fields = tl.join(x & 15, x >> 4)
    else:
        fields = tl.join(tl.join(x & 3, (x >> 4) & 3), tl.join((x >> 2) & 3, x >> 6))
    return tl.reshape(fields, (ROWS, HEAD_DIM)).to(tl.float32)


@triton.jit
def _store_group(
    k,
    v,
    block,
    offset,
    head,
    K_CODES,
    K_SCALE,
    K_ZERO,
    V_CODES,
    V_SCALE,
    V_ZERO,
    stride_kc_block,
    stride_kc_head,
    stride_kc_tok,
    stride_ks_block,
    stride_ks_head,
    stride_ks_group,
    stride_vc_block,
    stride_vc_head,
    stride_vc_tok,
    stride_vs_block,
    stride_vs_head,
    stride_vs_tok,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BITS: tl.constexpr,
):
    # Quantize one (GROUP, HEAD_DIM) fp32 group of K and V for one KV head and
    # write it at token `offset` (a multiple of GROUP) of paged block `block`.
    QMAX: tl.constexpr = (1 << BITS) - 1
    tok = tl.arange(0, GROUP)
    dim = tl.arange(0, HEAD_DIM)

    # Keys: per-channel range over the group's tokens.
    k_min = tl.min(k, axis=0)
    k_scale = tl.minimum((tl.max(k, axis=0) - k_min) / QMAX, _FP16_MAX).to(tl.float16)
    k_zero = tl.minimum(tl.maximum(k_min, -_FP16_MAX), _FP16_MAX).to(tl.float16)
    k_codes = _quantize(k, k_zero.to(tl.float32)[None, :], k_scale.to(tl.float32)[None, :], QMAX)

    # Values: per-token range over each GROUP-channel slice of the head.
    v_grouped = tl.reshape(v, (GROUP, HEAD_DIM // GROUP, GROUP))
    v_min = tl.min(v_grouped, axis=2)
    v_scale = tl.minimum((tl.max(v_grouped, axis=2) - v_min) / QMAX, _FP16_MAX).to(tl.float16)
    v_zero = tl.minimum(tl.maximum(v_min, -_FP16_MAX), _FP16_MAX).to(tl.float16)
    v_codes = tl.reshape(
        _quantize(v_grouped, v_zero.to(tl.float32)[:, :, None], v_scale.to(tl.float32)[:, :, None], QMAX),
        (GROUP, HEAD_DIM),
    )

    packed = tl.arange(0, HEAD_DIM * BITS // 8)
    rows = (offset + tok)[:, None]
    kc = block * stride_kc_block + head * stride_kc_head + rows * stride_kc_tok + packed[None, :]
    tl.store(K_CODES + kc, _pack(k_codes, GROUP, HEAD_DIM, BITS))
    ks = block * stride_ks_block + head * stride_ks_head + (offset // GROUP) * stride_ks_group + dim
    tl.store(K_SCALE + ks, k_scale)
    tl.store(K_ZERO + ks, k_zero)
    vc = block * stride_vc_block + head * stride_vc_head + rows * stride_vc_tok + packed[None, :]
    tl.store(V_CODES + vc, _pack(v_codes, GROUP, HEAD_DIM, BITS))
    vgroup = tl.arange(0, HEAD_DIM // GROUP)
    vs = block * stride_vs_block + head * stride_vs_head + rows * stride_vs_tok + vgroup[None, :]
    tl.store(V_SCALE + vs, v_scale)
    tl.store(V_ZERO + vs, v_zero)


@triton.jit
def _kivi_store_groups_kernel(
    K_SRC,
    V_SRC,
    SRC_POS,
    DST_BLOCK,
    DST_OFFSET,
    K_CODES,
    K_SCALE,
    K_ZERO,
    V_CODES,
    V_SCALE,
    V_ZERO,
    stride_src_head,
    stride_src_pos,
    stride_kc_block,
    stride_kc_head,
    stride_kc_tok,
    stride_ks_block,
    stride_ks_head,
    stride_ks_group,
    stride_vc_block,
    stride_vc_head,
    stride_vc_tok,
    stride_vs_block,
    stride_vs_head,
    stride_vs_tok,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BITS: tl.constexpr,
):
    # Grid: (group, KV head). Group g reads tokens [SRC_POS[g], SRC_POS[g] + GROUP)
    # of the token-major prefill staging tensor.
    group_id = tl.program_id(0)
    head = tl.program_id(1)
    src_pos = tl.load(SRC_POS + group_id).to(tl.int64)
    tok = tl.arange(0, GROUP)
    dim = tl.arange(0, HEAD_DIM)
    src = head * stride_src_head + (src_pos + tok)[:, None] * stride_src_pos + dim[None, :]
    _store_group(
        tl.load(K_SRC + src).to(tl.float32),
        tl.load(V_SRC + src).to(tl.float32),
        tl.load(DST_BLOCK + group_id).to(tl.int64),
        tl.load(DST_OFFSET + group_id).to(tl.int64),
        head,
        K_CODES,
        K_SCALE,
        K_ZERO,
        V_CODES,
        V_SCALE,
        V_ZERO,
        stride_kc_block,
        stride_kc_head,
        stride_kc_tok,
        stride_ks_block,
        stride_ks_head,
        stride_ks_group,
        stride_vc_block,
        stride_vc_head,
        stride_vc_tok,
        stride_vs_block,
        stride_vs_head,
        stride_vs_tok,
        GROUP,
        HEAD_DIM,
        BITS,
    )


@triton.jit
def _kivi_append_kernel(
    K_NEW,
    V_NEW,
    CACHE_SEQLENS,
    SLOTS,
    BLOCK_TABLE,
    K_RING,
    V_RING,
    K_CODES,
    K_SCALE,
    K_ZERO,
    V_CODES,
    V_SCALE,
    V_ZERO,
    residual,
    ring_size,
    stride_new_row,
    stride_new_head,
    stride_bt_row,
    stride_ring_slot,
    stride_ring_head,
    stride_ring_pos,
    stride_kc_block,
    stride_kc_head,
    stride_kc_tok,
    stride_ks_block,
    stride_ks_head,
    stride_ks_group,
    stride_vc_block,
    stride_vc_head,
    stride_vc_tok,
    stride_vs_block,
    stride_vs_head,
    stride_vs_tok,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (row, KV head). CACHE_SEQLENS holds n, the tokens cached before
    # this step; the new token takes position n. When n - R is a positive
    # multiple of GROUP the ring is full: its oldest group [n - R - GROUP, n - R)
    # is quantized into the paged cache first, then the new token reuses the
    # ring entry of position n - R - GROUP.
    row = tl.program_id(0)
    head = tl.program_id(1)
    cached = tl.load(CACHE_SEQLENS + row)
    slot = tl.load(SLOTS + row).to(tl.int64)
    dim = tl.arange(0, HEAD_DIM)
    ring = K_RING + slot * stride_ring_slot + head * stride_ring_head
    ring_v = V_RING + slot * stride_ring_slot + head * stride_ring_head
    past = cached - residual
    if (past >= GROUP) & (past % GROUP == 0):
        start = past - GROUP
        tok = tl.arange(0, GROUP)
        src = ((start + tok) % ring_size)[:, None] * stride_ring_pos + dim[None, :]
        block = tl.load(BLOCK_TABLE + row * stride_bt_row + start // BLOCK_SIZE).to(tl.int64)
        _store_group(
            tl.load(ring + src).to(tl.float32),
            tl.load(ring_v + src).to(tl.float32),
            block,
            (start % BLOCK_SIZE).to(tl.int64),
            head,
            K_CODES,
            K_SCALE,
            K_ZERO,
            V_CODES,
            V_SCALE,
            V_ZERO,
            stride_kc_block,
            stride_kc_head,
            stride_kc_tok,
            stride_ks_block,
            stride_ks_head,
            stride_ks_group,
            stride_vc_block,
            stride_vc_head,
            stride_vc_tok,
            stride_vs_block,
            stride_vs_head,
            stride_vs_tok,
            GROUP,
            HEAD_DIM,
            BITS,
        )
        # Every thread must finish reading the flushed group before any thread
        # overwrites its first ring entry with the new token.
        tl.debug_barrier()
    new = row * stride_new_row + head * stride_new_head + dim
    dst = (cached % ring_size) * stride_ring_pos + dim
    tl.store(ring + dst, tl.load(K_NEW + new))
    tl.store(ring_v + dst, tl.load(V_NEW + new))


@triton.jit
def _attend(q, k, v, valid, m_i, l_i, acc, qk_scale):
    # One online-softmax update of (m_i, l_i, acc) with a (BLOCK_N, HEAD_DIM)
    # fp32 K/V tile; `q` is fp16. Dots run on fp16 tensor cores (10 mantissa
    # bits; dequantized K/V and probabilities are well inside its range). Keys
    # are split into high and low fp16 halves so the scores stay close to fp32;
    # P @ V needs a single fp16 dot for the same tolerance.
    k_hi = k.to(tl.float16)
    k_lo = (k - k_hi.to(tl.float32)).to(tl.float16)
    scores = (tl.dot(q, tl.trans(k_hi)) + tl.dot(q, tl.trans(k_lo))) * qk_scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(scores - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
    return m_new, l_i, acc


@triton.jit
def _kivi_decode_kernel(
    Q,
    K_CODES,
    K_SCALE,
    K_ZERO,
    V_CODES,
    V_SCALE,
    V_ZERO,
    K_RING,
    V_RING,
    BLOCK_TABLE,
    CACHE_SEQLENS,
    SLOTS,
    OUT,
    PART_ACC,
    PART_MAX,
    PART_SUM,
    qk_scale,
    residual,
    ring_size,
    stride_q_row,
    stride_q_head,
    stride_o_row,
    stride_o_head,
    stride_kc_block,
    stride_kc_head,
    stride_kc_tok,
    stride_ks_block,
    stride_ks_head,
    stride_ks_group,
    stride_vc_block,
    stride_vc_head,
    stride_vc_tok,
    stride_vs_block,
    stride_vs_head,
    stride_vs_tok,
    stride_ring_slot,
    stride_ring_head,
    stride_ring_pos,
    stride_bt_row,
    stride_pa_row,
    stride_pa_head,
    stride_pa_split,
    stride_pm_row,
    stride_pm_head,
    QUERIES_PER_KV: tl.constexpr,
    HEADS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # Grid: (row, KV head, split). One program serves the QUERIES_PER_KV query
    # heads sharing this KV head (GQA), padded to HEADS_PAD >= 16 rows for
    # tensor-core dots. CACHE_SEQLENS holds the tokens cached before this step;
    # the new token is already in the ring, so the row attends to positions
    # [0, total) with total = cached + 1: the quantized prefix [0, quantized)
    # through the block table and [quantized, total) through the ring.
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    total = tl.load(CACHE_SEQLENS + row) + 1
    slot = tl.load(SLOTS + row).to(tl.int64)
    past = total - 1 - residual
    quantized = tl.where(past >= 0, (past // GROUP) * GROUP, 0)
    # Whole quantized tiles first (no masks, no ring reads), then ring tiles.
    quant_tiles = quantized // BLOCK_N
    num_tiles = quant_tiles + tl.cdiv(total - quantized, BLOCK_N)
    per_split = tl.cdiv(num_tiles, NUM_SPLITS)
    tile_begin = split * per_split
    tile_end = tl.minimum(tile_begin + per_split, num_tiles)

    heads = tl.arange(0, HEADS_PAD)
    head_mask = heads < QUERIES_PER_KV
    q_heads = kv_head * QUERIES_PER_KV + heads
    dim = tl.arange(0, HEAD_DIM)
    tok = tl.arange(0, BLOCK_N)
    packed = tl.arange(0, HEAD_DIM * BITS // 8)
    vgroup = tl.arange(0, HEAD_DIM // GROUP)
    q = tl.load(
        Q + row * stride_q_row + q_heads[:, None] * stride_q_head + dim[None, :],
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.float16)
    ring_k = K_RING + slot * stride_ring_slot + kv_head * stride_ring_head
    ring_v = V_RING + slot * stride_ring_slot + kv_head * stride_ring_head
    m_i = tl.full([HEADS_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([HEADS_PAD], tl.float32)
    acc = tl.zeros([HEADS_PAD, HEAD_DIM], tl.float32)
    # Whole quantized tiles: per-block base pointers are 64-bit scalars and the
    # per-element offsets stay 32-bit. BLOCK_N divides GROUP, so a tile shares
    # one key scale row and never straddles a paged block.
    tile_rows = tok[:, None]
    all_valid = tok < BLOCK_N
    for tile in range(tile_begin, tl.minimum(tile_end, quant_tiles)):
        start = tile * BLOCK_N
        block = tl.load(BLOCK_TABLE + row * stride_bt_row + start // BLOCK_SIZE).to(tl.int64)
        first = start % BLOCK_SIZE
        k_base = K_CODES + block * stride_kc_block + kv_head * stride_kc_head + first * stride_kc_tok
        ks_base = block * stride_ks_block + kv_head * stride_ks_head + (first // GROUP) * stride_ks_group
        k = _unpack(tl.load(k_base + tile_rows * stride_kc_tok + packed[None, :]), BLOCK_N, HEAD_DIM, BITS)
        k = (
            k * tl.load(K_SCALE + ks_base + dim).to(tl.float32)[None, :]
            + tl.load(K_ZERO + ks_base + dim).to(tl.float32)[None, :]
        )
        v_base = V_CODES + block * stride_vc_block + kv_head * stride_vc_head + first * stride_vc_tok
        vs_base = block * stride_vs_block + kv_head * stride_vs_head + first * stride_vs_tok
        vs_offsets = tile_rows * stride_vs_tok + vgroup[None, :]
        v_scale = tl.load(V_SCALE + vs_base + vs_offsets).to(tl.float32)
        v_zero = tl.load(V_ZERO + vs_base + vs_offsets).to(tl.float32)
        # Broadcast the per-(token, channel-group) metadata over each group.
        v_scale = tl.reshape(
            tl.broadcast_to(v_scale[:, :, None], (BLOCK_N, HEAD_DIM // GROUP, GROUP)), (BLOCK_N, HEAD_DIM)
        )
        v_zero = tl.reshape(
            tl.broadcast_to(v_zero[:, :, None], (BLOCK_N, HEAD_DIM // GROUP, GROUP)), (BLOCK_N, HEAD_DIM)
        )
        v = _unpack(tl.load(v_base + tile_rows * stride_vc_tok + packed[None, :]), BLOCK_N, HEAD_DIM, BITS)
        m_i, l_i, acc = _attend(q, k, v * v_scale + v_zero, all_valid, m_i, l_i, acc, qk_scale)

    # Residual tiles from the ring, masked past the newest token.
    for tile in range(tl.maximum(tile_begin, quant_tiles), tile_end):
        positions = quantized + (tile - quant_tiles) * BLOCK_N + tok
        valid = positions < total
        ring = (positions % ring_size)[:, None] * stride_ring_pos + dim[None, :]
        k = tl.load(ring_k + ring, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(ring_v + ring, mask=valid[:, None], other=0.0).to(tl.float32)
        m_i, l_i, acc = _attend(q, k, v, valid, m_i, l_i, acc, qk_scale)

    if NUM_SPLITS == 1:
        out = OUT + row * stride_o_row + q_heads[:, None] * stride_o_head + dim[None, :]
        tl.store(out, acc / l_i[:, None], mask=head_mask[:, None])
    else:
        pa = row * stride_pa_row + q_heads[:, None] * stride_pa_head + split * stride_pa_split + dim[None, :]
        tl.store(PART_ACC + pa, acc, mask=head_mask[:, None])
        pm = row * stride_pm_row + q_heads * stride_pm_head + split
        tl.store(PART_MAX + pm, m_i, mask=head_mask)
        tl.store(PART_SUM + pm, l_i, mask=head_mask)


@triton.jit
def _kivi_combine_kernel(
    PART_ACC,
    PART_MAX,
    PART_SUM,
    OUT,
    stride_pa_row,
    stride_pa_head,
    stride_pa_split,
    stride_pm_row,
    stride_pm_head,
    stride_o_row,
    stride_o_head,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # Grid: (row, query head). Merge the split partials with the usual
    # log-sum-exp rescaling. Empty splits carry (max=-inf, sum=0) and are
    # masked out so their accumulators never contribute.
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.arange(0, NUM_SPLITS)
    pm = row * stride_pm_row + head * stride_pm_head + split
    m = tl.load(PART_MAX + pm)
    l = tl.load(PART_SUM + pm)
    live = l > 0
    m_max = tl.max(tl.where(live, m, float("-inf")), axis=0)
    weight = tl.where(live, tl.exp2(m - m_max), 0.0)
    norm = tl.sum(l * weight, axis=0)
    dim = tl.arange(0, HEAD_DIM)
    pa = row * stride_pa_row + head * stride_pa_head + split[:, None] * stride_pa_split + dim[None, :]
    acc = tl.load(PART_ACC + pa, mask=live[:, None], other=0.0)
    out = tl.sum(acc * weight[:, None], axis=0) / norm
    tl.store(OUT + row * stride_o_row + head * stride_o_head + dim, out)


def _cache_strides(cache_tensors: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    k_codes, k_scale, _, v_codes, v_scale, _ = cache_tensors
    return (
        k_codes.stride(0),
        k_codes.stride(1),
        k_codes.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_codes.stride(0),
        v_codes.stride(1),
        v_codes.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
    )


def store_groups(
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    src_pos: torch.Tensor,
    dst_block: torch.Tensor,
    dst_offset: torch.Tensor,
    cache_tensors: tuple[torch.Tensor, ...],
    *,
    group_size: int,
    bits: int,
) -> None:
    """Quantize ``GROUP``-token groups of token-major ``(tokens, kv_heads, head_dim)`` K/V.

    Group ``g`` covers tokens ``[src_pos[g], src_pos[g] + group_size)`` and is
    written at token ``dst_offset[g]`` of paged block ``dst_block[g]``.
    ``cache_tensors`` is ``(k_codes, k_scale, k_zero, v_codes, v_scale, v_zero)``.
    """

    num_groups = int(src_pos.numel())
    if num_groups == 0:
        return
    if k_src.stride() != v_src.stride() or k_src.stride(-1) != 1:
        raise ValueError("store_groups expects K/V sources with identical strides and contiguous channels")
    _kivi_store_groups_kernel[(num_groups, k_src.shape[1])](
        k_src,
        v_src,
        src_pos,
        dst_block,
        dst_offset,
        *cache_tensors,
        k_src.stride(1),
        k_src.stride(0),
        *_cache_strides(cache_tensors),
        GROUP=group_size,
        HEAD_DIM=k_src.shape[2],
        BITS=bits,
        num_warps=4,
    )


def append_tokens(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache_seqlens: torch.Tensor,
    slots: torch.Tensor,
    block_table: torch.Tensor,
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    cache_tensors: tuple[torch.Tensor, ...],
    *,
    residual_length: int,
    group_size: int,
    bits: int,
    block_size: int,
) -> None:
    """Decode-time update: flush full residual groups, then append ``(rows, kv_heads, head_dim)`` tokens."""

    rows, num_kv_heads, head_dim = k_new.shape
    if rows == 0:
        return
    if k_new.stride() != v_new.stride() or k_new.stride(-1) != 1:
        raise ValueError("append_tokens expects K/V with identical strides and contiguous channels")
    _kivi_append_kernel[(rows, num_kv_heads)](
        k_new,
        v_new,
        cache_seqlens,
        slots,
        block_table,
        k_ring,
        v_ring,
        *cache_tensors,
        residual_length,
        k_ring.shape[2],
        k_new.stride(0),
        k_new.stride(1),
        block_table.stride(0),
        k_ring.stride(0),
        k_ring.stride(1),
        k_ring.stride(2),
        *_cache_strides(cache_tensors),
        GROUP=group_size,
        HEAD_DIM=head_dim,
        BITS=bits,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


def decode_attention(
    q: torch.Tensor,
    cache_tensors: tuple[torch.Tensor, ...],
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    slots: torch.Tensor,
    *,
    softmax_scale: float,
    residual_length: int,
    group_size: int,
    bits: int,
    block_size: int,
    num_splits: int,
    block_n: int = 32,
    num_warps: int = 2,
) -> torch.Tensor:
    """Single-token decode attention over a KIVI paged cache.

    ``q`` is ``(rows, query_heads, head_dim)``. Each row attends to positions
    ``[0, cache_seqlens[row]]``: the quantized prefix through ``block_table``
    and the residual tail through ``k_ring``/``v_ring[slots[row]]``.
    ``num_splits`` must be a power of two; ``block_n`` (tokens per tile) must
    divide the group size. Tiles narrower than 32 tokens are not supported.
    """

    rows, query_heads, head_dim = q.shape
    num_kv_heads = k_ring.shape[1]
    queries_per_kv = query_heads // num_kv_heads
    out = torch.empty_like(q)
    if rows == 0:
        return out
    if group_size % block_n or block_n < 32:
        raise ValueError(f"block_n {block_n} must be >= 32 and divide the KIVI group size {group_size}")
    if num_splits > 1:
        part_acc = torch.empty((rows, query_heads, num_splits, head_dim), device=q.device, dtype=torch.float32)
        part_max = torch.empty((rows, query_heads, num_splits), device=q.device, dtype=torch.float32)
        part_sum = torch.empty_like(part_max)
    else:
        part_acc = part_max = part_sum = out
    _kivi_decode_kernel[(rows, num_kv_heads, num_splits)](
        q,
        *cache_tensors,
        k_ring,
        v_ring,
        block_table,
        cache_seqlens,
        slots,
        out,
        part_acc,
        part_max,
        part_sum,
        softmax_scale * math.log2(math.e),
        residual_length,
        k_ring.shape[2],
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        *_cache_strides(cache_tensors),
        k_ring.stride(0),
        k_ring.stride(1),
        k_ring.stride(2),
        block_table.stride(0),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2) if num_splits > 1 else 0,
        part_max.stride(0),
        part_max.stride(1),
        QUERIES_PER_KV=queries_per_kv,
        HEADS_PAD=max(16, triton.next_power_of_2(queries_per_kv)),
        HEAD_DIM=head_dim,
        GROUP=group_size,
        BITS=bits,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        num_warps=num_warps,
        # Deeper software pipelining buffers several dequantization tiles in
        # shared memory and exceeds the ~100 KiB per-block limit of sm_120.
        num_stages=1,
    )
    if num_splits > 1:
        _kivi_combine_kernel[(rows, query_heads)](
            part_acc,
            part_max,
            part_sum,
            out,
            part_acc.stride(0),
            part_acc.stride(1),
            part_acc.stride(2),
            part_max.stride(0),
            part_max.stride(1),
            out.stride(0),
            out.stride(1),
            HEAD_DIM=head_dim,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
    return out
