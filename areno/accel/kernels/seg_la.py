"""
Segmented Linear Attention (seg-LA) Triton kernels.

Linear attention reformulates softmax attention as a cumulative outer-product
state update so that each new token has O(d^2) work instead of O(L*d). This
file implements the *segmented* variant: multiple variable-length requests
share one launch, indexed via packed offset arrays (``q_offsets`` /
``q_lengths`` / ``s_offsets`` / ``s_scales``). A persistent per-request state
matrix ``S`` of shape (head_dim, head_dim) is carried across chunks so that
prefill can be split into pieces and decode can append one token at a time.

Mathematical core (per head, per request):
    state_t = state_{t-1} * exp(-decay) + k_t^T v_t
    o_t     = q_t @ state_t * softmax_scale
For multi-token chunks we apply the matrix exponential form
``state * exp(-decay * b) + sum_i k_i * exp(-decay * (b-1-i)) v_i`` and a
masked-causal Q @ K^T tile to express the in-block contribution.

Kernels in this file:
    * ``seg_la_kernel``     — generic prefill+decode fallback (kept for the
      commented-out reference path); supports an optional DECOUPLE rescale.
    * ``seg_la_p_kernel``   — prefill, partitioned along (K_SPLIT_DIM,
      V_SPLIT_DIM) so the outer product fits in SRAM.
    * ``seg_la_s_kernel``   — speculative decoding with a precomputed
      tree/position mask.
    * ``seg_la_d_kernel``   — single-token decode.
    * ``seg_la_mtp_kernel`` — multi-token-prediction (MTP) decode that also
      snapshots the per-step state into ``CACHES`` for downstream
      verification.
    * ``seg_la_sum_kernel`` — reduce-across-K-partitions when the head_dim
      is split into multiple K-blocks.

``EVEN`` constexpr branches: when the block length divides BLOCK exactly we
skip the load/store mask construction, which speeds up the tight inner loop
materially.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


# arg `meta` of `seg_la_fwd` is SegLaMeta
@dataclass
class SegLaMeta:
    """Per-batch metadata required to dispatch segmented linear attention.

    The kernel processes many variable-length requests in one launch; this
    struct packs the descriptors. ``q_offsets`` gives the start index of each
    request inside the flattened (sum_l, heads, head_dim) Q/K/V tensors,
    ``q_lengths`` gives each request's length. ``s_offsets`` is the slot id
    in the persistent state pool (or ``-1`` to skip an entry), and
    ``s_scales`` is 0 for the very first prefill chunk of a request (state
    zero-initialised inside the kernel) or 1 for continuation chunks (state
    loaded from ``S``).
    """

    batch_size: int  # batch size, num of requests
    max_q_length: int  # max(seq_lens)
    q_offsets: torch.Tensor  # [bs+1], query_start_locations,
    s_offsets: torch.Tensor  # [bs], slot_ids
    q_lengths: torch.Tensor  # [bs], query length
    s_scales: torch.Tensor  # [bs], prefill = 0, decode = 1
    s_offsets_stride: int = 0
    q_offsets_stride: int = 0
    s_scales_stride: int = 0
    decay_scales_stride: int = 0
    mask: torch.Tensor | None = None  # Currently not supported


# fused
@triton.jit
def seg_la_kernel(
    # ------------------------------------------------------------------
    # Generic segmented-LA kernel supporting both prefill (BLOCK > 1) and
    # decode (BLOCK == 1) on the same launch. Retained as a reference /
    # fallback path; in production we dispatch the specialised P/D/S/MTP
    # variants below for better occupancy.
    #
    # Grid: (bid=batch, hid=qo head, sid=value-dim split).
    # Per program, we own one request's slice for one (head, value-split).
    # The K-dim is *not* split here (HEAD_DIM full); see seg_la_p_kernel for
    # the K-split version used during prefill.
    # ------------------------------------------------------------------
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    q_offsets,
    q_lengths,
    s_scales,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    SPLIT_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    DECOUPLE: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    # Load this request's metadata. s_scale=0 means we start from a zero
    # state; s_scale=1 means we resume from the persistent slot.
    # s_scale is 0 (prefill) or 1 (decode)
    s_scale = tl.load(s_scales + bid)
    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)
    # decay_scale is negated up front so subsequent ``exp(decay_scale * x)``
    # is a *decay* (exp(-rate * x)). One scale per head.
    decay_scale = -tl.load(decay_scales + hid)

    # Lane indices: offs_b runs along the in-block token axis, offs_d along
    # the full HEAD_DIM (Q/K side), offs_s along the V-side split.
    offs_b = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLIT_DIM)

    # Sentinel: -1 marks a slot that should be skipped entirely (e.g. padding
    # request added to keep the grid rectangular).
    if s_offset == -1:
        return

    # Pointer setup. q_offset addresses the request's row in the packed
    # (sum_l, heads, head_dim) layout; hid * HEAD_DIM selects this head.
    # offs_b broadcasts over rows via stride_q (the per-token stride).
    q_ptrs = Q + q_offset * stride_q + hid * HEAD_DIM + (offs_b[:, None] * stride_q + offs_d[None, :])
    k_ptrs = K + q_offset * stride_k + hid * HEAD_DIM + (offs_b[:, None] * stride_k + offs_d[None, :])
    # V and Out additionally use sid * SPLIT_DIM to take this program's
    # value-dim split (V is the dimension we parallelise over for occupancy).
    v_ptrs = V + q_offset * stride_v + hid * HEAD_DIM + sid * SPLIT_DIM + (offs_b[:, None] * stride_v + offs_s[None, :])
    out_ptrs = (
        Out + q_offset * stride_o + hid * HEAD_DIM + sid * SPLIT_DIM + (offs_b[:, None] * stride_o + offs_s[None, :])
    )
    # State layout: (slots, heads, HEAD_DIM, HEAD_DIM). For each slot, head we
    # load the (HEAD_DIM, SPLIT_DIM) slice owned by this program.
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + sid * SPLIT_DIM
        + (offs_d[:, None] * HEAD_DIM + offs_s[None, :])
    )
    # mask=s_scale>0 zeros the state on the very first chunk (cold start).
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    if BLOCK > 1:
        # ----- Prefill / multi-token path -----
        for n in range(0, q_length, BLOCK):
            n = tl.multiple_of(n, BLOCK)

            # EVEN branch removes the per-element bounds mask in the hot
            # path. Triton hoists the mask compute out otherwise.
            if EVEN:
                q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
                k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)
                v = tl.load(v_ptrs + n * stride_k).to(tl.float32)
            else:
                q = tl.load(
                    q_ptrs + n * stride_q,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                ).to(tl.float32)
                k = tl.trans(
                    tl.load(
                        k_ptrs + n * stride_k,
                        mask=(n + offs_b)[:, None] < q_length,
                        other=0.0,
                    )
                ).to(tl.float32)
                v = tl.load(
                    v_ptrs + n * stride_k,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                ).to(tl.float32)

            if DECOUPLE:
                # Decoupled form rescales Q and K independently so the in-block
                # dot product avoids the per-pair exp factor entirely. Only
                # numerically safe when the decay magnitudes are small (the
                # rescaled values must fit in fp16/bf16 dynamic range).
                # only work with small scales
                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)
                # b_offs counts positions from end-of-block backwards: the
                # last token has offset 0, the first has offset b-1.
                b_offs = b - 1 - offs_b

                # Per-row decay factors: inv_decays scales Q so that the
                # implicit decay of K cancels out in the bilinear form.
                edb = tl.exp(decay_scale * b_offs)
                decays = tl.where(b_offs >= 0, edb, 0)
                inv_decays = tl.where(b_offs >= 0, 1 / edb, 0)

                q = q * inv_decays[:, None]
                k = k * decays[None, :]
                qk = tl.dot(q, k) * softmax_scale
                # Lower-triangular causal mask within the block.
                qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
                o = tl.dot(qk, v)

                # Add the contribution from the prior accumulated state,
                # decayed once per block step (block_decay * softmax_scale).
                block_decay = tl.exp(decay_scale * b)
                block_decay_plus = block_decay * softmax_scale
                o = tl.dot(q, state) * block_decay_plus + o

                # Recursive state update: decay then accumulate this block's
                # k^T v outer products.
                state = state * block_decay + tl.dot(k, v)
            else:
                # Coupled (default) form: in-block decay is folded into the
                # qk matrix via an explicit (i - j) decay grid, then masked
                # causally. More general but extra exp calls.
                qk = tl.dot(q, k) * softmax_scale
                decays = tl.exp(decay_scale * (offs_b[:, None] - offs_b[None, :]))
                decays = tl.where(offs_b[None, :] <= offs_b[:, None], decays, 0.0)
                qk *= decays
                o = tl.dot(qk, v)

                # State-attention contribution decays with token position
                # within block (+1 because state is "one step behind"). The
                # ``acc=o`` form lets the FMA-style dot accumulate in place.
                decay_arr = tl.exp(decay_scale * (offs_b[:, None] + 1)) * softmax_scale
                o = tl.dot(q * decay_arr, state, acc=o)

                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)
                # Sentinel 10000 keeps inactive lanes' exp values negligible
                # without branching; the corresponding k entries are zero.
                b_offs = b - 1 - offs_b
                b_offs = tl.where(b_offs >= 0, b_offs, 10000)
                decays = tl.exp(decay_scale * b_offs)
                block_decay = tl.exp(decay_scale * b)
                state = state * block_decay + tl.dot(k * decays[None, :], v)

            if EVEN:
                tl.store(out_ptrs + n * stride_o, o.to(Out.dtype.element_ty))
            else:
                tl.store(
                    out_ptrs + n * stride_o,
                    o.to(Out.dtype.element_ty),
                    mask=(n + offs_b)[:, None] < q_length,
                )

        # Persist the updated state back to the slot pool.
        tl.store(s_ptrs, state.to(S.dtype.element_ty))

    else:
        # ----- Decode path (BLOCK == 1, one token) -----
        # Single-token update: no in-block causal mask, just one outer product
        # and one matvec read of the state. q is transposed because the
        # state is laid out as (K, V) and we want a (V,) output via a sum
        # over K.
        q = tl.trans(tl.load(q_ptrs)).to(tl.float32) * softmax_scale
        k = tl.trans(tl.load(k_ptrs)).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        state = state * tl.exp(decay_scale) + k * v

        o = tl.sum(q * state, axis=0, keep_dims=True)

        tl.store(out_ptrs, o.to(Out.dtype.element_ty))

        tl.store(s_ptrs, state.to(S.dtype.element_ty))


# used for prefilling
@triton.jit
def seg_la_p_kernel(
    # ------------------------------------------------------------------
    # Prefill-specialised seg-LA kernel. Compared with the generic kernel
    # this version splits HEAD_DIM into K_SPLIT_DIM and V_SPLIT_DIM tiles so
    # the (K_SPLIT_DIM, V_SPLIT_DIM) state slice and the per-block scratch
    # fit comfortably in registers/SMEM, enabling higher occupancy.
    #
    # Grid: (bid=request, hid=qo-head, kvid=combined K-split * V-split index).
    # The K-split partial results are summed across (kid) afterwards in
    # ``seg_la_sum_kernel`` or via ``tmp.sum(0)`` for short sequences.
    # ------------------------------------------------------------------
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    q_offsets,
    q_lengths,
    s_scales,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    # kvid encodes (kid, vid) jointly so the 3D grid covers both K- and
    # V-side splits without nesting another program dim.
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)  # number of qo heads, needed for output strides

    # s_scale is 0 (first prefill chunk) or 1 (next prefill chunk)
    s_scale = tl.load(s_scales + bid)
    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)
    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    if s_offset == -1:
        return

    # Q/K addressing: ``kid * K_SPLIT_DIM`` picks this program's K-stripe
    # of the head's QK dimension.
    q_ptrs = (
        Q + q_offset * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K + q_offset * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    # V uses ``vid * V_SPLIT_DIM`` along the V-side dim.
    v_ptrs = (
        V + q_offset * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    # Output is laid out (k_dim_block, length, qo_heads, d) so each kid
    # writes into its own slab; later the slabs are summed to produce the
    # final (length, qo_heads, d) tensor.
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = (
        Out
        + kid * stride_o
        + q_offset * HEAD_DIM * H
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * H * HEAD_DIM + offs_v[None, :])
    )
    # State is (slots, heads, HEAD_DIM, HEAD_DIM); a (K_SPLIT_DIM, V_SPLIT_DIM)
    # tile is what this program owns. The outer stride between K rows is
    # HEAD_DIM (the V dim) which is the inner stride of the matrix.
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    # Main prefill loop: iterate BLOCK tokens at a time over the request.
    for n in range(0, q_length, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        if EVEN:
            # No bounds mask. b_offs / decays are pure functions of constexpr
            # BLOCK so Triton can constant-fold the exp computations.
            q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
            k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)
            v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
            b = BLOCK
            b_offs = b - 1 - offs_b
            decays = tl.exp(decay_scale * b_offs)
            inv_decays = 1 / decays
        else:
            # Tail block: ``b`` may be < BLOCK. Negative b_offs (past valid
            # length) get masked to 0 so they don't contribute to the
            # outer-product update.
            q = tl.load(q_ptrs + n * stride_q, mask=(n + offs_b)[:, None] < q_length, other=0.0).to(tl.float32)
            k = tl.trans(
                tl.load(
                    k_ptrs + n * stride_k,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                )
            ).to(tl.float32)
            v = tl.load(v_ptrs + n * stride_v, mask=(n + offs_b)[:, None] < q_length, other=0.0).to(tl.float32)
            b = min(BLOCK, q_length - n)
            b_offs = b - 1 - offs_b
            block_decays = tl.exp(decay_scale * b_offs)
            decays = tl.where(b_offs >= 0, block_decays, 0)
            inv_decays = tl.where(b_offs >= 0, 1 / block_decays, 0)

        # Decoupled rescaling (always used in prefill): pre-multiply Q by
        # inv_decays and K by decays so the bilinear form QK^T reproduces
        # the desired (i - j) decay matrix when masked to lower-triangular.
        q = q * inv_decays[:, None]
        k = k * decays[None, :]
        qk = tl.dot(q, k) * softmax_scale
        qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
        o = tl.dot(qk, v)

        # Cross-block term: prior state times current Q, decayed by the
        # number of tokens spanned by this block.
        block_decay = tl.exp(decay_scale * b)
        o = tl.dot(q, state) * block_decay * softmax_scale + o

        # State update for the next block.
        state = state * block_decay + tl.dot(k, v)

        # Stride between consecutive (length) rows in the output is
        # ``H * HEAD_DIM`` because the layout interleaves heads inside each
        # length slot.
        if EVEN:
            tl.store(out_ptrs + n * H * HEAD_DIM, o.to(Out.dtype.element_ty))
        else:
            tl.store(
                out_ptrs + n * H * HEAD_DIM,
                o.to(Out.dtype.element_ty),
                mask=(n + offs_b)[:, None] < q_length,
            )

    # Persist updated state for use by the next chunk / decode step.
    tl.store(s_ptrs, state.to(S.dtype.element_ty))


# used for speculative
@triton.jit
def seg_la_s_kernel(
    # ------------------------------------------------------------------
    # Speculative-decoding variant. The input ``Mask`` encodes a candidate
    # tree: each row gives the binary visibility of every other speculation
    # position from the perspective of one verification step. We use the row
    # sums as "positions in the candidate path" to compute the correct decay
    # exponent for every (query, key) pair.
    #
    # Unlike seg_la_p_kernel this is a *single-block* kernel — the entire
    # speculative window fits in one BLOCK. Therefore no ``n`` loop and no
    # state write-back: the persistent state is only consumed (the verifier
    # decides afterwards which step "wins" and updates state accordingly).
    # ------------------------------------------------------------------
    Q,
    K,
    V,
    S,
    Out,
    Mask,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    q_offsets,
    q_lengths,
    s_scales,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    # Same (kid, vid) joint indexing as prefill kernel.
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    # s_scale is 0 (first prefill chunk) or 1 (next prefill chunk)
    s_scale = tl.load(s_scales + bid)
    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)
    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    if s_offset == -1:
        return

    # Same pointer scheme as prefill kernel; see seg_la_p_kernel for layout
    # notes. No ``n`` offset — the entire candidate window is in one block.
    q_ptrs = (
        Q + q_offset * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K + q_offset * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V + q_offset * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = (
        Out
        + kid * stride_o
        + q_offset * HEAD_DIM * H
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * H * HEAD_DIM + offs_v[None, :])
    )
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    if EVEN:
        # Fast path: candidate length matches BLOCK exactly, so no row/col
        # bounds masking is needed on Q/K/V loads.
        q = tl.load(q_ptrs).to(tl.float32)
        k = tl.trans(tl.load(k_ptrs)).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        # Load the (BLOCK x BLOCK) speculative mask — typically a tree mask
        # where row i has 1s for the ancestors of candidate i.
        mask = tl.load(
            Mask + bid * BLOCK * BLOCK + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
        ).to(tl.int32)
        # Each row's sum is the candidate's depth in the tree (= its position
        # in the resulting sequence). max_pos is the longest path.
        positions = tl.sum(mask, 1) - 1
        max_pos = tl.max(positions)
        # b_offs gives, per row, the decay distance from the (virtual) end
        # of the block. Used identically to the prefill kernel.
        b_offs = max_pos - positions
    else:
        # Generic path: bounded mask reads. Note the mask stride is the
        # request's actual q_length here (not BLOCK), because each request
        # owns a square mask sized to its window.
        q = tl.load(q_ptrs, mask=offs_b[:, None] < q_length).to(tl.float32)
        k = tl.trans(tl.load(k_ptrs, mask=offs_b[:, None] < q_length)).to(tl.float32)
        v = tl.load(v_ptrs, mask=offs_b[:, None] < q_length).to(tl.float32)
        mask = tl.load(
            Mask + bid * q_length * q_length + tl.arange(0, BLOCK)[:, None] * q_length + tl.arange(0, BLOCK)[None, :],
            mask=(tl.arange(0, BLOCK)[:, None] < q_length) & (tl.arange(0, BLOCK)[None, :] < q_length),
        ).to(tl.int32)
        positions = tl.sum(mask, 1) - 1
        max_pos = tl.max(positions)
        b_offs = max_pos - positions

    # Same decoupled Q/K rescaling as prefill, then a *tree-causal* mask
    # multiply replaces the in-block triangular mask.
    decays = tl.exp(decay_scale * b_offs)
    inv_decays = 1 / decays

    q = q * inv_decays[:, None]
    k = k * decays[None, :]
    qk = tl.dot(q, k) * softmax_scale
    qk = qk * mask.to(tl.float32)
    o = tl.dot(qk, v)

    # Cross-state contribution decays by ``max_pos + 1`` because that's the
    # number of "virtual" steps the state has advanced past, plus one for the
    # current step.
    block_decay = tl.exp(decay_scale * (max_pos + 1))
    o = tl.dot(q, state) * block_decay * softmax_scale + o

    if EVEN:
        tl.store(out_ptrs, o.to(Out.dtype.element_ty))
    else:
        tl.store(out_ptrs, o.to(Out.dtype.element_ty), mask=offs_b[:, None] < q_length)


# used for decode
@triton.jit
def seg_la_d_kernel(
    # ------------------------------------------------------------------
    # Single-token decode kernel. Each request emits exactly one token, so
    # there is no in-block causal mask and no inner loop over tokens. The
    # state update is a single outer product, the output is a single
    # matvec along K_SPLIT_DIM, and ``stride_o`` is the K-split slab stride
    # (output layout (k_dim_block, batch, qo_heads, d)).
    # ------------------------------------------------------------------
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    # (kid, vid) joint indexing — same scheme as the prefill kernel.
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    # s_scale is 0 (first prefill chunk) or 1 (next prefill chunk)
    s_offset = tl.load(s_offsets + bid)
    if s_offset == -1:
        return

    decay_scale = -tl.load(decay_scales + hid)

    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    # Each pointer addresses a single token (no offs_b in the row dim
    # because BLOCK is implicitly 1). Q/K rows are length K_SPLIT_DIM, V
    # is length V_SPLIT_DIM.
    q_ptrs = Q + bid * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    k_ptrs = K + bid * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    v_ptrs = V + bid * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = Out + kid * stride_o + bid * H * HEAD_DIM + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # State tile this program owns: (K_SPLIT_DIM, V_SPLIT_DIM).
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs).to(tl.float32)

    k = tl.load(k_ptrs).to(tl.float32)
    v = tl.load(v_ptrs).to(tl.float32)
    # softmax_scale folded into Q early so we don't multiply the (K, V)
    # state update by it.
    q = tl.load(q_ptrs).to(tl.float32) * softmax_scale

    # state = state * exp(-decay) + k^T v (rank-1 outer product per program).
    state = state * tl.exp(decay_scale) + k[:, None] * v
    # o[v] = sum_k q[k] * state[k, v]
    o = tl.sum(q[:, None] * state, axis=0)

    tl.store(out_ptrs, o.to(Out.dtype.element_ty))
    tl.store(s_ptrs, state.to(S.dtype.element_ty))


# used for MTP with only spec-topk=1.
@triton.jit
def seg_la_mtp_kernel(
    # ------------------------------------------------------------------
    # Multi-Token-Prediction (MTP) decode kernel. Unlike the regular decode,
    # we predict ``step`` tokens per request in one launch and snapshot the
    # state after each step into ``CACHES`` so the downstream MTP head can
    # verify candidates against arbitrary prefixes.
    #
    # Only supports spec-topk == 1 (single chain). For tree-shaped MTP use
    # seg_la_s_kernel.
    # ------------------------------------------------------------------
    Q,
    K,
    V,
    S,
    CACHES,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_c,
    stride_o,
    s_offsets,
    decay_scales,
    step,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    s_offset = tl.load(s_offsets + bid)
    if s_offset == -1:
        return

    # decay_scale precomputed as exp(-rate); avoids one exp call per step
    # inside the hot loop below.
    decay_scale = tl.exp(-tl.load(decay_scales + hid))

    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    # Input layout: (length, qo_heads, d) but each request owns ``step``
    # consecutive length slots. bid * step jumps to the request's first slot.
    # (length, qo_heads, d)
    q_ptrs = Q + bid * step * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    k_ptrs = K + bid * step * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    v_ptrs = V + bid * step * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # Output layout same as decode but with stride ``H * HEAD_DIM`` between
    # consecutive predicted steps.
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = Out + kid * stride_o + bid * step * H * HEAD_DIM + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # Persistent state: one (HEAD_DIM, HEAD_DIM) matrix per (slot, head).
    # (bs, qo_heads, d, d)
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs).to(tl.float32)
    # Per-step state snapshots: shape (bs, step, qo_heads, d, d).
    # stride_c is the leading slot stride; H * HEAD_DIM * HEAD_DIM jumps to
    # the next step's slab within the slot.
    # (bs, step, kv_heads, d, d)
    c_ptrs = (
        CACHES
        + s_offset * stride_c
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )

    # Roll-out loop: ``step`` decode-like updates, each writing its own
    # output token and state snapshot. The state advances in place.
    for i in range(step):
        q = tl.load(q_ptrs).to(tl.float32) * softmax_scale
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)

        # decay_scale here is already exp(-rate), so this is exactly one
        # step of state attenuation per iteration.
        state = state * decay_scale + k[:, None] * v
        o = tl.sum(q[:, None] * state, axis=0)

        tl.store(out_ptrs, o.to(Out.dtype.element_ty))
        # Snapshot the post-update state for verification by the MTP head.
        tl.store(c_ptrs, state.to(CACHES.dtype.element_ty))
        # Advance pointers by one length slot (Q/K/V) and one snapshot slot
        # (Out / CACHES).
        q_ptrs += stride_q
        k_ptrs += stride_k
        v_ptrs += stride_v
        out_ptrs += H * HEAD_DIM
        c_ptrs += H * HEAD_DIM * HEAD_DIM


# (k_dim_block, length, qo_heads, d)
@triton.jit
def seg_la_sum_kernel(T, O, DIM: tl.constexpr, NUM_BLOCK: tl.constexpr):
    """Reduce K-dim partial outputs across the ``k_dim_block`` axis.

    The prefill / decode / MTP kernels write per-K-split partials to
    ``tmp`` shaped (k_dim_block, length, qo_heads, d). For large lengths
    where ``tmp.sum(0)`` is wasteful we launch this kernel: one program per
    length slot summing NUM_BLOCK partials of DIM=(qo_heads * head_dim)
    lanes into the final output O.
    """
    pid = tl.program_id(0)
    length = tl.num_programs(0)
    # Accumulate in fp32 across all K-partitions.
    x = tl.zeros((DIM,), dtype=tl.float32)
    for i in range(NUM_BLOCK):
        # Stride across blocks is ``length * DIM`` since the layout is
        # (k_dim_block, length, qo_heads * head_dim).
        x += tl.load(T + i * length * DIM + pid * DIM + tl.arange(0, DIM)).to(tl.float32)
    tl.store(O + pid * DIM + tl.arange(0, DIM), x)


def seg_la_fwd(q, k, v, s, decay_scales, meta, caches=None, softmax_scale=None):
    """Forward dispatcher for segmented linear attention.

    Chooses one of the four specialised kernels based on the batch shape and
    the presence of optional inputs:

    * ``MAX_LENGTH == 1`` (decode):  ``seg_la_d_kernel``.
    * ``MAX_LENGTH > 1`` and ``caches`` provided: ``seg_la_mtp_kernel`` —
      multi-token prediction roll-out with per-step state snapshots.
    * ``MAX_LENGTH > 1`` and ``meta.mask`` provided: ``seg_la_s_kernel`` —
      speculative decoding under a tree mask.
    * Otherwise: ``seg_la_p_kernel`` — vanilla chunked prefill.

    Args:
        q/k/v: Packed (sum_l, qo_heads, head_dim) tensors. Tokens for all
            requests are concatenated along the length axis.
        s: Persistent state pool of shape (num_slots, qo_heads, head_dim,
            head_dim) — the rolling state matrices for each request.
        decay_scales: Per-head decay rates (positive scalars; the kernels
            negate internally so larger means faster decay).
        meta: ``SegLaMeta`` packed descriptors.
        caches: For MTP only — preallocated tensor of shape
            (num_slots, step, qo_heads, head_dim, head_dim) to hold per-step
            state snapshots.
        softmax_scale: Defaults to ``1/sqrt(head_dim)`` if not provided.

    Returns:
        Output tensor of shape (sum_l, qo_heads, head_dim).
    """
    length, qo_heads, HEAD_DIM = q.shape
    _, kv_heads, _ = k.shape
    bs = meta.batch_size
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** (-0.5)

    # MAX_LENGTH is the per-request token count assuming uniform batch.
    # For decode this is 1; anything >1 hits the prefill/MTP/spec paths.
    # MAX_LENGTH = meta.max_q_length
    MAX_LENGTH = triton.cdiv(length, bs)

    assert qo_heads == kv_heads, "seg_la does NOT support GQA currently"

    if MAX_LENGTH > 1:
        # Prefill / spec / MTP path. K and V dims are tiled into
        # K_SPLIT_DIM/V_SPLIT_DIM chunks so the (K, V) state slice fits in
        # SRAM at the chosen num_stages.
        # prefill with partitioning q/k/v
        # BLOCK should <= 64 with decouple
        K_SPLIT_DIM = 32
        # Bigger V split for larger batches where occupancy is already high.
        V_SPLIT_DIM = 32 if bs <= 2 else 64

        num_warps = 2  # 2
        num_stages = 3  # 3

        k_dim_block = HEAD_DIM // K_SPLIT_DIM
        v_dim_block = HEAD_DIM // V_SPLIT_DIM
        # Per-K-split temporary outputs to be summed in seg_la_sum_kernel.
        tmp = torch.empty((k_dim_block, length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype)
        # Grid covers (request, head, K-split * V-split). The kernel
        # decomposes pid 2 into (kid, vid) internally.
        grid = (bs, kv_heads, k_dim_block * v_dim_block)

        if caches is not None:
            # MTP roll-out: ``step`` tokens per request, snapshot state into
            # caches at each step.
            # mtp
            EVEN = False
            BLOCK = 32
            step = length // bs

            seg_la_mtp_kernel[grid](
                q,
                k,
                v,
                s,
                caches,
                tmp,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                caches.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                decay_scales,
                step,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        elif meta.mask is not None:
            # Speculative verification: BLOCK is rounded up to a multiple of
            # 16 (warp width on tensor cores). EVEN is True when the candidate
            # window matches BLOCK exactly so we can skip mask building.
            # spec
            ms = meta.mask.size(-1)
            BLOCK = (ms + 15) // 16 * 16
            EVEN = BLOCK == ms

            seg_la_s_kernel[grid](
                q,
                k,
                v,
                s,
                tmp,
                meta.mask,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                meta.q_offsets,
                meta.q_lengths,
                meta.s_scales,
                decay_scales,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                BLOCK=BLOCK,
                EVEN=EVEN,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        else:
            # Generic chunked prefill. EVEN requires bs==1 and the per-block
            # length divides BLOCK exactly; in batched mode we conservatively
            # disable it because each request can hit a tail block.
            # prefill
            BLOCK = 32
            EVEN = MAX_LENGTH % BLOCK == 0 if bs == 1 else False

            seg_la_p_kernel[grid](
                q,
                k,
                v,
                s,
                tmp,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                meta.q_offsets,
                meta.q_lengths,
                meta.s_scales,
                decay_scales,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                BLOCK=BLOCK,
                EVEN=EVEN,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        # Reduce across K-splits if we tiled the K dim. For short total
        # lengths a torch sum is cheaper than launching the reduce kernel.
        if k_dim_block > 1:
            if length < 2048:
                o = tmp.sum(0)
            else:
                o = torch.empty((length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype)
                seg_la_sum_kernel[(length,)](
                    tmp,
                    o,
                    DIM=qo_heads * HEAD_DIM,
                    NUM_BLOCK=k_dim_block,
                    num_warps=2,
                    num_stages=3,
                )
        else:
            # Single K-split: tmp[0] already holds the result.
            o = tmp[0]

    else:
        # Decode path: a wider K-split is OK since each program does very
        # little work, and we want to keep N small enough for cache locality.
        # decode with partitioning q/k/v
        if bs <= 128:
            K_SPLIT_DIM = 128  # 128
            V_SPLIT_DIM = 32  # 32
            num_warps = 2  # 2
            num_stages = 2  # 3
        else:
            # Larger batches benefit from a bigger V split (fewer programs
            # per (request, head) but more lanes per program).
            K_SPLIT_DIM = 128  # 128
            V_SPLIT_DIM = 64  # 32
            num_warps = 2  # 2
            num_stages = 3  # 3
        k_dim_block = HEAD_DIM // K_SPLIT_DIM
        v_dim_block = HEAD_DIM // V_SPLIT_DIM
        tmp = torch.empty((k_dim_block, length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype)
        grid = (bs, kv_heads, k_dim_block * v_dim_block)

        seg_la_d_kernel[grid](
            q,
            k,
            v,
            s,
            tmp,
            softmax_scale,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            s.stride(0),
            tmp.stride(0),
            meta.s_offsets,
            decay_scales,
            HEAD_DIM=HEAD_DIM,
            K_SPLIT_DIM=K_SPLIT_DIM,
            V_SPLIT_DIM=V_SPLIT_DIM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        # K-split reduction (torch sum is sufficient for decode shapes).
        if k_dim_block > 1:
            o = tmp.sum(0)
        else:
            o = tmp[0]

    # if fallback:
    #     # prefill/decode with partitioning v only
    #     o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    #     if MAX_LENGTH == 1:
    #         # decode
    #         BLOCK = 1
    #         EVEN = False
    #         SPLIT_DIM = 32
    #         num_warps = 8
    #         num_stages = 2
    #         num_dim_block = HEAD_DIM // SPLIT_DIM
    #         grid = (batch, kv_heads, num_dim_block)
    #     else:
    #         # prefill
    #         if decouple:
    #             BLOCK = 64
    #             SPLIT_DIM = 16
    #         else:
    #             BLOCK = HEAD_DIM
    #             SPLIT_DIM = 32
    #         # EVEN = all([x % BLOCK == 0 for x in meta.qls])
    #         EVEN = False
    #         num_warps = 8
    #         num_stages = 2
    #         # prop = torch.cuda.get_device_properties(q.device.index)
    #         # arch = prop.major * 10 + prop.minor
    #         # if arch not in (80, 90):
    #         #     num_stages = 1

    #         num_dim_block = HEAD_DIM // SPLIT_DIM
    #         grid = (batch, kv_heads, num_dim_block)

    #     seg_la_kernel[grid](
    #         q,
    #         k,
    #         v,
    #         s,
    #         o,
    #         softmax_scale,
    #         q.stride(0),
    #         k.stride(0),
    #         v.stride(0),
    #         s.stride(0),
    #         o.stride(0),
    #         meta.s_offsets,
    #         meta.q_offsets,
    #         meta.q_lengths,
    #         meta.s_scales,
    #         decay_scales,
    #         HEAD_DIM=HEAD_DIM,
    #         SPLIT_DIM=SPLIT_DIM,
    #         BLOCK=BLOCK,
    #         EVEN=EVEN,
    #         DECOUPLE=decouple,
    #         num_warps=num_warps,
    #         num_stages=num_stages
    #     )
    return o
