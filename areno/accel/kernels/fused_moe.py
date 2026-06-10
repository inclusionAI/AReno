"""Fused Mixture-of-Experts (MoE) expert evaluation kernels.

This file implements the Triton-backed core of a token-dispatched MoE layer:

1. ``_align_block_size`` (delegated to ``areno_moe_align``) reorders the flat
   list of ``(token, expert)`` pairs produced by the router so that all tokens
   routed to the same expert are contiguous, padded out to multiples of
   ``BLOCK_SIZE_M``. The output also includes a per-block expert id, so each
   matmul tile knows which expert weight slice to read.

2. ``_fused_moe_matmul_kernel`` performs a grouped GEMM:
   ``C[token, n] = sum_k A[token, k] * B[expert_of_token, k, n]``
   over the sorted-and-padded token list. Padding slots (``offs_token >=
   num_valid_tokens``) and entirely-skipped blocks (``expert_id == -1``) are
   handled with masks / a fast zero-write path.

3. ``_moe_sum_reduce_kernel`` collapses the per-(token, topk) intermediate
   outputs back into a single tensor per token, applying the routed scaling
   factor in fp32.

The fused-experts pipeline runs gate-MLP -> SiLU+mul (external CUDA op) ->
down-MLP -> weighted sum, sharing one preallocated cache buffer across both
matmuls.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from areno.accel import areno_moe_align
from areno.accel import areno_gelu_tanh_and_mul, areno_silu_and_mul


@dataclass(slots=True)
class FusedMoeConfig:
    """Static configuration for one fused MoE layer.

    Attributes:
        num_experts: Total number of experts in this layer.
        hidden_size: Model hidden dimension (size of the per-token MoE input
            and the final per-token output).
        intermediate_size: Width of the MLP expansion inside each expert,
            i.e. the column count of ``w1`` (before the SiLU+mul halving).
        top_k: Number of experts each token is routed to.
        routed_scaling_factor: Multiplier applied during the top-k sum-reduce
            (DeepSeek-style routed-expert rescaling).
        block_size_m: M tile size of the grouped matmul. Tokens are padded to
            multiples of this so each tile sees one expert exclusively.
        block_size_n: N tile size (output features per program).
        block_size_k: K tile size (reduction inner dim per iteration).
        group_size_m: L2-friendly M-supergroup factor; controls the
            ``pid_m / pid_n`` swizzle inside the matmul kernel.
    """
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    routed_scaling_factor: float = 1.0
    block_size_m: int = 16
    block_size_n: int = 64
    block_size_k: int = 64
    group_size_m: int = 8


def is_available() -> bool:
    """Whether the fused MoE kernels are usable in this build (stub returns True)."""
    return True


def _align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort flat (token, expert) pairs by expert and pad to ``block_size``.

    Wraps the external ``areno_moe_align`` CUDA op. Returns:
        sorted_ids: Length ``max_num_tokens_padded`` array where every
            ``block_size``-chunk contains tokens routed to a single expert,
            padded with ``num_valid_tokens`` (a sentinel out-of-range index)
            when a chunk is short.
        expert_ids: One expert id per M-block (or -1 for empty blocks).
        num_tokens_post_pad: Scalar total length actually used.

    The padding rationale: each Triton matmul program processes exactly
    ``BLOCK_SIZE_M`` rows; aligning to that boundary lets us assign one
    expert weight slice per program with no branching mid-tile.
    """
    # Worst-case padding: every expert boundary can waste up to block_size-1
    # rows, plus a guard at each end. This bounds the output buffer up-front.
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    # For small buffers the C++ op handles padding internally; for large ones
    # we pre-fill with the sentinel to skip a memset inside the op.
    pad_sorted_token_ids = sorted_ids.shape[0] <= 4096
    if not pad_sorted_token_ids:
        sorted_ids.fill_(topk_ids.numel())
    areno_moe_align(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


@triton.jit
def _write_zeros_to_output(
    # Helper called when a whole M-block has no assigned expert (expert_id == -1).
    # We still need to write a deterministic value into the C tile so that the
    # downstream reduce step (which always reads all top-k slots) sees zero
    # contribution. Faster than launching a separate memset.
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    n_cols,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Same C-tile addressing scheme as the main matmul kernel: stride_cm is
    # the per-token stride (along the M axis), stride_cn the per-column.
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < n_cols)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _fused_moe_matmul_kernel(
    # Grouped batched matmul over a token list that has already been sorted
    # by expert id (see _align_block_size). Each program computes one
    # (BLOCK_SIZE_M x BLOCK_SIZE_N) tile of the output for a single expert.
    #
    # Layout of A: (num_tokens, K) — but we index it via sorted_token_ids,
    # which contains routed token ids divided by top_k (since each source
    # token appears top_k times in the sorted list, all pointing back to the
    # same A row).
    # Layout of B: (num_experts, K, N) — stride_be selects the expert slab,
    # stride_bk/stride_bn span the inner GEMM dims.
    # Layout of C: (num_tokens_total, top_k, N) flattened to (M', N); we
    # write back via offs_token which is already a flat token-slot index.
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_k: tl.constexpr,
):
    # Linear program id is decomposed into a (pid_m, pid_n) tile coordinate
    # using a GROUP_SIZE_M swizzle. This reorders programs so that adjacent
    # pids tend to hit the same B columns first, improving L2 reuse on N.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Early-out for M-blocks past the actually-used (post-pad) tail. The host
    # over-allocates conservatively; this check skips dead work.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # offs_token_id are positions in the sorted list. Loading sorted_token_ids
    # at those positions yields the (top_k * source_token) ids, with sentinel
    # values >= num_valid_tokens for padding slots — these get masked off.
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    # Each M-block was guaranteed by the aligner to be assigned to a single
    # expert. -1 means the block is entirely padding; fast-path to a zero write.
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        _write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
        return

    # offs_bn wraps with % N so out-of-range columns reuse valid pointers.
    # Combined with c_mask below, masked stores keep the result correct.
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # A pointers: offs_token // top_k recovers the original token index so all
    # top_k copies of the same source token share the same hidden_state row.
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    # B pointers: select the expert slab (off_expert * stride_be) then index
    # (k, n) within it. stride_bk and stride_bn are the inner-matrix strides.
    b_ptrs = b_ptr + off_expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Accumulate in fp32 for accuracy; the dot product is bf16/fp16 inputs.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Fast path when K divides evenly: no K-dim masking on B is needed,
        # and A only needs the row mask. The else-branch handles the tail
        # block where ``offs_k < K - k * BLOCK_SIZE_K`` clamps overshoot.
        if even_k:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            k_mask = offs_k < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Optionally fold the per-token routing weight into the accumulator. This
    # is enabled for the *second* matmul (down-proj) so the sum-reduce step
    # can simply add up top_k pre-weighted contributions.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc *= moe_weight[:, None]
    acc = acc.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _moe_sum_reduce_kernel(
    # Reduce over the top_k axis of (token_num, topk_num, hidden_dim) into
    # (token_num, hidden_dim), applying routed_scaling_factor.
    # Grid layout: program_id(0) = token block, program_id(1) = dim block.
    # In practice the wrapper picks BLOCK_M=1 (one token per program) and
    # BLOCK_DIM=2048 (one hidden-dim slab per program).
    input_ptr,
    input_stride_0,
    input_stride_1,
    output_ptr,
    output_stride_0,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)
    # Spatial offsets for the (token, hidden_dim) tile this program owns.
    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim
    # input layout is (token, topk, hidden). Stride 0 advances tokens,
    # stride 1 advances within the topk axis (the dimension we're reducing).
    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
    # Accumulate in fp32 then downcast on store for numerical stability.
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for i in tl.range(0, topk_num):
        tile = tl.load(base_ptrs + i * input_stride_1, mask=mask_token[:, None] & mask_dim[None, :], other=0.0)
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor
    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(store_ptrs, accumulator.to(input_ptr.dtype.element_ty), mask=mask_token[:, None] & mask_dim[None, :])


def _invoke_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor, *, mul_routed_weight: bool, top_k: int, config: FusedMoeConfig) -> None:
    """Launch ``_fused_moe_matmul_kernel`` over the sorted token list.

    Grid size is computed from the padded M length (so every assignable block
    has a program) and the output N. ``compute_type`` follows the input dtype
    so accumulators downcast correctly on store. ``even_k`` selects the fast
    unmasked K-load path when N divides BLOCK_SIZE_K evenly.
    """
    grid = lambda meta: (
        triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"]) * triton.cdiv(b.shape[1], meta["BLOCK_SIZE_N"]),
    )
    compute_type = tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float16
    _fused_moe_matmul_kernel[grid](
        a,
        b,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        b.shape[1],
        b.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(1),
        c.stride(2),
        BLOCK_SIZE_M=config.block_size_m,
        BLOCK_SIZE_N=config.block_size_n,
        BLOCK_SIZE_K=config.block_size_k,
        GROUP_SIZE_M=config.group_size_m,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_k=b.shape[2] % config.block_size_k == 0,
        num_warps=4,
    )


def _sum_reduce(x: torch.Tensor, out: torch.Tensor, routed_scaling_factor: float) -> None:
    """Reduce ``x`` (token, topk, hidden) along the topk axis into ``out``.

    Grid: (token_num, hidden_dim // 2048). One program owns one token and a
    2048-wide hidden-dim slab, scanning topk_num sequentially in-program.
    """
    token_num, topk_num, hidden_dim = x.shape
    _moe_sum_reduce_kernel[(triton.cdiv(token_num, 1), triton.cdiv(hidden_dim, 2048))](
        x,
        x.stride(0),
        x.stride(1),
        out,
        out.stride(0),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=1,
        BLOCK_DIM=2048,
        num_warps=16,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: FusedMoeConfig,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """End-to-end fused MoE forward.

    Pipeline:
        1. Sort tokens by expert with ``_align_block_size``.
        2. matmul1 (gate-up): ``hidden_states @ w1[expert]`` produces a
           ``(num_tokens, top_k, 2*intermediate)`` tile written through one
           shared cache buffer.
        3. gated activation + element-wise multiply collapses
           the gate/up pair into ``(N*top_k, intermediate)``.
        4. matmul2 (down): multiply by ``w2[expert]`` with the routed weight
           folded in (``mul_routed_weight=True``, ``top_k=1`` because tokens
           are no longer duplicated post sum-reduce input).
        5. ``_sum_reduce`` collapses the per-(token, topk) outputs into the
           final ``(num_tokens, hidden)`` result scaled by
           ``routed_scaling_factor``.

    Both intermediate matmul outputs share one preallocated ``cache`` buffer,
    sliced into two views to save memory. ``intermediate2`` is the only
    separately allocated buffer because its width (``intermediate / 2``)
    differs from the matmul1 output.
    """
    if not is_available():
        raise RuntimeError("ARENO Nano MoE primitives are unavailable")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"fused_experts requires fp16/bf16 hidden_states, got {hidden_states.dtype}")
    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.int().contiguous()
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(topk_ids, config.block_size_m, config.num_experts)

    # Single backing buffer reused across both matmul outputs; w1 and w2 may
    # have different output widths so we size to the max and slice as views.
    max_intermediate = max(w1.shape[1], w2.shape[1])
    cache = torch.empty(num_tokens * top_k * max_intermediate, device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate1 = cache[: num_tokens * top_k * w1.shape[1]].view(num_tokens, top_k, w1.shape[1])
    intermediate2 = torch.empty((num_tokens * top_k, w1.shape[1] // 2), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate3 = cache[: num_tokens * top_k * w2.shape[1]].view(num_tokens, top_k, w2.shape[1])

    # Gate-up projection (no routed weight applied here — that happens in matmul2).
    _invoke_matmul(hidden_states, w1, intermediate1, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight=False, top_k=top_k, config=config)
    # Gated activation: out[:, i] = act(in[:, i]) * in[:, intermediate + i].
    if activation == "silu":
        areno_silu_and_mul(intermediate1.view(-1, w1.shape[1]), intermediate2)
    elif activation == "gelu_tanh":
        areno_gelu_tanh_and_mul(intermediate1.view(-1, w1.shape[1]), intermediate2)
    else:
        raise ValueError(f"unsupported fused MoE activation {activation!r}")
    # Down projection with routed weight folded in. top_k=1 because each
    # source token has already been expanded across the M axis.
    _invoke_matmul(intermediate2, w2, intermediate3, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight=True, top_k=1, config=config)

    out = torch.empty_like(hidden_states)
    _sum_reduce(intermediate3, out, config.routed_scaling_factor)
    return out
