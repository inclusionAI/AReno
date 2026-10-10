"""Graph-safe NF4 expert inference, with live, unmerged LoRA adapters.

Routes remain on the GPU. Decode only the weight tile used by each GEMM;
neither dense expert banks nor CPU token counts are materialized. Training
continues through the differentiable grouped path in ``accel.nf4``.
"""

import torch
import triton
import triton.language as tl

from areno.accel.kernels.fused_moe import _align_block_size, _apply_gated_activation, _invoke_matmul, _sum_reduce


@triton.jit
def _nf4_routed_matmul(
    X,
    Packed,
    Codes,
    Absmax,
    Offset,
    Codebook,
    ScaleCodebook,
    Sorted,
    Experts,
    Used,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    ROUTES: tl.constexpr,
    TOP_K: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m, n = tl.program_id(0), tl.program_id(1)
    if m * BM >= tl.load(Used):
        return
    expert = tl.load(Experts + m).to(tl.int64)
    rows = tl.load(Sorted + m * BM + tl.arange(0, BM)).to(tl.int64)
    cols = n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    if expert >= 0:
        for start in range(tl.cdiv(K, BK)):
            ks = start * BK + tl.arange(0, BK)
            x = tl.load(
                X + rows[:, None] // TOP_K * SX0 + ks[None, :] * SX1,
                (rows[:, None] < ROUTES) & (ks[None, :] < K),
                other=0,
            )
            indices = expert * N * K + cols[None, :] * K + ks[:, None]
            mask = (cols[None, :] < N) & (ks[:, None] < K)
            byte = tl.load(Packed + indices // 2, mask, other=0).to(tl.int32)
            code = tl.where(indices % 2 == 0, byte >> 4, byte & 15)
            scale_code = tl.load(Codes + indices // 64, mask, other=0).to(tl.int32)
            scale = tl.load(ScaleCodebook + scale_code) * tl.load(Absmax + indices // 16384, mask, other=0)
            scale = scale + tl.load(Offset)
            weight = tl.load(Codebook + code) * scale
            weight = tl.where(mask, weight, 0).to(x.dtype)
            acc += tl.dot(x, weight)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < ROUTES) & (cols[None, :] < N))


def _project(x, weight, routes, top_k, config):
    sorted_ids, expert_ids, used, ids, _ = routes
    _, n, k = weight.shape
    out = torch.empty((*ids.shape, n), device=x.device, dtype=x.dtype)
    _nf4_routed_matmul[(triton.cdiv(sorted_ids.numel(), config.block_size_m), triton.cdiv(n, config.block_size_n))](
        x,
        weight.packed,
        weight.scale_codes,
        weight.scale_absmax,
        weight.scale_offset,
        weight.codebook,
        weight.scale_codebook,
        sorted_ids,
        expert_ids,
        used,
        out,
        n,
        k,
        ids.numel(),
        top_k,
        x.stride(0),
        x.stride(1),
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


def _adapter(x, slot, routes, top_k, config):
    sorted_ids, expert_ids, used, ids, weights = routes
    low = torch.empty((*ids.shape, slot.rank), device=x.device, dtype=x.dtype)
    out = torch.empty((*ids.shape, slot.out_features), device=x.device, dtype=x.dtype)
    for source, matrix, dest, copies in (
        (x, slot.lora_A, low, top_k),
        (low.view(-1, slot.rank), slot.lora_B, out, 1),
    ):
        _invoke_matmul(
            source,
            matrix,
            dest,
            weights,
            ids,
            sorted_ids,
            expert_ids,
            used,
            mul_routed_weight=False,
            top_k=copies,
            config=config,
        )
    return out * slot.scale


@torch.no_grad()
def nf4_experts(hidden, fc1, fc2, ids, weights, slots, config):
    """Inference only; routes and adapter values may change between replays."""
    if hidden.dtype not in (torch.bfloat16, torch.float16) or not hidden.is_cuda:
        raise ValueError("NF4 expert inference requires CUDA BF16/FP16 activations")
    hidden = hidden.contiguous()
    ids, weights = ids.int().contiguous(), weights.contiguous()
    sorted_ids, expert_ids, used = _align_block_size(ids, config.block_size_m, config.num_experts)
    routes = sorted_ids, expert_ids, used, ids, weights
    gate_up = _project(hidden, fc1, routes, ids.shape[1], config)
    width = fc1.shape[1] // 2
    if "linear_fc1" in slots:
        gate_up.add_(_adapter(hidden, slots["linear_fc1"], routes, ids.shape[1], config))
    if "gate_proj" in slots:
        gate_up[..., :width].add_(_adapter(hidden, slots["gate_proj"], routes, ids.shape[1], config))
    if "up_proj" in slots:
        gate_up[..., width:].add_(_adapter(hidden, slots["up_proj"], routes, ids.shape[1], config))
    activated = torch.empty((ids.numel(), width), device=hidden.device, dtype=hidden.dtype)
    _apply_gated_activation(gate_up.view(-1, 2 * width), activated, activation="silu", swiglu_limit=config.swiglu_limit)
    # Match training: round the weighted activation before the down projection.
    activated.mul_(weights.reshape(-1, 1).to(hidden.dtype))
    expert_out = _project(activated, fc2, routes, 1, config)
    if "linear_fc2" in slots:
        expert_out.add_(_adapter(activated, slots["linear_fc2"], routes, 1, config))
    if "down_proj" in slots:
        expert_out.add_(_adapter(activated, slots["down_proj"], routes, 1, config))
    out = torch.empty_like(hidden)
    _sum_reduce(expert_out, out, 1.0)
    return out * config.routed_scaling_factor
