"""Graph-safe routed expert projections with live, unmerged LoRA adapters.

Routes remain on the GPU. Decode only the weight tile used by each GEMM;
neither dense expert banks nor CPU token counts are needed in forward.
"""

import torch
import triton
import triton.language as tl

from areno.accel import areno_silu_and_mul
from areno.accel.kernels.fused_moe import _align_block_size, _invoke_matmul, _sum_reduce


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
    TRANSPOSE_WEIGHT: tl.constexpr,
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
            if TRANSPOSE_WEIGHT:
                indices = expert * N * K + ks[:, None] * N + cols[None, :]
            else:
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


def _project(x, weight, routes, top_k, config, *, transpose_weight=False):
    sorted_ids, expert_ids, used, ids, _ = routes
    _, n, k = weight.shape
    if transpose_weight:
        n, k = k, n
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
        transpose_weight,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _adapter_weight_grad(
    X,
    DY,
    ORDER,
    BOUNDS,
    DW,
    N: tl.constexpr,
    K: tl.constexpr,
    COPIES: tl.constexpr,
    SY0: tl.constexpr,
    SY1: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    expert = tl.program_id(0)
    out_features = tl.program_id(1) * BN + tl.arange(0, BN)
    in_features = tl.program_id(2) * BK + tl.arange(0, BK)
    first = tl.load(BOUNDS + expert)
    end = tl.load(BOUNDS + expert + 1)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(first, end, BT):
        positions = start + tl.arange(0, BT)
        rows = tl.load(ORDER + positions, positions < end, other=0)
        x = tl.load(
            X + (rows[:, None] // COPIES) * K + in_features[None, :],
            (positions[:, None] < end) & (in_features[None, :] < K),
            other=0,
        )
        dy = tl.load(
            DY + rows[None, :] * SY0 + out_features[:, None] * SY1,
            (positions[None, :] < end) & (out_features[:, None] < N),
            other=0,
        )
        acc = tl.dot(dy, x, acc)
    tl.store(
        DW + (expert * N + out_features[:, None]) * K + in_features[None, :],
        acc,
        (out_features[:, None] < N) & (in_features[None, :] < K),
    )


def _adapter_backward(grad, x, weight, routes, copies, config, need_x, need_weight):
    sorted_ids, expert_ids, used, ids, weights = routes
    num_experts, n, k = weight.shape
    dx = dw = None
    if need_x:
        routed_dx = x.new_empty((*ids.shape, k))
        _invoke_matmul(
            grad,
            weight.transpose(1, 2),
            routed_dx,
            weights,
            ids,
            sorted_ids,
            expert_ids,
            used,
            mul_routed_weight=False,
            top_k=1,
            config=config,
        )
        dx = _sum_input_grad(routed_dx, copies)
    if need_weight:
        # Stable route order matches the grouped GEMM reduction order. Keep
        # expert offsets on the GPU instead of launching one GEMM per expert.
        expert_order, route_order = torch.sort(ids.flatten(), stable=True)
        bounds = torch.searchsorted(expert_order, torch.arange(num_experts + 1, device=ids.device, dtype=ids.dtype))
        dw = torch.empty_like(weight)
        block_n = 16 if n <= 16 else 32
        block_k = 16 if k <= 16 else 32
        _adapter_weight_grad[(num_experts, triton.cdiv(n, block_n), triton.cdiv(k, block_k))](
            x,
            grad,
            route_order,
            bounds,
            dw,
            n,
            k,
            copies,
            *grad.stride(),
            block_n,
            block_k,
            32,
        )
    return dx, dw


def _sum_input_grad(routed_dx, copies):
    if copies == 1:
        return routed_dx.flatten(0, 1)
    return routed_dx.sum(1, dtype=torch.float32).to(routed_dx.dtype)


class _RoutedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, routes, copies, config, adapter_projection):
        sorted_ids, expert_ids, used, ids, weights = routes
        ctx.copies = copies
        ctx.config = config
        ctx.adapter_projection = adapter_projection
        if isinstance(weight, torch.Tensor):
            ctx.quantized = None
            if adapter_projection:
                ctx.save_for_backward(x, weight, sorted_ids, expert_ids, used, ids, weights)
            else:
                ctx.save_for_backward(x, ids, weight)
            out = x.new_empty((*ids.shape, weight.shape[1]))
            _invoke_matmul(
                x,
                weight,
                out,
                weights,
                ids,
                sorted_ids,
                expert_ids,
                used,
                mul_routed_weight=False,
                top_k=copies,
                config=config,
            )
            return out
        ctx.quantized = weight
        ctx.save_for_backward(sorted_ids, expert_ids, used, ids)
        return _project(x, weight, routes, copies, config)

    @staticmethod
    def backward(ctx, grad):
        if ctx.quantized is not None:
            if not ctx.needs_input_grad[0]:
                return None, None, None, None, None, None
            sorted_ids, experts, used, ids = ctx.saved_tensors
            grad = grad.reshape(ids.numel(), ctx.quantized.shape[1])
            dx = _project(
                grad, ctx.quantized, (sorted_ids, experts, used, ids, None), 1, ctx.config, transpose_weight=True
            )
            return _sum_input_grad(dx, ctx.copies), None, None, None, None, None

        if ctx.adapter_projection:
            x, weight, *routes = ctx.saved_tensors
            grad = grad.reshape(-1, weight.shape[1])
            dx, dw = _adapter_backward(grad, x, weight, routes, ctx.copies, ctx.config, *ctx.needs_input_grad[:2])
            return dx, dw, None, None, None, None

        from areno.accel._extension import extension

        x, ids, weight = ctx.saved_tensors
        ids = ids.reshape(-1).long()
        order = torch.argsort(ids, stable=True)
        counts = torch.bincount(ids, minlength=weight.shape[0])
        rows = order // ctx.copies
        dy = grad.reshape(-1, weight.shape[1]).index_select(0, order)
        need_dx, need_dw = ctx.needs_input_grad[:2]
        dx, dw = extension(x.device).areno_grouped_linear_backward_counts(
            dy, x.index_select(0, rows), weight, counts, need_dx, need_dw
        )
        if need_dx:
            # Undo the expert sort, then accumulate the copies of each input row.
            routed_dx = torch.empty_like(dx)
            routed_dx.index_copy_(0, order, dx)
            dx = routed_dx.view(len(x), ctx.copies, -1).sum(1, dtype=torch.float32).to(x.dtype)
        return dx if need_dx else None, dw if need_dw else None, None, None, None, None


def _adapter(x, slot, routes, copies, config):
    low = _RoutedLinear.apply(x, slot.lora_A, routes, copies, config, True)
    return _RoutedLinear.apply(low.reshape(-1, slot.rank), slot.lora_B, routes, 1, config, True) * slot.scale


class _SumRoutes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        ctx.shape = values.shape
        out = values.new_empty(values.shape[0], values.shape[-1])
        _sum_reduce(values, out, 1.0)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad.unsqueeze(1).expand(ctx.shape)


def routed_limited_swiglu_experts(hidden, fc1, fc2, bias1, bias2, ids, weights, config, alpha, limit):
    """Differentiate the same biased projections and route sum used by fused GPT-OSS decode."""
    from areno.accel.grouped_bias import limited_swiglu

    ids, weights = ids.int().contiguous(), weights.contiguous()
    sorted_ids, expert_ids, used = _align_block_size(ids, config.block_size_m, config.num_experts)
    routes = sorted_ids, expert_ids, used, ids, weights
    gate_up = _RoutedLinear.apply(hidden.contiguous(), fc1, routes, ids.shape[1], config, False)
    # Fused decode rounds the matmul to the activation dtype before adding the bias.
    gate_up = gate_up + bias1[ids]
    activated = limited_swiglu(gate_up.flatten(0, 1), alpha, limit)
    down = _RoutedLinear.apply(activated, fc2, routes, 1, config, False)
    weighted = (down + bias2[ids]) * weights.unsqueeze(-1).to(down.dtype)
    return _SumRoutes.apply(weighted) * config.routed_scaling_factor


def routed_adapter_experts(hidden, fc1, fc2, ids, weights, slots, config):
    """Apply base and adapter matrices in identical route order during training and decode."""
    if hidden.dtype not in (torch.bfloat16, torch.float16) or not hidden.is_cuda:
        raise ValueError("routed adapter experts require CUDA BF16/FP16 activations")
    hidden = hidden.contiguous()
    ids, weights = ids.int().contiguous(), weights.contiguous()
    sorted_ids, expert_ids, used = _align_block_size(ids, config.block_size_m, config.num_experts)
    routes = sorted_ids, expert_ids, used, ids, weights
    gate_up = _RoutedLinear.apply(hidden, fc1, routes, ids.shape[1], config, False)
    if "gate_proj" in slots or "up_proj" in slots:
        gate, up = gate_up.chunk(2, dim=-1)
        if "gate_proj" in slots:
            gate = gate + _adapter(hidden, slots["gate_proj"], routes, ids.shape[1], config)
        if "up_proj" in slots:
            up = up + _adapter(hidden, slots["up_proj"], routes, ids.shape[1], config)
        gate_up = torch.cat((gate, up), dim=-1)
    activated = areno_silu_and_mul(gate_up.flatten(0, 1))
    activated = activated * weights.reshape(-1, 1).to(hidden.dtype)
    expert_out = _RoutedLinear.apply(activated, fc2, routes, 1, config, False)
    if "down_proj" in slots:
        expert_out = expert_out + _adapter(activated, slots["down_proj"], routes, 1, config)
    return _SumRoutes.apply(expert_out) * config.routed_scaling_factor


@torch.no_grad()
def nf4_experts(hidden, fc1, fc2, ids, weights, slots, config):
    return routed_adapter_experts(hidden, fc1, fc2, ids, weights, slots, config)
