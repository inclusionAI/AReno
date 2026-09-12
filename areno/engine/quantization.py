"""FP8 (E4M3) weight quantization for the decode path.

The quantize/dequant math is dtype-agnostic so it runs on CPU (unit tests) as
well as on GPU. :func:`mark_fp8_weight` attaches the quantized payload to a
parallel-linear weight and :func:`quantize_infer_weights_fp8` walks a model at
decode-session start; the FP8 linear backends live in ``areno.accel`` and are
selected by device capability.
"""

from __future__ import annotations

import torch
from torch import nn

# E4M3 has 3 exponent bits + 4 mantissa bits + sign. Max finite magnitude is
# 448.0; magnitude range is [2^-9, 448]. We clamp to the representable range.
_FP8_E4M3_MAX = 448.0


def compute_fp8_scale(weight: torch.Tensor, group_size: int = -1) -> torch.Tensor:
    """Compute the FP8 scale for `weight`, per-tensor or per-group.

    ``group_size <= 0`` (default) uses a single per-tensor scale. Otherwise the
    weight's last dim is split into groups of ``group_size`` and a scale is
    computed per group. Scale is chosen so the max magnitude in the group maps
    to ``_FP8_E4M3_MAX``; a zero group falls back to 1.0 to avoid inf/nan.
    """
    w = weight.detach().float()
    if group_size is None or group_size <= 0:
        amax = w.abs().amax()
        scale = amax / _FP8_E4M3_MAX
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return scale
    n = w.numel()
    g = int(group_size)
    if n % g != 0:
        raise ValueError(f"weight numel {n} must be divisible by group_size {g}")
    flat = w.reshape(-1, g)
    amax = flat.abs().amax(dim=1, keepdim=True)
    scale = amax / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.reshape(-1)


def quantize_to_fp8(weight: torch.Tensor, group_size: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `weight` (bf16/fp32) to FP8 grid; returns (fp8, scale).

    ``fp8`` is returned in ``torch.float8_e4m3fn`` when this torch build supports
    it; otherwise it is returned as a float tensor already snapped to the FP8
    grid (so the math is testable on CPU). ``scale`` is float32, shaped per the
    group layout (scalar for per-tensor, ``(n/g,)`` for per-group).
    """
    scale = compute_fp8_scale(weight, group_size)
    w = weight.detach().to(torch.float32)
    if group_size is None or group_size <= 0:
        snapped = torch.clamp((w / scale).round(), -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    else:
        low = w.reshape(-1, int(group_size))
        scale2 = scale.view(-1, 1)
        snapped = torch.clamp((low / scale2).round(), -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        snapped = snapped.reshape_as(w)
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return snapped.to(torch.float8_e4m3fn), scale
        except (RuntimeError, TypeError, ValueError):
            pass
    return snapped, scale


def dequant_fp8(fp8: torch.Tensor, scale: torch.Tensor, group_size: int = -1) -> torch.Tensor:
    """Dequantize FP8 weights back to float; opposite of :func:`quantize_to_fp8`.

    ``scale`` is per-tensor (scalar) or per-group (``(n/g,)``) and broadcasts
    down to the weight shape. Output dtype is float32 (caller casts).
    """
    q = fp8.float()
    if group_size is None or group_size <= 0:
        return q * scale
    return (q.reshape(-1, int(group_size)) * scale.view(-1, 1)).reshape_as(q)


def mark_fp8_weight(weight: nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16 weight to FP8 and stash the payload for the linear hook.

    Sets ``weight._areno_fp8`` (E4M3 tensor, shape ``(N, K)``) and
    ``weight._areno_fp8_scale`` (per-tensor float32 scalar) so
    ``_areno_linear_forward`` routes decode forwards through the FP8 linear.
    On refresh the payload buffers are updated **in place** so tensors captured
    by decode CUDA graphs keep their storage. Returns (payload, scale).

    The scale is synchronized across the TP group (MAX of the local amax) and
    the payload is quantized with that shared scale, so every rank dequantizes
    identically; row-parallel partial sums are then scale-consistent before the
    cross-rank reduction.
    """
    w = weight.detach().float()
    amax = w.abs().amax()
    amax = _tp_max(amax, weight)
    scale = amax / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).to(torch.float32).reshape(())

    payload_new = torch.clamp((w / scale).round(), -_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    payload = getattr(weight, "_areno_fp8", None)
    scale_buf = getattr(weight, "_areno_fp8_scale", None)
    if payload is not None and payload.shape == payload_new.shape and payload.device == payload_new.device:
        payload.copy_(payload_new)
        scale_buf.copy_(scale)
    else:
        payload = payload_new.contiguous()
        scale_buf = scale.to(device=weight.device).contiguous()
        weight._areno_fp8 = payload
        weight._areno_fp8_scale = scale_buf
    return payload, scale_buf


def _tp_max(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """MAX-reduce ``value`` across the TP group (no-op outside distributed TP)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    from areno.engine.parallel.context import get_tp_context

    ctx = get_tp_context()
    if ctx.world_size <= 1:
        return value
    import torch.distributed as dist

    value = value.to(ref.device)
    if value.device.type == "cpu" and dist.get_backend(ctx.group) == "nccl":
        # NCCL groups cannot collectivize CPU tensors (the unit-test path on a
        # GPU host); hop through the current device and bring the result back.
        gpu_value = value.to(torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(gpu_value, op=dist.ReduceOp.MAX, group=ctx.group)
        return gpu_value.cpu()
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=ctx.group)
    return value


def quantize_infer_weights_fp8(model: nn.Module) -> int:
    """Create-or-refresh the FP8 payload of every parallel-linear weight.

    Called when a decode session (re)materializes infer weights, so the payload
    always reflects the current weights (RL rollout refreshes them every policy
    sync). Training forwards are unaffected: ``_areno_linear_forward`` routes to
    the FP8 kernel only when gradients are disabled. Returns the count.
    """
    from areno.engine.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

    count = 0
    for module in model.modules():
        if isinstance(module, (ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear)):
            weight = getattr(module, "weight", None)
            if weight is not None and weight.ndim == 2 and weight.numel() > 0:
                mark_fp8_weight(weight)
                count += 1
    return count
