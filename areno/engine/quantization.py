"""FP8 (E4M3) weight quantization for the decode path.

`mark_fp8_weight` quantizes one parallel-linear weight and attaches the payload
that `_areno_linear_forward` routes on; `quantize_infer_weights_fp8` walks a
model at decode-session start. The payload is only routed while a decode scope
is open, so scoring and training forwards stay bf16.

The quantize/dequant math is dtype-agnostic, so it runs on CPU as well as GPU.
"""

from __future__ import annotations

import torch
from torch import nn

from areno.accel.utils import FP8_E4M3_MAX, warn_once

PAYLOAD_ATTR = "_areno_fp8"
SCALE_ATTR = "_areno_fp8_scale"

# Decode scope: the worker opens it for the lifetime of a rollout session and
# closes it before scoring or training starts, so this is the only thing that
# decides whether a payload-carrying weight routes to the FP8 backend. It is
# process-global rather than a contextvar because the session command that opens
# it and the forwards it enables are not guaranteed to run on the same thread.
_decode_active = False


def set_fp8_decode_active(active: bool) -> None:
    """Open or close the FP8 decode scope."""

    global _decode_active
    _decode_active = active


def fp8_decode_active() -> bool:
    """Whether the current region is decode, i.e. whether FP8 routing applies."""

    return _decode_active


def quantize_weight_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 quantize a weight into (payload, float32 scalar scale)."""

    w = weight.detach().float()
    amax = w.abs().amax()
    scale = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax)).to(torch.float32)
    payload = torch.clamp((w / scale).round(), -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return payload, scale.reshape(())


def dequantize_weight_fp8(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an E4M3 weight payload back to float32."""

    return payload.float() * scale


def mark_fp8_weight(weight: nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight and attach the payload consumed by the linear hook.

    Repeated calls refresh the payload buffers in place, so tensors captured by
    decode CUDA graphs keep their storage. Returns the (payload, scale) pair.
    """

    payload_new, scale_new = quantize_weight_fp8(weight)
    payload = getattr(weight, PAYLOAD_ATTR, None)
    scale = getattr(weight, SCALE_ATTR, None)
    if payload is None or payload.shape != payload_new.shape or payload.device != weight.device:
        payload = payload_new.contiguous().to(device=weight.device)
        scale = scale_new.contiguous().to(device=weight.device)
        setattr(weight, PAYLOAD_ATTR, payload)
        setattr(weight, SCALE_ATTR, scale)
        return payload, scale

    payload.copy_(payload_new)
    scale.copy_(scale_new.to(device=scale.device))
    return payload, scale


def release_fp8_weights(model: nn.Module) -> int:
    """Drop every FP8 payload in `model`, returning how many were released.

    The payload is a CUDA allocation the module does not own, so it survives
    `to("cpu")` and `empty_cache()` unless it is removed explicitly.
    """

    released = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, PAYLOAD_ATTR):
            continue
        delattr(weight, PAYLOAD_ATTR)
        delattr(weight, SCALE_ATTR)
        released += 1
    return released


def quantize_infer_weights_fp8(model: nn.Module) -> int:
    """Create-or-refresh the FP8 payload of every parallel-linear weight.

    Called when a decode session materializes infer weights, so the payload
    tracks the weights current as of the last policy sync. Only the parallel
    linear classes are covered; expert, embedding and output-projection weights
    keep full precision, which the coverage warning reports. Returns the number
    of marked weights.
    """

    from areno.engine.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

    parallel_linear_types = (ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear)
    marked_params = 0
    count = 0
    for module in model.modules():
        if not isinstance(module, parallel_linear_types):
            continue
        mark_fp8_weight(module.weight)
        marked_params += module.weight.numel()
        count += 1

    total_params = sum(parameter.numel() for parameter in model.parameters())
    if total_params and marked_params * 2 < total_params:
        warn_once(
            "fp8-decode-partial-coverage",
            f"quant_method='fp8' covers {marked_params / total_params:.0%} of model parameters; "
            "expert, embedding and output-projection weights keep full precision",
        )
    return count
