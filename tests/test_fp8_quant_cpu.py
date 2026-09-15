"""CPU tests for FP8 weight quantization.

The quantize/dequant math runs on float tensors, so these need no GPU. They cover
the contract the decode path depends on: quantize/dequantize round-trips within
FP8 tolerance, a dequantized weight matmul stays close to the bf16 linear, scales
stay finite, and payloads can be released again.
"""

from __future__ import annotations

import torch

from areno.engine.quantization import (
    PAYLOAD_ATTR,
    SCALE_ATTR,
    dequantize_weight_fp8,
    mark_fp8_weight,
    quantize_weight_fp8,
    release_fp8_weights,
)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6))


def _weights(shape=(64, 96)) -> torch.Tensor:
    # Unit-scale weights so the FP8 grid covers the range consistently.
    w = torch.randn(*shape)
    return (w / w.abs().amax()) * 10.0


def test_quant_dequant_roundtrip_within_fp8_tolerance():
    torch.manual_seed(0)
    w = _weights()
    payload, scale = quantize_weight_fp8(w)
    dequantized = dequantize_weight_fp8(payload, scale)
    assert dequantized.shape == w.shape
    # E4M3 keeps ~4 mantissa bits, so relative error stays under ~1/16.
    assert _rel_err(dequantized, w) < 0.12, f"roundtrip rel err too high: {_rel_err(dequantized, w)}"
    # The maximum-magnitude element survives the round trip by construction.
    amax_index = torch.argmax(w.abs())
    assert abs(float(dequantized.reshape(-1)[amax_index]) - float(w.reshape(-1)[amax_index])) < 1e-3


def test_dequant_weight_matmul_matches_bf16():
    torch.manual_seed(1)
    x = torch.randn(3, 96)
    w = _weights((64, 96))
    payload, scale = quantize_weight_fp8(w)
    dequantized = dequantize_weight_fp8(payload, scale).to(torch.bfloat16)
    reference = x.to(torch.bfloat16) @ w.to(torch.bfloat16).T
    out = x.to(torch.bfloat16) @ dequantized.T
    # The output error is bounded by the weight error (~6%).
    assert _rel_err(out, reference) < 0.2, f"matmul rel err too high: {_rel_err(out, reference)}"


def test_scales_are_positive_and_finite():
    scale = quantize_weight_fp8(_weights())[1]
    assert scale.item() > 0
    assert torch.isfinite(scale).all()
    # An all-zero weight must not produce an inf/nan scale.
    zero_scale = quantize_weight_fp8(torch.zeros(8, 8))[1]
    assert torch.isfinite(zero_scale).all() and zero_scale.item() > 0


def test_scale_tracks_largest_magnitude_not_largest_value():
    # A negative-heavy weight must not be scaled by its much smaller positive max.
    w = torch.full((4, 4), -100.0)
    w[0, 0] = 1.0
    payload, scale = quantize_weight_fp8(w)
    assert abs(float(scale) - 100.0 / 448.0) < 1e-6
    # Nothing saturates: the -100 elements keep their magnitude.
    dequantized = dequantize_weight_fp8(payload, scale)
    assert abs(float(dequantized[1, 1]) + 100.0) < 1.0


def test_release_fp8_weights_drops_payloads():
    model = torch.nn.ModuleDict({"linear": torch.nn.Linear(8, 8), "untouched": torch.nn.Linear(8, 8)})
    mark_fp8_weight(model.linear.weight)
    assert hasattr(model.linear.weight, PAYLOAD_ATTR)
    assert release_fp8_weights(model) == 1
    assert not hasattr(model.linear.weight, PAYLOAD_ATTR)
    assert not hasattr(model.linear.weight, SCALE_ATTR)
    # Releasing twice is a no-op, and the payload can be rebuilt afterwards.
    assert release_fp8_weights(model) == 0
    mark_fp8_weight(model.linear.weight)
    assert hasattr(model.linear.weight, PAYLOAD_ATTR)
