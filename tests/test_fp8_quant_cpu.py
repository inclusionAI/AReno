"""CPU tests for FP8 weight quantization.

These run without a GPU: the quantize/dequant math is implemented on float
tensors. They assert:
  * dequant(quantize(w)) is within FP8 tolerance of w,
  * a dequantized weight matmul is close to the bf16 linear,
  * the config default is opt-in unchanged (quant_method == "none").
"""

from __future__ import annotations

import torch

from areno.engine.config import ModelConfig
from areno.engine.quantization import compute_fp8_scale, dequant_fp8, quantize_to_fp8


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6))


def _weights(shape=(64, 96)) -> torch.Tensor:
    # Unit-scale weights so the FP8 grid covers the range consistently.
    w = torch.randn(*shape)
    return (w / w.abs().amax()) * 10.0


def test_quant_dequant_roundtrip_within_fp8_tolerance():
    w = _weights()
    q, scale = quantize_to_fp8(w, group_size=-1)
    dq = dequant_fp8(q, scale, group_size=-1)
    # E4M3 has ~4 mantissa bits => relative error <= ~2^-4. Use a safe bound.
    assert _rel_err(dq, w) < 0.12, f"roundtrip rel err too high: {_rel_err(dq, w)}"
    # The maximum-magnitude element is preserved exactly by construction.
    amax_idx = torch.argmax(w.abs())
    assert abs(float(dq.reshape(-1)[amax_idx]) - float(w.reshape(-1)[amax_idx])) < 1e-3


def test_quant_dequant_per_group_roundtrip():
    w = _weights((48, 128))
    g = 64
    q, scale = quantize_to_fp8(w, group_size=g)
    dq = dequant_fp8(q, scale, group_size=g)
    assert dq.shape == w.shape
    assert float(scale.numel()) == w.numel() // g
    assert _rel_err(dq, w) < 0.12


def test_dequant_weight_matmul_matches_bf16():
    x = torch.randn(3, 96)
    w = _weights((64, 96))
    q, scale = quantize_to_fp8(w, group_size=-1)
    dq = dequant_fp8(q, scale, group_size=-1).to(torch.bfloat16)
    ref = x.to(torch.bfloat16) @ w.to(torch.bfloat16).T
    out = x.to(torch.bfloat16) @ dq.T
    # Relative error on the matmul output is bounded by the weight error (~6%).
    rel = float((out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6))
    assert rel < 0.2, f"matmul rel err too high: {rel}"


def test_scales_nonzero_and_finite():
    w = _weights()
    scale = compute_fp8_scale(w, group_size=-1)
    assert scale.item() > 0
    assert torch.isfinite(scale).all()
    # A zero weight must not produce inf/nan scale.
    zero = torch.zeros(8, 8)
    scale0 = compute_fp8_scale(zero, group_size=-1)
    assert torch.isfinite(scale0).all() and scale0.item() > 0


def test_config_default_is_opt_in_unchanged():
    cfg = ModelConfig()
    assert cfg.quant_method == "none"
    assert cfg.quant_group_size == 128
    # Setting it to fp8 changes the field without changing any existing value.
    cfg.quant_method = "fp8"
    assert cfg.quant_method == "fp8"
