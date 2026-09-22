"""GPU tests for the FP8 decode backend.

Skipped without an FP8-capable GPU: torch._scaled_mm requires compute capability
>= 8.9 (Hopper/Ada) and the fused activation quantizer emits fp8e4nv, which
Ampere's Triton backend cannot compile. These tests are the only coverage of the
Triton kernel, so they assert it against the torch fallback rather than only
against an end-to-end tolerance.
"""

from __future__ import annotations

import pytest
import torch

import areno.accel.kernels.fp8_scaled_mm as fp8_module
from areno.engine.quantization import dequantize_weight_fp8, mark_fp8_weight

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 9)),
    reason="requires an FP8-capable GPU (compute capability >= 8.9)",
)


def test_scaled_mm_available_reports_the_device_capability():
    assert fp8_module.scaled_mm_available(torch.cuda.current_device()) is True
    assert fp8_module.scaled_mm_available(torch.device("cuda", torch.cuda.current_device())) is True
    assert fp8_module.scaled_mm_available(torch.device("cpu")) is False


def test_fused_activation_quantizer_matches_the_torch_reference():
    """The fused kernel must reproduce the torch fallback, sign and scale alike."""

    torch.manual_seed(4)
    asymmetric = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16) * 3.0
    # A negative tail that a signed max would miss when choosing the scale.
    asymmetric[..., : 4096 // 4] *= -8.0

    for x in (asymmetric, torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)):
        fused_fp8, fused_scale = fp8_module._quantize_act(x)
        fp8_module._HAS_TRITON = False
        try:
            reference_fp8, reference_scale = fp8_module._quantize_act(x)
        finally:
            fp8_module._HAS_TRITON = True

        expected_scale = x.abs().amax().float() / 448.0
        assert torch.allclose(fused_scale.float(), expected_scale, rtol=1e-5, atol=0.0)
        assert torch.allclose(reference_scale.float(), expected_scale, rtol=1e-5, atol=0.0)

        fused = fused_fp8.float() * fused_scale
        reference = reference_fp8.float() * reference_scale
        assert torch.allclose(fused, reference, rtol=0.0, atol=1e-3 * float(x.abs().max()))
        assert torch.allclose(fused, x.float(), rtol=0.0, atol=0.1 * float(x.abs().max()))
        # Negative activations must stay negative and unsaturated.
        assert bool((fused[x < -1.0] < 0).all())


def test_quantized_fp8_scaled_mm_matches_the_dequantized_reference():
    torch.manual_seed(5)
    rows, out_features, in_features = 4, 4096, 4096
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * (4.0 / in_features**0.5)
    payload, scale = mark_fp8_weight(torch.nn.Parameter(weight))

    out = fp8_module.quantized_fp8_scaled_mm(x, payload, scale)

    activation_fp8, activation_scale = fp8_module._quantize_act(x)
    reference = (activation_fp8.float() * activation_scale) @ dequantize_weight_fp8(payload, scale).T
    assert out.dtype == torch.bfloat16
    assert torch.allclose(
        out.float(), reference.to(torch.bfloat16).float(), rtol=0.0, atol=0.02 * float(reference.abs().max())
    )
