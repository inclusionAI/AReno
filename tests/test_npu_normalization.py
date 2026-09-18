"""Ascend C RMSNorm acceptance against independent PyTorch math on CPU."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from areno.accel import normalization
from areno.accel._extension import extension
from areno.accel.ops import rms_norm_gate_fwd


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    assert extension("npu").normalization_implementation == "ascendc"


def reference(x, weight, eps, gate=None):
    normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        normalized = normalized * weight
    return normalized if gate is None else normalized * F.silu(gate)


def candidate(variant, x, weight, eps, gate):
    if variant == "plain":
        return normalization.areno_rmsnorm(x, weight, eps)
    if variant == "silu_gate":
        return normalization.areno_rmsnorm_silu_gate(x, gate, weight, eps)
    return normalization.areno_optional_scale_rmsnorm(x, weight if variant == "optional" else None, eps)


@pytest.mark.parametrize("variant", ["plain", "optional", "unscaled", "silu_gate"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 7, 65, 1023, 1024, 1025, 4097, 16385])
@pytest.mark.parametrize("transposed", [False, True])
def test_forward_and_all_gradients(variant, dtype, width, transposed):
    generator = torch.Generator().manual_seed(603)
    values = torch.randn(2, 3, width, generator=generator).to(dtype)
    gates = (torch.randn(2, 3, width, generator=generator) * 3).to(dtype)
    weights = torch.randn(width, generator=generator)
    x, gate = values.to("npu"), gates.to("npu")
    if transposed:
        values, gates = values.transpose(0, 1), gates.transpose(0, 1)
        x, gate = x.transpose(0, 1), gate.transpose(0, 1)
    x.requires_grad_()
    gate.requires_grad_()
    weight = weights.to("npu").requires_grad_()
    rx, rg, rw = (v.float().detach().requires_grad_() for v in (values, gates, weights))
    actual = candidate(variant, x, weight, 1e-6, gate)
    expected = reference(rx, rw if variant != "unscaled" else None, 1e-6, rg if variant == "silu_gate" else None)
    gradient = torch.randn(actual.shape, generator=generator).to(dtype)
    actual.backward(gradient.to("npu"))
    expected.backward(gradient.float())
    tolerance = {torch.float32: 8e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual.cpu(), expected.detach().to(dtype), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x.grad.cpu(), rx.grad.to(dtype), atol=tolerance, rtol=tolerance)
    if variant != "unscaled":
        # Weight gradient is FP32 even for low-precision activations.
        torch.testing.assert_close(weight.grad.cpu(), rw.grad, atol=2e-4, rtol=8e-5)
    if variant == "silu_gate":
        torch.testing.assert_close(gate.grad.cpu(), rg.grad.to(dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weight_cast_preserves_original_parameter_gradient(dtype):
    values = torch.linspace(-2, 3, 7 * 65).reshape(7, 65).to(dtype)
    weights = torch.linspace(0.1, 1.1, 65).to(dtype)
    x = values.to("npu").requires_grad_()
    weight = weights.to("npu").requires_grad_()
    rw = weights.float().requires_grad_()
    expected = reference(values.float(), rw, 1e-5)
    result = normalization.areno_rmsnorm(x, weight, 1e-5)
    result.sum().backward()
    expected.sum().backward()
    assert weight.grad.dtype == dtype
    tolerance = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(weight.grad.cpu(), rw.grad.to(dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("variant", ["plain", "optional", "unscaled", "silu_gate"])
def test_empty_rows_and_zero_inputs(variant):
    for rows in (0, 3):
        x = torch.zeros(rows, 65, device="npu", requires_grad=True)
        gate = torch.full_like(x, -2, requires_grad=True)
        weight = torch.ones(65, device="npu", requires_grad=True)
        result = candidate(variant, x, weight, 1e-6, gate)
        result.sum().backward()
        assert result.shape == x.shape and x.grad.shape == x.shape
        torch.testing.assert_close(result.cpu(), torch.zeros(rows, 65))
        gain = F.silu(torch.tensor(-2.0)) if variant == "silu_gate" else 1.0
        torch.testing.assert_close(x.grad.cpu(), torch.full((rows, 65), 1000.0) * gain, rtol=8e-5, atol=8e-5)
        if variant != "unscaled":
            torch.testing.assert_close(weight.grad.cpu(), torch.zeros(65))
        if variant == "silu_gate":
            torch.testing.assert_close(gate.grad.cpu(), torch.zeros(rows, 65))


@pytest.mark.parametrize("gated", [False, True])
def test_many_rows_accumulate_fp32_weight_gradient(gated):
    generator = torch.Generator().manual_seed(94)
    values = torch.randn(257, 1025, generator=generator)
    gates = torch.randn(values.shape, generator=generator)
    gradient = torch.randn(values.shape, generator=generator)
    weight_values = torch.randn(1025, generator=generator)
    x, gate, weight = (t.to("npu").requires_grad_() for t in (values, gates, weight_values))
    rw = weight_values.detach().requires_grad_()
    result = candidate("silu_gate" if gated else "plain", x, weight, 1e-5, gate)
    expected = reference(values, rw, 1e-5, gates if gated else None)
    result.backward(gradient.to("npu"))
    expected.backward(gradient)
    torch.testing.assert_close(weight.grad.cpu(), rw.grad, atol=5e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 65, 1025, 8193])
def test_group_gate_matches_sigmoid_and_group_specific_weights(dtype, weight_dtype, width):
    generator = torch.Generator().manual_seed(212)
    values = torch.randn(3, 2, 2 * width, generator=generator).to(dtype)
    gates = torch.randn(3, 2, 2 * width, generator=generator).to(dtype)
    weights = (torch.arange(3).unsqueeze(1) + torch.linspace(0.5, 1.5, 2 * width)).to(weight_dtype)
    # Exercise non-contiguous feature, row and weight strides through the
    # public group wrapper, which does not pack its inputs before dispatch.
    x = values.to("npu").transpose(0, 1)[..., ::2]
    gate = gates.to("npu").transpose(0, 1)[..., ::2]
    weight = weights.to("npu")[:, ::2]
    rx = values.transpose(0, 1)[..., ::2].float()
    rg = gates.transpose(0, 1)[..., ::2].float()
    rw = weights[:, ::2].float()
    inv = torch.rsqrt(rx.square().mean(dim=-1) + 1e-6)
    expected = rx * inv.unsqueeze(-1) * rw * rg.sigmoid()
    with torch.no_grad():
        actual, saved = rms_norm_gate_fwd(x, gate, weight, 1e-6)
    tolerance = {torch.float32: 8e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    assert saved.dtype == torch.float32 and saved.shape == (2, 3)
    torch.testing.assert_close(saved.cpu(), inv, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.cpu(), expected.to(dtype), atol=tolerance, rtol=tolerance)


def test_norm_uses_current_stream_and_tensor_device():
    device = 1 if torch.npu.device_count() >= 2 else 0
    x = torch.empty(37, 1025, device=f"npu:{device}", requires_grad=True)
    weight = torch.ones(1025, device=x.device, requires_grad=True)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        with torch.no_grad():
            x.fill_(0.75)
        result = normalization.areno_rmsnorm(x, weight, 1e-5)
        result.sum().backward()
    stream.synchronize()
    rx = torch.full((37, 1025), 0.75, requires_grad=True)
    rw = torch.ones(1025, requires_grad=True)
    expected = reference(rx, rw, 1e-5)
    expected.sum().backward()
    assert result.device == x.device
    torch.testing.assert_close(result.cpu(), expected.detach(), atol=8e-5, rtol=8e-5)
    torch.testing.assert_close(x.grad.cpu(), rx.grad, atol=8e-5, rtol=8e-5)
    torch.testing.assert_close(weight.grad.cpu(), rw.grad, atol=2e-4, rtol=8e-5)


def test_native_input_contract_and_saved_statistics():
    native = extension("npu")
    x = torch.ones(3, 65, device="npu")
    w = torch.ones(65, device="npu")
    output, inv = native.areno_rmsnorm_forward(x, w, 1e-6)
    assert output.shape == x.shape and inv.shape == (3,) and inv.dtype == torch.float32
    torch.testing.assert_close(inv.cpu(), torch.rsqrt(torch.ones(3) + 1e-6))
    with pytest.raises(RuntimeError, match="nonempty feature"):
        native.areno_rmsnorm_forward(torch.empty(3, 0, device="npu"), torch.empty(0, device="npu"), 1e-6)
    with pytest.raises(RuntimeError, match="weight must be FP32"):
        native.areno_rmsnorm_forward(x, w.to(torch.float16), 1e-6)
    with pytest.raises(RuntimeError, match="epsilon"):
        native.areno_rmsnorm_forward(x, w, -1)
    with pytest.raises(RuntimeError, match="saved inv_rms"):
        native.areno_rmsnorm_backward(x, x, w, torch.ones(4, device="npu"))
