"""Ascend native extension acceptance through the existing accel/autograd API."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from areno.accel import activations
from areno.accel._extension import extension


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    native = extension("npu")
    assert native.activation_implementation == "ascendc"
    assert not native.supports_training_and_serving


@pytest.mark.parametrize("name", ["silu", "sigmoid", "softplus"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 65, 1023, 1024, 1025, 32769])
@pytest.mark.parametrize("transposed", [False, True])
def test_forward_backward(name, dtype, width, transposed):
    values = torch.linspace(-30, 30, 6 * width).reshape(2, 3, width).to(dtype)
    source = values.to("npu")
    if transposed:
        source = source.transpose(0, 1)
        values = values.transpose(0, 1)
    source.requires_grad_()
    expected_input = values.float().detach().requires_grad_()
    reference = {"silu": F.silu, "sigmoid": torch.sigmoid, "softplus": F.softplus}[name]
    expected = reference(expected_input)
    actual = getattr(activations, f"areno_{name}")(source)
    grad = torch.linspace(-2, 2, actual.numel()).reshape(actual.shape).to(dtype)
    actual.backward(grad.to("npu"))
    expected.backward(grad.float())
    expected_grad = expected_input.grad
    if name == "sigmoid":
        saved = expected.detach().to(dtype).float()
        expected_grad = grad.float() * saved * (1 - saved)
    tolerance = {torch.float32: 3e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.detach().to(dtype), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(source.grad.cpu(), expected_grad.to(dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("name", ["silu", "sigmoid", "softplus"])
def test_empty_input(name):
    x = torch.empty((2, 0, 65), device="npu", requires_grad=True)
    result = getattr(activations, f"areno_{name}")(x)
    result.sum().backward()
    assert result.shape == x.shape
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("name", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 65, 1023, 1024, 1025, 8193])
@pytest.mark.parametrize("transposed", [False, True])
def test_gated_forward_backward(name, dtype, width, transposed):
    generator = torch.Generator().manual_seed(410)
    values = (torch.randn(2, 3, 2 * width, generator=generator) * 4).to(dtype)
    source = values.to("npu")
    if transposed:
        source = source.transpose(0, 1)
        values = values.transpose(0, 1)
    source.requires_grad_()
    reference = values.float().detach().requires_grad_()
    gate, up = reference.chunk(2, dim=-1)
    expected = (F.silu(gate) if name == "silu" else F.gelu(gate, approximate="tanh")) * up
    actual = getattr(activations, f"areno_{name}_and_mul")(source)
    grad = torch.randn(actual.shape, generator=generator).to(dtype)
    actual.backward(grad.to("npu"))
    expected.backward(grad.float())
    tolerance = {torch.float32: 4e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual.cpu(), expected.detach().to(dtype), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(source.grad.cpu(), reference.grad.to(dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("name", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("shape", [(0, 130), (3, 0)])
def test_gated_empty_input(name, shape):
    x = torch.empty(shape, device="npu", requires_grad=True)
    result = getattr(activations, f"areno_{name}_and_mul")(x)
    result.sum().backward()
    assert result.shape == (*shape[:-1], shape[-1] // 2)
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_gated_out_buffer_respects_strides_offsets_and_canaries(dtype, strided):
    x = torch.linspace(-8, 8, 6 * 130).reshape(2, 3, 130).to(dtype)
    storage = torch.full((6 * 65 + 6,), 47, dtype=dtype, device="npu")
    output = storage[3:-3].view(3, 2, 65).transpose(0, 1) if strided else storage[3:-3].view(2, 3, 65)
    actual = activations.areno_silu_and_mul(x.to("npu"), out=output)
    gate, up = x.float().chunk(2, dim=-1)
    expected = (F.silu(gate) * up).to(dtype)
    tolerance = {torch.float32: 3e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    assert torch.equal(storage[:3].cpu(), torch.full((3,), 47, dtype=dtype))
    assert torch.equal(storage[-3:].cpu(), torch.full((3,), 47, dtype=dtype))


def test_softplus_negative_tail_and_large_inputs():
    values = torch.tensor([-80.0, -40.0, -30.0, -20.0, -10.0, 0.0, 20.0, 21.0, 80.0, 1000.0])
    source = values.to("npu").requires_grad_()
    expected_input = values.clone().requires_grad_()
    expected = F.softplus(expected_input)
    actual = activations.areno_softplus(source)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected.detach(), atol=0, rtol=3e-5)
    torch.testing.assert_close(source.grad.cpu(), expected_input.grad, atol=0, rtol=3e-5)


def test_native_launch_uses_current_stream():
    stream = torch.npu.Stream()
    # Both the input-producing fill and the output-consuming add must be
    # ordered around the native launch on this non-default stream.
    with torch.npu.stream(stream):
        x = torch.empty(65539, device="npu")
        x.fill_(0.75)
        actual = activations.areno_silu(x) + 2
    stream.synchronize()
    torch.testing.assert_close(actual.cpu(), F.silu(torch.full((65539,), 0.75)) + 2)


def test_launch_selects_tensor_device_and_restores_callers_device():
    if torch.npu.device_count() < 2:
        pytest.skip("two Ascend devices required")
    x = torch.ones((2, 130), device="npu:1", requires_grad=True)
    torch.npu.set_device(0)
    actual = activations.areno_silu_and_mul(x)
    actual.sum().backward()
    assert torch.npu.current_device() == 0
    assert actual.device == x.device and x.grad.device == x.device
    expected_input = torch.ones((2, 130), requires_grad=True)
    gate, up = expected_input.chunk(2, dim=-1)
    expected = F.silu(gate) * up
    expected.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected.detach())
    torch.testing.assert_close(x.grad.cpu(), expected_input.grad)


def test_native_entry_rejects_invalid_metadata():
    native = extension("npu")
    with pytest.raises(RuntimeError, match="Ascend NPU tensor"):
        native.areno_silu(torch.ones(3))
    with pytest.raises(RuntimeError, match="FP32, FP16 or BF16"):
        native.areno_silu(torch.ones(3, dtype=torch.int32, device="npu"))
    with pytest.raises(RuntimeError, match="shape, dtype and device"):
        native.areno_d_silu(torch.ones(4, device="npu"), torch.ones(3, device="npu"))
    with pytest.raises(RuntimeError, match="contiguous"):
        native.areno_silu(torch.ones(3, 4, device="npu").t())


@pytest.mark.parametrize("storage_format", [0, 2, 29], ids=["NCHW", "ND", "FRACTAL_NZ"])
def test_native_entry_checks_physical_storage_format(storage_format):
    import torch_npu

    # Permit creation of a packed tensor specifically to test the raw-pointer
    # boundary. Restore TorchNPU's setting even when the kernel/guard fails.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    try:
        values = torch.linspace(-3, 3, 2 * 17 * 3 * 19).reshape(2, 17, 3, 19).half()
        source = torch_npu.npu_format_cast(values.to("npu"), storage_format)
        assert torch_npu.get_npu_format(source) == storage_format
        assert source.is_contiguous()
        native = extension("npu")
        if storage_format == 29:
            with pytest.raises(RuntimeError, match="base storage format"):
                native.areno_silu(source)
        else:
            result = native.areno_silu(source)
            torch.testing.assert_close(result.cpu(), F.silu(values.float()).half(), atol=3e-3, rtol=3e-3)
    finally:
        torch.npu.config.allow_internal_format = previous == b"enable"
