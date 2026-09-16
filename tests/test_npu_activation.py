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
    assert not native.supports_training_and_serving


@pytest.mark.parametrize("name", ["silu", "sigmoid", "softplus"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 65, 257])
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
