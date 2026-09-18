"""Dense linear uses the shared autograd wrapper and CANN/Ascend C operators."""

import importlib.util
import itertools

import pytest
import torch

from areno.accel._extension import extension
from areno.accel.linear import areno_linear


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    assert extension("npu").linear_implementation == "aclnn_ascendc"


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def values(shape, dtype, *, strided=False, device="npu", seed=37):
    physical = (*shape[:-1], shape[-1] * 2) if strided else shape
    data = (torch.randint(-8, 9, physical, generator=torch.Generator().manual_seed(seed)) / 64).to(dtype)
    tensor = data.to(device)
    if strided:
        tensor, data = tensor[..., ::2], data[..., ::2]
    return tensor, data


def reference(input, weight, bias=None):
    # CUDA first rounds the GEMM result to storage dtype, then adds bias in
    # FP32 and rounds again. Fused F.linear/addmm is not this reference.
    result = (input.double() @ weight.double().t()).to(input.dtype)
    return result if bias is None else (result.float() + bias.float()).to(input.dtype)


def check_gradients(input, weight, bias, x, w, b, gradient):
    rows = gradient.numel() // w.shape[0] if w.shape[0] else 0
    g = gradient.reshape(rows, w.shape[0]).double()
    if input.requires_grad:
        expected = (g @ w.double()).reshape(x.shape).to(x.dtype)
        torch.testing.assert_close(input.grad.cpu(), expected, atol=0, rtol=0)
    if weight.requires_grad:
        expected = (g.t() @ x.reshape(rows, w.shape[1]).double()).to(w.dtype)
        torch.testing.assert_close(weight.grad.cpu(), expected, atol=0, rtol=0)
    if bias is not None and bias.requires_grad:
        torch.testing.assert_close(bias.grad.cpu(), g.sum(0).to(b.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,n", [((1, 1), 1), ((7, 13), 17), ((2, 3, 33), 65), ((3, 1025), 1027), ((2, 2053), 9), ((2049, 7), 65)]
)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_linear_forward_backward(dtype, shape, n, use_bias, strided):
    input, x = values(shape, dtype, strided=strided)
    weight, w = values((n, shape[-1]), dtype, strided=strided, seed=41)
    bias, b = values((n,), dtype, strided=strided, seed=43) if use_bias else (None, None)
    input.requires_grad_()
    weight.requires_grad_()
    if bias is not None:
        bias.requires_grad_()
    output = areno_linear(input, weight, bias)
    torch.testing.assert_close(output.cpu(), reference(x, w, b), atol=0, rtol=0)
    grad, g = values(tuple(output.shape), dtype, strided=True, seed=47)
    output.backward(grad)
    # Small binary fractions have exact FP32 accumulation for these shapes.
    check_gradients(input, weight, bias, x, w, b, g)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("needs", list(itertools.product((False, True), repeat=3))[1:])
def test_linear_selective_gradients(dtype, needs):
    input, x = values((2, 3, 17), dtype)
    weight, w = values((65, 17), dtype, seed=41)
    bias, b = values((65,), dtype, seed=43)
    for tensor, need in zip((input, weight, bias), needs, strict=True):
        tensor.requires_grad_(need)
    output = areno_linear(input, weight, bias)
    grad, g = values(tuple(output.shape), dtype, seed=47)
    output.backward(grad)
    check_gradients(input, weight, bias, x, w, b, g)
    for tensor, need in zip((input, weight, bias), needs, strict=True):
        assert (tensor.grad is not None) == need


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,n", [((0, 17), 65), ((2, 0, 17), 65), ((3, 0), 65), ((3, 17), 0)])
def test_linear_empty_dimensions(dtype, shape, n):
    input = torch.empty(shape, dtype=dtype, device="npu", requires_grad=True)
    weight = torch.empty(n, shape[-1], dtype=dtype, device="npu", requires_grad=True)
    bias = torch.ones(n, dtype=dtype, device="npu", requires_grad=True)
    output = areno_linear(input, weight, bias)
    assert output.shape == (*shape[:-1], n)
    torch.testing.assert_close(output.cpu(), torch.ones(output.shape, dtype=dtype), atol=0, rtol=0)
    output.backward(torch.ones_like(output))
    assert input.grad.shape == input.shape and torch.count_nonzero(input.grad.cpu()) == 0
    assert weight.grad.shape == weight.shape and torch.count_nonzero(weight.grad.cpu()) == 0
    rows = 1
    for dimension in shape[:-1]:
        rows *= dimension
    torch.testing.assert_close(bias.grad.cpu(), torch.full((n,), rows, dtype=dtype), atol=0, rtol=0)


def test_linear_fp32_precision_and_storage_offsets():
    rng = torch.Generator().manual_seed(59)
    x = torch.randn(17, 4097, generator=rng) * 0.125
    w = torch.randn(65, 4097, generator=rng) * 0.125
    g = torch.randn(17, 65, generator=rng) * 0.125
    # Contiguous row slices with nonzero storage offsets pass through the
    # shared wrapper without copies, including the transposed GEMM operand.
    input = torch.cat((torch.zeros(1, 4097), x)).to("npu")[1:].requires_grad_()
    weight = torch.cat((torch.zeros(1, 4097), w)).to("npu")[1:].requires_grad_()
    output = areno_linear(input, weight)
    output.backward(g.to("npu"))
    torch.testing.assert_close(output.cpu(), (x.double() @ w.double().t()).float(), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(input.grad.cpu(), (g.double() @ w.double()).float(), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(weight.grad.cpu(), (g.double().t() @ x.double()).float(), atol=2e-6, rtol=2e-5)


def test_linear_current_stream_and_tensor_device():
    device = 1 if torch.npu.device_count() > 1 else 0
    input = torch.empty(3, 1025, device=f"npu:{device}", requires_grad=True)
    weight = torch.empty(65, 1025, device=input.device, requires_grad=True)
    bias = torch.empty(65, device=input.device, requires_grad=True)
    stream = torch.npu.Stream(device=device)
    with torch.npu.stream(stream):
        with torch.no_grad():
            input.fill_(0.125)
            weight.fill_(0.25)
            bias.fill_(0.5)
        result = areno_linear(input, weight, bias)
        result.sum().backward()
    stream.synchronize()
    assert result.device == input.device
    torch.testing.assert_close(result.cpu(), torch.full((3, 65), 1025 / 32 + 0.5), atol=0, rtol=0)
    torch.testing.assert_close(input.grad.cpu(), torch.full((3, 1025), 65 / 4), atol=0, rtol=0)
    torch.testing.assert_close(weight.grad.cpu(), torch.full((65, 1025), 3 / 8), atol=0, rtol=0)
    torch.testing.assert_close(bias.grad.cpu(), torch.full((65,), 3.0), atol=0, rtol=0)


def test_linear_native_contract_and_unrequested_gradients():
    native = extension("npu")
    input = torch.ones(3, 17, device="npu")
    weight = torch.ones(65, 17, device="npu")
    bias = torch.ones(65, device="npu")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        native.areno_linear_forward(input, weight[:, :-1].contiguous(), bias, True)
    with pytest.raises(RuntimeError, match="bias size mismatch"):
        native.areno_linear_forward(input, weight, bias[:-1], True)
    with pytest.raises(RuntimeError, match="dtype must match"):
        native.areno_linear_forward(input, weight.half(), bias, True)
    with pytest.raises(RuntimeError, match="contiguous"):
        native.areno_linear_forward(input[:, ::2], weight[:, ::2], bias, True)
    with pytest.raises(RuntimeError, match="gradient shape mismatch"):
        native.areno_linear_backward(torch.ones(3, 64, device="npu"), input, weight, False, True, True, False)
    results = native.areno_linear_backward(torch.ones(3, 65, device="npu"), input, weight, True, False, False, False)
    assert len(results) == 3 and all(value.numel() == 0 for value in results)
