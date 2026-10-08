"""Ascend C causal convolution acceptance against independent CPU conv1d."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from areno.accel._extension import extension
from areno.accel.conv import (
    areno_depthwise_causal_conv1d_silu,
    areno_depthwise_causal_conv1d_silu_decode,
    areno_packed_depthwise_causal_conv1d_silu,
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Ascend torch_npu and hardware are required")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "torch_npu is installed but no NPU is available"
    torch.npu.set_device(0)
    assert extension("npu").conv_implementation == "ascendc"


def values(shape, dtype, *, strided=False, device="npu", seed=37):
    physical = (*shape[:-1], 2 * shape[-1]) if strided else shape
    cpu = (torch.randn(physical, generator=torch.Generator().manual_seed(seed)) * 0.2).to(dtype)
    tensor = cpu.to(device)
    if strided:
        tensor, cpu = tensor[..., ::2], cpu[..., ::2]
    return tensor, cpu


def preactivation(x, w, lengths=None):
    if lengths is not None:
        return torch.cat([preactivation(part, w) for part in x.split(lengths, dim=1)], dim=1)
    if x.numel() == 0:
        return x.float() * 0 + w.float().sum() * 0
    padded = F.pad(x.float().transpose(1, 2), (w.shape[-1] - 1, 0))
    return F.conv1d(padded, w.float(), groups=x.shape[-1]).transpose(1, 2)


def reference(x, w, gradient, lengths=None):
    input, weight = x.detach().requires_grad_(), w.detach().requires_grad_()
    preact = preactivation(input, weight, lengths)
    output = F.silu(preact).to(x.dtype)
    dx, dw = torch.autograd.grad(output, (input, weight), gradient)
    return output.detach(), preact.detach(), dx, dw


def close(actual, expected):
    tolerance = {
        torch.float32: (5e-6, 1e-4),
        torch.float16: (3e-4, 2e-3),
        torch.bfloat16: (2e-3, 1e-2),
    }
    atol, rtol = tolerance[expected.dtype]
    torch.testing.assert_close(actual.detach().cpu(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "b,t,c,k",
    [
        (1, 1, 1, 1),
        (2, 7, 17, 4),
        (3, 3, 31, 9),
        (1, 257, 1, 4),
        (1, 5, 255, 3),
        (2, 9, 256, 4),
        (1, 7, 257, 5),
        (1, 3, 1025, 4),
    ],
)
def test_conv_forward_and_both_gradients(dtype, strided, b, t, c, k):
    input, x = values((b, t, c), dtype, strided=strided)
    weight, w = values((c, 1, k), torch.float32, strided=strided, seed=41)
    gradient, g = values((b, t, c), dtype, strided=True, seed=43)
    input.requires_grad_()
    weight.requires_grad_()
    output = areno_depthwise_causal_conv1d_silu(input, weight)
    output.backward(gradient)
    y, _, dx, dw = reference(x, w, g)
    for actual, expected in ((output, y), (input.grad, dx), (weight.grad, dw)):
        close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "lengths,c,k", [([0, 1, 0, 5, 0, 3, 0], 17, 4), ([2, 0, 0, 1], 257, 9), ([9], 255, 1), ([0, 0], 3, 4)]
)
def test_conv_packed_boundaries_empty_segments_and_gradients(dtype, strided, lengths, c, k):
    input, x = values((1, sum(lengths), c), dtype, strided=strided)
    weight, w = values((c, 1, k), torch.float32, strided=strided, seed=41)
    gradient, g = values(tuple(input.shape), dtype, strided=True, seed=43)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device="npu")
    input.requires_grad_()
    weight.requires_grad_()
    output = areno_packed_depthwise_causal_conv1d_silu(input, weight, cu)
    output.backward(gradient)
    y, _, dx, dw = reference(x, w, g, lengths)
    for actual, expected in ((output, y), (input.grad, dx), (weight.grad, dw)):
        close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("c,k", [(1, 1), (17, 2), (257, 4), (31, 9)])
def test_conv_decode_matches_prefill_without_mutating_history(dtype, c, k):
    input, x = values((2, 7, c), dtype)
    weight, w = values((c, 1, k), torch.float32, strided=True, seed=41)
    history = torch.zeros(2, c, k - 1, dtype=dtype, device="npu")
    results = []
    with torch.no_grad():
        for token in range(input.shape[1]):
            old = history.clone()
            results.append(areno_depthwise_causal_conv1d_silu_decode(input[:, token], history, weight))
            torch.testing.assert_close(history, old, atol=0, rtol=0)
            if k > 1:
                history = torch.cat((history[..., 1:], input[:, token, :, None]), dim=-1)
        prefill = areno_depthwise_causal_conv1d_silu(input, weight)
    decoded = torch.stack(results, dim=1)
    close(decoded, prefill.cpu())
    close(decoded, F.silu(preactivation(x, w)).to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("packed", [False, True])
def test_conv_saved_fp32_preact_and_storage_offsets(dtype, packed):
    # Binary fractions make the short convolution sum exactly representable.
    x = (torch.arange(1, 1 + 5 * 17).reshape(1, 5, 17) % 13 / 32).to(dtype)
    w = (torch.arange(17 * 4).reshape(17, 1, 4) % 7 / 32).float()
    input = torch.cat((torch.zeros(1, 1, 17, dtype=dtype), x), dim=1).to("npu")[:, 1:]
    weight = torch.cat((torch.zeros(1, 1, 4), w)).to("npu")[1:]
    native = extension("npu")
    gradient = torch.cat((torch.zeros(1, 1, 17, dtype=dtype), torch.ones_like(x)), dim=1).to("npu")[:, 1:]
    if packed:
        cu = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device="npu")
        y, preact = native.areno_packed_depthwise_causal_conv1d_silu_forward(input, weight, cu)
        expected = preactivation(x, w, [2, 0, 3])
        dx, dw = native.areno_packed_depthwise_causal_conv1d_silu_backward(gradient, input, weight, cu, preact)
    else:
        y, preact = native.areno_depthwise_causal_conv1d_silu_forward(input, weight)
        expected = preactivation(x, w)
        dx, dw = native.areno_depthwise_causal_conv1d_silu_backward(gradient, input, weight, preact)
    assert preact.dtype == torch.float32
    torch.testing.assert_close(preact.cpu(), expected, atol=0, rtol=0)
    close(y, F.silu(expected).to(dtype))
    _, _, dx_ref, dw_ref = reference(x, w, torch.ones_like(x), [2, 0, 3] if packed else None)
    close(dx, dx_ref)
    close(dw, dw_ref)


@pytest.mark.parametrize("dtype", DTYPES)
def test_conv_decode_strided_nonzero_history(dtype):
    current, x = values((2, 257), dtype, strided=True)
    history, h = values((2, 257, 3), dtype, strided=True, seed=41)
    weight, w = values((257, 1, 4), torch.float32, strided=True, seed=43)
    expected = F.silu((h.float() * w[:, 0, :3]).sum(-1) + x.float() * w[:, 0, -1]).to(dtype)
    close(areno_depthwise_causal_conv1d_silu_decode(current, history, weight), expected)
    torch.testing.assert_close(history.cpu(), h, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_conv_shared_autograd_casts_low_precision_weights(dtype):
    input, x = values((2, 9, 17), dtype)
    weight, w = values((17, 1, 4), dtype, seed=41)
    gradient, g = values(tuple(input.shape), dtype, seed=43)
    input.requires_grad_()
    weight.requires_grad_()
    output = areno_depthwise_causal_conv1d_silu(input, weight)
    output.backward(gradient)
    y, _, dx, dw = reference(x, w, g)
    for actual, expected in ((output, y), (input.grad, dx), (weight.grad, dw)):
        close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0, 3, 17), (2, 0, 17), (2, 3, 0)])
def test_conv_empty_shapes_have_zero_weight_gradients(dtype, shape):
    input = torch.empty(shape, dtype=dtype, device="npu", requires_grad=True)
    weight = torch.ones(shape[-1], 1, 4, device="npu", requires_grad=True)
    output = areno_depthwise_causal_conv1d_silu(input, weight)
    output.backward(torch.ones_like(output))
    assert output.shape == input.shape and input.grad.shape == input.shape
    torch.testing.assert_close(weight.grad.cpu(), torch.zeros_like(weight.cpu()), atol=0, rtol=0)


def test_conv_does_not_mix_sequences_or_future_tokens():
    input, _ = values((1, 7, 17), torch.float32)
    weight, _ = values((17, 1, 4), torch.float32, seed=41)
    cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32, device="npu")
    input.requires_grad_()
    output = areno_packed_depthwise_causal_conv1d_silu(input, weight, cu)
    output[:, 3:].sum().backward()
    torch.testing.assert_close(input.grad[:, :3], torch.zeros_like(input.grad[:, :3]), atol=0, rtol=0)
    with torch.no_grad():
        changed = input.clone()
        changed[:, :3].add_(100)
        after = areno_packed_depthwise_causal_conv1d_silu(changed, weight, cu)
        torch.testing.assert_close(after[:, 3:], output[:, 3:], atol=0, rtol=0)
        changed.copy_(input)
        changed[:, 5:].add_(100)
        after = areno_packed_depthwise_causal_conv1d_silu(changed, weight, cu)
        torch.testing.assert_close(after[:, :5], output[:, :5], atol=0, rtol=0)


def test_conv_padding_skips_nonfinite_unused_weight_taps():
    input = torch.tensor([[[2.0]]], device="npu", requires_grad=True)
    weight = torch.tensor([[[float("nan"), float("inf"), 0.5]]], device="npu", requires_grad=True)
    output = areno_depthwise_causal_conv1d_silu(input, weight)
    output.sum().backward()
    # CUDA skips out-of-sequence taps entirely: it does not multiply them by
    # zero padding. Unused taps have zero gradients even for nonfinite weights.
    close(output, F.silu(torch.ones(1, 1, 1)))
    torch.testing.assert_close(weight.grad.cpu()[..., :2], torch.zeros(1, 1, 2), atol=0, rtol=0)
    assert torch.isfinite(input.grad.cpu()).all()


def test_conv_current_stream_and_tensor_device():
    index = 1 if torch.npu.device_count() > 1 else 0
    device = f"npu:{index}"
    input = torch.empty(1, 5, 257, device=device, requires_grad=True)
    weight = torch.empty(257, 1, 4, device=device, requires_grad=True)
    cu = torch.empty(3, dtype=torch.int32, device=device)
    stream = torch.npu.Stream(device=index)
    with torch.npu.stream(stream):
        with torch.no_grad():
            input.fill_(0.125)
            weight.fill_(0.25)
            cu.copy_(torch.tensor([0, 2, 5], dtype=torch.int32))
        output = areno_packed_depthwise_causal_conv1d_silu(input, weight, cu)
        output.sum().backward()
    stream.synchronize()
    x, w = torch.full((1, 5, 257), 0.125), torch.full((257, 1, 4), 0.25)
    y, _, dx, dw = reference(x, w, torch.ones_like(x), [2, 3])
    for actual, expected in ((output, y), (input.grad, dx), (weight.grad, dw)):
        assert actual.device == input.device
        close(actual, expected)


def test_conv_native_contract():
    native = extension("npu")
    x = torch.ones(1, 5, 17, device="npu")
    w = torch.ones(17, 1, 4, device="npu")
    forward = native.areno_depthwise_causal_conv1d_silu_forward
    for input, weight, error in (
        (x, w.half(), "dtype"),
        (x[0], w, "rank"),
        (x, w[:-1], "channel"),
        (x, w[..., :0], "positive kernel"),
        (x[..., ::2], w[::2], "contiguous"),
    ):
        with pytest.raises(RuntimeError, match=error):
            forward(input, weight)
    _, preact = forward(x, w)
    with pytest.raises(RuntimeError, match="dtype"):
        native.areno_depthwise_causal_conv1d_silu_backward(x, x, w, preact.half())
    with pytest.raises(RuntimeError, match="shape"):
        native.areno_depthwise_causal_conv1d_silu_backward(x[:, :4], x, w, preact)
    with pytest.raises(RuntimeError, match="history shape"):
        native.areno_depthwise_causal_conv1d_silu_decode(x[:, 0], torch.ones(1, 17, 4, device="npu"), w)
    with pytest.raises(RuntimeError, match="dtype"):
        native.areno_packed_depthwise_causal_conv1d_silu_forward(x, w, torch.tensor([0, 5], device="npu"))
    with pytest.raises(RuntimeError, match="at least two"):
        native.areno_packed_depthwise_causal_conv1d_silu_forward(
            x, w, torch.tensor([0], dtype=torch.int32, device="npu")
        )
