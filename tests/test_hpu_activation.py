"""Native Gaudi activation validation; requires a matching, built HPU extension.

The default lazy mode matches the extension's default build mode.
An installed bridge with missing/broken native kernels fails instead of skipping.
"""

import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

from areno.accel import activations
from areno.accel._extension import configure_hpu_kernel_library, extension

OPS = ("silu", "sigmoid", "softplus", "silu_and_mul", "gelu_tanh_and_mul")
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
TOLERANCES = {
    torch.float32: {"atol": 2e-5, "rtol": 2e-4},
    torch.bfloat16: {"atol": 1.5e-2, "rtol": 2e-2},
    torch.float16: {"atol": 1.5e-3, "rtol": 5e-3},
}


@pytest.fixture(scope="module", autouse=True)
def hpu_core():
    if importlib.util.find_spec("habana_frameworks") is None:
        pytest.skip("Gaudi PyTorch bridge and HPU hardware are required")
    configure_hpu_kernel_library()
    assert os.environ.get("PT_HPU_LAZY_MODE") in {"0", "1"}
    import habana_frameworks.torch.core as core

    assert torch.hpu.is_available(), "The bridge is installed but no HPU is available"
    extension("hpu")
    return core


def reference(name, x):
    if name == "silu":
        return F.silu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "softplus":
        return F.softplus(x)
    gate, up = x.chunk(2, dim=-1)
    if name == "silu_and_mul":
        return F.silu(gate) * up
    return F.gelu(gate, approximate="tanh") * up


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [1, 63, 64, 65, 127, 128, 129, 257])
@pytest.mark.parametrize("transposed", [False, True], ids=["contiguous", "transposed"])
def test_forward_backward_and_vector_tails(hpu_core, name, dtype, width, transposed):
    channels = 2 if name.endswith("and_mul") else 1
    generator = torch.Generator().manual_seed(71)
    shape = (2, 3, channels * width)
    values = (torch.randn(shape, generator=generator) * 3).to(dtype)
    source = values.to("hpu")
    if transposed:
        source = source.transpose(0, 1)
        values = values.transpose(0, 1)
    source.requires_grad_()
    expected_input = values.float().detach().requires_grad_()
    expected = reference(name, expected_input)
    actual = getattr(activations, f"areno_{name}")(source)
    grad = torch.randn(expected.shape, generator=generator).to(dtype)
    actual.backward(grad.to("hpu"))
    expected.backward(grad.float())
    expected_grad = expected_input.grad
    if name == "sigmoid":
        # CUDA and HPU backward both consume the saved, storage-dtype output.
        saved = expected.detach().to(dtype).float()
        expected_grad = grad.float() * saved * (1 - saved)
    hpu_core.mark_step()
    torch.hpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.detach().to(dtype), **TOLERANCES[dtype])
    torch.testing.assert_close(source.grad.cpu(), expected_grad.to(dtype), **TOLERANCES[dtype])


@pytest.mark.parametrize("name", ["silu_and_mul", "gelu_tanh_and_mul"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_out_preserves_alias_and_strides(hpu_core, name, dtype):
    values = torch.linspace(-3, 3, 3 * 130).reshape(3, 130).to(dtype)
    source = values.to("hpu")
    storage = torch.full((3, 130), -42, device="hpu", dtype=dtype)
    out = storage[:, ::2]
    result = getattr(activations, f"areno_{name}")(source, out=out)
    assert result is out
    hpu_core.mark_step()
    torch.hpu.synchronize()
    expected = reference(name, values.float()).to(dtype)
    torch.testing.assert_close(result.cpu(), expected, **TOLERANCES[dtype])
    torch.testing.assert_close(storage[:, 1::2].cpu(), torch.full_like(expected, -42))


@pytest.mark.parametrize("name", OPS)
def test_empty_input(hpu_core, name):
    x = torch.empty((2, 0), device="hpu", requires_grad=True)
    output = getattr(activations, f"areno_{name}")(x)
    output.backward(torch.empty_like(output))
    hpu_core.mark_step()
    assert output.shape == x.shape
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("name", ["silu", "sigmoid", "softplus"])
def test_scalar_and_extreme_values(hpu_core, name):
    op = getattr(activations, f"areno_{name}")
    for values in (torch.tensor(0.75), torch.tensor([-80.0, -30.0, -20.0, -1e-6, 0.0, 20.0, 20.001, 30.0, 80.0])):
        x = values.to("hpu").requires_grad_()
        ref = values.clone().requires_grad_()
        actual = op(x)
        expected = reference(name, ref)
        actual.backward(torch.ones_like(actual))
        expected.backward(torch.ones_like(expected))
        hpu_core.mark_step()
        torch.testing.assert_close(actual.cpu(), expected.detach(), **TOLERANCES[torch.float32])
        torch.testing.assert_close(x.grad.cpu(), ref.grad, **TOLERANCES[torch.float32])
