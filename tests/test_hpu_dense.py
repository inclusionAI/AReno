"""Native RMSNorm and MME linear checks; a working Gaudi installation is required."""

import pytest
import torch
import torch.nn.functional as F

from areno.accel.linear import areno_linear
from areno.accel.normalization import areno_optional_scale_rmsnorm, areno_rmsnorm, areno_rmsnorm_silu_gate
from tests.test_hpu_activation import DTYPES, TOLERANCES
from tests.test_hpu_activation import hpu_core as hpu_core


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["scaled", "unscaled", "gate"])
@pytest.mark.parametrize("shape", [(2, 3, 1), (2, 3, 65), (2, 3, 129), (2, 0, 65)])
def test_rmsnorm_forward_backward(hpu_core, dtype, kind, shape):
    rng = torch.Generator().manual_seed(16)
    values = torch.randn(shape, generator=rng).to(dtype).transpose(0, 1)
    gate = torch.randn(shape, generator=rng).to(dtype).transpose(0, 1)
    weight = torch.randn(shape[-1], generator=rng)
    actual_inputs = [x.to("hpu").requires_grad_() for x in (values, gate, weight)]
    ref_inputs = [x.float().detach().requires_grad_() for x in (values, gate, weight)]
    x, g, w = actual_inputs
    rx, rg, rw = ref_inputs
    expected = rx * torch.rsqrt(rx.square().mean(-1, keepdim=True) + 1e-6)
    if kind == "gate":
        actual = areno_rmsnorm_silu_gate(x, g, w, 1e-6)
        expected = expected * F.silu(rg) * rw
    elif kind == "scaled":
        actual = areno_rmsnorm(x, w, 1e-6)
        expected = expected * rw
    else:
        actual = areno_optional_scale_rmsnorm(x, None, 1e-6)
    grad = torch.randn(expected.shape, generator=rng).to(dtype)
    actual.backward(grad.to("hpu"))
    expected.backward(grad.float())
    hpu_core.mark_step()
    torch.hpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.detach().to(dtype), **TOLERANCES[dtype])
    for source, ref in zip(actual_inputs, ref_inputs):
        if ref.grad is not None:
            torch.testing.assert_close(source.grad.cpu(), ref.grad.to(source.dtype), **TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2, 3, 65), (2, 0, 65)])
@pytest.mark.parametrize("trainable", [(True, True, True), (False, True, False), (True, False, False)])
def test_linear_forward_backward(hpu_core, dtype, shape, trainable):
    rng = torch.Generator().manual_seed(92)
    values = torch.randn(shape, generator=rng).to(dtype).transpose(0, 1)
    weight = torch.randn(33, shape[-1], generator=rng).to(dtype)
    bias = torch.randn(33, generator=rng).to(dtype)
    source = [v.to("hpu").requires_grad_(need) for v, need in zip((values, weight, bias), trainable)]
    ref = [v.float().detach().requires_grad_(need) for v, need in zip((values, weight, bias), trainable)]
    actual = areno_linear(source[0], source[1], source[2] if trainable[2] else None)
    expected = F.linear(ref[0], ref[1], ref[2] if trainable[2] else None)
    grad = torch.randn(expected.shape, generator=rng).to(dtype)
    actual.backward(grad.to("hpu"))
    expected.backward(grad.float())
    hpu_core.mark_step()
    torch.hpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.detach().to(dtype), **TOLERANCES[dtype])
    for tensor, reference in zip(source, ref):
        if reference.grad is not None:
            torch.testing.assert_close(tensor.grad.cpu(), reference.grad.to(dtype), **TOLERANCES[dtype])
