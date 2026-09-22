"""CPU tests for the FP8 decode routing path.

The FP8 backend needs an FP8-capable GPU, so it is stubbed here and these tests
cover the routing contract instead (the quantize math lives in
test_fp8_quant_cpu.py, and the real backend in test_fp8_decode_path_gpu.py):
  * a payload-carrying weight takes the FP8 path only while a decode scope is
    open, so scoring and training forwards keep bf16 math,
  * unmarked weights never enter the FP8 path,
  * mark_fp8_weight refreshes the payload in place, keeping storage stable for
    captured decode CUDA graphs,
  * quantize_infer_weights_fp8 marks every parallel-linear weight.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import areno.engine.layers.linear as linear_module
from areno.engine.quantization import (
    PAYLOAD_ATTR,
    SCALE_ATTR,
    mark_fp8_weight,
    quantize_infer_weights_fp8,
    set_fp8_decode_active,
)


@pytest.fixture(autouse=True)
def closed_decode_scope():
    """Keep the process-global decode scope closed around every test."""

    set_fp8_decode_active(False)
    try:
        yield
    finally:
        set_fp8_decode_active(False)


def _weights(shape=(16, 32)) -> torch.Tensor:
    torch.manual_seed(2)
    w = torch.randn(*shape)
    return (w / w.abs().amax() * 4.0).to(torch.bfloat16)


class _StubBackend:
    """Records calls so the routing decision is observable without a GPU."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor, fp8_weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(*x.shape[:-1], fp8_weight.shape[0], dtype=torch.bfloat16, device=x.device)

    @staticmethod
    def available(device: torch.device) -> bool:
        return True


@pytest.fixture
def stub_backend(monkeypatch):
    backend = _StubBackend()
    monkeypatch.setattr(linear_module, "_FP8_BACKEND", (backend, backend.available))
    # Stand in for the compiled areno_linear so the bf16 path runs extension-free.
    monkeypatch.setattr(linear_module, "areno_linear", F.linear)
    return backend


def test_decode_scope_alone_decides_fp8_routing(stub_backend):
    weight = torch.nn.Parameter(_weights())
    mark_fp8_weight(weight)
    x = torch.randn(1, 4, weight.shape[1], dtype=torch.bfloat16)
    reference = F.linear(x, weight)

    # Scope closed: exact bf16 math, with or without gradients enabled.
    assert torch.equal(linear_module._areno_linear_forward(x, weight, None), reference)
    with torch.no_grad():
        assert torch.equal(linear_module._areno_linear_forward(x, weight, None), reference)
    assert stub_backend.calls == 0

    # Scope open: the FP8 backend takes over and restores the leading dims.
    set_fp8_decode_active(True)
    out = linear_module._areno_linear_forward(x, weight, None)
    assert stub_backend.calls == 1
    assert out.shape == reference.shape


def test_unmarked_weight_never_takes_the_fp8_path(stub_backend):
    weight = torch.nn.Parameter(_weights())
    x = torch.randn(2, weight.shape[1], dtype=torch.bfloat16)
    set_fp8_decode_active(True)
    assert torch.equal(linear_module._areno_linear_forward(x, weight, None), F.linear(x, weight))
    assert stub_backend.calls == 0


def test_mark_fp8_weight_refreshes_payload_in_place():
    parameter = torch.nn.Parameter(_weights((8, 16)))
    payload, scale = mark_fp8_weight(parameter)
    assert payload.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32

    with torch.no_grad():
        parameter.copy_(torch.randn(8, 16).to(torch.bfloat16))
    refreshed, refreshed_scale = mark_fp8_weight(parameter)
    # Same tensor objects: decode CUDA graphs keep their captured pointers valid.
    assert refreshed is payload
    assert refreshed_scale is scale


def test_quantize_infer_weights_fp8_marks_parallel_linears(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(linear_module, "get_tp_context", lambda: SimpleNamespace(rank=0, world_size=1))
    model = torch.nn.ModuleDict(
        {
            "col": linear_module.ColumnParallelLinear(8, 8, input_grad_allreduce=False),
            "row": linear_module.RowParallelLinear(8, 8),
            "plain": torch.nn.Linear(8, 8),
        }
    )
    count = quantize_infer_weights_fp8(model)
    assert count == 2
    assert hasattr(model.col.weight, PAYLOAD_ATTR)
    assert hasattr(model.row.weight, PAYLOAD_ATTR)
    assert hasattr(model.col.weight, SCALE_ATTR)
    assert not hasattr(model.plain.weight, PAYLOAD_ATTR)
