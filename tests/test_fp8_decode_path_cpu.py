"""CPU tests for the FP8 decode path wiring.

Covers the plumbing around the FP8 linear backends (the quantize math lives in
``test_fp8_quant_cpu.py``):
  * ``_areno_linear_forward`` routes a marked weight to FP8 math only when
    gradients are disabled (decode) and ignores the payload during training,
  * ``mark_fp8_weight`` refreshes the payload in place (stable storage for
    captured decode CUDA graphs),
  * ``quantize_infer_weights_fp8`` marks every parallel-linear weight,
  * the per-tensor scale is identical across a real 2-rank TP group, so
    row-parallel partial sums stay scale-consistent.
"""

from __future__ import annotations

import multiprocessing as mp

import torch


def _weights(shape=(16, 32)) -> torch.Tensor:
    w = torch.randn(*shape)
    return (w / w.abs().amax() * 4.0).to(torch.bfloat16)


def test_linear_hook_ignores_payload_while_gradients_enabled(monkeypatch):
    import torch.nn.functional as F

    import areno.engine.layers.linear as linear_mod
    from areno.engine.layers.linear import _areno_linear_forward
    from areno.engine.quantization import mark_fp8_weight

    # Stand in for the compiled areno_linear so the test runs extension-free.
    monkeypatch.setattr(linear_mod, "areno_linear", F.linear)

    w = _weights()
    x = torch.randn(1, 4, w.shape[1], dtype=torch.bfloat16)
    ref = F.linear(x, w)

    marked = torch.nn.Parameter(w.clone())
    mark_fp8_weight(marked)
    # Training forward (grad enabled): exact bf16 math even with a payload present.
    assert torch.equal(_areno_linear_forward(x, marked, None), ref)

    with torch.no_grad():
        # Unmarked weight: unchanged path.
        assert torch.equal(_areno_linear_forward(x, w, None), ref)
        # Marked weight (decode): CPU FP8 reference math within tolerance; the
        # 3-D activation must come back with its leading dims restored.
        out = _areno_linear_forward(x, marked, None)
    assert out.shape == ref.shape
    rel = float((out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6))
    assert rel < 0.2, f"fp8 decode rel err too high: {rel}"


def test_mark_fp8_weight_refreshes_payload_in_place():
    from areno.engine.quantization import mark_fp8_weight

    p = torch.nn.Parameter(_weights((8, 16)))
    payload1, scale1 = mark_fp8_weight(p)
    assert payload1.dtype == torch.float8_e4m3fn and scale1.dtype == torch.float32

    with torch.no_grad():
        p.copy_(torch.randn(8, 16).to(torch.bfloat16))
    payload2, scale2 = mark_fp8_weight(p)
    # Same tensor objects: decode CUDA graphs keep captured pointers valid.
    assert payload2 is payload1
    assert scale2 is scale1


def test_quantize_infer_weights_fp8_marks_parallel_linears(monkeypatch):
    from types import SimpleNamespace

    import areno.engine.layers.linear as linear_mod
    from areno.engine.quantization import quantize_infer_weights_fp8

    monkeypatch.setattr(linear_mod, "get_tp_context", lambda: SimpleNamespace(rank=0, world_size=1))
    model = torch.nn.ModuleDict(
        {
            "col": linear_mod.ColumnParallelLinear(8, 8, input_grad_allreduce=False),
            "row": linear_mod.RowParallelLinear(8, 8),
            "plain": torch.nn.Linear(8, 8),
        }
    )
    count = quantize_infer_weights_fp8(model)
    assert count == 2
    assert hasattr(model.col.weight, "_areno_fp8")
    assert hasattr(model.row.weight, "_areno_fp8")
    assert not hasattr(model.plain.weight, "_areno_fp8")


def _run_tp_scale_rank(rank: int, port: int, output_queue) -> None:
    from areno.engine.parallel import context
    from areno.engine.quantization import mark_fp8_weight

    context.init_process_group(
        rank=rank,
        world_size=2,
        master_addr="127.0.0.1",
        master_port=port,
        device_id=rank,
        tp_size=2,
    )
    try:
        # Different local amax per rank: only a TP-group MAX sync makes the
        # scales (and therefore the row-parallel reduction) consistent.
        amax = 400.0 if rank == 0 else 100.0
        w = torch.randn(4, 8)
        w = (w / w.abs().amax() * amax).to(torch.bfloat16)
        payload, scale = mark_fp8_weight(torch.nn.Parameter(w))
        output_queue.put((rank, float(scale), bool(torch.isfinite(payload.float()).all())))
    finally:
        context.destroy_process_group()


def test_mark_fp8_weight_scale_is_tp_consistent_gloo() -> None:
    from areno.engine.protocol import _create_rendezvous_store

    spawn = mp.get_context("spawn")
    output_queue = spawn.Queue()
    # The coordinator holds the server store (port=0) so the resolved port is
    # genuinely reserved before the workers join as client stores.
    store = _create_rendezvous_store("127.0.0.1", 2)
    port = int(store.port)
    processes = [spawn.Process(target=_run_tp_scale_rank, args=(rank, port, output_queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = {}
    for _ in processes:
        rank, scale, finite = output_queue.get(timeout=30)
        results[rank] = (scale, finite)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    # results[rank] == (scale, payload_finite)
    scale0, scale1 = results[0][0], results[1][0]
    assert results[0][1] and results[1][1]
    # Both ranks must agree on the group-max scale.
    assert scale0 == scale1
    assert scale0 > 0


def test_scaled_mm_module_is_hopper_gated() -> None:
    import pytest as pytest_module

    from areno.accel.kernels import fp8_scaled_mm
    from areno.engine.quantization import mark_fp8_weight

    payload, _scale = mark_fp8_weight(torch.nn.Parameter(_weights((8, 16))))
    if not torch.cuda.is_available():
        # Without a GPU the Hopper path must report unavailable and refuse to run.
        assert fp8_scaled_mm.scaled_mm_available() is False
        with pytest_module.raises(RuntimeError, match="scaled_mm"):
            fp8_scaled_mm.quantized_fp8_scaled_mm(torch.randn(1, 16).to(torch.bfloat16), payload, torch.tensor(0.01))


def test_quantize_act_cpu_matches_reference() -> None:
    from areno.accel.kernels.fp8_scaled_mm import _quantize_act

    x = torch.randn(2, 32) * 3.0
    xq, scale = _quantize_act(x)
    assert xq.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    dq = xq.float() * scale
    rel = float((dq - x).abs().max() / (x.abs().max() + 1e-6))
    assert rel < 0.12, f"activation quantize rel err too high: {rel}"
