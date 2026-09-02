from types import SimpleNamespace

import torch

import areno.engine.layers.linear as linear
import areno.engine.parallel.collectives as collectives


def test_column_parallel_can_move_non_sp_gradient_boundary(monkeypatch):
    calls = []
    monkeypatch.setattr(linear, "get_tp_context", lambda: SimpleNamespace(rank=0, world_size=2))
    monkeypatch.setattr(linear, "is_sequence_parallel_active", lambda: False)
    monkeypatch.setattr(linear, "copy_to_tensor_parallel_region", lambda x: calls.append("copy") or x)
    monkeypatch.setattr(linear, "gather_from_sequence_parallel_region", lambda x: calls.append("gather") or x)
    layer = linear.ColumnParallelLinear(4, 8, input_grad_allreduce=False)

    layer(torch.zeros(1, 2, 4))

    assert calls == []


def test_column_parallel_always_gathers_sequence_parallel_input(monkeypatch):
    calls = []
    monkeypatch.setattr(linear, "get_tp_context", lambda: SimpleNamespace(rank=0, world_size=2))
    monkeypatch.setattr(linear, "is_sequence_parallel_active", lambda: True)
    monkeypatch.setattr(linear, "copy_to_tensor_parallel_region", lambda x: calls.append("copy") or x)
    monkeypatch.setattr(linear, "gather_from_sequence_parallel_region", lambda x: calls.append("gather") or x)
    layer = linear.ColumnParallelLinear(4, 8, input_grad_allreduce=False)

    layer(torch.zeros(1, 2, 4))

    assert calls == ["gather"]


def test_reduce_scatter_reduces_matching_sequence_chunks(monkeypatch):
    rank = 1
    world_size = 2
    x = torch.arange(8, dtype=torch.float32).view(1, 4, 2)

    def fake_reduce_scatter(output, sequence_first, *, group):
        assert group == "tp"
        assert sequence_first.shape == (4, 1, 2)
        chunk = sequence_first.shape[0] // world_size
        output.copy_(sequence_first[rank * chunk : (rank + 1) * chunk] * world_size)

    monkeypatch.setattr(collectives.dist, "reduce_scatter_tensor", fake_reduce_scatter)

    actual = collectives._reduce_scatter_sequence(x, "tp", rank, world_size)

    torch.testing.assert_close(actual, x[:, 2:] * world_size)
