from types import SimpleNamespace

import torch

import areno.engine.layers.linear as linear


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
