"""CUDA phase boundaries must release views as well as their storage owners."""

import weakref

import pytest
import torch
from torch import nn

from areno.engine.worker import ArenoWorker


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weights_and_buffer_aliases_survive_repeated_phase_switches():
    device = torch.device("cuda", torch.cuda.current_device())
    worker = object.__new__(ArenoWorker)
    worker.device = device
    worker.model = nn.Module()
    worker._actor_on_device = True
    worker._idle_buffers = None
    worker.model.weight = nn.Parameter(torch.arange(32, dtype=torch.float32, device=device))
    worker.model.register_buffer("quantized_weight", torch.ones(4, device=device, dtype=torch.uint8))
    worker.model.register_buffer("weight_view", worker.model.weight.detach()[2:], persistent=False)
    table = torch.arange(64, device=device).reshape(8, 8)
    worker.model.register_buffer("table", table, persistent=False)
    worker.model.register_buffer("alias", table, persistent=False)
    worker.model.register_buffer("column", table[:, 3], persistent=False)
    del table

    weight_ptr = worker.model.weight.data_ptr()
    for _ in range(2):
        old_table = weakref.ref(worker.model.table)
        worker._park_cuda_phase_buffers()
        assert old_table() is None
        assert worker.model.table.device.type == "cpu"
        assert worker.model.alias is worker.model.table
        assert worker.model.column.untyped_storage() == worker.model.table.untyped_storage()
        assert worker.model.weight.data_ptr() == weight_ptr
        assert worker.model.weight_view.untyped_storage() == worker.model.weight.untyped_storage()
        assert worker.model.quantized_weight.is_cuda
        worker._prepare_actor_onloaded()
        assert worker.model.table.device == device
        assert worker.model.table is worker.model.alias
        assert worker.model.column.untyped_storage() == worker.model.table.untyped_storage()
        torch.testing.assert_close(worker.model.column.cpu(), torch.arange(3, 64, 8))
