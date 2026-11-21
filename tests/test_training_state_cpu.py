import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from areno.engine.checkpoints.training_state import load_worker_state, save_worker_state
from areno.engine.config import OptimizerConfig
from areno.engine.optim import AdamWFP32Master


def _worker():
    model = torch.nn.Linear(3, 2)
    config = OptimizerConfig(lr=0.01, betas=(0.9, 0.99))
    optimizer = AdamWFP32Master(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        config=SimpleNamespace(optimizer=config),
        _global_step=0,
        adapter_registry=None,
        device=torch.device("cpu"),
    )


def _step(worker):
    loss = worker.model(torch.tensor([[0.2, 0.5, 1.0]])).square().sum()
    loss.backward()
    worker.optimizer.step()
    worker.optimizer.zero_grad(set_to_none=True)
    worker._global_step += 1


def test_worker_state_restores_optimizer_rng_and_continues_training(tmp_path):
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    original = _worker()
    _step(original)
    path = tmp_path / "rank.pt"
    save_worker_state(original, str(path))
    expected_rng = (random.random(), np.random.random(), torch.rand(1))
    _step(original)

    restored = _worker()
    load_worker_state(restored, str(path))
    actual_rng = (random.random(), np.random.random(), torch.rand(1))
    assert actual_rng[:2] == expected_rng[:2]
    torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
    assert restored._global_step == 1
    _step(restored)
    for actual, expected in zip(restored.model.parameters(), original.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_state = restored.optimizer.state_dict()
    expected_state = original.optimizer.state_dict()
    assert actual_state["state"][0]["step"] == 2
    for actual, expected in zip(actual_state["state"], expected_state["state"], strict=True):
        assert actual["step"] == expected["step"]
        torch.testing.assert_close(actual["exp_avg"], expected["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(actual["exp_avg_sq"], expected["exp_avg_sq"], rtol=0, atol=0)


def test_worker_state_rejects_changed_layout_and_pending_gradients(tmp_path):
    worker = _worker()
    path = tmp_path / "rank.pt"
    worker.model(torch.ones(1, 3)).sum().backward()
    with pytest.raises(ValueError, match="uncommitted gradients"):
        save_worker_state(worker, str(path))
    worker.optimizer.zero_grad(set_to_none=True)
    save_worker_state(worker, str(path))
    other = _worker()
    other.config.optimizer.weight_decay = 0.4
    with pytest.raises(ValueError, match="layout or optimizer configuration"):
        load_worker_state(other, str(path))
