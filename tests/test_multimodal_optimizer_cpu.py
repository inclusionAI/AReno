from __future__ import annotations

import pytest
import torch

from areno.engine.optim import AdamW8bit, AdamWFP32Master


@pytest.mark.parametrize("optimizer_type", [AdamWFP32Master, AdamW8bit])
def test_multimodal_parameter_learning_rates_are_independent(optimizer_type):
    text = torch.nn.Parameter(torch.tensor([1.0]))
    tower = torch.nn.Parameter(torch.tensor([1.0]))
    projector = torch.nn.Parameter(torch.tensor([1.0]))
    tower._areno_lr = 0.01
    projector._areno_lr = 0.001
    optimizer = optimizer_type(
        [text, tower, projector],
        lr=0.1,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        bucket_numel=16,
    )
    for param in (text, tower, projector):
        param.grad = torch.ones_like(param)

    optimizer.step()

    assert text.item() == pytest.approx(0.9, abs=2e-3)
    assert tower.item() == pytest.approx(0.99, abs=2e-3)
    assert projector.item() == pytest.approx(0.999, abs=2e-3)
