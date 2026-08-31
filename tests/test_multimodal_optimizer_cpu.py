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


def test_cuda_adam8bit_matches_fp32_master_bias_corrected_updates():
    reference_param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    quantized_param = torch.nn.Parameter(reference_param.detach().clone())
    torch_param = torch.nn.Parameter(reference_param.detach().clone())
    kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01,
        "bucket_numel": 16,
    }
    reference = AdamWFP32Master([reference_param], **kwargs)
    quantized = AdamW8bit([quantized_param], **kwargs)
    torch_reference = torch.optim.AdamW(
        [torch_param],
        lr=kwargs["lr"],
        betas=kwargs["betas"],
        weight_decay=kwargs["weight_decay"],
    )

    for gradient in (0.25, -0.5, 0.125, 1.0):
        reference_param.grad = torch.tensor([gradient])
        quantized_param.grad = torch.tensor([gradient])
        torch_param.grad = torch.tensor([gradient])
        reference.step()
        quantized.step()
        torch_reference.step()
        torch.testing.assert_close(quantized_param, reference_param, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(quantized_param, torch_param, atol=1e-6, rtol=1e-6)
