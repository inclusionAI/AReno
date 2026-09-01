from __future__ import annotations

import torch

from areno.engine.optim import AdamW8bit
from areno.engine.optim.adamw_8bit import _dequantize_positive, _quantize_positive


def test_adamw8bit_uses_parameter_local_block_scales() -> None:
    first = torch.nn.Parameter(torch.zeros(9))
    second = torch.nn.Parameter(torch.zeros(3))
    optimizer = AdamW8bit(
        [first, second],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=32,
        quant_block_size=4,
    )
    first.grad = torch.tensor([1000.0, 1.0, -1.0, 0.5, 0.25, -0.5, 0.75, -0.25, 0.125])
    second.grad = torch.tensor([0.01, -0.02, 0.03])

    optimizer.step()

    state = optimizer.state_dict()["state"][0]
    assert state["exp_avg_scale"].shape == (4,)
    assert state["exp_avg_sq_scale"].shape == (4,)
    assert state["exp_avg_scale"][0] > 1000 * state["exp_avg_scale"][-1]
    assert state["exp_avg_sq_scale"][0] > 1000 * state["exp_avg_sq_scale"][-1]
    torch.testing.assert_close(second, torch.tensor([-1.0e-3, 1.0e-3, -1.0e-3]), atol=1.0e-6, rtol=0.0)


def test_adamw8bit_second_moment_uses_full_linear_range_per_block() -> None:
    values = torch.tensor([0.0, 1.0 / 255.0, 128.0 / 255.0, 1.0])

    quantized, scale = _quantize_positive(values)
    restored = _dequantize_positive(quantized, scale)

    torch.testing.assert_close(restored, values)


def test_adamw8bit_same_lr_does_not_amplify_constant_gradient_step() -> None:
    parameter = torch.nn.Parameter(torch.ones(4096))
    optimizer = AdamW8bit(
        [parameter],
        lr=1.0e-3,
        betas=(0.0, 0.0),
        weight_decay=0.0,
        quant_block_size=2048,
    )
    parameter.grad = torch.full_like(parameter, 0.25)

    optimizer.step()

    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.999), rtol=0.0, atol=1.0e-6)
