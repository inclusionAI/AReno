from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from areno.api.trainer_config import TrainerConfig
from areno.cli.train import train_command
from areno.engine.config import OptimizerConfig
from areno.engine.modeling import build_optimizer
from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master
from areno.engine.optim.adamw_4bit import (
    _quantize_positive_4bit,
    _quantize_signed_4bit,
    _unpack_positive_4bit,
    _unpack_signed_4bit,
)


def _optimizer(param: torch.nn.Parameter, *, block_size: int = 128) -> AdamW4bit:
    return AdamW4bit(
        [param],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=max(param.numel(), 1),
        quant_block_size=block_size,
    )


def test_adamw4bit_packs_two_moments_within_storage_budget() -> None:
    param = torch.nn.Parameter(torch.zeros(8192))
    optimizer = _optimizer(param)
    param.grad = torch.linspace(-1.0, 1.0, param.numel())

    optimizer.step()

    assert optimizer.persistent_moment_bytes() / param.numel() <= 1.25
    state = optimizer._states[0]
    assert state.exp_avg_q.numel() == param.numel() // 2
    assert state.exp_avg_sq_q.numel() == param.numel() // 2
    assert state.exp_avg_scale.numel() == param.numel() // 128

    eight_bit_param = torch.nn.Parameter(torch.zeros_like(param))
    eight_bit = AdamW8bit(
        [eight_bit_param],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=param.numel(),
        quant_block_size=128,
    )
    eight_bit_param.grad = torch.ones_like(eight_bit_param)
    eight_bit.step()
    eight_bit_state = eight_bit.state_dict()["state"][0]
    eight_bit_bytes = sum(
        eight_bit_state[name].numel() * eight_bit_state[name].element_size()
        for name in ("exp_avg_q", "exp_avg_scale", "exp_avg_sq_q", "exp_avg_sq_scale")
    )
    assert optimizer.persistent_moment_bytes() <= eight_bit_bytes * 0.6


def test_adamw4bit_second_moment_mapping_excludes_zero() -> None:
    values = torch.tensor([0.0, 1.0 / 16.0, 0.5, 1.0])

    packed, scale = _quantize_positive_4bit(values)
    restored = _unpack_positive_4bit(packed, values.numel(), scale)

    assert scale.item() == 1.0
    assert restored[0].item() == pytest.approx(1.0 / 16.0)
    assert torch.all(restored > 0)
    torch.testing.assert_close(restored[1:], values[1:])


def test_adamw4bit_signed_quantizer_preserves_dynamic_map_points() -> None:
    values = torch.tensor([-0.8875, -0.2125, -0.0055, 0.0, 0.0325, 0.4375, 1.0])

    packed, scale = _quantize_signed_4bit(values)
    restored = _unpack_signed_4bit(packed, values.numel(), scale)

    torch.testing.assert_close(restored, values, rtol=1.0e-6, atol=1.0e-6)


def test_adamw4bit_checkpoint_round_trip_preserves_next_update() -> None:
    initial = torch.linspace(-0.5, 0.5, 257).to(torch.bfloat16)
    first_param = torch.nn.Parameter(initial.clone())
    first = _optimizer(first_param)
    first_param.grad = torch.linspace(-0.3, 0.7, first_param.numel()).to(torch.bfloat16)
    first.step()
    checkpoint = copy.deepcopy(first.state_dict())

    restored_param = torch.nn.Parameter(first_param.detach().clone())
    restored = _optimizer(restored_param)
    restored.load_state_dict(checkpoint)
    next_gradient = torch.linspace(0.8, -0.4, first_param.numel()).to(torch.bfloat16)
    first_param.grad = next_gradient.clone()
    restored_param.grad = next_gradient.clone()
    first.step()
    restored.step()

    torch.testing.assert_close(restored_param, first_param, rtol=0.0, atol=0.0)
    assert restored.state_dict()["state_format_version"] == 1


def test_adamw4bit_disk_offload_preserves_update(tmp_path: Path) -> None:
    initial = torch.linspace(-0.5, 0.5, 257).to(torch.bfloat16)
    candidate_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())
    candidate = _optimizer(candidate_param)
    reference = _optimizer(reference_param)
    candidate.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)

    for gradient in (
        torch.linspace(-0.4, 0.7, initial.numel()),
        torch.linspace(0.8, -0.2, initial.numel()),
    ):
        candidate_param.grad = gradient.to(torch.bfloat16)
        reference_param.grad = gradient.to(torch.bfloat16)
        candidate.step()
        reference.step()

    torch.testing.assert_close(candidate_param, reference_param, rtol=0.0, atol=0.0)
    assert all(state.offload_file is not None for state in candidate._states)
    candidate.onload_state(torch.device("cpu"))
    assert all(state.exp_avg_q is not None for state in candidate._states)
    assert not list(tmp_path.rglob("*.mmap"))


def test_adamw4bit_tracks_fp32_adamw_on_smooth_gradients() -> None:
    initial = torch.linspace(-1.0, 1.0, 1024)
    quantized_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())
    quantized = _optimizer(quantized_param)
    reference = AdamWFP32Master(
        [reference_param],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=initial.numel(),
    )

    for step in range(20):
        gradient = torch.sin(torch.linspace(-2.0, 2.0, initial.numel()) + step * 0.1)
        quantized_param.grad = gradient.clone()
        reference_param.grad = gradient.clone()
        quantized.step()
        reference.step()

    torch.testing.assert_close(quantized_param, reference_param, rtol=3.0e-3, atol=3.0e-3)


def test_adamw4bit_nonfinite_gradient_skips_only_affected_block() -> None:
    parameter = torch.nn.Parameter(torch.zeros(256))
    optimizer = _optimizer(parameter, block_size=128)
    gradient = torch.ones_like(parameter)
    gradient[4] = torch.inf
    parameter.grad = gradient

    optimizer.step()

    torch.testing.assert_close(parameter[:128], torch.zeros(128))
    assert torch.all(parameter[128:] < 0)


def test_optimizer_config_rejects_multiple_low_bit_modes() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        OptimizerConfig(adam_4bit=True, adam_8bit=True)


def test_build_optimizer_selects_adamw4bit() -> None:
    class Context:
        dp_rank = 0
        dp_size = 1
        dp_group = None

    param = torch.nn.Parameter(torch.ones(4))
    optimizer = build_optimizer([param], OptimizerConfig(adam_4bit=True), Context())

    assert isinstance(optimizer, AdamW4bit)


def test_trainer_config_propagates_adamw4bit() -> None:
    config = TrainerConfig(
        algo="sft",
        ckpt="unused",
        dataset_path="unused",
        backend="cuda",
        adam_4bit=True,
    )

    assert config.optimizer_config()["adam_4bit"] is True
    assert config.cuda_config().optimizer["adam_4bit"] is True


def test_train_cli_exposes_adamw4bit_flag() -> None:
    result = CliRunner().invoke(train_command, ["--help"])

    assert result.exit_code == 0
    assert "--adam-4bit" in result.output
