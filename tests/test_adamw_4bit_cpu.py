from __future__ import annotations

import copy
import multiprocessing as mp
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from click.testing import CliRunner

from areno.api.trainer_config import TrainerConfig
from areno.cli.train import train_command
from areno.engine.config import OptimizerConfig
from areno.engine.modeling import build_optimizer
from areno.engine.optim import AdamW4bit, AdamW8bit, AdamWFP32Master
from areno.engine.optim.adamw_4bit import (
    _factored_state_numel_for_parameter,
    _quantize_positive_4bit,
    _quantize_signed_4bit,
    _unpack_positive_4bit,
    _unpack_signed_4bit,
)


def _optimizer(param: torch.nn.Parameter, *, block_size: int = 128, bucket_numel: int | None = None) -> AdamW4bit:
    return AdamW4bit(
        [param],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=max(param.numel(), 1) if bucket_numel is None else bucket_numel,
        quant_block_size=block_size,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _expected_first_step_factors(gradient: torch.Tensor, beta2: float = 0.99) -> torch.Tensor:
    matrix = gradient.float().reshape(gradient.shape[0], -1).square().mul(1.0 - beta2)
    return torch.cat((matrix.mean(dim=1), matrix.mean(dim=0)))


def _gloo_factored_worker(rank: int, port: int, output_queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        parameter = torch.nn.Parameter(torch.zeros(5, 7))
        optimizer = AdamW4bit(
            [parameter],
            lr=3.0e-4,
            betas=(0.9, 0.99),
            weight_decay=0.0,
            bucket_numel=16,
            quant_block_size=128,
            dp_rank=rank,
            dp_size=2,
            dp_group=dist.group.WORLD,
        )
        parameter.grad = torch.arange(1, 36, dtype=torch.float32).reshape_as(parameter) + rank * 3.0
        optimizer.reduce_scatter_gradients()
        gradient_dtype = optimizer.buckets[0].grad_shard.dtype
        optimizer.step()
        output_queue.put(
            (
                rank,
                parameter.detach().tolist(),
                optimizer._factored_second_moments[id(parameter)].tolist(),
                gradient_dtype == torch.bfloat16,
                optimizer.buckets[0].refs[0].shard_start,
                optimizer.buckets[0].refs[0].shard_numel,
            )
        )
    finally:
        dist.destroy_process_group()


def test_adamw4bit_vector_packs_two_moments_within_storage_budget() -> None:
    parameter = torch.nn.Parameter(torch.zeros(8192))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.linspace(-1.0, 1.0, parameter.numel())

    optimizer.step()

    state = optimizer._states[0]
    assert optimizer.persistent_moment_bytes() / parameter.numel() <= 1.25
    assert state.exp_avg_q.numel() == parameter.numel() // 2
    assert state.exp_avg_sq_q.numel() == parameter.numel() // 2
    assert state.exp_avg_scale.numel() == parameter.numel() // 128

    eight_bit_parameter = torch.nn.Parameter(torch.zeros_like(parameter))
    eight_bit = AdamW8bit(
        [eight_bit_parameter],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=parameter.numel(),
        quant_block_size=128,
    )
    eight_bit_parameter.grad = torch.ones_like(eight_bit_parameter)
    eight_bit.step()
    assert optimizer.persistent_moment_bytes() <= eight_bit.state_memory_metrics()["total_bytes"] * 0.6


def test_adamw4bit_matrix_stores_only_packed_first_moment_and_factors() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1024, 1024))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    state = optimizer._states[0]
    factors = optimizer._factored_second_moments[id(parameter)]
    assert factors is not None and factors.numel() == 2048
    assert state.exp_avg_q.numel() == parameter.numel() // 2
    assert state.exp_avg_sq_q.numel() == 0
    assert state.exp_avg_sq_scale.numel() == 0
    assert optimizer.persistent_moment_bytes() / parameter.numel() < 0.55


def test_adamw4bit_uses_bf16_gradient_shards_without_changing_adam8_default() -> None:
    four_bit_parameter = torch.nn.Parameter(torch.ones(257, dtype=torch.bfloat16))
    four_bit = _optimizer(four_bit_parameter)
    four_bit_parameter.grad = torch.linspace(-1.0, 1.0, 257).to(torch.bfloat16)
    four_bit.reduce_scatter_gradients()

    eight_bit_parameter = torch.nn.Parameter(torch.ones(257, dtype=torch.bfloat16))
    eight_bit = AdamW8bit(
        [eight_bit_parameter],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=257,
    )
    eight_bit_parameter.grad = torch.linspace(-1.0, 1.0, 257).to(torch.bfloat16)
    eight_bit.reduce_scatter_gradients()

    assert four_bit.stream_gradient_shards is True
    assert four_bit_parameter.grad is None
    assert all(bucket.grad_shard.dtype == torch.bfloat16 for bucket in four_bit.buckets)
    assert eight_bit.stream_gradient_shards is False
    assert all(bucket.grad_shard.dtype == torch.float32 for bucket in eight_bit.buckets)


def test_adamw4bit_initializes_and_releases_state_per_parameter() -> None:
    parameters = [torch.nn.Parameter(torch.zeros(32, 32)) for _ in range(2)]
    optimizer = AdamW4bit(
        parameters,
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=1024,
        quant_block_size=128,
    )
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    live_gradients_at_initialization: list[int] = []
    ensure_bucket_state = optimizer._ensure_bucket_state

    def tracked_ensure_bucket_state(bucket, state) -> None:
        live_gradients_at_initialization.append(sum(parameter.grad is not None for parameter in parameters))
        ensure_bucket_state(bucket, state)

    optimizer._ensure_bucket_state = tracked_ensure_bucket_state
    optimizer.step()

    assert live_gradients_at_initialization == [2, 1]
    assert all(parameter.grad is None for parameter in parameters)


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


def test_adamw4bit_vector_checkpoint_round_trip_preserves_next_update() -> None:
    initial = torch.linspace(-0.5, 0.5, 257).to(torch.bfloat16)
    first_parameter = torch.nn.Parameter(initial.clone())
    first = _optimizer(first_parameter)
    first_parameter.grad = torch.linspace(-0.3, 0.7, first_parameter.numel()).to(torch.bfloat16)
    first.step()
    checkpoint = copy.deepcopy(first.state_dict())

    restored_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    restored = _optimizer(restored_parameter)
    restored.load_state_dict(checkpoint)
    next_gradient = torch.linspace(0.8, -0.4, first_parameter.numel()).to(torch.bfloat16)
    first_parameter.grad = next_gradient.clone()
    restored_parameter.grad = next_gradient.clone()
    first.step()
    restored.step()

    torch.testing.assert_close(restored_parameter, first_parameter, rtol=0.0, atol=0.0)
    assert restored.state_dict()["state_format_version"] == 3


def test_adamw4bit_factored_checkpoint_round_trip_preserves_next_update() -> None:
    initial = torch.linspace(-0.5, 0.5, 35).reshape(5, 7).to(torch.bfloat16)
    first_parameter = torch.nn.Parameter(initial.clone())
    first = _optimizer(first_parameter)
    first_parameter.grad = torch.linspace(-0.3, 0.7, 35).reshape_as(initial).to(torch.bfloat16)
    first.step()
    checkpoint = copy.deepcopy(first.state_dict())

    restored_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    restored = _optimizer(restored_parameter)
    restored.load_state_dict(checkpoint)
    next_gradient = torch.linspace(0.8, -0.4, 35).reshape_as(initial).to(torch.bfloat16)
    first_parameter.grad = next_gradient.clone()
    restored_parameter.grad = next_gradient.clone()
    first.step()
    restored.step()

    torch.testing.assert_close(restored_parameter, first_parameter, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored._factored_second_moments[id(restored_parameter)],
        first._factored_second_moments[id(first_parameter)],
    )


def test_adamw4bit_rejects_legacy_optimizer_state_format() -> None:
    parameter = torch.nn.Parameter(torch.zeros(8))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    checkpoint = optimizer.state_dict()
    checkpoint["state_format_version"] = 2

    with pytest.raises(ValueError, match="unsupported AdamW4bit state format"):
        _optimizer(torch.nn.Parameter(torch.zeros(8))).load_state_dict(checkpoint)


def test_adamw4bit_disk_offload_preserves_vector_update(tmp_path: Path) -> None:
    initial = torch.linspace(-0.5, 0.5, 257).to(torch.bfloat16)
    candidate_parameter = torch.nn.Parameter(initial.clone())
    reference_parameter = torch.nn.Parameter(initial.clone())
    candidate = _optimizer(candidate_parameter)
    reference = _optimizer(reference_parameter)
    candidate.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)

    for gradient in (
        torch.linspace(-0.4, 0.7, initial.numel()),
        torch.linspace(0.8, -0.2, initial.numel()),
    ):
        candidate_parameter.grad = gradient.to(torch.bfloat16)
        reference_parameter.grad = gradient.to(torch.bfloat16)
        candidate.step()
        reference.step()

    torch.testing.assert_close(candidate_parameter, reference_parameter, rtol=0.0, atol=0.0)
    assert all(state.offload_file is not None for state in candidate._states)
    candidate.onload_state(torch.device("cpu"))
    assert all(state.exp_avg_q is not None for state in candidate._states)
    assert not list(tmp_path.rglob("*.mmap"))


def test_adamw4bit_disk_offload_preserves_factored_update(tmp_path: Path) -> None:
    initial = torch.linspace(-0.5, 0.5, 35).reshape(5, 7).to(torch.bfloat16)
    candidate_parameter = torch.nn.Parameter(initial.clone())
    reference_parameter = torch.nn.Parameter(initial.clone())
    candidate = _optimizer(candidate_parameter)
    reference = _optimizer(reference_parameter)
    candidate.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=1)

    for gradient in (
        torch.linspace(-0.4, 0.7, 35).reshape_as(initial),
        torch.linspace(0.8, -0.2, 35).reshape_as(initial),
    ):
        candidate_parameter.grad = gradient.to(torch.bfloat16)
        reference_parameter.grad = gradient.to(torch.bfloat16)
        candidate.step()
        reference.step()

    candidate.onload_state(torch.device("cpu"))
    torch.testing.assert_close(candidate_parameter, reference_parameter, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        candidate._factored_second_moments[id(candidate_parameter)],
        reference._factored_second_moments[id(reference_parameter)],
    )
    assert not list(tmp_path.rglob("*.mmap"))


def test_adamw4bit_vector_tracks_fp32_adamw_on_smooth_gradients() -> None:
    initial = torch.linspace(-1.0, 1.0, 1024)
    quantized_parameter = torch.nn.Parameter(initial.clone())
    reference_parameter = torch.nn.Parameter(initial.clone())
    quantized = _optimizer(quantized_parameter)
    reference = AdamWFP32Master(
        [reference_parameter],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=initial.numel(),
    )

    for step in range(20):
        gradient = torch.sin(torch.linspace(-2.0, 2.0, initial.numel()) + step * 0.1)
        quantized_parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        quantized.step()
        reference.step()

    torch.testing.assert_close(quantized_parameter, reference_parameter, rtol=3.0e-3, atol=3.0e-3)


def test_adamw4bit_nonfinite_vector_gradient_skips_only_affected_block() -> None:
    parameter = torch.nn.Parameter(torch.zeros(256))
    optimizer = _optimizer(parameter, block_size=128)
    gradient = torch.ones_like(parameter)
    gradient[4] = torch.inf
    parameter.grad = gradient

    optimizer.step()

    torch.testing.assert_close(parameter[:128], torch.zeros(128))
    assert torch.all(parameter[128:] < 0)


def test_adamw4bit_nonfinite_matrix_gradient_skips_whole_parameter() -> None:
    parameter = torch.nn.Parameter(torch.zeros(16, 16))
    optimizer = _optimizer(parameter)
    gradient = torch.ones_like(parameter)
    gradient[4, 4] = torch.inf
    parameter.grad = gradient

    optimizer.step()

    torch.testing.assert_close(parameter, torch.zeros_like(parameter))
    torch.testing.assert_close(
        optimizer._factored_second_moments[id(parameter)],
        torch.zeros(parameter.shape[0] + parameter.shape[1]),
    )


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 5), (3, 5, 7)])
def test_factored_statistics_match_row_column_means(shape: tuple[int, ...]) -> None:
    parameter = torch.nn.Parameter(torch.zeros(shape))
    optimizer = _optimizer(parameter, bucket_numel=7)
    gradient = torch.arange(1, parameter.numel() + 1, dtype=torch.float32).reshape(shape)
    parameter.grad = gradient.clone()

    optimizer.step()

    torch.testing.assert_close(
        optimizer._factored_second_moments[id(parameter)],
        _expected_first_step_factors(gradient),
    )


def test_factored_statistics_combine_partial_parameter_chunks() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2048, 2049))
    optimizer = _optimizer(parameter, bucket_numel=1)

    refs = [ref for bucket in optimizer.buckets for ref in bucket.refs]
    assert len(optimizer.buckets) == 2
    assert [ref.param_start for ref in refs] == [0, 4 * 1024 * 1024]
    assert sum(ref.numel for ref in refs) == parameter.numel()
    factors = optimizer._ensure_factored_second_moment(parameter)
    assert factors.numel() == 2048 + 2049
    assert _factored_state_numel_for_parameter(parameter) == factors.numel()


def test_adamw4bit_mixed_matrix_vector_bucket_uses_disjoint_state_layouts() -> None:
    matrix = torch.nn.Parameter(torch.zeros(8, 8))
    vector = torch.nn.Parameter(torch.zeros(17))
    optimizer = AdamW4bit(
        [matrix, vector],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=1024,
        quant_block_size=128,
    )
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)

    optimizer.step()

    state = optimizer._states[0]
    assert state.exp_avg_q.numel() == 32 + 9
    assert state.exp_avg_sq_q.numel() == 9
    assert state.exp_avg_sq_scale.numel() == 1


def test_adamw4bit_factored_memory_metrics_match_resident_tensors() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1024, 1024))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    metrics = optimizer.state_memory_metrics()
    assert metrics["total_bytes"] == optimizer.persistent_moment_bytes()
    assert metrics["quantized_state_bytes"] == parameter.numel() // 2
    assert metrics["scale_metadata_bytes"] == (parameter.numel() // 128 + 2048) * 4


def test_adamw4bit_clear_state_drops_factored_state() -> None:
    parameter = torch.nn.Parameter(torch.zeros(5, 7))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert optimizer._factored_second_moments[id(parameter)] is not None
    optimizer.clear_state()

    assert optimizer._factored_second_moments[id(parameter)] is None
    assert all(state.step == 0 and state.exp_avg_q is None for state in optimizer._states)


def test_real_gloo_factored_statistics_match_unsharded_reference() -> None:
    spawn = mp.get_context("spawn")
    output_queue = spawn.Queue()
    port = _free_port()
    processes = [spawn.Process(target=_gloo_factored_worker, args=(rank, port, output_queue)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = dict((item[0], item[1:]) for item in (output_queue.get(timeout=30) for _ in processes))
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)

    rank0_model, rank0_factors, rank0_bf16, rank0_start, rank0_count = results[0]
    rank1_model, rank1_factors, rank1_bf16, rank1_start, rank1_count = results[1]
    averaged_gradient = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7) + 1.5
    expected_factors = _expected_first_step_factors(averaged_gradient)
    assert rank0_model == rank1_model
    assert rank0_bf16 and rank1_bf16
    torch.testing.assert_close(torch.tensor(rank0_factors), expected_factors)
    torch.testing.assert_close(torch.tensor(rank1_factors), expected_factors)
    assert (rank0_start, rank0_count) == (0, 18)
    assert (rank1_start, rank1_count) == (18, 17)


def test_optimizer_config_rejects_multiple_low_bit_modes() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        OptimizerConfig(adam_4bit=True, adam_8bit=True)


def test_build_optimizer_selects_adamw4bit() -> None:
    class Context:
        dp_rank = 0
        dp_size = 1
        dp_group = None

    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = build_optimizer([parameter], OptimizerConfig(adam_4bit=True), Context())

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
