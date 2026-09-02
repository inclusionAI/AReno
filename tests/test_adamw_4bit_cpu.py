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
    _accumulate_rank1_maxima,
    _quantize_positive_4bit,
    _quantize_positive_4bit_elementwise,
    _quantize_signed_4bit,
    _rank1_element_scales,
    _unpack_positive_4bit,
    _unpack_positive_4bit_elementwise,
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gloo_rank1_worker(rank: int, port: int, output_queue) -> None:
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
        optimizer.step()
        output_queue.put(
            (
                rank,
                parameter.detach().tolist(),
                optimizer._rank1_scales[id(parameter)].tolist(),
                optimizer.buckets[0].refs[0].shard_start,
                optimizer.buckets[0].refs[0].shard_numel,
            )
        )
    finally:
        dist.destroy_process_group()


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


def test_adamw4bit_fresh_rank1_state_is_initialized_during_streaming_update() -> None:
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

    # The statistics pass consumes the implicit zero second moment without
    # allocating packed state. State is initialized only in the update pass,
    # where each completed parameter releases its gradient before the next.
    assert live_gradients_at_initialization == [2, 1]


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
    assert restored.state_dict()["state_format_version"] == 2


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


@pytest.mark.parametrize(
    "values",
    [
        torch.tensor([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0]]),
        torch.tensor([[1.0, 2.0, 8.0], [1.0, 2.0, 8.0]]),
        torch.tensor([[1.0, 8.0, 1.0], [8.0, 1.0, 8.0], [1.0, 8.0, 1.0]]),
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1000.0, 1.0]]),
        torch.zeros(3, 5),
        torch.full((3, 5), 1.0e-20),
    ],
    ids=["row", "column", "checkerboard", "isolated-outlier", "all-zero", "tiny-positive"],
)
def test_rank1_statistics_match_paper_sm3_algorithm(values: torch.Tensor) -> None:
    flattened = values.flatten()
    statistics = torch.zeros(sum(values.shape))

    for start in range(0, flattened.numel(), 4):
        _accumulate_rank1_maxima(statistics, flattened[start : start + 4], tuple(values.shape), start)

    expected = torch.cat((values.amax(dim=1), values.amax(dim=0)))
    torch.testing.assert_close(statistics, expected)
    expanded = _rank1_element_scales(statistics, tuple(values.shape), 0, values.numel()).reshape(values.shape)
    torch.testing.assert_close(
        expanded,
        torch.minimum(expected[: values.shape[0], None], expected[values.shape[0] :]),
    )


def test_rank1_statistics_generalize_to_higher_rank() -> None:
    values = torch.arange(1, 31, dtype=torch.float32).reshape(2, 3, 5)
    statistics = torch.zeros(sum(values.shape))
    _accumulate_rank1_maxima(statistics, values.flatten(), tuple(values.shape), 0)

    expected = torch.cat(
        (
            values.amax(dim=(1, 2)),
            values.amax(dim=(0, 2)),
            values.amax(dim=(0, 1)),
        )
    )
    torch.testing.assert_close(statistics, expected)


def test_adamw4bit_matrix_uses_rank1_second_moment_scales() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 5))
    optimizer = _optimizer(parameter)
    gradient = torch.arange(1, 16, dtype=torch.float32).reshape_as(parameter)
    parameter.grad = gradient.clone()

    optimizer.step()

    state = optimizer._states[0]
    variance = (1.0 - optimizer.betas[1]) * gradient.square()
    expected = torch.cat((variance.amax(dim=1), variance.amax(dim=0)))
    rank1_scales = optimizer._rank1_scales[id(parameter)]
    assert rank1_scales is not None
    torch.testing.assert_close(rank1_scales, expected)
    assert state.exp_avg_scale.numel() == 1
    assert state.exp_avg_sq_scale.numel() == 0


def test_adamw4bit_rank1_second_moment_never_decodes_nonzero_scale_to_zero() -> None:
    values = torch.tensor([[0.0, 0.01, 0.5], [0.02, 0.25, 1.0]])
    statistics = torch.cat((values.amax(dim=1), values.amax(dim=0)))
    scales = _rank1_element_scales(statistics, tuple(values.shape), 0, values.numel())

    packed = _quantize_positive_4bit_elementwise(values.flatten(), scales)
    restored = _unpack_positive_4bit_elementwise(packed, values.numel(), scales)

    assert torch.all(restored[scales > 0] > 0)


def test_adamw4bit_preserves_chunking_and_shares_rank1_scales_across_chunks() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2048, 2049))
    optimizer = AdamW4bit(
        [parameter],
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        bucket_numel=1,
        quant_block_size=128,
    )

    refs = [ref for bucket in optimizer.buckets for ref in bucket.refs]
    assert len(optimizer.buckets) == 2
    assert len(refs) == 2
    assert [ref.param_start for ref in refs] == [0, 4 * 1024 * 1024]
    assert sum(ref.numel for ref in refs) == parameter.numel()
    assert all(ref.model_param is parameter for ref in refs)
    scales = optimizer._ensure_rank1_scales(parameter)
    assert scales.numel() == sum(parameter.shape)
    assert optimizer._rank1_scales[id(parameter)] is scales


def test_adamw4bit_rank1_metadata_stays_within_large_matrix_budget() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1024, 1024))
    optimizer = _optimizer(parameter)
    optimizer._ensure_bucket_state(optimizer.buckets[0], optimizer._states[0])
    optimizer._ensure_rank1_scales(parameter)
    optimizer._states[0].step = 1

    assert optimizer.persistent_moment_bytes() / parameter.numel() <= 1.25
    metrics = optimizer.state_memory_metrics()
    assert metrics["total_bytes"] == optimizer.persistent_moment_bytes()
    assert metrics["scale_metadata_bytes"] == (parameter.numel() // 128 + sum(parameter.shape)) * 4


def test_adamw4bit_rank1_checkpoint_round_trip_preserves_next_update() -> None:
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
    torch.testing.assert_close(restored._rank1_scales[id(restored_parameter)], first._rank1_scales[id(first_parameter)])


def test_adamw4bit_clear_state_drops_rank1_scales() -> None:
    parameter = torch.nn.Parameter(torch.zeros(5, 7))
    optimizer = _optimizer(parameter)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert optimizer._rank1_scales[id(parameter)] is not None
    optimizer.clear_state()

    assert optimizer._rank1_scales[id(parameter)] is None
    assert all(state.step == 0 and state.exp_avg_q is None for state in optimizer._states)


def test_adamw4bit_rank1_disk_offload_preserves_axis_scales(tmp_path: Path) -> None:
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
        candidate._rank1_scales[id(candidate_parameter)], reference._rank1_scales[id(reference_parameter)]
    )
    assert not list(tmp_path.rglob("*.mmap"))


def test_rank1_partial_shards_combine_to_unsharded_statistics() -> None:
    values = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7)
    combined = torch.zeros(sum(values.shape))
    for start, count in ((0, 13), (13, 12), (25, 10)):
        partial = torch.zeros_like(combined)
        _accumulate_rank1_maxima(partial, values.flatten().narrow(0, start, count), tuple(values.shape), start)
        torch.maximum(combined, partial, out=combined)

    expected = torch.cat((values.amax(dim=1), values.amax(dim=0)))
    torch.testing.assert_close(combined, expected)


def test_real_gloo_rank1_statistics_match_unsharded_reference_across_split_row() -> None:
    spawn = mp.get_context("spawn")
    output_queue = spawn.Queue()
    port = _free_port()
    processes = [spawn.Process(target=_gloo_rank1_worker, args=(rank, port, output_queue)) for rank in range(2)]
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

    rank0_model, rank0_scales, rank0_start, rank0_count = results[0]
    rank1_model, rank1_scales, rank1_start, rank1_count = results[1]
    averaged_gradient = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7) + 1.5
    variance = 0.01 * averaged_gradient.square()
    expected_scales = torch.cat((variance.amax(dim=1), variance.amax(dim=0)))
    assert rank0_model == rank1_model
    torch.testing.assert_close(torch.tensor(rank0_scales), expected_scales)
    torch.testing.assert_close(torch.tensor(rank1_scales), expected_scales)
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
