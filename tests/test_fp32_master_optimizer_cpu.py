from __future__ import annotations

import copy
import multiprocessing as mp
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from areno.engine.optim import AdamWFP32Master
from areno.engine.runtime.train_step import _grad_norms_from_shards


def _flatten_master_state(optimizer: AdamWFP32Master) -> torch.Tensor:
    return torch.cat([value for value in optimizer.state_dict()["master_params"] if value is not None])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gloo_sharded_optimizer_worker(rank: int, port: int, output_queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        initial = torch.linspace(-0.75, 0.75, 17).to(torch.bfloat16)
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = AdamWFP32Master(
            [parameter],
            lr=4.0e-4,
            betas=(0.9, 0.98),
            weight_decay=0.02,
            bucket_numel=32,
            dp_rank=rank,
            dp_size=2,
            dp_group=dist.group.WORLD,
        )
        parameter.grad = (torch.linspace(-0.5, 0.5, 17) + rank * 0.25).to(torch.bfloat16)
        optimizer.reduce_scatter_gradients()
        grad_dtype = optimizer.buckets[0].grad_shard.dtype
        shard_numel = optimizer.buckets[0].grad_shard.numel()
        optimizer.step()
        state = optimizer.state_dict()
        output_queue.put(
            (
                rank,
                parameter.detach().float().tolist(),
                state["master_params"][0].tolist(),
                grad_dtype == torch.float32,
                shard_numel,
            )
        )
    finally:
        dist.destroy_process_group()


def test_fp32_master_adamw_matches_torch_reference_across_buckets() -> None:
    initial = torch.linspace(-1.5, 1.5, 37, dtype=torch.float32).to(torch.bfloat16)
    candidate_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.float())
    kwargs = {
        "lr": 3.0e-4,
        "betas": (0.9, 0.97),
        "weight_decay": 0.03,
    }
    candidate = AdamWFP32Master([candidate_param], bucket_numel=11, **kwargs)
    reference = torch.optim.AdamW([reference_param], eps=1.0e-8, **kwargs)
    generator = torch.Generator().manual_seed(17)

    for _ in range(7):
        gradient = torch.randn(candidate_param.shape, generator=generator).to(torch.bfloat16)
        candidate_param.grad = gradient.clone()
        reference_param.grad = gradient.float()

        candidate.step()
        reference.step()

        torch.testing.assert_close(_flatten_master_state(candidate), reference_param, rtol=2e-6, atol=2e-7)
        assert torch.equal(candidate_param, reference_param.detach().to(torch.bfloat16))
        assert all(bucket.master is None for bucket in candidate.buckets)


def test_fp32_master_runtime_state_is_sharded_and_compact() -> None:
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 1024).to(torch.bfloat16))
    optimizer = AdamWFP32Master(
        [parameter],
        lr=1.0e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        bucket_numel=2048,
        dp_rank=1,
        dp_size=4,
    )
    bucket = optimizer.buckets[0]
    optimizer._ensure_bucket_state(bucket)

    assert bucket.shard_numel == 256
    assert bucket.master is None
    assert bucket.master_storage is not None
    assert bucket.master_storage.nbytes == 2 * bucket.shard_numel + (bucket.shard_numel + 7) // 8
    assert bucket.exp_avg is not None and bucket.exp_avg.numel() == bucket.shard_numel
    assert bucket.exp_avg_sq is not None and bucket.exp_avg_sq.numel() == bucket.shard_numel


def test_fp32_master_checkpoint_round_trip_preserves_next_update() -> None:
    initial = torch.tensor([0.25, -0.5, 1.0, -2.0, 4.0], dtype=torch.bfloat16)
    first_param = torch.nn.Parameter(initial.clone())
    first = AdamWFP32Master(
        [first_param],
        lr=2.0e-4,
        betas=(0.85, 0.995),
        weight_decay=0.1,
        bucket_numel=3,
    )
    first_param.grad = torch.tensor([0.5, -0.25, 1.5, -2.0, 0.125], dtype=torch.bfloat16)
    first.step()
    checkpoint = copy.deepcopy(first.state_dict())

    restored_param = torch.nn.Parameter(torch.zeros_like(first_param))
    restored = AdamWFP32Master(
        [restored_param],
        lr=2.0e-4,
        betas=(0.85, 0.995),
        weight_decay=0.1,
        bucket_numel=3,
    )
    restored.load_state_dict(checkpoint)
    torch.testing.assert_close(restored_param, first_param, rtol=0.0, atol=0.0)

    next_gradient = torch.tensor([-1.0, 0.75, -0.5, 0.25, 2.0], dtype=torch.bfloat16)
    first_param.grad = next_gradient.clone()
    restored_param.grad = next_gradient.clone()
    first.step()
    restored.step()

    torch.testing.assert_close(restored_param, first_param, rtol=0.0, atol=0.0)
    torch.testing.assert_close(_flatten_master_state(restored), _flatten_master_state(first), rtol=0.0, atol=0.0)


def test_gradient_is_streamed_into_fp32_dp_shard_without_full_main_grad() -> None:
    direct_param = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 19).to(torch.bfloat16))
    sharded_param = torch.nn.Parameter(direct_param.detach().clone())
    kwargs = {"lr": 1.0e-3, "betas": (0.9, 0.99), "weight_decay": 0.02, "bucket_numel": 7}
    direct = AdamWFP32Master([direct_param], **kwargs)
    sharded = AdamWFP32Master([sharded_param], **kwargs)
    gradient = torch.linspace(0.75, -0.5, 19).to(torch.bfloat16)
    direct_param.grad = gradient.clone()
    sharded_param.grad = gradient.clone()

    sharded.reduce_scatter_gradients()

    assert sharded_param.grad is None
    assert getattr(sharded_param, "main_grad", None) is None
    assert all(bucket.grad_shard is not None and bucket.grad_shard.dtype == torch.float32 for bucket in sharded.buckets)
    with patch(
        "areno.engine.runtime.train_step.get_tp_context",
        return_value=SimpleNamespace(world_size=1, rank=0, group=None, dp_size=1, dp_group=None),
    ):
        norms = _grad_norms_from_shards(sharded.grad_shards())
    expected_norm = torch.linalg.vector_norm(gradient.float()).item()
    assert norms["global"] == pytest.approx(expected_norm, rel=1e-6)

    direct.step()
    sharded.step()

    torch.testing.assert_close(sharded_param, direct_param, rtol=0.0, atol=0.0)
    torch.testing.assert_close(_flatten_master_state(sharded), _flatten_master_state(direct), rtol=0.0, atol=0.0)


def test_bf16_autograd_accumulation_does_not_create_fp32_gradient_copy() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))

    (parameter.float().square().sum() / 2.0).backward()
    (parameter.float().square().sum() / 2.0).backward()

    assert parameter.grad is not None
    assert parameter.grad.dtype == torch.bfloat16
    assert getattr(parameter, "main_grad", None) is None


def test_two_microbatches_match_one_fp32_accumulated_optimizer_update() -> None:
    initial = torch.linspace(-1.0, 1.0, 23).to(torch.bfloat16)
    candidate_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.float())
    kwargs = {"lr": 7.0e-4, "betas": (0.9, 0.97), "weight_decay": 0.01}
    candidate = AdamWFP32Master([candidate_param], bucket_numel=8, **kwargs)
    reference = torch.optim.AdamW([reference_param], eps=1.0e-8, **kwargs)
    first = torch.linspace(-0.75, 0.5, 23).to(torch.bfloat16)
    second = torch.linspace(0.25, -1.0, 23).to(torch.bfloat16)

    candidate_param.grad = (first / 2).to(torch.bfloat16)
    candidate.reduce_scatter_gradients()
    candidate_param.grad = (second / 2).to(torch.bfloat16)
    candidate.reduce_scatter_gradients()
    reference_param.grad = (first / 2).to(torch.bfloat16).float() + (second / 2).to(torch.bfloat16).float()

    candidate.step()
    reference.step()

    torch.testing.assert_close(_flatten_master_state(candidate), reference_param, rtol=2e-6, atol=2e-7)
    assert torch.equal(candidate_param, reference_param.detach().to(torch.bfloat16))


def test_real_gloo_dp_reduce_scatter_matches_averaged_reference() -> None:
    spawn = mp.get_context("spawn")
    output_queue = spawn.Queue()
    port = _free_port()
    processes = [
        spawn.Process(target=_gloo_sharded_optimizer_worker, args=(rank, port, output_queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = dict((item[0], item[1:]) for item in (output_queue.get(timeout=30) for _ in processes))
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    initial = torch.linspace(-0.75, 0.75, 17).to(torch.bfloat16)
    reference_param = torch.nn.Parameter(initial.float())
    reference = torch.optim.AdamW(
        [reference_param],
        lr=4.0e-4,
        betas=(0.9, 0.98),
        weight_decay=0.02,
        eps=1.0e-8,
    )
    rank0_grad = torch.linspace(-0.5, 0.5, 17).to(torch.bfloat16)
    rank1_grad = (torch.linspace(-0.5, 0.5, 17) + 0.25).to(torch.bfloat16)
    reference_param.grad = ((rank0_grad + rank1_grad) / 2).float()
    reference.step()

    rank0_model, rank0_master, rank0_is_fp32, rank0_shard_numel = results[0]
    rank1_model, rank1_master, rank1_is_fp32, rank1_shard_numel = results[1]
    assert rank0_model == rank1_model
    torch.testing.assert_close(torch.tensor(rank0_model), reference_param.detach().to(torch.bfloat16).float())
    joined_master = torch.tensor(rank0_master + rank1_master)
    torch.testing.assert_close(joined_master, reference_param, rtol=2e-6, atol=2e-7)
    assert rank0_is_fp32 and rank1_is_fp32
    assert (rank0_shard_numel, rank1_shard_numel) == (9, 8)
