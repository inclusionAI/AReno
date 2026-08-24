from __future__ import annotations

import copy
import multiprocessing as mp
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from areno.engine.optim import AdamW8bit, AdamWFP32Master
from areno.engine.runtime.train_step import _grad_norms_from_shards
from areno.engine.training import TrainingManager


@pytest.mark.parametrize("optimizer_cls", [AdamWFP32Master, AdamW8bit])
def test_optimizer_state_offload_batch_size_defaults_to_one(tmp_path, optimizer_cls) -> None:
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = optimizer_cls(
        [parameter],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    assert optimizer._active_offload_batch_size == 1
    optimizer.configure_state_offload(mode="disk", directory=str(tmp_path))
    assert optimizer._active_offload_batch_size == 1


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
        gradient = (torch.linspace(-0.5, 0.5, 17) + rank * 0.25).to(torch.bfloat16)
        # This pair has an exactly representable FP32 mean that differs from a
        # BF16-rounded cross-rank sum, so the test detects collective dtype.
        gradient[0] = 2.53125 if rank == 0 else 2.203125
        parameter.grad = gradient
        optimizer.reduce_scatter_gradients()
        grad_dtype = optimizer.buckets[0].grad_shard.dtype
        shard_numel = optimizer.buckets[0].grad_shard.numel()
        optimizer.step()
        state = optimizer.state_dict()
        restored_param = torch.nn.Parameter(torch.full_like(parameter, 10.0 + rank))
        restored = AdamWFP32Master(
            [restored_param],
            lr=4.0e-4,
            betas=(0.9, 0.98),
            weight_decay=0.02,
            bucket_numel=32,
            dp_rank=rank,
            dp_size=2,
            dp_group=dist.group.WORLD,
        )
        restored.load_state_dict(state)
        output_queue.put(
            (
                rank,
                parameter.detach().float().tolist(),
                state["master_params"][0].tolist(),
                grad_dtype == torch.float32,
                shard_numel,
                restored_param.detach().float().tolist(),
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


@pytest.mark.parametrize(
    ("keep_rollout_state", "optimizer_state_offload", "expected_offloads"),
    [
        (True, "none", []),
        (False, "none", [("cpu", None)]),
        (True, "cpu", [("cpu", None)]),
        (True, "disk", [("disk", "/tmp/areno-test-offload")]),
    ],
)
def test_training_manager_offloads_optimizer_state_when_requested(
    keep_rollout_state: bool,
    optimizer_state_offload: str,
    expected_offloads: list[tuple[str, str | None]],
) -> None:
    calls = {"events": [], "offload": []}

    class _Optimizer:
        def configure_state_offload(self, *, mode: str, directory: str | None, batch_size: int) -> None:
            assert batch_size == 32
            calls["events"].append(("configure", mode, directory))

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            calls["events"].append(("zero_grad",))

        def prefetch_state(self) -> None:
            calls["events"].append(("prefetch",))

        def offload_state(self, *, mode: str, directory: str | None, batch_size: int) -> None:
            assert batch_size == 32
            calls["offload"].append((mode, directory))

    worker = SimpleNamespace(
        optimizer=_Optimizer(),
        device=torch.device("cpu"),
        _train_state_ready=False,
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                keep_rollout_state=keep_rollout_state,
                optimizer_state_offload=optimizer_state_offload,
                optimizer_state_offload_dir=("/tmp/areno-test-offload" if optimizer_state_offload == "disk" else None),
                optimizer_state_offload_batch_size=32,
            )
        ),
    )

    def _prepare_for_train() -> None:
        calls["events"].append(("prepare",))
        worker._train_state_ready = True

    worker._prepare_for_train = _prepare_for_train
    manager = TrainingManager(worker)
    manager._train_step = lambda *_args, **_kwargs: {"ok": True}
    payload = SimpleNamespace(data_packs_by_dp=[[{}]], gradient_accumulation_steps=1)

    assert manager.train(payload) == [{"ok": True}]
    expected_events = [("prepare",)]
    if expected_offloads:
        expected_events.append(("configure", expected_offloads[0][0], expected_offloads[0][1]))
    if optimizer_state_offload == "disk":
        expected_events.append(("prefetch",))
    expected_events.append(("zero_grad",))
    assert calls == {"events": expected_events, "offload": expected_offloads}


def test_fp32_master_disk_offload_is_lazy_and_preserves_next_update(tmp_path) -> None:
    initial = torch.linspace(-1.0, 1.0, 29).to(torch.bfloat16)
    candidate_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())
    kwargs = {"lr": 7.0e-4, "betas": (0.9, 0.97), "weight_decay": 0.01, "bucket_numel": 8}
    candidate = AdamWFP32Master([candidate_param], **kwargs)
    reference = AdamWFP32Master([reference_param], **kwargs)
    candidate.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)

    first_gradient = torch.linspace(-0.75, 0.5, 29).to(torch.bfloat16)
    candidate_param.grad = first_gradient.clone()
    reference_param.grad = first_gradient.clone()
    candidate.step()
    reference.step()
    expected_state = copy.deepcopy(reference.state_dict())

    offload_files = [bucket.offload_file for bucket in candidate.buckets]
    assert all(path is not None and Path(path).is_file() for path in offload_files)
    assert len(set(offload_files)) == (len(candidate.buckets) + 1) // 2
    assert all(Path(path).suffix == ".mmap" for path in set(offload_files))
    offload_inodes = {path: Path(path).stat().st_ino for path in set(offload_files)}
    assert all(bucket.master_storage is None for bucket in candidate.buckets)
    assert all(bucket.exp_avg is None and bucket.exp_avg_sq is None for bucket in candidate.buckets)
    disk_state = candidate.state_dict()
    for actual, expected in zip(disk_state["master_params"], expected_state["master_params"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for actual, expected in zip(disk_state["state"], expected_state["state"], strict=True):
        torch.testing.assert_close(actual["exp_avg"], expected["exp_avg"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual["exp_avg_sq"], expected["exp_avg_sq"], rtol=0.0, atol=0.0)
    assert [bucket.offload_file for bucket in candidate.buckets] == offload_files

    candidate.prefetch_state()
    assert len(candidate._disk_prefetch_futures) == min(2, len(candidate.buckets))
    second_gradient = torch.linspace(0.25, -0.9, 29).to(torch.bfloat16)
    candidate_param.grad = second_gradient.clone()
    reference_param.grad = second_gradient.clone()
    candidate.step()
    reference.step()

    torch.testing.assert_close(candidate_param, reference_param, rtol=0.0, atol=0.0)
    torch.testing.assert_close(_flatten_master_state(candidate), _flatten_master_state(reference), rtol=0.0, atol=0.0)
    assert [bucket.offload_file for bucket in candidate.buckets] == offload_files
    assert {path: Path(path).stat().st_ino for path in set(offload_files)} == offload_inodes
    assert all(bucket.master_storage is None for bucket in candidate.buckets)
    assert all(bucket.exp_avg is None and bucket.exp_avg_sq is None for bucket in candidate.buckets)
    assert not candidate._disk_prefetch_futures
    assert not candidate._disk_prefetch_in_use
    candidate.onload_state(torch.device("cpu"))
    assert all(bucket.offload_file is None for bucket in candidate.buckets)
    assert all(bucket.master_storage is not None for bucket in candidate.buckets)
    assert not list(tmp_path.rglob("*.mmap"))


def test_adam8bit_disk_offload_preserves_quantized_update(tmp_path) -> None:
    initial = torch.linspace(-0.5, 0.5, 31).to(torch.bfloat16)
    candidate_param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())
    kwargs = {"lr": 4.0e-4, "betas": (0.9, 0.99), "weight_decay": 0.02, "bucket_numel": 9}
    candidate = AdamW8bit([candidate_param], **kwargs)
    reference = AdamW8bit([reference_param], **kwargs)
    candidate.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)
    for index, gradient in enumerate(
        (
            torch.linspace(-0.4, 0.7, 31),
            torch.linspace(0.8, -0.2, 31),
        )
    ):
        candidate_param.grad = gradient.to(torch.bfloat16)
        reference_param.grad = gradient.to(torch.bfloat16)
        candidate.step()
        reference.step()
        if index == 0:
            assert all(state.offload_file is not None for state in candidate._states)
            assert len({state.offload_file for state in candidate._states}) == (len(candidate._states) + 1) // 2
            assert all(Path(path).suffix == ".mmap" for path in {state.offload_file for state in candidate._states})
            offload_files = [state.offload_file for state in candidate._states]
            offload_inodes = {path: Path(path).stat().st_ino for path in set(offload_files)}
            assert all(state.exp_avg_q is None for state in candidate._states)
            candidate.prefetch_state()
            assert len(candidate._disk_prefetch_futures) == min(2, len(candidate.buckets))

    torch.testing.assert_close(candidate_param, reference_param, rtol=0.0, atol=0.0)
    candidate_state = candidate.state_dict()
    reference_state = reference.state_dict()
    for actual, expected in zip(candidate_state["state"], reference_state["state"], strict=True):
        for key in ("exp_avg_q", "exp_avg_scale", "exp_avg_sq_q", "exp_avg_sq_scale"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)
    assert [state.offload_file for state in candidate._states] == offload_files
    assert {path: Path(path).stat().st_ino for path in set(offload_files)} == offload_inodes
    assert not candidate._disk_prefetch_futures
    assert not candidate._disk_prefetch_in_use
    candidate.onload_state(torch.device("cpu"))
    assert all(state.offload_file is None for state in candidate._states)
    assert all(state.exp_avg_q is not None for state in candidate._states)
    assert not list(tmp_path.rglob("*.mmap"))


def test_disk_prefetch_window_is_bounded_and_pending_reads_clean_up(tmp_path) -> None:
    parameters = [torch.nn.Parameter(torch.full((4,), float(index), dtype=torch.bfloat16)) for index in range(3)]
    optimizer = AdamWFP32Master(
        parameters,
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=4,
    )
    assert len(optimizer.buckets) == 3
    optimizer.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    optimizer.prefetch_state()
    assert len(optimizer._disk_prefetch_futures) == 2
    optimizer.clear_state()

    assert not optimizer._disk_prefetch_futures
    assert not optimizer._disk_prefetch_in_use
    assert not list(tmp_path.rglob("*.mmap"))


def test_disk_save_returns_before_background_mmap_flush(tmp_path) -> None:
    from areno.engine.optim.adamw_fp32_master import _write_mmap_payloads

    write_started = threading.Event()
    allow_write = threading.Event()

    def delayed_write(*args, **kwargs) -> None:
        write_started.set()
        assert allow_write.wait(timeout=5.0)
        _write_mmap_payloads(*args, **kwargs)

    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 16).to(torch.bfloat16))
    optimizer = AdamWFP32Master(
        [parameter],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=32,
    )
    optimizer.configure_state_offload(mode="disk", directory=str(tmp_path), batch_size=2)
    parameter.grad = torch.ones_like(parameter)
    with patch("areno.engine.optim.adamw_fp32_master._write_mmap_payloads", delayed_write):
        optimizer.step()
        assert write_started.wait(timeout=2.0)
        assert len(optimizer._disk_write_futures) == 1
        assert not next(iter(optimizer._disk_write_futures.values())).done()
        allow_write.set()
        optimizer.onload_state(torch.device("cpu"))

    assert not optimizer._disk_write_futures
    assert optimizer.buckets[0].exp_avg is not None
    assert not list(tmp_path.rglob("*.mmap"))


def test_disk_writer_waits_for_staged_payload_event(tmp_path) -> None:
    class _ReadyEvent:
        def __init__(self) -> None:
            self.synchronized = threading.Event()

        def synchronize(self) -> None:
            self.synchronized.set()

    ready_event = _ReadyEvent()
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 16).to(torch.bfloat16))
    optimizer = AdamWFP32Master(
        [parameter],
        lr=1.0e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        bucket_numel=32,
    )
    optimizer.configure_state_offload(mode="disk", directory=str(tmp_path))

    def stage_with_event(payload):
        return {name: tensor.to(device="cpu") for name, tensor in payload.items()}, (ready_event,)

    parameter.grad = torch.ones_like(parameter)
    with patch.object(optimizer, "_stage_payload_on_cpu", side_effect=stage_with_event):
        optimizer.step()
        optimizer._shutdown_disk_writes()

    assert ready_event.synchronized.is_set()
    optimizer.onload_state(torch.device("cpu"))


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
    rank0_grad[0] = 2.53125
    rank1_grad[0] = 2.203125
    reference_param.grad = (rank0_grad.float() + rank1_grad.float()) / 2
    reference.step()

    rank0_model, rank0_master, rank0_is_fp32, rank0_shard_numel, rank0_restored = results[0]
    rank1_model, rank1_master, rank1_is_fp32, rank1_shard_numel, rank1_restored = results[1]
    assert rank0_model == rank1_model
    assert rank0_restored == rank1_restored == rank0_model
    torch.testing.assert_close(torch.tensor(rank0_model), reference_param.detach().to(torch.bfloat16).float())
    joined_master = torch.tensor(rank0_master + rank1_master)
    torch.testing.assert_close(joined_master, reference_param, rtol=2e-6, atol=2e-7)
    assert rank0_is_fp32 and rank1_is_fp32
    assert (rank0_shard_numel, rank1_shard_numel) == (9, 8)
