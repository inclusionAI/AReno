from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areno.engine.checkpoints.io import (
    PolicyFlatPiece,
    PolicyTensorLayout,
    _RangedTensorParallelGatherTask,
    _ReplicatedTensorTask,
    _SplitColumnGatherTask,
    _TensorParallelGatherTask,
)
from areno.engine.parallel.context import TPContext, destroy_process_group, init_process_group, set_tp_context
from areno.engine.policy_sync import PolicyTensorMeta, assign_policy_owners, transfer_policy_weights
from areno.engine.protocol import PolicySyncPayload, find_free_port


def _set_rank(rank: int, world_size: int) -> None:
    set_tp_context(
        TPContext(
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
            group=None,
        )
    )


def _canonical(layouts: list[PolicyTensorLayout], *, chunk_size: int = 5) -> torch.Tensor:
    output = torch.empty(layouts[0].numel, dtype=layouts[0].dtype)
    for offset in range(0, output.numel(), chunk_size):
        chunk = torch.zeros(min(chunk_size, output.numel() - offset), dtype=output.dtype)
        for layout in layouts:
            contribution = torch.empty_like(chunk)
            layout.read_chunk(offset, contribution)
            chunk.add_(contribution)
        output[offset : offset + chunk.numel()].copy_(chunk)
    return output.reshape(layouts[0].shape)


def _write(layouts: list[PolicyTensorLayout], canonical: torch.Tensor, *, chunk_size: int = 7) -> None:
    flat = canonical.reshape(-1)
    for offset in range(0, flat.numel(), chunk_size):
        chunk = flat[offset : offset + chunk_size]
        for layout in layouts:
            layout.write_chunk(offset, chunk)


def test_column_parallel_layout_reshards_tp2_to_tp1() -> None:
    full = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    train_layouts = []
    for rank, local in enumerate(full.chunk(2, dim=0)):
        _set_rank(rank, 2)
        train_layouts.append(_TensorParallelGatherTask(local.clone(), dim=0).policy_layout())
    canonical = _canonical(train_layouts)

    target = torch.zeros_like(full)
    _set_rank(0, 1)
    rollout_layout = _TensorParallelGatherTask(target, dim=0).policy_layout()
    _write([rollout_layout], canonical)

    torch.testing.assert_close(canonical, full)
    torch.testing.assert_close(target, full)


def test_row_parallel_layout_reshards_tp1_to_tp2() -> None:
    full = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    _set_rank(0, 1)
    canonical = _canonical([_TensorParallelGatherTask(full.clone(), dim=1).policy_layout()])

    targets = [torch.zeros(4, 3), torch.zeros(4, 3)]
    layouts = []
    for rank, target in enumerate(targets):
        _set_rank(rank, 2)
        layouts.append(_TensorParallelGatherTask(target, dim=1).policy_layout())
    _write(layouts, canonical)

    torch.testing.assert_close(torch.cat(targets, dim=1), full)


def test_split_column_layout_preserves_fused_qkv_order() -> None:
    q = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    k = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)
    v = torch.arange(16, 24, dtype=torch.float32).reshape(2, 4)
    layouts = []
    for rank in range(2):
        _set_rank(rank, 2)
        parts = [part.chunk(2, dim=0)[rank].clone() for part in (q, k, v)]
        layouts.append(_SplitColumnGatherTask(parts).policy_layout())

    torch.testing.assert_close(_canonical(layouts), torch.cat((q, k, v), dim=0))


def test_ranged_layout_publishes_replicated_kv_once_and_writes_all_replicas() -> None:
    head0 = torch.tensor([[1.0, 2.0]])
    head1 = torch.tensor([[3.0, 4.0]])
    train_layouts = []
    for rank, local in enumerate((head0, head0, head1, head1)):
        _set_rank(rank, 4)
        start = 0 if rank < 2 else 1
        train_layouts.append(_RangedTensorParallelGatherTask(local.clone(), start, start + 1, 2).policy_layout())
    canonical = _canonical(train_layouts)
    torch.testing.assert_close(canonical, torch.cat((head0, head1), dim=0))

    targets = [torch.zeros_like(head0) for _ in range(4)]
    rollout_layouts = []
    for rank, target in enumerate(targets):
        _set_rank(rank, 4)
        start = 0 if rank < 2 else 1
        rollout_layouts.append(_RangedTensorParallelGatherTask(target, start, start + 1, 2).policy_layout())
    _write(rollout_layouts, canonical)
    torch.testing.assert_close(targets[0], head0)
    torch.testing.assert_close(targets[1], head0)
    torch.testing.assert_close(targets[2], head1)
    torch.testing.assert_close(targets[3], head1)


def test_replicated_and_flat_piece_layouts_write_live_tensors() -> None:
    replicated = torch.zeros(5)
    layout = _ReplicatedTensorTask(replicated).policy_layout()
    _write([layout], torch.arange(5, dtype=torch.float32), chunk_size=2)
    torch.testing.assert_close(replicated, torch.arange(5, dtype=torch.float32))

    first = torch.zeros(2)
    second = torch.zeros(2)
    packed = PolicyTensorLayout(
        shape=(6,),
        dtype=torch.float32,
        pieces=(),
        flat_pieces=(PolicyFlatPiece(first, 1), PolicyFlatPiece(second, 4)),
    )
    _write([packed], torch.arange(6, dtype=torch.float32), chunk_size=3)
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second, torch.tensor([4.0, 5.0]))


def test_policy_owner_assignment_is_deterministic_and_byte_balanced() -> None:
    metadata = tuple(
        PolicyTensorMeta(f"tensor.{index}", (size,), "float32", size * 4)
        for index, size in enumerate((100, 80, 60, 40, 20, 10))
    )
    owners = assign_policy_owners(metadata, 2)
    assert owners == assign_policy_owners(metadata, 2)
    loads = [sum(meta.nbytes for meta, owner in zip(metadata, owners, strict=True) if owner == dp) for dp in range(2)]
    assert max(loads) - min(loads) <= max(meta.nbytes for meta in metadata)


def _gloo_policy_sync_worker(global_rank: int, port: int, output_queue) -> None:
    role = "train" if global_rank < 2 else "rollout"
    local_rank = global_rank if role == "train" else 0
    init_process_group(
        rank=local_rank,
        world_size=2 if role == "train" else 1,
        master_addr="127.0.0.1",
        master_port=port,
        device_id=0,
        tp_size=2 if role == "train" else 1,
        global_rank=global_rank,
        global_world_size=3,
        train_world_size=2,
        train_tp_size=2,
        rollout_world_size=1,
        rollout_tp_size=1,
        role=role,
    )
    try:
        if role == "train":
            full = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            local = full.chunk(2, dim=0)[local_rank].clone()
        else:
            local = torch.zeros(4, 2)
        task = _TensorParallelGatherTask(local, dim=0)
        layout = task.policy_layout()
        meta = PolicyTensorMeta("weight", layout.shape, "float32", layout.nbytes)
        worker = type("Worker", (), {})()
        worker._policy_sync_plan = {"weight": task}
        worker._policy_sync_metadata = (meta,)
        worker._policy_sync_buffer = None
        transfer_policy_weights(worker, PolicySyncPayload(version=1, bucket_bytes=16))
        dist.barrier()
        if role == "rollout":
            output_queue.put(local.tolist())
    finally:
        destroy_process_group()


def test_real_gloo_collectives_reshard_train_tp2_to_rollout_tp1() -> None:
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    port = find_free_port()
    processes = [ctx.Process(target=_gloo_policy_sync_worker, args=(rank, port, output_queue)) for rank in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    assert [process.exitcode for process in processes] == [0, 0, 0]
    assert output_queue.get(timeout=5) == torch.arange(8, dtype=torch.float32).reshape(4, 2).tolist()


def _gloo_policy_sync_reverse_worker(global_rank: int, port: int, output_queue) -> None:
    role = "train" if global_rank == 0 else "rollout"
    local_rank = 0 if role == "train" else global_rank - 1
    init_process_group(
        rank=local_rank,
        world_size=1 if role == "train" else 2,
        master_addr="127.0.0.1",
        master_port=port,
        device_id=0,
        tp_size=1 if role == "train" else 2,
        global_rank=global_rank,
        global_world_size=3,
        train_world_size=1,
        train_tp_size=1,
        rollout_world_size=2,
        rollout_tp_size=2,
        role=role,
    )
    try:
        full = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        local = full.clone() if role == "train" else torch.zeros(2, 2)
        task = _TensorParallelGatherTask(local, dim=0)
        layout = task.policy_layout()
        meta = PolicyTensorMeta("weight", layout.shape, "float32", layout.nbytes)
        worker = type("Worker", (), {})()
        worker._policy_sync_plan = {"weight": task}
        worker._policy_sync_metadata = (meta,)
        worker._policy_sync_buffer = None
        transfer_policy_weights(worker, PolicySyncPayload(version=1, bucket_bytes=16))
        dist.barrier()
        if role == "rollout":
            output_queue.put((local_rank, local.tolist()))
    finally:
        destroy_process_group()


def test_real_gloo_collectives_reshard_train_tp1_to_rollout_tp2() -> None:
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    port = find_free_port()
    processes = [
        ctx.Process(target=_gloo_policy_sync_reverse_worker, args=(rank, port, output_queue)) for rank in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    assert [process.exitcode for process in processes] == [0, 0, 0]
    shards = sorted((output_queue.get(timeout=5) for _ in range(2)), key=lambda item: item[0])
    combined = torch.cat([torch.tensor(rows) for _, rows in shards], dim=0)
    torch.testing.assert_close(combined, torch.arange(8, dtype=torch.float32).reshape(4, 2))
