"""Direct GPU policy synchronization between train and rollout partitions."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from areno.engine.checkpoints.io import PolicyTensorLayout
from areno.engine.parallel.context import get_tp_context
from areno.engine.protocol import PolicySyncPayload
from areno.models.registry import build_policy_weight_plan


@dataclass(slots=True, frozen=True)
class PolicyTensorMeta:
    """Canonical metadata shared by every train and rollout rank."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def build_policy_plan(worker) -> tuple[dict[str, object], tuple[PolicyTensorMeta, ...]]:
    """Build and cache live adapter tasks plus transport metadata."""

    plan = build_policy_weight_plan(worker.model, worker.config.model)
    metadata = []
    for key, task in plan.items():
        layout = task.policy_layout()
        metadata.append(
            PolicyTensorMeta(
                key=key,
                shape=layout.shape,
                dtype=str(layout.dtype).removeprefix("torch."),
                nbytes=layout.nbytes,
            )
        )
    worker._policy_sync_plan = plan
    worker._policy_sync_metadata = tuple(metadata)
    return plan, tuple(metadata)


def policy_plan_metadata(worker) -> tuple[PolicyTensorMeta, ...]:
    """Return canonical metadata without transferring weights."""

    _, metadata = build_policy_plan(worker)
    return metadata


def assign_policy_owners(metadata: tuple[PolicyTensorMeta, ...], train_dp_size: int) -> tuple[int, ...]:
    """Greedily balance canonical bytes over train DP publishers."""

    if train_dp_size < 1:
        raise ValueError("train_dp_size must be positive")
    loads = [0] * train_dp_size
    owners = [0] * len(metadata)
    for index in sorted(range(len(metadata)), key=lambda item: (-metadata[item].nbytes, metadata[item].key)):
        owner = min(range(train_dp_size), key=lambda candidate: (loads[candidate], candidate))
        owners[index] = owner
        loads[owner] += metadata[index].nbytes
    return tuple(owners)


@torch.no_grad()
def transfer_policy_weights(worker, payload: PolicySyncPayload) -> dict[str, object]:
    """Publish or receive every canonical chunk through NCCL groups."""

    ctx = get_tp_context()
    if not ctx.policy_publisher_groups:
        raise RuntimeError("policy synchronization requires partitioned publisher groups")
    plan = getattr(worker, "_policy_sync_plan", None)
    metadata = getattr(worker, "_policy_sync_metadata", None)
    if plan is None or metadata is None:
        plan, metadata = build_policy_plan(worker)
    train_dp_size = len(ctx.policy_publisher_groups)
    owners = assign_policy_owners(metadata, train_dp_size)
    bucket_bytes = int(payload.bucket_bytes)
    if bucket_bytes < 1:
        raise ValueError("policy sync bucket_bytes must be positive")
    started = time.perf_counter()
    published_by_dp = [0] * train_dp_size
    buffer = getattr(worker, "_policy_sync_buffer", None)

    for meta, owner in zip(metadata, owners, strict=True):
        if ctx.role == "train" and ctx.dp_rank != owner:
            continue
        task = plan[meta.key]
        layout: PolicyTensorLayout = task.policy_layout()
        element_size = torch.empty((), dtype=layout.dtype).element_size()
        capacity = max(bucket_bytes // element_size, 1)
        publisher_group = ctx.policy_publisher_groups[owner]
        if publisher_group is None:
            raise RuntimeError(f"rank {ctx.global_rank} is missing policy publisher group {owner}")
        source_rank = owner * ctx.train_tp_size
        for offset in range(0, layout.numel, capacity):
            count = min(capacity, layout.numel - offset)
            if buffer is None or buffer.dtype != layout.dtype or buffer.device != ctx.device or buffer.numel() < count:
                buffer = torch.empty(capacity, dtype=layout.dtype, device=ctx.device)
                worker._policy_sync_buffer = buffer
            chunk = buffer[:count]
            if ctx.role == "train":
                layout.read_chunk(offset, chunk, include_replicated=ctx.rank == 0)
                if ctx.world_size > 1:
                    dist.reduce(chunk, dst=source_rank, group=ctx.group)
            dist.broadcast(chunk, src=source_rank, group=publisher_group)
            if ctx.role == "rollout":
                layout.write_chunk(offset, chunk)
        published_by_dp[owner] += meta.nbytes

    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)
    return {
        "version": payload.version,
        "bytes": sum(meta.nbytes for meta in metadata),
        "tensors": len(metadata),
        "elapsed_s": time.perf_counter() - started,
        "published_bytes_by_train_dp": tuple(published_by_dp),
    }
