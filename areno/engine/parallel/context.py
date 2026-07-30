"""Tensor-parallel and data-parallel rank context.

A worker process needs to know both its place inside a tensor-parallel group
(used for sharded layers and TP collectives) and its place inside the matching
data-parallel group (used for DP gradient averaging and DP-strided result
merging). `TPContext` carries both views, and `init_process_group` derives the
two groups from a global rank layout where ranks are laid out as
`dp_rank * tp_size + tp_rank`.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

import torch
import torch.distributed as dist


@dataclass(slots=True)
class TPContext:
    """Rank-local view of the tensor-parallel and data-parallel groups."""

    rank: int
    world_size: int
    device: torch.device
    group: dist.ProcessGroup | None
    global_rank: int = 0
    global_world_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    dp_group: dist.ProcessGroup | None = None
    role: Literal["train", "rollout"] = "train"
    policy_publisher_groups: tuple[dist.ProcessGroup | None, ...] = ()
    policy_publisher_ranks: tuple[tuple[int, ...], ...] = ()
    policy_source_ranks: tuple[int, ...] = ()
    policy_bridge_ranks: tuple[int, ...] = ()
    policy_bridge_dp_ranks: tuple[int, ...] = ()
    train_tp_size: int = 1
    rollout_tp_size: int = 1

    @property
    def is_rank0(self) -> bool:
        """True for the rank that owns user-visible TP outputs."""

        return self.rank == 0

    def tp_global_rank(self, rank: int) -> int:
        """Translate a rank local to this TP group into a default-world rank."""

        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank={rank} is outside TP group size {self.world_size}")
        return self.global_rank - self.rank + rank

    def partition_global_rank(self, rank: int) -> int:
        """Translate a role-partition-local rank into a default-world rank."""

        partition_rank = self.dp_rank * self.world_size + self.rank
        return self.global_rank - partition_rank + rank


_TP_CONTEXT = TPContext(
    rank=0,
    world_size=1,
    device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
    group=None,
)
_TP_CONTEXT_LOCK = Lock()


def get_tp_context() -> TPContext:
    """Return the rank-local TP/DP context set up by `init_process_group`."""

    with _TP_CONTEXT_LOCK:
        return _TP_CONTEXT


def set_tp_context(ctx: TPContext) -> None:
    """Replace the module-global TP/DP context."""

    global _TP_CONTEXT
    with _TP_CONTEXT_LOCK:
        _TP_CONTEXT = ctx


def init_process_group(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    device_id: int,
    tp_size: int,
    *,
    global_rank: int | None = None,
    global_world_size: int | None = None,
    train_world_size: int | None = None,
    train_tp_size: int | None = None,
    rollout_world_size: int | None = None,
    rollout_tp_size: int | None = None,
    train_devices: tuple[int, ...] | None = None,
    rollout_devices: tuple[int, ...] | None = None,
    role: Literal["train", "rollout"] = "train",
) -> TPContext:
    """Initialize process groups and derive local TP/DP rank coordinates."""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device("cuda", device_id)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    resolved_global_rank = rank if global_rank is None else global_rank
    resolved_global_world_size = world_size if global_world_size is None else global_world_size
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=resolved_global_rank,
        world_size=resolved_global_world_size,
    )
    if world_size % tp_size != 0:
        raise ValueError("distributed world_size must be divisible by tp_size")
    dp_size = world_size // tp_size
    # Global rank layout is row-major over DP first, then TP within each DP
    # row, so `dp_rank * tp_size + tp_rank == global_rank`.
    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    resolved_train_world = world_size if train_world_size is None else train_world_size
    resolved_train_tp = tp_size if train_tp_size is None else train_tp_size
    resolved_rollout_world = 0 if rollout_world_size is None else rollout_world_size
    resolved_rollout_tp = resolved_train_tp if rollout_tp_size is None else rollout_tp_size
    train_tp_groups, train_dp_groups = _create_partition_groups(
        offset=0,
        world_size=resolved_train_world,
        tp_size=resolved_train_tp,
        global_rank=resolved_global_rank,
    )
    rollout_tp_groups, rollout_dp_groups = _create_partition_groups(
        offset=resolved_train_world,
        world_size=resolved_rollout_world,
        tp_size=resolved_rollout_tp,
        global_rank=resolved_global_rank,
    )
    if role == "train":
        tp_group = train_tp_groups[dp_rank]
        dp_group = train_dp_groups[tp_rank]
    else:
        tp_group = rollout_tp_groups[dp_rank]
        dp_group = rollout_dp_groups[tp_rank]

    publisher_groups = []
    publisher_ranks = []
    source_ranks = []
    bridge_ranks = []
    bridge_dp_ranks = []
    if resolved_rollout_world:
        resolved_train_devices = train_devices or tuple(range(resolved_train_world))
        resolved_rollout_devices = rollout_devices or tuple(range(resolved_rollout_world))
        for train_dp_rank in range(resolved_train_world // resolved_train_tp):
            pair = _select_policy_relay_pair(
                train_dp_rank=train_dp_rank,
                train_tp_size=resolved_train_tp,
                train_devices=resolved_train_devices,
                rollout_devices=resolved_rollout_devices,
                rollout_rank_offset=resolved_train_world,
                rollout_tp_size=resolved_rollout_tp,
            )
            source_rank, bridge_rank, bridge_dp_rank = pair
            ranks = [source_rank, bridge_rank]
            group = dist.new_group(ranks=ranks)
            publisher_ranks.append(tuple(ranks))
            publisher_groups.append(group if resolved_global_rank in ranks else None)
            source_ranks.append(source_rank)
            bridge_ranks.append(bridge_rank)
            bridge_dp_ranks.append(bridge_dp_rank)

    ctx = TPContext(
        rank=tp_rank,
        world_size=tp_size,
        device=device,
        group=tp_group,
        global_rank=resolved_global_rank,
        global_world_size=resolved_global_world_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        dp_group=dp_group,
        role=role,
        policy_publisher_groups=tuple(publisher_groups),
        policy_publisher_ranks=tuple(publisher_ranks),
        policy_source_ranks=tuple(source_ranks),
        policy_bridge_ranks=tuple(bridge_ranks),
        policy_bridge_dp_ranks=tuple(bridge_dp_ranks),
        train_tp_size=resolved_train_tp,
        rollout_tp_size=resolved_rollout_tp,
    )
    set_tp_context(ctx)
    return ctx


def _create_partition_groups(
    *,
    offset: int,
    world_size: int,
    tp_size: int,
    global_rank: int,
) -> tuple[list[dist.ProcessGroup | None], list[dist.ProcessGroup | None]]:
    """Create all TP/DP groups for one contiguous partition."""

    if world_size == 0:
        return [], []
    if world_size % tp_size != 0:
        raise ValueError("partition world_size must be divisible by tp_size")
    dp_size = world_size // tp_size
    tp_groups: list[dist.ProcessGroup | None] = []
    for dp_rank in range(dp_size):
        ranks = list(range(offset + dp_rank * tp_size, offset + (dp_rank + 1) * tp_size))
        group = dist.new_group(ranks=ranks)
        tp_groups.append(group if global_rank in ranks else None)
    dp_groups: list[dist.ProcessGroup | None] = []
    for tp_rank in range(tp_size):
        ranks = [offset + dp_rank * tp_size + tp_rank for dp_rank in range(dp_size)]
        group = dist.new_group(ranks=ranks)
        dp_groups.append(group if global_rank in ranks else None)
    return tp_groups, dp_groups


def _select_policy_relay_pair(
    *,
    train_dp_rank: int,
    train_tp_size: int,
    train_devices: tuple[int, ...],
    rollout_devices: tuple[int, ...],
    rollout_rank_offset: int,
    rollout_tp_size: int,
) -> tuple[int, int, int]:
    """Choose train/rollout ranks on different GPUs for one NCCL relay."""

    train_start = train_dp_rank * train_tp_size
    for train_local_rank in range(train_start, train_start + train_tp_size):
        for rollout_local_rank, rollout_device in enumerate(rollout_devices):
            if train_devices[train_local_rank] == rollout_device:
                continue
            return (
                train_local_rank,
                rollout_rank_offset + rollout_local_rank,
                rollout_local_rank // rollout_tp_size,
            )
    raise ValueError(
        "overlapping train and rollout engines need at least two distinct CUDA devices for NCCL policy sync"
    )


def destroy_process_group() -> None:
    """Tear down NCCL/Gloo state and reset the module-global TP context."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    set_tp_context(
        TPContext(
            rank=0,
            world_size=1,
            device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
            group=None,
        )
    )
