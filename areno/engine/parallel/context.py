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

    @property
    def is_rank0(self) -> bool:
        """True for the rank that owns user-visible TP outputs."""

        return self.rank == 0


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
) -> TPContext:
    """Initialize process groups and derive local TP/DP rank coordinates."""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device("cuda", device_id)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    if world_size % tp_size != 0:
        raise ValueError("distributed world_size must be divisible by tp_size")
    dp_size = world_size // tp_size
    # Global rank layout is row-major over DP first, then TP within each DP
    # row, so `dp_rank * tp_size + tp_rank == global_rank`.
    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    # Create one TP group per DP row. Every rank participates in `new_group`
    # for every row so all ranks agree on group construction order, but only
    # keeps a handle to the group that contains this rank.
    tp_group = None
    for group_dp_rank in range(dp_size):
        ranks = list(range(group_dp_rank * tp_size, (group_dp_rank + 1) * tp_size))
        group = dist.new_group(ranks=ranks)
        if group_dp_rank == dp_rank:
            tp_group = group

    # Create one DP group per TP column; same all-ranks-participate pattern as
    # above so collective initialization is symmetric across the cluster.
    dp_group = None
    for group_tp_rank in range(tp_size):
        ranks = [group_dp_rank * tp_size + group_tp_rank for group_dp_rank in range(dp_size)]
        group = dist.new_group(ranks=ranks)
        if group_tp_rank == tp_rank:
            dp_group = group

    ctx = TPContext(
        rank=tp_rank,
        world_size=tp_size,
        device=device,
        group=tp_group,
        global_rank=rank,
        global_world_size=world_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        dp_group=dp_group,
    )
    set_tp_context(ctx)
    return ctx


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
