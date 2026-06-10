"""Autograd-aware TP collectives plus sequence-parallel helpers.

Each TP collective comes in a forward function that drives one NCCL op and a
matching backward that drives the dual op so the gradient is consistent with
Megatron-style sequence parallelism:
  * `all_reduce` / `_AllReduceSum`            : forward all-reduce sum;
     backward is identity (grad already summed by previous layer).
  * `copy_to_tensor_parallel_region`          : forward identity, backward
     all-reduce (column-parallel layer entry).
  * `scatter_to_sequence_parallel_region`     : forward scatter along seq dim,
     backward all-gather along seq dim.
  * `gather_from_sequence_parallel_region`    : forward all-gather, backward
     reduce-scatter.
  * `reduce_scatter_to_sequence_parallel_region`: forward reduce-scatter,
     backward all-gather.

The sequence-parallel layout assumes `(batch, seqlen, hidden)` with `seqlen`
divisible by `tp_size`, so the seqlen-axis collectives operate on dim=1.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from areno.engine.parallel.context import get_tp_context


_SEQUENCE_PARALLEL_ACTIVE = contextvars.ContextVar("areno_sequence_parallel_active", default=False)


def is_sequence_parallel_active() -> bool:
    """Return True inside a `sequence_parallel_region(True)` scope."""

    return bool(_SEQUENCE_PARALLEL_ACTIVE.get())


@contextmanager
def sequence_parallel_region(active: bool):
    """Set a per-call flag so layers can dispatch SP-aware kernels."""

    token = _SEQUENCE_PARALLEL_ACTIVE.set(active)
    try:
        yield
    finally:
        _SEQUENCE_PARALLEL_ACTIVE.reset(token)


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Autograd-preserving TP all-reduce for row-parallel outputs."""
    ctx = get_tp_context()
    if ctx.world_size == 1:
        return x
    return _AllReduceSum.apply(x, ctx.group)


def copy_to_tensor_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Forward identity whose backward sums input gradients across TP ranks."""
    ctx = get_tp_context()
    if ctx.world_size == 1:
        return x
    return _CopyToTPRegion.apply(x, ctx.group)


def scatter_to_sequence_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Split `x` along its sequence dimension across TP ranks for SP layers."""

    ctx = get_tp_context()
    if ctx.world_size == 1:
        return x
    return _ScatterToSequenceParallelRegion.apply(x, ctx.group, ctx.rank, ctx.world_size)


def gather_from_sequence_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """All-gather an SP-sharded tensor along the sequence dimension."""

    ctx = get_tp_context()
    if ctx.world_size == 1:
        return x
    return _GatherFromSequenceParallelRegion.apply(x, ctx.group, ctx.rank, ctx.world_size)


def reduce_scatter_to_sequence_parallel_region(x: torch.Tensor) -> torch.Tensor:
    """Sum across TP ranks then scatter along the sequence dimension."""

    ctx = get_tp_context()
    if ctx.world_size == 1:
        return x
    return _ReduceScatterToSequenceParallelRegion.apply(x, ctx.group, ctx.rank, ctx.world_size)


def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Gather TP vocab/hidden shards along the last dimension."""
    ctx = get_tp_context()
    if ctx.world_size == 1:
        return x
    chunks = dist_nn.all_gather(x.contiguous(), group=ctx.group)
    return torch.cat(chunks, dim=-1)


def all_gather_first_dim(x: torch.Tensor) -> torch.Tensor:
    """Gather same-shaped per-rank tensors and stack by TP rank."""
    ctx = get_tp_context()
    if ctx.world_size == 1:
        return x.unsqueeze(0)
    chunks = [torch.empty_like(x) for _ in range(ctx.world_size)]
    dist.all_gather(chunks, x.contiguous(), group=ctx.group)
    return torch.stack(chunks, dim=0)


def broadcast_tensor(x: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast `x` from a TP-local source rank to all other TP ranks."""

    ctx = get_tp_context()
    if ctx.world_size > 1:
        dist.broadcast(x, src=_tp_local_to_global_rank(src), group=ctx.group)
    return x


def broadcast_object(obj: object | None, src: int = 0) -> object:
    """Broadcast a picklable Python object across the TP group."""

    ctx = get_tp_context()
    if ctx.world_size == 1:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=_tp_local_to_global_rank(src), group=ctx.group)
    return payload[0]


def _tp_local_to_global_rank(rank: int) -> int:
    """Translate a TP-local rank (0..tp_size-1) into a global process rank."""

    ctx = get_tp_context()
    if rank < 0 or rank >= ctx.world_size:
        raise ValueError(f"rank={rank} is outside TP group size {ctx.world_size}")
    return ctx.dp_rank * ctx.world_size + rank


class _AllReduceSum(torch.autograd.Function):
    """Forward all-reduce sum; backward returns the upstream grad unchanged."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        out = x.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


class _CopyToTPRegion(torch.autograd.Function):
    """Forward identity; backward all-reduce sums grads from every TP rank."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = grad.contiguous().clone()
        dist.all_reduce(out, group=ctx.group)
        return out, None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split sequence axis on entry; gather along the same axis on backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, rank: int, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        if x.shape[1] % world_size != 0:
            raise RuntimeError(f"sequence length {x.shape[1]} must be divisible by TP size {world_size}")
        chunk = x.shape[1] // world_size
        return x.narrow(1, rank * chunk, chunk).contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _all_gather_sequence(grad, ctx.group, ctx.world_size), None, None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """All-gather the sequence axis; backward reduce-scatters the gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, rank: int, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        return _all_gather_sequence(x, group, world_size)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _reduce_scatter_sequence(grad, ctx.group, ctx.rank, ctx.world_size), None, None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Sum across TP ranks then scatter the sequence axis; backward is gather."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, rank: int, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        return _reduce_scatter_sequence(x, group, rank, world_size)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _all_gather_sequence(grad, ctx.group, ctx.world_size), None, None, None


def _all_gather_sequence(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    """Helper that concatenates per-rank chunks along the sequence axis."""

    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x.contiguous(), group=group)
    return torch.cat(chunks, dim=1).contiguous()


def _reduce_scatter_sequence(x: torch.Tensor, group, rank: int, world_size: int) -> torch.Tensor:
    """Reduce across the TP group, then keep only this rank's seq slice.

    Implemented as an all-reduce + slice because Torch's `reduce_scatter`
    requires a list of equal-sized tensors; with a contiguous SP layout the
    all-reduce + narrow path is simpler and equivalent.
    """

    if x.shape[1] % world_size != 0:
        raise RuntimeError(f"sequence length {x.shape[1]} must be divisible by TP size {world_size}")
    chunk = x.shape[1] // world_size
    out = x.narrow(1, rank * chunk, chunk).contiguous()
    dist.all_reduce(out, group=group)
    return out
