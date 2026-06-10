"""Tensor-parallel linear projections.

Three building blocks share the same `_areno_linear_forward` matmul: column,
merged-column (a stack of column projections that share the same input) and
row parallel linears. Column parallel splits the output dim across ranks,
producing locally distinct activations that can either stay sharded or be
all-gathered. Row parallel splits the input dim, so each rank computes a
partial result that must be all-reduced (or reduce-scattered when sequence
parallelism is on) to recover the global output.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from areno.accel import areno_linear
from areno.engine.parallel.collectives import (
    all_gather_last_dim,
    all_reduce,
    copy_to_tensor_parallel_region,
    gather_from_sequence_parallel_region,
    is_sequence_parallel_active,
    reduce_scatter_to_sequence_parallel_region,
)
from areno.engine.parallel.context import get_tp_context


def mark_tensor_parallel_parameter(
    param: nn.Parameter | None,
    is_parallel: bool,
    *,
    sequence_parallel: bool = False,
    tp_grad_allreduce: bool = False,
) -> None:
    """Tag a parameter with TP/SP attributes consumed by the optimizer."""

    if param is not None:
        setattr(param, "tensor_model_parallel", is_parallel)
        setattr(param, "sequence_parallel", sequence_parallel)
        setattr(param, "tp_grad_allreduce", tp_grad_allreduce)


def _shard_range(size: int, rank: int, world_size: int) -> tuple[int, int]:
    """Compute ``[start, end)`` of the local shard for an even partition."""

    if size % world_size != 0:
        raise ValueError(f"cannot shard size {size} across {world_size} ranks")
    part = size // world_size
    return rank * part, (rank + 1) * part


def _kv_shard_range(num_kv_heads: int, head_dim: int, rank: int, world_size: int) -> tuple[int, int]:
    """Shard KV heads, replicating whole heads when TP is wider than KV count."""

    kv_size = num_kv_heads * head_dim
    if num_kv_heads % world_size == 0:
        return _shard_range(kv_size, rank, world_size)
    if world_size % num_kv_heads != 0:
        raise ValueError(f"cannot shard or replicate {num_kv_heads} KV heads across {world_size} ranks")
    ranks_per_kv_head = world_size // num_kv_heads
    kv_head = rank // ranks_per_kv_head
    start = kv_head * head_dim
    return start, start + head_dim


class ColumnParallelLinear(nn.Module):
    """Linear that splits ``out_features`` across the TP group.

    Each rank stores a ``(local_out, in)`` slice of the weight. The input is
    replicated (or gathered from a sequence-parallel region) before the
    matmul; the result stays sharded along the last dim unless
    ``gather_output`` requests an all-gather to materialize the full output.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, gather_output: bool = False):
        super().__init__()
        ctx = get_tp_context()
        start, end = _shard_range(out_features, ctx.rank, ctx.world_size)
        self.in_features = in_features
        self.out_features = out_features
        self.local_out_features = end - start
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        self.bias = nn.Parameter(torch.empty(self.local_out_features)) if bias else None
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=True)
        mark_tensor_parallel_parameter(self.bias, True, sequence_parallel=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In sequence-parallel mode the activation is sharded along the
        # sequence dim, so we all-gather it back to the full sequence before
        # the matmul; otherwise we just copy through the TP region.
        x = gather_from_sequence_parallel_region(x) if is_sequence_parallel_active() else copy_to_tensor_parallel_region(x)
        out = _areno_linear_forward(x, self.weight, self.bias)
        if self.gather_output:
            # Concatenate column-shards along the last dim to recover the
            # full output (only used when downstream code needs it dense).
            out = all_gather_last_dim(out)
        return out


class MergedColumnParallelLinear(nn.Module):
    """Column-parallel projection over a tuple of stacked output sizes.

    Useful for fusing several column-parallel weights that share the same
    input (e.g. gate/up in SwiGLU). Each entry of ``out_features`` is sharded
    independently and concatenated into a single weight to amortize the
    matmul, while ``local_out_features`` records the per-entry slice sizes
    so callers can split the output back.
    """

    def __init__(self, in_features: int, out_features: list[int] | tuple[int, ...], bias: bool = False):
        super().__init__()
        ctx = get_tp_context()
        if not out_features:
            raise ValueError("out_features must not be empty")
        self.in_features = in_features
        self.out_features = tuple(out_features)
        self.local_out_features = []
        # Shard each output sub-block independently so that the fused weight
        # is the concatenation of per-block local shards.
        for size in self.out_features:
            start, end = _shard_range(size, ctx.rank, ctx.world_size)
            self.local_out_features.append(end - start)
        self.weight = nn.Parameter(torch.empty(sum(self.local_out_features), in_features))
        self.bias = nn.Parameter(torch.empty(sum(self.local_out_features))) if bias else None
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=True)
        mark_tensor_parallel_parameter(self.bias, True, sequence_parallel=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = gather_from_sequence_parallel_region(x) if is_sequence_parallel_active() else copy_to_tensor_parallel_region(x)
        return _areno_linear_forward(x, self.weight, self.bias)


class QKVParallelLinear(MergedColumnParallelLinear):
    """Specialized merged column-parallel projection for fused Q/K/V.

    Builds a single ``[Q | K | V]`` weight on each rank where the Q block is
    sharded across ``num_heads`` query heads and the K/V blocks across
    ``num_kv_heads`` KV heads. Exposes ``shard_ranges`` for weight loaders
    that need to map global head indices to local slices.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
    ):
        nn.Module.__init__(self)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        ctx = get_tp_context()
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        # Independent shard ranges for Q, K, V so GQA configs (where K/V have
        # fewer heads than Q) shard each block over the same TP group.
        q_range = _shard_range(q_size, ctx.rank, ctx.world_size)
        k_range = _kv_shard_range(num_kv_heads, head_dim, ctx.rank, ctx.world_size)
        self.in_features = hidden_size
        self.out_features = (q_size, kv_size, kv_size)
        self.shard_ranges = (q_range, k_range, k_range)
        self.local_out_features = [end - start for start, end in self.shard_ranges]
        self.weight = nn.Parameter(torch.empty(sum(self.local_out_features), hidden_size))
        self.bias = nn.Parameter(torch.empty(sum(self.local_out_features))) if bias else None
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=True)
        mark_tensor_parallel_parameter(self.bias, True, sequence_parallel=True)
        self.reset_parameters()


class RowParallelLinear(nn.Module):
    """Linear that splits ``in_features`` across the TP group.

    Each rank multiplies its locally-owned input slice by a ``(out, local_in)``
    weight, producing a partial sum. To recover the full output we either
    all-reduce (default) or reduce-scatter back into a sequence-parallel
    region. The bias is replicated and added only after the cross-rank sum.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, input_is_parallel: bool = True):
        super().__init__()
        ctx = get_tp_context()
        start, end = _shard_range(in_features, ctx.rank, ctx.world_size)
        self.in_features = in_features
        self.out_features = out_features
        self.local_in_features = end - start
        self.input_is_parallel = input_is_parallel
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        # Bias lives on each rank as a replica (not TP-sharded) and is added
        # post-reduction so it is not summed `world_size` times.
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=True)
        mark_tensor_parallel_parameter(self.bias, False, sequence_parallel=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # Caller passed the full input; slice out the local input shard.
            ctx = get_tp_context()
            start, end = _shard_range(self.in_features, ctx.rank, ctx.world_size)
            x = x[..., start:end]
        out = _areno_linear_forward(x, self.weight, None)
        # Partial sum -> cross-rank reduction. SP mode also re-shards along
        # the sequence dim via reduce-scatter, saving activation memory.
        out = reduce_scatter_to_sequence_parallel_region(out) if is_sequence_parallel_active() else all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


def _areno_linear_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Single entry point so all parallel linears share the areno.accel matmul."""

    return areno_linear(x, weight, bias)
