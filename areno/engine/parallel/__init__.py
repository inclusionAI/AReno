"""Tensor-parallel / data-parallel context and collective primitives.

`context` owns the per-rank `TPContext` object that records this rank's place
inside the TP and DP groups. `collectives` exposes the autograd-aware
all-reduce/all-gather/scatter primitives that TP layers and rollout sampling
build on top of.
"""

from areno.engine.parallel.context import TPContext, get_tp_context

__all__ = ["TPContext", "get_tp_context"]

