"""Rollout-routing capture and training replay for sparse MoE routers.

The runtime installs one short-lived context around a model forward.  MoE
routers use :func:`resolve_softmax_routes` or :func:`resolve_sigmoid_routes`
to either record their selected expert ids during rollout, or replace the
fresh training top-k decision with the recorded ids.  Replayed weights are
always recomputed from the current router logits so gradients still flow to
the current router parameters.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta


@dataclass(slots=True)
class _RoutingState:
    replay: torch.Tensor | None = None
    captured: dict[int, torch.Tensor] = field(default_factory=dict)


_ROUTING_STATE: ContextVar[_RoutingState | None] = ContextVar("areno_routing_replay", default=None)


@contextmanager
def routing_replay_context(meta: TrainMeta | InferMeta | None) -> Iterator[None]:
    """Expose capture/replay metadata to routers for one model forward."""

    replay = meta.routing_replay if isinstance(meta, TrainMeta) else None
    capture = bool(isinstance(meta, InferMeta) and meta.capture_routing)
    if replay is None and not capture:
        yield
        return
    state = _RoutingState(replay=replay)
    if isinstance(meta, InferMeta):
        meta.captured_routing = None
    token = _ROUTING_STATE.set(state)
    try:
        yield
    finally:
        _ROUTING_STATE.reset(token)
        if isinstance(meta, InferMeta):
            meta.captured_routing = state.captured


def captured_routing(meta: InferMeta) -> torch.Tensor | None:
    """Return captured routes as ``(tokens, moe_layers, top_k)``."""

    captured = meta.captured_routing
    if not captured:
        return None
    slots = sorted(captured)
    if slots != list(range(len(slots))):
        raise RuntimeError(f"MoE routing slots must be contiguous from zero, got {slots}")
    tensors = [captured[slot] for slot in slots]
    token_count = int(tensors[0].shape[0])
    top_k = int(tensors[0].shape[1])
    for slot, routes in enumerate(tensors):
        if routes.ndim != 2 or tuple(routes.shape) != (token_count, top_k):
            raise RuntimeError(
                f"captured MoE routes at slot {slot} have shape {tuple(routes.shape)}, expected {(token_count, top_k)}"
            )
    return torch.stack(tensors, dim=1)


# The routing state is Python-owned and its capture dictionary mutates once per
# MoE layer. Letting Dynamo inspect it specializes a graph for every dictionary
# state/layer slot and quickly exhausts the compile cache on deep MoE models.
# CUDA graph capture still records the tensor operations launched here.
@torch._dynamo.disable
def resolve_softmax_routes(
    layer_slot: int,
    logits: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    *,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture or replay a softmax router's expert ids."""

    state = _ROUTING_STATE.get()
    if state is None:
        return topk_idx, topk_weight
    replayed = _replayed_ids(state, layer_slot, logits, topk_idx)
    if replayed is not None:
        replayed_ids, replay_mask = replayed
        if renormalize:
            replayed_weight = torch.softmax(logits.float().gather(-1, replayed_ids), dim=-1)
        else:
            replayed_weight = torch.softmax(logits.float(), dim=-1).gather(-1, replayed_ids)
        topk_idx = replayed_ids
        topk_weight = torch.where(replay_mask.unsqueeze(-1), replayed_weight, topk_weight)
    _capture_ids(state, layer_slot, topk_idx)
    return topk_idx, topk_weight


@torch._dynamo.disable
def resolve_sigmoid_routes(
    layer_slot: int,
    logits: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture or replay a normalized sigmoid router's expert ids."""

    state = _ROUTING_STATE.get()
    if state is None:
        return topk_idx, topk_weight
    replayed = _replayed_ids(state, layer_slot, logits, topk_idx)
    if replayed is not None:
        replayed_ids, replay_mask = replayed
        scores = torch.sigmoid(logits.float().gather(-1, replayed_ids))
        replayed_weight = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        topk_idx = replayed_ids
        topk_weight = torch.where(replay_mask.unsqueeze(-1), replayed_weight, topk_weight)
    _capture_ids(state, layer_slot, topk_idx)
    return topk_idx, topk_weight


def _replayed_ids(
    state: _RoutingState,
    layer_slot: int,
    logits: torch.Tensor,
    dynamic_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    replay = state.replay
    if replay is None:
        return None
    if replay.ndim != 3:
        raise ValueError(f"routing replay must have shape (tokens, layers, top_k), got {tuple(replay.shape)}")
    if layer_slot < 0 or layer_slot >= int(replay.shape[1]):
        raise ValueError(f"routing replay is missing MoE layer slot {layer_slot}")
    layer_replay = replay[:, layer_slot, :]
    if int(layer_replay.shape[0]) != int(logits.shape[0]):
        ctx = get_tp_context()
        if int(layer_replay.shape[0]) != int(logits.shape[0]) * ctx.world_size:
            raise ValueError(
                f"routing replay token count {layer_replay.shape[0]} does not match router token count "
                f"{logits.shape[0]} (or its TP-sharded layout)"
            )
        start = ctx.rank * int(logits.shape[0])
        layer_replay = layer_replay.narrow(0, start, int(logits.shape[0]))
    ids = layer_replay.to(device=logits.device, dtype=torch.long)
    missing = ids < 0
    replay_mask = ~missing.all(dim=-1)
    resolved = torch.where(replay_mask.unsqueeze(-1), ids, dynamic_ids.to(dtype=torch.long))
    return resolved, replay_mask


def _capture_ids(state: _RoutingState, layer_slot: int, topk_idx: torch.Tensor) -> None:
    if state.replay is not None:
        return
    if layer_slot < 0:
        raise ValueError("MoE routing layer slot must be non-negative")
    if layer_slot in state.captured:
        raise RuntimeError(f"MoE routing layer slot {layer_slot} was captured more than once in one forward")
    state.captured[layer_slot] = topk_idx.detach()
