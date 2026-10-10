"""Activation checkpoint helpers for model training forwards."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from areno.engine.parallel.collectives import sequence_parallel_region
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.mst import mini_sequence_forward


def _disable_dynamo_frame(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Keep the high-order checkpoint wrapper out of Dynamo's guard cache."""

    try:
        return torch._dynamo.disable(fn, recursive=False)
    except AttributeError:
        return fn


def should_checkpoint_layer(train_meta: TrainMeta | None, infer_meta: InferMeta | None) -> bool:
    """Return true when layer activation recompute is enabled for this forward."""

    return bool(
        torch.is_grad_enabled()
        and infer_meta is None
        and train_meta is not None
        and getattr(train_meta, "activation_checkpointing", False)
    )


def tokenwise_forward(
    function: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    *token_args: torch.Tensor,
    train_meta: TrainMeta | None = None,
    infer_meta: InferMeta | None = None,
) -> torch.Tensor:
    """Schedule an explicitly token-independent block; keep attention outside.

    Extra arguments must have one leading row per flattened token (for
    example fixed routes, route weights or modality masks). Routing itself
    must execute before this boundary, so replay and counters run only once.
    """
    size = getattr(train_meta, "mst_chunk_size", 0) if infer_meta is None else 0
    if size:
        return mini_sequence_forward(function, hidden_states, *token_args, chunk_size=size)
    return function(hidden_states, *token_args)


@_disable_dynamo_frame
def checkpoint_layer(
    layer_fn: Callable[..., Any],
    hidden_states: torch.Tensor,
    *args: Any,
    train_meta: TrainMeta | None = None,
    infer_meta: InferMeta | None = None,
    tokenwise: bool = False,
) -> Any:
    """Checkpoint one decoder layer, recomputing its activations in backward."""

    if tokenwise and infer_meta is None and getattr(train_meta, "mst_chunk_size", 0):
        return mini_sequence_forward(layer_fn, hidden_states, *args, chunk_size=train_meta.mst_chunk_size)
    if not should_checkpoint_layer(train_meta, infer_meta):
        return layer_fn(hidden_states, *args)

    def recompute(states: torch.Tensor) -> Any:
        # The backward recompute runs after the model's outer SP context has
        # exited. Restore it so column/row-parallel layers use the same
        # activation layout as the original forward.
        with sequence_parallel_region(bool(train_meta.sequence_parallel)):
            return layer_fn(states, *args)

    return checkpoint(
        recompute,
        hidden_states,
        use_reentrant=False,
        preserve_rng_state=True,
    )


@_disable_dynamo_frame
def checkpoint_routed_moe_layer(
    attention_fn: Callable[..., torch.Tensor],
    post_attention_norm: Callable[[torch.Tensor], torch.Tensor],
    route_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    expert_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    *attention_args: Any,
    train_meta: TrainMeta | None = None,
    infer_meta: InferMeta | None = None,
) -> torch.Tensor:
    """Checkpoint attention and experts while keeping dynamic routing fixed."""

    attended = checkpoint_layer(
        attention_fn,
        hidden_states,
        *attention_args,
        train_meta=train_meta,
        infer_meta=infer_meta,
    )
    normalized = post_attention_norm(attended)
    topk_idx, topk_weight = route_fn(normalized)
    expert_output = checkpoint_layer(
        expert_fn,
        normalized,
        topk_idx,
        topk_weight,
        train_meta=train_meta,
        infer_meta=infer_meta,
        tokenwise=True,
    )
    return attended + expert_output
