"""Activation checkpoint helpers for model training forwards."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from areno.engine.parallel.collectives import sequence_parallel_region
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.mst import mini_sequence_forward
from areno.engine.runtime.tokenwise_checkpoint import tokenwise_checkpoint


def _disable_dynamo_frame(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Keep the high-order checkpoint wrapper out of Dynamo's guard cache."""

    try:
        return torch._dynamo.disable(fn, recursive=False)
    except AttributeError:
        return fn


def should_checkpoint_layer(
    train_meta: TrainMeta | None,
    infer_meta: InferMeta | None,
    *,
    module: str | None = None,
    layer_idx: int | None = None,
) -> bool:
    """Return true when layer activation recompute is enabled for this forward."""

    enabled = bool(
        torch.is_grad_enabled()
        and infer_meta is None
        and train_meta is not None
        and getattr(train_meta, "activation_checkpointing", False)
    )
    policy = getattr(train_meta, "recompute_policy", None)
    if enabled and policy is not None and module is not None and layer_idx is not None:
        return policy.selects(module, layer_idx)
    return enabled


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
    module: str | None = None,
    layer_idx: int | None = None,
) -> Any:
    """Checkpoint one decoder layer, recomputing its activations in backward."""

    selected = should_checkpoint_layer(train_meta, infer_meta, module=module, layer_idx=layer_idx)
    owner = layer_fn if isinstance(layer_fn, torch.nn.Module) else getattr(layer_fn, "__self__", None)
    if (
        tokenwise
        and train_meta is not None
        and infer_meta is None
        and getattr(owner, "mst_consistent_tiles", False)
        and hidden_states.device.type in ("cuda", "cpu")
        and get_tp_context().world_size == 1
        and not train_meta.sequence_parallel
    ):
        return tokenwise_checkpoint(
            layer_fn,
            hidden_states,
            *args,
            chunk_size=train_meta.mst_chunk_size or 256,
            recompute=selected,
            checkpoint_chunks=bool(train_meta.mst_chunk_size),
            sequence_parallel=False,
        )
    # An explicit selection must also gate MST, which otherwise precedes the
    # legacy global checkpoint switch and can still recompute excluded MLPs.
    if getattr(train_meta, "recompute_policy", None) is not None and module is not None and not selected:
        return layer_fn(hidden_states, *args)
    if tokenwise and infer_meta is None and getattr(train_meta, "mst_chunk_size", 0):
        return mini_sequence_forward(layer_fn, hidden_states, *args, chunk_size=train_meta.mst_chunk_size)
    if not selected:
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
