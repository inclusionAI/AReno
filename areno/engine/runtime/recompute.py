"""Activation checkpoint helpers for model training forwards."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from areno.engine.runtime.metadata import InferMeta, TrainMeta


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
        and train_meta.activation_checkpointing
    )


@_disable_dynamo_frame
def checkpoint_layer(
    layer_fn: Callable[..., Any],
    hidden_states: torch.Tensor,
    *args: Any,
    train_meta: TrainMeta | None = None,
    infer_meta: InferMeta | None = None,
) -> Any:
    """Checkpoint one decoder layer, recomputing its activations in backward."""

    if not should_checkpoint_layer(train_meta, infer_meta):
        return layer_fn(hidden_states, *args)
    return checkpoint(
        lambda states: layer_fn(states, *args),
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
    )
    return attended + expert_output
