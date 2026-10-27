"""Replicated scalar heads on a tensor/sequence-parallel backbone."""

import torch

from areno.engine.parallel.collectives import gather_from_sequence_parallel_region
from areno.engine.parallel.context import get_tp_context


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def gather_packed_hidden(hidden_states, train_meta):
    """Gather tokens; avoid multiplying backbone gradients by the TP size."""
    if train_meta.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)
        return _ScaleGradient.apply(hidden_states, 1.0 / get_tp_context().world_size)
    return hidden_states
