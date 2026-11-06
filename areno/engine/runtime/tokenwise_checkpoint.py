"""Use identical tokenwise arithmetic with saved or recomputed activations."""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from areno.engine.parallel.collectives import sequence_parallel_region


class _GradientAccumulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter):
        ctx.dtype = parameter.dtype
        # Only the gradient needs parameter-sized FP32 storage. The forward
        # value is a one-element expanded placeholder, never a weight copy.
        return parameter.new_zeros((), dtype=torch.float32).expand(parameter.shape)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to(ctx.dtype)


class _ParameterValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, accumulator, data):
        return data

    @staticmethod
    def backward(ctx, gradient):
        return gradient.float(), None


class _Method(nn.Module):
    def __init__(self, function, owner):
        super().__init__()
        self.block = owner
        self.function = function.__call__ if isinstance(function, nn.Module) else function

    def forward(self, *args):
        return self.function(*args)


@torch._dynamo.disable
def tokenwise_checkpoint(
    function,
    hidden_states,
    *token_args,
    chunk_size: int,
    recompute: bool,
    checkpoint_chunks: bool,
    sequence_parallel: bool = False,
):
    """Keep the same tile graph, changing only activation retention.

    Routing is computed by the caller and its fixed results are sliced with
    the input. Low-precision parameter partials accumulate in FP32 before a
    single cast, for both direct execution and activation checkpointing.
    """
    if chunk_size < 1:
        raise ValueError("Tokenwise chunk size must be positive")
    rows = hidden_states.numel() // hidden_states.shape[-1]
    if any(value.shape[0] != rows for value in token_args):
        raise ValueError("Tokenwise arguments must have one row per token")
    owner = function if isinstance(function, nn.Module) else getattr(function, "__self__", None)
    low_precision = {}
    if torch.is_grad_enabled():
        # Resolve tied aliases once. Passing all of them explicitly lets
        # functional_call skip a full module-tree scan in every tile/replay.
        for name, parameter in owner.named_parameters(remove_duplicate=False):
            if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
                key = id(parameter)
                if key not in low_precision:
                    low_precision[key] = ([], _GradientAccumulator.apply(parameter), parameter.detach())
                low_precision[key][0].append(f"block.{name}")
    method = _Method(function, owner) if low_precision else None

    def forward_tiles(states, *arguments):
        flat = states.reshape(-1, states.shape[-1])
        outputs = []
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            chunk = flat[start:end]
            if states.ndim == 3:
                chunk = chunk.unsqueeze(0)
            values = (chunk, *(arg[start:end] for arg in arguments))
            parameters = {}
            for names, accumulator, data in low_precision.values():
                value = _ParameterValue.apply(accumulator, data)
                parameters.update((name, value) for name in names)

            def execute(*inputs, parameters=parameters):
                with sequence_parallel_region(sequence_parallel):
                    return (
                        functional_call(method, parameters, inputs, tie_weights=False)
                        if parameters
                        else function(*inputs)
                    )

            result = (
                checkpoint(execute, *values, use_reentrant=False, preserve_rng_state=True)
                if recompute and checkpoint_chunks
                else execute(*values)
            )
            outputs.append(result.reshape(-1, states.shape[-1]))
        if not outputs:
            return function(states, *arguments)
        return torch.cat(outputs, dim=0).view_as(states)

    if recompute and not checkpoint_chunks:
        return checkpoint(forward_tiles, hidden_states, *token_args, use_reentrant=False, preserve_rng_state=True)
    return forward_tiles(hidden_states, *token_args)
