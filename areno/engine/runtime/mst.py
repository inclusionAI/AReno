"""Mini-sequence execution for token-independent training blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from areno.engine.parallel.collectives import is_sequence_parallel_active, sequence_parallel_region


def training_chunk_size(device_type: str, tp_size: int) -> int:
    """Select the internal training schedule only for supported execution paths."""
    # Token partitioning under TP/SP needs a separate collective schedule.
    # This policy depends on execution layout, never on the model family.
    return 1024 if device_type == "cuda" and tp_size == 1 else 0


class _RecomputedBlock(torch.autograd.Function):
    """Preserve the canonical forward geometry; chunk its recomputed VJP."""

    @staticmethod
    def forward(ctx, execute, chunk_size, input_count, *tensors):
        ctx.execute = execute
        ctx.chunk_size = chunk_size
        ctx.input_count = input_count
        ctx.save_for_backward(*tensors)
        inputs = tensors[:input_count]
        ctx.device = inputs[0].device
        ctx.cpu_rng = torch.get_rng_state()
        ctx.cuda_rng = torch.cuda.get_rng_state(ctx.device) if ctx.device.type == "cuda" else None
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device.type)
        # Grad-enabled dispatch must match the ordinary model forward. The
        # checkpoint suppresses saved expanded tensors before we discard its
        # temporary graph; only boundary inputs survive in this Function.
        with torch.enable_grad():
            output = checkpoint(execute, *inputs, use_reentrant=False).detach()
        ctx.stochastic = not torch.equal(ctx.cpu_rng, torch.get_rng_state())
        if ctx.cuda_rng is not None:
            ctx.stochastic |= not torch.equal(ctx.cuda_rng, torch.cuda.get_rng_state(ctx.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        inputs = tensors[: ctx.input_count]
        parameters = tensors[ctx.input_count :]
        flat = inputs[0].reshape(-1, inputs[0].shape[-1])
        input_grads = [torch.zeros_like(value) if value.requires_grad else None for value in inputs]
        parameter_grads = [None] * len(parameters)
        size = flat.shape[0] if ctx.stochastic else ctx.chunk_size
        # Stochastic blocks must replay the whole draw, since splitting a
        # dropout draw changes its mask. Keep the caller's RNG state intact.
        devices = [ctx.device.index] if ctx.cuda_rng is not None else []
        with torch.random.fork_rng(devices=devices, enabled=ctx.stochastic):
            if ctx.stochastic:
                torch.set_rng_state(ctx.cpu_rng)
                if ctx.cuda_rng is not None:
                    torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device)
            for start in range(0, flat.shape[0], size):
                end = min(start + size, flat.shape[0])
                states = flat[start:end]
                if inputs[0].ndim == 3:
                    states = states.unsqueeze(0)
                chunks = [states, *(value[start:end] for value in inputs[1:])]
                chunks = [
                    value.detach().requires_grad_(original.requires_grad) for value, original in zip(chunks, inputs)
                ]
                targets = [value for value in chunks if value.requires_grad] + list(parameters)
                with (
                    torch.enable_grad(),
                    torch.autocast(ctx.device.type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype),
                ):
                    output = ctx.execute(*chunks)
                    grads = torch.autograd.grad(
                        output,
                        targets,
                        grad_output.reshape(-1, grad_output.shape[-1])[start:end].view_as(output),
                        allow_unused=True,
                    )
                offset = 0
                for index, value in enumerate(chunks):
                    if not value.requires_grad:
                        continue
                    gradient = grads[offset]
                    offset += 1
                    if gradient is not None:
                        destination = input_grads[index]
                        if index == 0:
                            destination = destination.reshape_as(flat)
                        destination[start:end] = gradient.reshape_as(destination[start:end])
                for index, gradient in enumerate(grads[offset:]):
                    if gradient is None:
                        continue
                    # Sum partial parameter gradients before the one cast to
                    # BF16/FP16; 512 + 1 - 512 must retain the contribution 1.
                    gradient = gradient.float() if gradient.dtype in (torch.bfloat16, torch.float16) else gradient
                    if parameter_grads[index] is None:
                        parameter_grads[index] = gradient
                    else:
                        parameter_grads[index].add_(gradient)
        parameter_grads = [
            value.to(parameter.dtype) if value is not None else None
            for value, parameter in zip(parameter_grads, parameters)
        ]
        return (None, None, None, *input_grads, *parameter_grads)


@torch._dynamo.disable
def mini_sequence_forward(function, hidden_states, *token_args, chunk_size: int):
    """Recompute each token chunk separately during backward.

    Attention must remain outside this boundary. Routed MoE arguments are
    computed once on the full input, then sliced alongside its tokens so
    routing replay and router counters retain their original semantics.
    """
    if chunk_size < 1:
        raise ValueError("MST chunk_size must be positive")
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if any(arg.shape[0] != flat.shape[0] for arg in token_args):
        raise ValueError("MST routing arguments must have one row per token")
    if not flat.shape[0]:
        return function(hidden_states, *token_args)
    sequence_parallel = is_sequence_parallel_active()
    owner = function if isinstance(function, nn.Module) else getattr(function, "__self__", None)

    def execute(*inputs):
        # Backward runs after the outer model scope has exited.
        with sequence_parallel_region(sequence_parallel):
            return function(*inputs)

    if torch.is_grad_enabled() and flat.shape[0] > chunk_size and isinstance(owner, nn.Module):
        parameters = tuple(parameter for parameter in owner.parameters() if parameter.requires_grad)
        return _RecomputedBlock.apply(execute, chunk_size, len(token_args) + 1, hidden_states, *token_args, *parameters)

    outputs = []
    for start in range(0, flat.shape[0], chunk_size):
        end = start + chunk_size
        states = flat[start:end]
        if hidden_states.ndim == 3:
            states = states.unsqueeze(0)
        inputs = (states, *(arg[start:end] for arg in token_args))
        if torch.is_grad_enabled():
            output = checkpoint(execute, *inputs, use_reentrant=False, preserve_rng_state=True)
        else:
            output = function(*inputs)
        outputs.append(output.reshape(-1, hidden_states.shape[-1]))
    return torch.cat(outputs, dim=0).view_as(hidden_states)
