"""Mini-sequence execution for token-independent training blocks."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from areno.engine.parallel.collectives import is_sequence_parallel_active, sequence_parallel_region


def training_chunk_size(device_type: str, tp_size: int) -> int:
    """Select the internal training schedule only for supported execution paths."""
    # Token partitioning under TP/SP needs a separate collective schedule.
    # This policy depends on execution layout, never on the model family.
    if device_type != "cuda" or tp_size != 1:
        return 0
    # Keep recomputation allocations small and predictable by default.
    size = int(os.environ.get("ARENO_MST_CHUNK_SIZE", "256"))
    if size < 0:
        raise ValueError("ARENO_MST_CHUNK_SIZE must be nonnegative")
    return size


logger = logging.getLogger(__name__)
_TRAIN_PROGRESS = ContextVar("mst_train_progress", default=None)


class _TrainingProgress:
    """Wall-clock feedback for the current microbatch, not the whole job."""

    def __init__(self):
        self.total = 0
        self.completed = 0
        self.started = None
        self.last_log = None

    def begin(self):
        if self.started is None:
            self.started = self.last_log = time.monotonic()

    def advance(self, count, chunk):
        self.completed += count
        now = time.monotonic()
        if self.started is None or now - self.last_log < 15 or not self.completed:
            return
        self.last_log = now
        elapsed = now - self.started
        # Attention VJPs and optimizer work are not part of this token count;
        # leave a margin and explicitly label this as a rough estimate.
        remaining = elapsed / self.completed * max(0, self.total - self.completed) * 1.25 + 5
        finish = datetime.now(timezone.utc) + timedelta(seconds=remaining)
        logger.info(
            "train ETA remaining~%.1fs estimated_finish=%s MST_progress=%d/%d chunk=%d (rough estimate)",
            remaining,
            finish.isoformat(timespec="seconds"),
            self.completed,
            self.total,
            chunk,
        )


@contextmanager
def mst_training_progress():
    token = _TRAIN_PROGRESS.set(_TrainingProgress())
    try:
        yield
    finally:
        _TRAIN_PROGRESS.reset(token)


def _chunk_vjp(ctx, inputs, parameters, flat, grad_output, start, end):
    # Release each temporary graph before processing the next chunk.
    # No input/parameter gradient is committed here.
    states = flat[start:end]
    if inputs[0].ndim == 3:
        states = states.unsqueeze(0)
    chunks = [states, *(value[start:end] for value in inputs[1:])]
    chunks = [value.detach().requires_grad_(original.requires_grad) for value, original in zip(chunks, inputs)]
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
    return grads


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
        ctx.progress = _TRAIN_PROGRESS.get() if ctx.device.type == "cuda" else None
        if ctx.progress is not None:
            ctx.progress.total += inputs[0].numel() // inputs[0].shape[-1]
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
        if ctx.progress is not None:
            ctx.progress.begin()
        devices = [ctx.device.index] if ctx.cuda_rng is not None else []
        with torch.random.fork_rng(devices=devices, enabled=ctx.stochastic):
            if ctx.stochastic:
                torch.set_rng_state(ctx.cpu_rng)
                if ctx.cuda_rng is not None:
                    torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device)
            start = 0
            while start < flat.shape[0]:
                end = min(start + size, flat.shape[0])
                grads = _chunk_vjp(ctx, inputs, parameters, flat, grad_output, start, end)
                offset = 0
                for index, value in enumerate(inputs):
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
                if ctx.progress is not None:
                    ctx.progress.advance(end - start, size)
                start = end
                del grads
                gradient = None
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
