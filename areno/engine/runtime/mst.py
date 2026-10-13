"""Mini-sequence execution for token-independent training blocks."""

from __future__ import annotations

import logging
import os

import psutil
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
    # Internal tuning for runs with memory headroom: larger chunks amortize
    # recomputation and repeated QLoRA dequantization without changing the CLI.
    size = int(os.environ.get("ARENO_MST_CHUNK_SIZE", "1024"))
    if size < 0:
        raise ValueError("ARENO_MST_CHUNK_SIZE must be nonnegative")
    return size


logger = logging.getLogger(__name__)


class _AdaptiveChunks:
    """Conservative feedback without resetting the trainer's peak counters.

    CUDA reserved memory retains transient allocation pressure after tensors
    die. Device-wide free memory also includes other processes; host available
    memory accounts for CPU allocations competing with CUDA on unified RAM.
    This is feedback, not a hard cap on allocations outside MST.
    """

    def __init__(self, device, initial, maximum):
        self.device = device
        self.size = min(initial, 1024, maximum)
        self.maximum = maximum
        self.stable = 0

    def pressure(self):
        free, total = torch.cuda.mem_get_info(self.device)
        host = psutil.virtual_memory()
        return max(1 - free / total, 1 - host.available / host.total)

    def observe(self, pressure):
        old = self.size
        if pressure >= 0.8:
            self.size = max(1, self.size // 2)
            # Remember an unsafe size for this block instead of oscillating.
            self.maximum = min(self.maximum, self.size)
            self.stable = 0
        elif pressure < 0.7:
            self.stable += 1
            if self.stable >= 2:
                self.size = min(self.maximum, self.size + max(1, self.size // 4))
                self.stable = 0
        else:
            self.stable = 0
        if self.size != old:
            logger.debug("MST chunk %d -> %d, memory pressure %.3f", old, self.size, pressure)

    def oom(self):
        self.size = max(1, self.size // 2)
        self.maximum = self.size
        self.stable = 0
        logger.warning("MST allocation failed; retrying uncommitted chunk with %d tokens", self.size)


def _adaptive_chunks(device, initial, maximum):
    return _AdaptiveChunks(device, initial, maximum) if device.type == "cuda" else None


def _chunk_vjp(ctx, inputs, parameters, flat, grad_output, start, end):
    # Keep temporary graphs in a separate frame so a failed allocation can be
    # released before retrying. No input/parameter gradient is committed here.
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
        adaptive = _adaptive_chunks(ctx.device, size, flat.shape[0]) if not ctx.stochastic else None
        # Stochastic blocks must replay the whole draw, since splitting a
        # dropout draw changes its mask. Keep the caller's RNG state intact.
        devices = [ctx.device.index] if ctx.cuda_rng is not None else []
        with torch.random.fork_rng(devices=devices, enabled=ctx.stochastic):
            if ctx.stochastic:
                torch.set_rng_state(ctx.cpu_rng)
                if ctx.cuda_rng is not None:
                    torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device)
            start = 0
            while start < flat.shape[0]:
                if adaptive is not None:
                    pressure = adaptive.pressure()
                    if pressure >= 0.8:
                        adaptive.observe(pressure)
                    size = adaptive.size
                end = min(start + size, flat.shape[0])
                retry = False
                try:
                    grads = _chunk_vjp(ctx, inputs, parameters, flat, grad_output, start, end)
                except torch.cuda.OutOfMemoryError:
                    if adaptive is None or size <= 1:
                        raise
                    adaptive.oom()
                    retry = True
                if retry:
                    # Leave the exception scope first: its traceback owns the
                    # failed graph. Successful chunks are never replayed.
                    torch.cuda.empty_cache()
                    continue
                if adaptive is not None:
                    pressure = adaptive.pressure()
                    adaptive.observe(pressure)
                    if pressure >= 0.8:
                        # Release the previous chunk's cached high-water mark;
                        # otherwise reserved memory would force repeated
                        # shrinking even after the live tensors are gone.
                        torch.cuda.empty_cache()
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
