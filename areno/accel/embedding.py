"""Tensor-parallel vocab embedding gather backed by an ARENO CUDA kernel.

In TP-sharded embedding tables each rank only holds a contiguous slice of the
vocabulary ``[vocab_start, vocab_end)``. The kernel gathers rows for ids that
fall inside that range and writes zeros for out-of-range ids so the subsequent
all-reduce reconstructs the full embedding.
"""

import torch

from areno.accel._extension import extension as _extension
from areno.accel.utils import on_kernel_device


class _VocabEmbedding(torch.autograd.Function):
    """Autograd glue for the vocab-parallel embedding gather/scatter."""

    @staticmethod
    def forward(ctx, input_ids: torch.Tensor, weight: torch.Tensor, vocab_start: int, vocab_end: int) -> torch.Tensor:
        out = _extension(input_ids.device).areno_vocab_embedding_forward(
            input_ids.contiguous(), weight.contiguous(), int(vocab_start), int(vocab_end)
        )
        ctx.save_for_backward(input_ids, weight)
        ctx.vocab_start = int(vocab_start)
        ctx.vocab_end = int(vocab_end)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None, None]:
        input_ids, weight = ctx.saved_tensors
        grad_weight = _extension(grad_output.device).areno_vocab_embedding_backward(
            grad_output.contiguous(),
            input_ids.contiguous(),
            weight,
            ctx.vocab_start,
            ctx.vocab_end,
        )
        return None, grad_weight, None, None


@torch._dynamo.disable
def areno_vocab_embedding(
    input_ids: torch.Tensor, weight: torch.Tensor, vocab_start: int, vocab_end: int
) -> torch.Tensor:
    """Gather rank-local vocab embeddings and zero out non-local token ids.

    ``input_ids`` must be int64 on CUDA. ``weight`` is the local shard with
    shape ``(vocab_end - vocab_start, hidden)``. Returns embeddings with shape
    ``(*input_ids.shape, hidden)`` ready for tensor-parallel reduction.
    """
    if not on_kernel_device(input_ids, weight):
        raise RuntimeError("areno_vocab_embedding requires CUDA or NPU input_ids and weight on the same device")
    if input_ids.dtype != torch.long:
        raise TypeError("areno_vocab_embedding input_ids must be int64")
    return _VocabEmbedding.apply(input_ids, weight, int(vocab_start), int(vocab_end))
