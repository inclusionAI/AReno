"""Vocabulary-parallel embedding and LM head.

Both modules shard the vocabulary across the TP group along the first
weight dim. The embedding masks out-of-range token IDs locally and uses an
all-reduce to assemble the global embedding from each rank's zero-padded
local contribution. The LM head produces local logits over each rank's
vocab shard; downstream cross-entropy code is expected to perform the
TP-aware softmax over the union of shards.
"""

from __future__ import annotations

import torch
from torch import nn

from areno.accel import areno_vocab_embedding
from areno.engine.parallel.collectives import all_reduce, copy_to_tensor_parallel_region, gather_from_sequence_parallel_region, is_sequence_parallel_active
from areno.engine.parallel.context import get_tp_context
from areno.engine.layers.linear import _areno_linear_forward, _shard_range, mark_tensor_parallel_parameter


class VocabParallelEmbedding(nn.Module):
    """Token embedding with the vocabulary split across the TP group.

    Each rank owns rows ``[vocab_start, vocab_end)``; the fused kernel
    returns the row for any in-range token and zero for tokens outside the
    local shard. A cross-rank all-reduce then sums the per-rank embeddings
    (only one rank contributes a non-zero row per token) to yield the full
    embedding everywhere.
    """

    def __init__(self, vocab_size: int, hidden_size: int, *, dtype: torch.dtype | None = None):
        super().__init__()
        ctx = get_tp_context()
        self.vocab_size = vocab_size
        # Local row range of this rank in the global vocabulary.
        self.vocab_start, self.vocab_end = _shard_range(vocab_size, ctx.rank, ctx.world_size)
        self.weight = nn.Parameter(torch.empty(self.vocab_end - self.vocab_start, hidden_size, dtype=dtype))
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=True)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Kernel handles the local mask: token < start or >= end yields zeros.
        out = areno_vocab_embedding(input_ids, self.weight, self.vocab_start, self.vocab_end)
        # All-reduce so each rank ends up with the full embedding (exactly
        # one rank contributed a non-zero row for each token).
        return all_reduce(out)


class VocabParallelLMHead(nn.Module):
    """LM head producing per-rank vocab-sharded logits.

    The weight shares the vocab-parallel layout with the embedding; the
    matmul output of each rank covers its local vocabulary slice. Callers
    typically feed the result to a TP-aware cross-entropy that gathers the
    local softmax denominators across ranks to compute the global softmax.
    """

    def __init__(self, hidden_size: int, vocab_size: int, *, dtype: torch.dtype | None = None):
        super().__init__()
        ctx = get_tp_context()
        self.vocab_size = vocab_size
        self.vocab_start, self.vocab_end = _shard_range(vocab_size, ctx.rank, ctx.world_size)
        self.weight = nn.Parameter(torch.empty(self.vocab_end - self.vocab_start, hidden_size, dtype=dtype))
        mark_tensor_parallel_parameter(self.weight, True, sequence_parallel=True)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Reassemble the full hidden activation: in SP mode it is sharded
        # along the sequence dim and must be gathered; otherwise we just
        # pass through the TP boundary.
        hidden_states = gather_from_sequence_parallel_region(hidden_states) if is_sequence_parallel_active() else copy_to_tensor_parallel_region(hidden_states)
        return _areno_linear_forward(hidden_states, self.weight, None)
