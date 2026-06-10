"""Lightweight metadata dataclasses passed into the model forward.

`TrainMeta` and `InferMeta` describe the attention layout for one forward
call. They are intentionally split so a single model module can dispatch
between dense/packed training and prefill/decode inference based on which
metadata object the caller hands in. The runtime never alters these objects
after construction; they are pure value carriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(slots=True)
class TrainMeta:
    """Attention metadata for dense or packed training batches."""

    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    packed: bool = False
    sequence_parallel: bool = False
    activation_checkpointing: bool = False


@dataclass(slots=True)
class InferMeta:
    """Attention metadata for prefill/decode with paged KV cache.

    Prefill uses sequence lengths and cache write locations. Decode uses one row
    per active sequence plus a block table that maps logical positions to KV
    cache blocks.
    """

    mode: Literal["prefill", "decode"]
    sample_indices: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    cache_seqlens: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    cache_block_ids: torch.Tensor | None = None
    cache_block_offsets: torch.Tensor | None = None
