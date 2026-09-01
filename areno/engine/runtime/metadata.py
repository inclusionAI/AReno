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
    fp8_checkpoint_activations: bool = False
    fp8_checkpoint_group_size: int = 128
    fp8_checkpoint_stochastic: bool = False
    fp8_checkpoint_warmup_steps: int = 0
    fp8_checkpoint_fallback_layers: tuple[int, ...] = ()
    global_step: int = 0
    num_padding_tokens: int = 0
    routing_replay: torch.Tensor | None = None


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
    recurrent_slots: torch.Tensor | None = None
    capture_routing: bool = False
    captured_routing: dict[int, torch.Tensor] | None = None
