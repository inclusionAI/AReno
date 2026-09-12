"""Lightweight metadata dataclasses passed into the model forward.

`TrainMeta` and `InferMeta` describe the attention layout for one forward
call. They are intentionally split so a single model module can dispatch
between dense/packed training and prefill/decode inference based on which
metadata object the caller hands in. The runtime never alters these objects
after construction; they are pure value carriers, except for the capture
fields a forward fills in (routing, speculative recurrent state).
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
    num_padding_tokens: int = 0
    routing_replay: torch.Tensor | None = None
    # True when the training step wants MTP logits from models with MTP layers.
    mtp_enabled: bool = False


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
    # Decode only. Speculative verify feeds each active sequence this many
    # consecutive tokens (the sampled token plus its drafts), packed
    # sequence-major in the flat token axis, so `tokens == rows * tokens_per_seq`.
    tokens_per_seq: int = 1
    # Filled by recurrent layers during a verify forward (tokens_per_seq > 1),
    # stacked over recurrent layers so the commit is a handful of kernels:
    # states after every fed token (layers, rows, tokens_per_seq, heads, k, v)
    # and conv windows (layers, rows, convs, channels, kernel - 1 + tokens_per_seq).
    # Read back by `commit_speculative_state` once the accepted prefix is known.
    speculative_recurrent_states: torch.Tensor | None = None
    speculative_conv_windows: torch.Tensor | None = None
