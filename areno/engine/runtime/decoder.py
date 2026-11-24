"""Shared memory optimizations for CUDA language decoders."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext

import torch
from torch import nn

from areno.engine.parallel.collectives import is_sequence_parallel_active
from areno.engine.runtime.metadata import TrainMeta


def project_lm_head(
    head: nn.Module,
    hidden_states: torch.Tensor,
    *,
    defer: bool,
    logits_indices: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Project requested prefill rows while leaving full decoder states accessible."""
    if defer:
        return None
    # Sequence-parallel training gathers global positions inside the head.
    if logits_indices is not None and not is_sequence_parallel_active():
        hidden_states = hidden_states.index_select(1, logits_indices)
        logits_indices = None
    logits = head(hidden_states if dtype is None else hidden_states.to(dtype=dtype))
    return logits if logits_indices is None else logits.index_select(1, logits_indices)


def decoder_activation_offload(
    states: torch.Tensor, train_meta: TrainMeta | None, *, legacy_bailing_v3: bool = False
) -> AbstractContextManager:
    """Optionally store the decoder's autograd-saved tensors in host memory."""
    if train_meta is not None and states.is_cuda and torch.is_grad_enabled():
        setting = os.getenv("ARENO_OFFLOAD_ACTIVATIONS")
        if setting is None and legacy_bailing_v3:
            setting = os.getenv("ARENO_BAILING_OFFLOAD_ACTIVATIONS")
        if setting == "1":
            return torch.autograd.graph.save_on_cpu()
    return nullcontext()
