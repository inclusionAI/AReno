"""Shared response-layout helpers for policy losses.

The trainer can feed either packed variable-length tensors or padded
rectangular tensors. Loss functions should operate on the same semantic
fields in both cases: response mask, old logprobs, advantages, reference
logprobs, and optional sequence ids for per-sequence reductions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ResponseLayout:
    """Normalized view over packed and padded response-token tensors."""

    packed: bool
    response_mask: torch.Tensor
    valid_count: torch.Tensor
    response_len: torch.Tensor
    old_logprobs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    seq_ids: torch.Tensor | None = None
    num_sequences: int | None = None


def response_layout(
    data_pack: dict,
    logprobs: torch.Tensor,
    *,
    need_old_logprobs: bool = False,
    need_advantages: bool = False,
    need_ref_logprobs: bool = False,
    need_sequences: bool = False,
) -> ResponseLayout:
    """Return a common response-token view for packed or padded batches."""

    if "packed_response_mask" in data_pack:
        return _packed_response_layout(
            data_pack,
            logprobs,
            need_old_logprobs=need_old_logprobs,
            need_advantages=need_advantages,
            need_ref_logprobs=need_ref_logprobs,
            need_sequences=need_sequences,
        )
    return _padded_response_layout(
        data_pack,
        logprobs,
        need_old_logprobs=need_old_logprobs,
        need_advantages=need_advantages,
        need_ref_logprobs=need_ref_logprobs,
    )


def sequence_sum(values: torch.Tensor, layout: ResponseLayout) -> torch.Tensor:
    """Sum response-token values into one scalar per sequence."""

    masked = values * layout.response_mask.to(dtype=values.dtype)
    if not layout.packed:
        return masked.sum(dim=-1)
    if layout.seq_ids is None or layout.num_sequences is None:
        raise ValueError("packed sequence_sum requires seq_ids and num_sequences")
    out = torch.zeros(layout.num_sequences, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, layout.seq_ids, masked)
    return out


def masked_mean(values: torch.Tensor, layout: ResponseLayout) -> torch.Tensor:
    """Average values over response tokens only."""

    return (values * layout.response_mask.to(dtype=values.dtype)).sum() / layout.valid_count.to(dtype=values.dtype)


def _packed_response_layout(
    data_pack: dict,
    logprobs: torch.Tensor,
    *,
    need_old_logprobs: bool,
    need_advantages: bool,
    need_ref_logprobs: bool,
    need_sequences: bool,
) -> ResponseLayout:
    device = logprobs.device
    response_mask = data_pack["packed_response_mask"].to(device=device, dtype=torch.float32)
    valid_count = response_mask.sum().clamp(min=1)

    seq_ids = None
    num_sequences = None
    if need_sequences:
        seq_ids = data_pack["packed_seq_ids"].to(device=device, dtype=torch.long)
        num_sequences = int(data_pack["packed_num_sequences"])
        response_len = torch.zeros(num_sequences, device=device, dtype=torch.float32)
        response_len.scatter_add_(0, seq_ids, response_mask)
        response_len = response_len.clamp(min=1)
    else:
        response_len = valid_count

    old_logprobs = None
    if need_old_logprobs:
        old_logprobs = data_pack["packed_logprobs"].to(device=device, dtype=torch.float32)

    advantages = None
    if need_advantages:
        advantages = data_pack["packed_advantages"].to(device=device, dtype=torch.float32)

    ref_logprobs = None
    if need_ref_logprobs and "packed_ref_logprobs" in data_pack:
        ref_logprobs = data_pack["packed_ref_logprobs"].to(device=device, dtype=torch.float32)

    return ResponseLayout(
        packed=True,
        response_mask=response_mask,
        valid_count=valid_count,
        response_len=response_len,
        old_logprobs=old_logprobs,
        advantages=advantages,
        ref_logprobs=ref_logprobs,
        seq_ids=seq_ids,
        num_sequences=num_sequences,
    )


def _padded_response_layout(
    data_pack: dict,
    logprobs: torch.Tensor,
    *,
    need_old_logprobs: bool,
    need_advantages: bool,
    need_ref_logprobs: bool,
) -> ResponseLayout:
    device = logprobs.device
    response_mask = (~data_pack["prompt_mask"][:, 1:]).to(device=device, dtype=torch.float32)
    valid_count = response_mask.sum().clamp(min=1)
    response_len = response_mask.sum(dim=-1).clamp(min=1)

    old_logprobs = None
    if need_old_logprobs:
        old_logprobs = data_pack["logprobs"][:, 1:].to(device=device, dtype=torch.float32)

    advantages = None
    if need_advantages:
        advantages = data_pack["advantages"][:, 1:].to(device=device, dtype=torch.float32)

    ref_logprobs = None
    if need_ref_logprobs and "ref_logprobs" in data_pack:
        ref_logprobs = data_pack["ref_logprobs"][:, 1:].to(device=device, dtype=torch.float32)

    return ResponseLayout(
        packed=False,
        response_mask=response_mask,
        valid_count=valid_count,
        response_len=response_len,
        old_logprobs=old_logprobs,
        advantages=advantages,
        ref_logprobs=ref_logprobs,
    )
