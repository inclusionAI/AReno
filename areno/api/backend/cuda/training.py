"""CUDA training-pack construction and metric reduction helpers."""

from __future__ import annotations

from collections.abc import Callable

import torch

from areno.api.algorithms import sft_loss_fn
from areno.api.backend.common import accumulation_steps
from areno.api.context import Context
from areno.api.models import TrainSequence


def make_train_pack(seqs: list[TrainSequence]) -> dict[str, torch.Tensor]:
    """Pack public training rows into right-padded CUDA engine tensors."""

    if not seqs:
        raise ValueError("train batch is empty")
    from areno.engine.runtime.common import pad_rows

    max_len = max(len(seq.tokens) for seq in seqs)
    lengths = torch.tensor([len(seq.tokens) for seq in seqs], dtype=torch.int32)
    input_ids = pad_rows([seq.tokens for seq in seqs], dtype=torch.long, fill_value=seqs[0].eos_token_id, width=max_len)
    prompt_mask = _make_prompt_mask(seqs, lengths, max_len)
    loss_mask = _make_loss_mask(seqs, lengths, max_len) if any(bool(seq.loss_mask) for seq in seqs) else None
    pack = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "lengths": lengths,
        "prompt_mask": prompt_mask,
        "logprobs": pad_rows([seq.logprobs for seq in seqs], dtype=torch.float32, width=max_len),
        "advantages": _make_advantages(seqs, prompt_mask, loss_mask, max_len),
    }
    if loss_mask is not None:
        pack["loss_mask"] = loss_mask
    for key, values in (
        ("returns", [seq.returns for seq in seqs]),
        ("values", [seq.values for seq in seqs]),
        ("ref_logprobs", [seq.ref_logprobs for seq in seqs]),
    ):
        if any(values):
            pack[key] = pad_rows(values, dtype=torch.float32, width=max_len)
    features = [seq.features for seq in seqs]
    if any(feature is not None for feature in features):
        pack["features"] = features
    return pack


def is_sft_loss_fn(loss_fn: Callable) -> bool:
    """Return whether a callable is the built-in SFT loss or its partial."""

    return loss_fn is sft_loss_fn or getattr(loss_fn, "func", None) is sft_loss_fn


def annotate_sft_token_mean_packs(
    packs: list[dict],
    target_counts: list[int],
    *,
    gradient_accumulation_steps: int | None,
) -> None:
    """Attach per-accumulation-group token normalizers for token-mean SFT."""

    if not packs:
        return
    accumulation = accumulation_steps(len(packs), gradient_accumulation_steps)
    for start in range(0, len(packs), accumulation):
        end = min(start + accumulation, len(packs))
        total = max(sum(target_counts[start:end]), 1)
        for pack in packs[start:end]:
            pack["_sft_total_target_tokens"] = total
            pack["_sft_grad_scale"] = end - start


def sft_target_token_count(seqs: list[TrainSequence]) -> int:
    """Count target tokens using the same shifted masks as training."""

    count = 0
    for seq in seqs:
        length = len(seq.tokens)
        for index in range(1, length):
            if seq.prompt_mask:
                target = index < len(seq.prompt_mask) and not seq.prompt_mask[index]
            else:
                target = index >= _sequence_prompt_len(seq)
            if seq.loss_mask:
                target = target and index < len(seq.loss_mask) and seq.loss_mask[index]
            count += int(target)
    return count


def pad_token_id(ctx: Context) -> int:
    """Resolve a tokenizer pad id for role scoring."""

    token_id = ctx.tokenizer.pad_token_id
    if token_id is None:
        token_id = ctx.tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(token_id)


def _make_prompt_mask(seqs: list[TrainSequence], lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    if all(not seq.prompt_mask and seq.prompt_len is not None for seq in seqs):
        prompt_lens = torch.tensor([int(seq.prompt_len or 0) for seq in seqs], dtype=torch.int64)
        positions = torch.arange(max_len, dtype=torch.int64).unsqueeze(0)
        return (positions < prompt_lens.unsqueeze(1)) | (positions >= lengths.to(torch.int64).unsqueeze(1))
    if any(not seq.prompt_mask and seq.prompt_len is not None for seq in seqs):
        result = torch.ones((len(seqs), max_len), dtype=torch.bool)
        positions = torch.arange(max_len, dtype=torch.int64)
        for row, seq in enumerate(seqs):
            if seq.prompt_mask:
                result[row, : len(seq.prompt_mask)] = torch.as_tensor(seq.prompt_mask, dtype=torch.bool)
            else:
                result[row] = (positions < int(seq.prompt_len or 0)) | (positions >= int(lengths[row].item()))
        return result
    from areno.engine.runtime.common import pad_rows

    return pad_rows([seq.prompt_mask for seq in seqs], dtype=torch.bool, fill_value=True, width=max_len)


def _make_loss_mask(seqs: list[TrainSequence], lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    result = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    positions = torch.arange(max_len, dtype=torch.int64)
    for row, seq in enumerate(seqs):
        if seq.loss_mask:
            result[row, : len(seq.loss_mask)] = torch.as_tensor(seq.loss_mask, dtype=torch.bool)
        else:
            result[row] = (positions >= _sequence_prompt_len(seq)) & (positions < int(lengths[row].item()))
    return result


def _make_advantages(
    seqs: list[TrainSequence],
    prompt_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    max_len: int,
) -> torch.Tensor:
    if all(not seq.advantages and seq.scalar_advantage is not None for seq in seqs):
        result = torch.zeros((len(seqs), max_len), dtype=torch.float32)
        active_mask = loss_mask if loss_mask is not None else ~prompt_mask
        for row, seq in enumerate(seqs):
            result[row, active_mask[row]] = float(seq.scalar_advantage or 0.0)
        return result
    from areno.engine.runtime.common import pad_rows

    return pad_rows([seq.advantages for seq in seqs], dtype=torch.float32, width=max_len)


def _sequence_prompt_len(seq: TrainSequence) -> int:
    if seq.prompt_len is not None:
        return int(seq.prompt_len)
    return next((index for index, is_prompt in enumerate(seq.prompt_mask) if not is_prompt), len(seq.prompt_mask))


__all__ = [
    "annotate_sft_token_mean_packs",
    "is_sft_loss_fn",
    "make_train_pack",
    "pad_token_id",
    "sft_target_token_count",
]
