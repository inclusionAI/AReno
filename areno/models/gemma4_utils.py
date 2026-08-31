"""Lightweight Gemma 4 helpers that do not import optional acceleration kernels."""

from __future__ import annotations

import torch


def keep_frozen_modules_in_eval(modules) -> None:
    """Keep frozen feature extractors deterministic when their parent trains."""

    for module in modules:
        if module is not None:
            module.eval()


def text_embedding_ids(
    input_ids: torch.Tensor,
    *,
    modality_token_ids: tuple[int | None, ...],
    pad_token_id: int,
) -> torch.Tensor:
    """Replace multimodal soft-token ids before the text embedding lookup."""

    token_ids = [int(value) for value in modality_token_ids if value is not None]
    if not token_ids:
        return input_ids
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in token_ids:
        mask |= input_ids == token_id
    return torch.where(mask, torch.full_like(input_ids, int(pad_token_id)), input_ids)
