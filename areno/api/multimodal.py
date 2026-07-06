"""Helpers for multimodal token/feature alignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def image_token_counts_from_features(features: dict[str, Any] | None) -> list[int]:
    """Return merged visual-token counts for each image in ``features``.

    Qwen-style image processors place one placeholder token per image in the
    rendered text, while the vision tower emits one embedding per merged patch
    group. The language input must therefore repeat each image placeholder by
    the corresponding merged visual-token count before the model can replace
    those token embeddings with visual embeddings.
    """

    if not features:
        return []
    grid = features.get("image_grid_thw")
    if grid is None:
        return []
    if not isinstance(grid, torch.Tensor):
        grid = torch.as_tensor(grid)
    grid = grid.detach().cpu().to(dtype=torch.long).reshape(-1, 3)
    merge = int(features.get("spatial_merge_size", features.get("merge_size", 2)) or 2)
    merge_unit = merge * merge
    counts: list[int] = []
    for t, h, w in grid.tolist():
        patches = int(t) * int(h) * int(w)
        if patches % merge_unit:
            raise ValueError(f"image_grid_thw patches={patches} is not divisible by spatial_merge_size**2={merge_unit}")
        counts.append(patches // merge_unit)
    return counts


def expand_image_tokens(
    tokens: Sequence[int],
    *,
    image_token_id: int | None,
    image_token_counts: Sequence[int],
    aligned_sequences: dict[str, Sequence[Any]] | None = None,
) -> tuple[list[int], dict[str, list[Any]]]:
    """Expand one image placeholder per image into merged visual-token slots.

    ``aligned_sequences`` may contain masks or per-token arrays with the same
    length as ``tokens``; each value at an expanded image-token position is
    repeated by the same visual-token count.
    """

    out_tokens: list[int] = []
    out_aligned = {name: [] for name in (aligned_sequences or {})}
    count_idx = 0
    image_token_id = int(image_token_id) if image_token_id is not None else None
    for idx, token in enumerate(tokens):
        repeat = 1
        if image_token_id is not None and int(token) == image_token_id and count_idx < len(image_token_counts):
            repeat = int(image_token_counts[count_idx])
            count_idx += 1
        out_tokens.extend([int(token)] * repeat)
        for name, values in (aligned_sequences or {}).items():
            out_aligned[name].extend([values[idx]] * repeat)
    if count_idx != len(image_token_counts):
        raise ValueError(
            "image feature count does not match prompt image token count: "
            f"features={len(image_token_counts)} prompt_tokens={count_idx}"
        )
    return out_tokens, out_aligned


def mrope_position_ids_from_image_grid(
    tokens: Sequence[int],
    *,
    image_token_id: int | None,
    features: dict[str, Any] | None,
) -> torch.Tensor | None:
    """Build Qwen3.5-VL MRoPE ids for tokens after image-token expansion.

    This follows the image-only fast path used by SGLang: text spans advance a
    scalar position, image spans use (t, h, w) grid positions at merged-patch
    resolution, and the following text resumes at max(t, h, w) after the image.
    """

    if image_token_id is None or not features:
        return None
    grid = features.get("image_grid_thw")
    if grid is None:
        return None
    if not isinstance(grid, torch.Tensor):
        grid = torch.as_tensor(grid)
    grid = grid.detach().cpu().to(dtype=torch.long).reshape(-1, 3)
    merge = int(features.get("spatial_merge_size", features.get("merge_size", 2)) or 2)
    image_token_id = int(image_token_id)
    token_list = [int(token) for token in tokens]
    segments: list[torch.Tensor] = []
    st = 0
    next_pos = 0
    for t_tensor, h_tensor, w_tensor in grid:
        t, h, w = int(t_tensor), int(h_tensor), int(w_tensor)
        if h % merge or w % merge:
            raise ValueError("image_grid_thw height/width must be divisible by spatial_merge_size")
        llm_t, llm_h, llm_w = t, h // merge, w // merge
        count = llm_t * llm_h * llm_w
        try:
            start = _find_image_span(token_list, image_token_id, st, count)
        except ValueError as exc:
            raise ValueError("image_grid_thw count does not match expanded image tokens") from exc
        text_len = start - st
        if text_len > 0:
            segments.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + next_pos)
            next_pos += text_len
        end = start + count
        t_index = torch.arange(llm_t, dtype=torch.long).view(-1, 1).expand(llm_t, llm_h * llm_w).reshape(-1)
        h_index = torch.arange(llm_h, dtype=torch.long).view(1, -1, 1).expand(llm_t, llm_h, llm_w).reshape(-1)
        w_index = torch.arange(llm_w, dtype=torch.long).view(1, 1, -1).expand(llm_t, llm_h, llm_w).reshape(-1)
        segments.append(torch.stack([t_index, h_index, w_index]) + next_pos)
        next_pos += max(llm_t, llm_h, llm_w)
        st = end
    if st < len(token_list):
        text_len = len(token_list) - st
        segments.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + next_pos)
    return torch.cat(segments, dim=1) if segments else None


def _find_image_span(tokens: list[int], image_token_id: int, start: int, count: int) -> int:
    if count <= 0:
        raise ValueError("image token count must be positive")
    for idx in range(start, len(tokens) - count + 1):
        if all(token == image_token_id for token in tokens[idx : idx + count]):
            return idx
    raise ValueError("expanded image token span does not match image_grid_thw")
