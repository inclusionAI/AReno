"""Helpers for multimodal token/feature alignment."""

from __future__ import annotations

from typing import Any, Sequence

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
            raise ValueError(
                f"image_grid_thw patches={patches} is not divisible by spatial_merge_size**2={merge_unit}"
            )
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
