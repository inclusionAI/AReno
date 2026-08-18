"""Framework-neutral helpers shared by execution backends."""

from __future__ import annotations

from areno.api.models import RolloutResult, RolloutSequence


def expand_prompts(prompt_tokens: list[list[int]], n_samples: int) -> list[list[int]]:
    """Expand prompts in prompt-major, sample-minor order."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return [tokens for tokens in prompt_tokens for _ in range(n_samples)]


def expand_prompt_features(
    prompt_features: list[dict | None] | None,
    prompt_count: int,
    n_samples: int,
) -> list[dict | None] | None:
    """Validate and expand prompt-aligned side inputs like ``expand_prompts``."""

    if prompt_features is None:
        return None
    if len(prompt_features) != prompt_count:
        raise ValueError("prompt_features must have the same length as prompt_tokens")
    return [feature for feature in prompt_features for _ in range(n_samples)]


def group_rollout_sequences(
    sequences: list[RolloutSequence],
    prompt_count: int,
    n_samples: int,
) -> list[RolloutResult]:
    """Restore a flat prompt-major sequence list to the public result layout."""

    expected = prompt_count * n_samples
    if len(sequences) != expected:
        raise ValueError(f"backend returned {len(sequences)} sequences; expected {expected}")
    return [RolloutResult(sequences=sequences[start : start + n_samples]) for start in range(0, expected, n_samples)]


__all__ = ["expand_prompt_features", "expand_prompts", "group_rollout_sequences"]
