"""Composed reward example: correctness + format + brevity.

Demonstrates the compose_reward_fn API for combining multiple weighted
reward dimensions. Use with:

    areno train --reward-fn-path examples/math/composed_reward.py ...

This file still exposes ``reward_fn`` as required by ``load_reward_fn``,
so the CLI and trainer code are completely unchanged.
"""

from __future__ import annotations

from areno.api.rewards import RewardRecord, compose_reward_fn


def correctness_reward(record: RewardRecord) -> float:
    """1.0 if the completion contains the boxed ground truth answer."""

    solutions = record.answer
    if solutions is None:
        return 0.0
    ground_truth = solutions[0] if isinstance(solutions, list) else solutions
    # Extract the last \boxed{...} expression, handling nested braces.
    boxed = _extract_boxed(record.completion)
    if not boxed:
        return 0.0
    return 1.0 if boxed[-1].strip() == str(ground_truth).strip() else 0.0


def _extract_boxed(text: str) -> list[str]:
    """Extract content inside all \\boxed{...} expressions, handling nested braces."""

    results = []
    search_from = 0
    prefix = r"\boxed{"
    while True:
        idx = text.find(prefix, search_from)
        if idx == -1:
            break
        brace_start = idx + len(prefix) - 1  # position of the opening '{'
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[brace_start + 1 : i])
                    search_from = i + 1
                    break
        else:
            # Unbalanced braces; stop searching.
            break
    return results


def format_reward(record: RewardRecord) -> float:
    """1.0 if the completion contains a \\boxed{} expression, else 0.0."""

    return 1.0 if r"\boxed{" in record.completion else 0.0


def brevity_reward(record: RewardRecord) -> float:
    """Shorter completions score higher, clamped to [0, 1].

    Completions of 2000+ characters score 0.0.
    """

    # 2000 chars is a rough cap for concise math answers; adjust per use case.
    return max(0.0, 1.0 - len(record.completion) / 2000.0)


# Compose: correctness 0.7, format 0.2, brevity 0.1
_composed = compose_reward_fn(
    [
        ("correctness", correctness_reward, 0.7),
        ("format", format_reward, 0.2),
        ("brevity", brevity_reward, 0.1),
    ]
)


def reward_fn(record: RewardRecord) -> float:
    """Weighted reward composed of correctness, format, and brevity."""

    return _composed(record)


# Expose component metadata for downstream metrics wiring.
reward_fn._reward_components = _composed._reward_components  # type: ignore[attr-defined]
