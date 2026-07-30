"""Classify and handle empty or invalid model completions.

During RL rollout the model may produce completions that carry no real
content: empty strings, whitespace-only text, special-token-only output, or
immediate-EOS generations.  These degenerate completions should not silently
reach ``reward_fn`` or training code because they pollute rewards and gradients.

This module provides a single entry point -- :func:`validate_completions` --
that inspects a batch of completions, classifies invalid ones, and applies a
configured policy (``filter`` / ``resample``).  The feature is
**off** by default; callers must pass ``policy != "off"`` to activate it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

InvalidType = Literal["empty", "whitespace", "special_token", "immediate_eos"]
Policy = Literal["off", "filter", "resample"]


@dataclass(slots=True)
class CompletionCheck:
    """Result of checking a single completion."""

    is_valid: bool
    invalid_type: InvalidType | None = None


@dataclass(slots=True)
class ValidationResult:
    """Outcome of :func:`validate_completions` for one batch.

    ``kept_indices`` / ``dropped_indices`` are positions into the original
    completions list.  ``metrics`` holds counters that callers should merge
    into step-level metrics.  ``quarantine_records`` holds structured records
    of every dropped completion for offline inspection.
    """

    kept_indices: list[int] = field(default_factory=list)
    dropped_indices: list[int] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    quarantine_records: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_completion(
    completion: str,
    resp_tokens: list[int],
    eos_token_ids: tuple[int, ...] = (),
    special_token_ids: tuple[int, ...] = (),
) -> CompletionCheck:
    """Classify a single completion as valid or invalid.

    Parameters
    ----------
    completion
        Decoded text of the model's response.
    resp_tokens
        Raw token ids of the response (before decoding).
    eos_token_ids
        Token ids that count as EOS for this tokenizer.
    special_token_ids
        Token ids that count as special (non-content) tokens.

    Notes
    -----
    All four invalid types are detectable from ``completion`` and
    ``resp_tokens`` alone; ``finish_reason`` from the engine is **not**
    required because ``RolloutSequence`` does not carry it.
    """

    # 1. No response tokens at all (some tokenizers decode this to "").
    if not resp_tokens:
        return CompletionCheck(is_valid=False, invalid_type="empty")

    # 2. Empty string.
    if completion == "":
        return CompletionCheck(is_valid=False, invalid_type="empty")

    # 3. Whitespace-only.
    if completion.strip() == "":
        return CompletionCheck(is_valid=False, invalid_type="whitespace")

    # 4. Immediate EOS: the response is a single token and it is an EOS token.
    if len(resp_tokens) == 1 and resp_tokens[0] in eos_token_ids:
        return CompletionCheck(is_valid=False, invalid_type="immediate_eos")

    # 5. Special-token-only: every token in the response is a special token.
    if resp_tokens and special_token_ids:
        if all(tid in special_token_ids for tid in resp_tokens):
            return CompletionCheck(is_valid=False, invalid_type="special_token")

    return CompletionCheck(is_valid=True)


# ---------------------------------------------------------------------------
# Batch validation + policy application
# ---------------------------------------------------------------------------

def validate_completions(
    completions: list[str],
    resp_tokens_list: list[list[int]],
    *,
    policy: Policy = "off",
    eos_token_ids: tuple[int, ...] = (),
    special_token_ids: tuple[int, ...] = (),
    resample_budget: int = 3,
    quarantine_path: str | Path | None = None,
    prompt: str | None = None,
) -> tuple[list[str], list[list[int]], ValidationResult]:
    """Validate a batch of completions and apply the configured policy.

    Returns a 3-tuple ``(filtered_completions, filtered_tokens, result)``
    containing only the completions that survived validation, plus the
    :class:`ValidationResult` with metrics and quarantine records.

    When ``policy == "off"`` (the default) the inputs are returned unchanged
    and ``result`` contains no dropped indices.
    """

    result = ValidationResult()

    if policy == "off":
        result.kept_indices = list(range(len(completions)))
        return completions, resp_tokens_list, result

    # Classify every completion.
    checks = [
        classify_completion(
            completion=completion,
            resp_tokens=tokens,
            eos_token_ids=eos_token_ids,
            special_token_ids=special_token_ids,
        )
        for completion, tokens in zip(completions, resp_tokens_list, strict=True)
    ]

    # Tally counters by invalid type.
    type_counts: dict[str, int] = {}
    for check in checks:
        if not check.is_valid and check.invalid_type is not None:
            type_counts[check.invalid_type] = type_counts.get(check.invalid_type, 0) + 1

    total_invalid = sum(type_counts.values())
    total_valid = len(completions) - total_invalid

    # Apply policy: filter and resample both drop invalid rows so they never
    # reach reward_fn or training code.  For resample, the caller is expected
    # to re-generate dropped rows up to ``resample_budget`` times; this
    # function only reports which rows need resampling.
    kept_completions: list[str] = []
    kept_tokens: list[list[int]] = []

    for idx, (completion, tokens, check) in enumerate(
        zip(completions, resp_tokens_list, checks, strict=True)
    ):
        if check.is_valid:
            result.kept_indices.append(idx)
            kept_completions.append(completion)
            kept_tokens.append(tokens)
        else:
            result.dropped_indices.append(idx)
            result.quarantine_records.append(
                {
                    "index": idx,
                    "invalid_type": check.invalid_type,
                    "completion": completion[:500],  # truncate to avoid huge records
                    "resp_token_count": len(tokens),
                    "prompt": prompt[:500] if prompt else None,
                    "policy": policy,
                }
            )

    # Build metrics.
    metrics: dict[str, float] = {
        "completion_total": float(len(completions)),
        "completion_valid": float(total_valid),
        "completion_invalid": float(total_invalid),
    }
    for invalid_type, count in type_counts.items():
        metrics[f"completion_invalid_{invalid_type}"] = float(count)

    if policy == "filter":
        metrics["completion_filtered"] = float(len(result.dropped_indices))
    elif policy == "resample":
        metrics["completion_resample_candidates"] = float(len(result.dropped_indices))
        metrics["completion_resample_budget"] = float(resample_budget)

    result.metrics = metrics

    # Write quarantine file if requested.
    if quarantine_path and result.quarantine_records:
        _write_quarantine(quarantine_path, result.quarantine_records)

    if result.dropped_indices:
        logger.warning(
            "completion_validator: dropped %d/%d completions (policy=%s, types=%s)",
            len(result.dropped_indices),
            len(completions),
            policy,
            type_counts,
        )

    return kept_completions, kept_tokens, result


def get_special_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Extract special token ids from a HuggingFace tokenizer.

    Covers ``all_special_ids`` and the EOS tokens that some tokenizers list
    separately.
    """

    ids: set[int] = set()
    all_special = getattr(tokenizer, "all_special_ids", [])
    ids.update(all_special)

    # Some tokenizers expose additional special tokens via added_tokens.
    added = getattr(tokenizer, "added_tokens_encoder", {})
    for token_id in added.values():
        ids.add(int(token_id))

    return tuple(sorted(ids))


def _write_quarantine(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Append invalid completion records to a JSONL quarantine file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
