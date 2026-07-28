"""Dataset tokenization helpers shared by offline trainers.

This module also provides configurable dataset field mapping, constant-field
injection, and sample filtering utilities used before AReno contract conversion.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from areno.api.tokenizer import apply_chat_template_with_options, encode_generation_prompt, normalize_token_ids

logger = logging.getLogger(__name__)


def apply_chat_template(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    """Encode full chat messages, with a plain-text fallback for base tokenizers."""

    if getattr(tokenizer, "chat_template", None):
        return normalize_token_ids(
            apply_chat_template_with_options(tokenizer, messages, tokenize=True, add_generation_prompt=False)
        )
    text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)
    return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))


def encode_prompt_value(tokenizer, prompt) -> list[int]:
    """Encode a DPO prompt that may be either plain text or chat messages."""

    if isinstance(prompt, list):
        if getattr(tokenizer, "chat_template", None):
            return normalize_token_ids(
                apply_chat_template_with_options(tokenizer, prompt, tokenize=True, add_generation_prompt=True)
            )
        text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in prompt)
        return encode_generation_prompt(tokenizer, text)
    return encode_generation_prompt(tokenizer, prompt)


def prompt_response_to_tokens_and_mask(
    prompt: str, response: str, tokenizer, eos_token_id: int
) -> tuple[list[int], list[bool]]:
    """Encode prompt text plus response text and mask the prompt prefix."""

    prompt_ids = encode_generation_prompt(tokenizer, prompt)
    return response_to_tokens_and_mask(prompt_ids, response, tokenizer, eos_token_id)


def response_to_tokens_and_mask(
    prompt_ids: list[int], response: str, tokenizer, eos_token_id: int
) -> tuple[list[int], list[bool]]:
    """Append a response to pre-tokenized prompt ids and mask prompt tokens."""

    response_ids = normalize_token_ids(tokenizer.encode(response, add_special_tokens=False))
    if eos_token_id is not None and (not response_ids or response_ids[-1] != eos_token_id):
        response_ids.append(eos_token_id)
    return prompt_ids + response_ids, [True] * len(prompt_ids) + [False] * len(response_ids)


def has_any(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    """Return whether a record has any string field in keys."""

    return any(isinstance(record.get(key), str) for key in keys)


def first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first string field for required text schemas."""

    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    raise KeyError(keys[0])


def first_value(record: dict[str, Any], keys: tuple[str, ...]):
    """Return the first string/list field for optional preference schemas."""

    for key in keys:
        value = record.get(key)
        if isinstance(value, str | list):
            return value
    return None


# ---------------------------------------------------------------------------
# Configurable dataset field mapping, constant fields, and sample filtering.
#
# These utilities let users declaratively rename dataset fields, inject
# constant fields, and filter samples by field presence or text length —
# all before AReno's internal contract conversion.  When none of the
# options are provided the dataset is returned unchanged so existing
# behaviour is fully preserved.
# ---------------------------------------------------------------------------


@dataclass
class FilterSummary:
    """Aggregated keep/drop counts and per-reason breakdowns.

    ``total_in`` is the number of records inspected; ``total_kept`` and
    ``total_dropped`` reconcile to it.  ``drop_reasons`` maps a human-readable
    reason string to the number of records dropped for that reason.
    """

    total_in: int = 0
    total_kept: int = 0
    total_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def as_log_lines(self) -> list[str]:
        """Return human-readable summary lines for CLI / log output."""

        lines = [
            f"field-mapping: scanned={self.total_in} kept={self.total_kept} dropped={self.total_dropped}",
        ]
        for reason, count in sorted(self.drop_reasons.items()):
            lines.append(f"  drop reason: {reason} ({count})")
        return lines


def apply_field_mapping(record: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Rename fields in *record* according to *mapping*.

    For each ``{source: target}`` pair, if *source* is present and *target*
    is not, the value is moved.  Fields that already have the *target* name
    are left untouched so users can safely include identity mappings.
    """

    for source, target in mapping.items():
        if source == target:
            continue
        if source in record and target not in record:
            record[target] = record.pop(source)
    return record


def apply_constant_fields(record: dict[str, Any], constants: dict[str, Any]) -> dict[str, Any]:
    """Inject constant key/value pairs into *record* without overwriting existing keys."""

    for key, value in constants.items():
        record.setdefault(key, value)
    return record


def check_sample_filter(
    record: dict[str, Any], filter_config: dict[str, Any]
) -> tuple[bool, str | None]:
    """Return ``(keep, reason)`` for *record* against *filter_config*.

    Supported keys in *filter_config*:
    - ``require_fields``: list[str] — all must be present (and non-empty for strings)
    - ``min_prompt_chars``: int — minimum character length of the ``prompt`` field
    - ``max_prompt_chars``: int — maximum character length of the ``prompt`` field
    - ``min_response_chars``: int — minimum character length of the ``response`` field
    """

    require_fields: list[str] = filter_config.get("require_fields", [])
    for fname in require_fields:
        if fname not in record:
            return False, f"missing field: {fname}"
        value = record[fname]
        if isinstance(value, str) and not value:
            return False, f"empty field: {fname}"

    min_prompt = filter_config.get("min_prompt_chars")
    if min_prompt is not None and "prompt" in record:
        prompt_val = record["prompt"]
        if isinstance(prompt_val, str) and len(prompt_val) < min_prompt:
            return False, f"prompt too short ({len(prompt_val)} < {min_prompt})"

    max_prompt = filter_config.get("max_prompt_chars")
    if max_prompt is not None and "prompt" in record:
        prompt_val = record["prompt"]
        if isinstance(prompt_val, str) and len(prompt_val) > max_prompt:
            return False, f"prompt too long ({len(prompt_val)} > {max_prompt})"

    min_response = filter_config.get("min_response_chars")
    if min_response is not None and "response" in record:
        resp_val = record["response"]
        if isinstance(resp_val, str) and len(resp_val) < min_response:
            return False, f"response too short ({len(resp_val)} < {min_response})"

    return True, None


def transform_dataset(
    dataset: Sequence[dict[str, Any]],
    *,
    field_mapping: dict[str, str] | None = None,
    constant_fields: dict[str, Any] | None = None,
    sample_filter: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], FilterSummary]:
    """Apply field mapping, constant injection, and filtering to a dataset.

    Returns a tuple of ``(kept_records, summary)``.  When all options are
    ``None`` the dataset is shallow-copied and returned unchanged with a
    summary that reports every record as kept — this guarantees backward
    compatibility when the feature is not used.

    The input dataset is **not** mutated; each kept record is a shallow copy
    so downstream code can freely modify it.
    """

    summary = FilterSummary(total_in=len(dataset))

    if not field_mapping and not constant_fields and not sample_filter:
        kept = [dict(record) for record in dataset]
        summary.total_kept = len(kept)
        return kept, summary

    field_mapping = field_mapping or {}
    constant_fields = constant_fields or {}
    sample_filter = sample_filter or {}

    kept: list[dict[str, Any]] = []
    for record in dataset:
        # Work on a copy so the original dataset is never mutated.
        transformed = dict(record)

        apply_field_mapping(transformed, field_mapping)
        apply_constant_fields(transformed, constant_fields)

        if sample_filter:
            ok, reason = check_sample_filter(transformed, sample_filter)
            if not ok:
                summary.total_dropped += 1
                summary.drop_reasons[reason] = summary.drop_reasons.get(reason, 0) + 1
                continue

        kept.append(transformed)

    summary.total_kept = len(kept)
    return kept, summary


def parse_json_option(value: str | None, option_name: str) -> dict[str, Any] | None:
    """Parse a CLI JSON-string option into a dict with a clear error on failure."""

    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must be a JSON object, got {type(parsed).__name__}")
    return parsed
