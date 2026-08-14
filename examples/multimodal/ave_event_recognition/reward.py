"""LLM-judged semantic reward for AVE event-label lists."""

from __future__ import annotations

import json
import logging
import math
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)
JSON_QUOTE_TRANSLATION = str.maketrans({"\u201c": '"', "\u201d": '"'})


def reward_fn(record: Any) -> float:
    """Return normalized semantic event-list similarity from 0 to 1."""

    prediction = _tool_prediction(getattr(record, "tool_calls", None))
    if prediction is None:
        return 0.0
    expected = _label_list(record.source_record.get("event_classes"))
    predicted = _label_list(prediction.get("events"))
    if expected is None or predicted is None:
        return 0.0
    return _judge_event_similarity(tuple(expected), tuple(predicted)) / 10.0


try:
    reward_fn.parallel_workers = max(1, min(64, int(os.environ.get("JUDGE_MAX_WORKERS", "16"))))
except ValueError as exc:
    raise RuntimeError("JUDGE_MAX_WORKERS must be an integer from 1 to 64") from exc


def _tool_prediction(tool_calls: Any) -> dict[str, Any] | None:
    matching = [call for call in tool_calls or [] if isinstance(call, dict) and call.get("name") == "report_events"]
    if len(matching) != 1:
        return None
    arguments = matching[0].get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return arguments if isinstance(arguments, dict) else None


def _label_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    labels = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    if len(labels) != len(value):
        return None
    return sorted(set(labels), key=str.casefold)


def _env(name: str) -> str:
    value = os.environ.get(name) or os.environ.get(name.lower())
    if not value:
        raise RuntimeError(f"AVE judge reward requires {name}")
    return value


def _judge_event_similarity(expected: tuple[str, ...], predicted: tuple[str, ...]) -> float:
    return _judge_request(
        _env("JUDGE_BASE_URL"),
        _env("JUDGE_MODEL"),
        _env("JUDGE_API_KEY"),
        expected,
        predicted,
    )


@lru_cache(maxsize=4096)
def _judge_request(
    base_url: str,
    model: str,
    api_key: str,
    expected: tuple[str, ...],
    predicted: tuple[str, ...],
) -> float:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("AVE judge reward requires the openai package included with AReno") from exc

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=2)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=32,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You score the semantic similarity of a predicted audiovisual event-label set against a "
                    "reference set from 0 to 10. Treat ordinary synonyms, singular/plural variants, and equivalent "
                    "specific/common names as matches. Score 10 only when the sets are semantically equivalent with "
                    "no missing or extra events; score 7-9 for a nearly complete match with a minor specificity "
                    "difference; score 4-6 for meaningful partial overlap; score 1-3 for weakly related events; "
                    "score 0 when unrelated. Penalize every missing or extra event. Use a floating-point score to "
                    'express partial similarity and reply with exactly one JSON object such as {"score": 7.35}, '
                    "with no other fields or text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"reference_event_labels": expected, "predicted_event_labels": predicted},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        body = json.loads(content.translate(JSON_QUOTE_TRANSLATION))
    except json.JSONDecodeError as exc:
        logger.warning(
            "AVE judge returned invalid response model=%s expected=%s predicted=%s response=%r",
            model,
            expected,
            predicted,
            content,
        )
        raise RuntimeError(f"judge returned invalid JSON: {content!r}") from exc
    if (
        not isinstance(body, dict)
        or set(body) != {"score"}
        or isinstance(body["score"], bool)
        or not isinstance(body["score"], (int, float))
    ):
        logger.warning(
            "AVE judge returned invalid body model=%s expected=%s predicted=%s response=%r",
            model,
            expected,
            predicted,
            content,
        )
        raise RuntimeError(f"judge returned an invalid similarity body: {body!r}")
    score = float(body["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 10.0:
        logger.warning(
            "AVE judge returned out-of-range score model=%s expected=%s predicted=%s response=%r",
            model,
            expected,
            predicted,
            content,
        )
        raise RuntimeError(f"judge returned an out-of-range similarity score: {score}")
    logger.info(
        "AVE judge result model=%s expected=%s predicted=%s response=%r score=%.3f normalized_reward=%.3f",
        model,
        expected,
        predicted,
        content,
        score,
        score / 10.0,
    )
    return score
