"""LLM-judged semantic reward for AVE event-label lists."""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def reward_fn(record: Any) -> float:
    """Return 1 when the judge accepts the predicted event set, otherwise -1."""

    prediction = _tool_prediction(getattr(record, "tool_calls", None))
    if prediction is None:
        return -1.0
    expected = _label_list(record.source_record.get("event_classes"))
    predicted = _label_list(prediction.get("events"))
    if expected is None or predicted is None:
        return -1.0
    return 1.0 if _judge_same_events(tuple(expected), tuple(predicted)) else -1.0


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


def _judge_same_events(expected: tuple[str, ...], predicted: tuple[str, ...]) -> bool:
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
) -> bool:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("AVE judge reward requires the openai package included with AReno") from exc

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=2)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=32,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict audiovisual event-label judge. Decide whether the predicted labels and "
                    "reference labels denote exactly the same event set. Accept ordinary synonyms, singular/plural "
                    "variants, and equivalent specific/common names. Reject missing events, extra events, merely "
                    "related objects, and explanations that are not labels. Reply with exactly SAME or DIFFERENT."
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
    content = response.choices[0].message.content or ""
    verdicts = re.findall(r"\b(SAME|DIFFERENT)\b", content.upper())
    if not verdicts:
        logger.warning(
            "AVE judge returned invalid response model=%s expected=%s predicted=%s response=%r",
            model,
            expected,
            predicted,
            content,
        )
        raise RuntimeError(f"judge returned an invalid verdict: {content!r}")
    equivalent = verdicts[-1] == "SAME"
    logger.info(
        "AVE judge result model=%s expected=%s predicted=%s response=%r equivalent=%s",
        model,
        expected,
        predicted,
        content,
        equivalent,
    )
    return equivalent
