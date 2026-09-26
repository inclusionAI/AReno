"""Convert JevForge records into `ClassifyTrainer` question rows.

Mirrors `jevforge/encode.py` and `jevforge/schema.py` so AReno sees the exact
candidate paths JevForge trains on:

    <|oj:state|>
    {state}
    <|oj:question|>
    {type}: {instructions}
    [yes := ... / no := ...]          (noul only, when criteria present)
    {candidate rendering}
    <|oj:answer|>

`prompt` is everything through the question block; each candidate is its
rendering plus the answer marker. The score head reads the last token.
"""

from __future__ import annotations

import json
from pathlib import Path

STATE_MARK = "<|oj:state|>"
QUESTION_MARK = "<|oj:question|>"
ANSWER_MARK = "<|oj:answer|>"
SMOOTHED_TARGET_KINDS = ("deterministic_truth", "uniform_over_positive_elements")


def candidate_ids(question: dict) -> list[str]:
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return [str(index) for index in range(len(question["criteria"]))]
    return ["false", "true"]


def render_candidate(question: dict, index: int) -> str:
    kind = question["type"]
    if kind == "choice":
        key = list(question["criteria"])[index]
        return f"[{key}] {question['criteria'][key]}"
    if kind == "score":
        return f"level {index}: {question['criteria'][index]}"
    return "yes" if index == 1 else "no"


def question_prefix(state: str, question: dict) -> str:
    lines = [STATE_MARK, state, QUESTION_MARK, f"{question['type']}: {question['instructions']}"]
    criteria = question.get("criteria")
    if question["type"] == "noul" and isinstance(criteria, dict):
        if "true" in criteria:
            lines.append(f"yes := {criteria['true']}")
        if "false" in criteria:
            lines.append(f"no := {criteria['false']}")
    return "\n".join(lines) + "\n"


def record_to_questions(record: dict, *, label_smoothing: float = 0.0) -> list[dict]:
    """One row per question that has a target in this record."""

    request = record["request"]
    kinds = record.get("target_kinds", {})
    rows = []
    for question_id, question in request["questions"].items():
        target_map = record["targets"].get(question_id)
        if target_map is None:
            continue
        ids = candidate_ids(question)
        target = [float(target_map[candidate]) for candidate in ids]
        if label_smoothing and kinds.get(question_id) in SMOOTHED_TARGET_KINDS:
            k = len(ids)
            target = [(1.0 - label_smoothing) * value + label_smoothing / k for value in target]
        rows.append(
            {
                "prompt": question_prefix(request["state"], question),
                "candidates": [render_candidate(question, index) + "\n" + ANSWER_MARK for index in range(len(ids))],
                "target": target,
                "type": question["type"],
                "record_id": record["id"],
                "question_id": question_id,
                "source_group": record.get("source_group", ""),
            }
        )
    return rows


def load_questions(records: str | Path, split: str = "train", *, label_smoothing: float = 0.0) -> list[dict]:
    """Read `<records>/<split>.jsonl` (or one jsonl file filtered by `split`)."""

    path = Path(records)
    files = [path] if path.is_file() else [path / f"{split}.jsonl"]
    rows = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split", split) != split:
                    continue
                rows.extend(record_to_questions(record, label_smoothing=label_smoothing))
    return rows
