"""Question-preserving candidate tokenization and grouped training rows."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from areno.api.models import TrainSequence


def encode_question(question: dict[str, Any], tokenizer: Any, max_seq_len: int) -> list[TrainSequence]:
    candidates, target = question["candidates"], question["target"]
    if len(candidates) < 2 or len(candidates) != len(target):
        raise ValueError("a question needs >=2 candidates and one target per candidate")
    if any(not math.isfinite(p) or p < 0 for p in target) or not math.isclose(sum(target), 1, abs_tol=1e-5):
        raise ValueError("target must be a finite nonnegative probability distribution")
    suffixes = [tokenizer.encode(candidate, add_special_tokens=False) for candidate in candidates]
    budget = max_seq_len - max(map(len, suffixes))
    if budget < 1:
        raise ValueError("max_seq_len cannot preserve the complete candidate and answer marker")
    prefix = tokenizer.encode(question["prompt"], add_special_tokens=False)
    if not prefix:
        raise ValueError("question prefix is empty")
    # Keep the question and full candidate/answer suffix even for long states.
    prefix = prefix[-budget:]
    return [
        TrainSequence(
            tokens=prefix + suffix,
            features={
                "jev_question": question["question_id"],
                "jev_target": float(probability),
                "jev_candidates": len(candidates),
                "jev_index": index,
            },
        )
        for index, (suffix, probability) in enumerate(zip(suffixes, target, strict=True))
    ]


def group_questions(rows: list[TrainSequence]) -> list[list[TrainSequence]]:
    groups = []
    index = 0
    while index < len(rows):
        first = rows[index].features or {}
        count = first.get("jev_candidates", 0)
        group = rows[index : index + count]
        if count < 2 or len(group) != count:
            raise ValueError("incomplete JEV question")
        for offset, row in enumerate(group):
            features = row.features or {}
            if (
                features.get("jev_index") != offset
                or features.get("jev_candidates") != count
                or features.get("jev_question") != first.get("jev_question")
            ):
                raise ValueError("JEV candidates must remain contiguous and complete")
        target = [row.features["jev_target"] for row in group]
        if any(not math.isfinite(p) or p < 0 for p in target) or not math.isclose(sum(target), 1, abs_tol=1e-5):
            raise ValueError("invalid JEV target distribution")
        groups.append(group)
        index += count
    return groups


def load_parquet_questions(path: str | Path) -> list[dict[str, Any]]:
    """Read typed questions with probability gold or deterministic candidate ids."""
    import pyarrow.parquet as pq

    return normalize_questions(
        pq.read_table(path, columns=["id", "workflow", "state", "questions", "gold"]).to_pylist()
    )


def normalize_questions(dataset) -> list[dict[str, Any]]:
    """Normalize CLI/SDK training records while preserving the selected split."""
    import json

    from areno.api.jev.schema import candidate_ids, record_to_questions

    if isinstance(dataset, Mapping):
        dataset = dataset["train"]
    result = []
    for row in dataset:
        if row.get("split", "train") != "train":
            continue
        if "prompt" in row and "candidates" in row and "target" in row:
            result.append(row)
            continue
        if "request" in row and "targets" in row:
            result.extend(record_to_questions(row))
            continue
        questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        state = row["state"]
        try:
            state = json.dumps(json.loads(state), ensure_ascii=False, sort_keys=True)
        except (ValueError, TypeError):
            pass
        targets = {}
        for qid, question in questions.items():
            value, ids = gold[qid], candidate_ids(question)
            if isinstance(value, dict) and "probabilities" in value:
                target = value["probabilities"]
                total = sum(float(target[key]) for key in ids)
                if total <= 0:
                    raise ValueError(f"{row['id']}: gold has no probability mass")
                targets[qid] = {key: float(target[key]) / total for key in ids}
            else:
                label = str(value).lower() if isinstance(value, bool) else str(value)
                if label not in ids:
                    raise ValueError(f"{row['id']}: gold {label!r} is not a candidate")
                targets[qid] = {key: float(key == label) for key in ids}
        result.extend(
            record_to_questions(
                {
                    "id": row["id"],
                    "source_group": row["workflow"],
                    "request": {"state": state, "questions": questions},
                    "targets": targets,
                }
            )
        )
    return result


class QuestionIterator:
    """Deterministic shuffled batches with checkpointable position and identity."""

    def __init__(self, questions, *, seed, batch_size, state=None):
        import hashlib
        import json

        self.questions, self.seed, self.batch_size = questions, seed, batch_size
        self.identity = {
            "dataset_sha256": hashlib.sha256(
                json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "seed": seed,
            "batch_size": batch_size,
        }
        self.epoch = self.next_index = self.completed_steps = 0
        if state is not None:
            if state.get("format_version") != 1 or any(state.get(k) != v for k, v in self.identity.items()):
                raise ValueError("JEV resume requires the same dataset, seed and batch_size")
            self.epoch, self.next_index, self.completed_steps = (
                state[key] for key in ("epoch", "next_index", "completed_steps")
            )
            if (
                any(
                    type(value) is not int or value < 0 for value in (self.epoch, self.next_index, self.completed_steps)
                )
                or self.next_index >= len(questions)
                or self.next_index % batch_size
            ):
                raise ValueError("invalid JEV dataset iterator position")
        self._order_epoch, self._order = None, []

    def take(self):
        import random

        if self._order_epoch != self.epoch:
            self._order = list(range(len(self.questions)))
            random.Random(self.seed + self.epoch).shuffle(self._order)
            self._order_epoch = self.epoch
        indices = self._order[self.next_index : self.next_index + self.batch_size]
        self.next_index += len(indices)
        self.completed_steps += 1
        if self.next_index == len(self.questions):
            self.epoch += 1
            self.next_index = 0
        return [self.questions[index] for index in indices]

    def state_dict(self):
        return {
            "format_version": 1,
            **self.identity,
            "epoch": self.epoch,
            "next_index": self.next_index,
            "completed_steps": self.completed_steps,
        }


def candidate_packs(groups, mini_bs):
    """Fill candidate microbatches while keeping each question's softmax intact."""
    # Sorting only changes execution within this optimizer update. Dataset
    # iteration and per-question loss weights remain unchanged.
    ordered = sorted(groups, key=lambda group: max(len(row.tokens) for row in group))
    pack, count, tokens, longest = [], 0, 0, 0
    for group in ordered:
        length = max(len(row.tokens) for row in group)
        padded_tokens = (count + len(group)) * max(longest, length)
        useful_tokens = tokens + len(group) * length
        if pack and (count + len(group) > mini_bs or padded_tokens > 1.25 * useful_tokens):
            yield pack
            pack, count, tokens, longest = [], 0, 0, 0
        if len(group) > mini_bs:
            yield [group]
            continue
        pack.append(group)
        count += len(group)
        tokens += len(group) * length
        longest = max(longest, length)
    if pack:
        yield pack
