"""Grouped-softmax classification trainer (JevForge-style decision models).

Dataset rows describe one question each:

- `{"prompt": str, "candidates": [str, ...], "target": [float, ...]}`, where
  candidate path `i` is `encode(prompt) + encode(candidates[i])`; or
- `{"candidate_tokens": [[int, ...], ...], "target": [float, ...]}`.

Every candidate path becomes one `TrainSequence`; the actor score head reads
its last token and `classify_loss_fn` normalizes the logits of one question
with a single softmax. A question must therefore never be split across DP
ranks or microbatches, so each step:

1. Packs whole questions into bins of at most `microbatch_tokens` real tokens
   (first-fit decreasing) and pads the bin count to a multiple of DP size.
2. Pads every bin to the same row count with two-token rows (`group=-1`).
3. Interleaves bins so the engine's strided DP split (`rows[rank::dp]`) hands
   bin `r` of each microbatch to DP rank `r`.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import areno.api
from areno.api.dashboard import record_dashboard_state


@dataclass(slots=True)
class EncodedQuestion:
    """Token paths and target distribution for one question."""

    leaves: list[list[int]]
    target: list[float]

    @property
    def cost(self) -> int:
        return sum(len(leaf) for leaf in self.leaves)


class ClassifyTrainer:
    """Offline loop: encode questions, pack whole groups, train the score head."""

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        del reward_fn
        from areno.experimental.classify.config import ClassifyTrainerConfig

        if not isinstance(config, ClassifyTrainerConfig):
            raise TypeError(
                "classify requires ClassifyTrainerConfig (it enables the actor score head); "
                "build the trainer through the SDK, see examples/classify/jev/train.py"
            )
        self.config = config
        self.areno = instance
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def fit(self) -> None:
        self.areno.init()
        try:
            self._fit_initialized()
        finally:
            self.areno.close()

    def _fit_initialized(self) -> None:
        tokenizer = self.areno.get_tokenizer()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_token_id = int(pad_token_id or 0)
        questions, skipped = encode_questions(self.dataset, tokenizer, max_seq_len=self.config.max_seq_len)
        self.logger.info("stage=classify_dataset questions=%d skipped_too_long=%d", len(questions), skipped)
        if not questions:
            raise ValueError("classify dataset produced no questions within max_seq_len")
        dp_size = self.areno.dp_size()
        rng = random.Random(self.config.seed)
        step = 0
        saved_step = None
        for epoch in range(self.config.epochs):
            record_dashboard_state(self.areno, stage="epoch_start", epoch=epoch, step=step, role="policy")
            order = list(range(len(questions)))
            rng.shuffle(order)
            for start in range(0, len(order), self.config.batch_size):
                chunk = [questions[index] for index in order[start : start + self.config.batch_size]]
                rows, mini_bs = build_step_rows(
                    chunk,
                    dp_size=dp_size,
                    microbatch_tokens=self.config.microbatch_tokens,
                    pad_token_id=pad_token_id,
                )
                record_dashboard_state(self.areno, stage="train_start", epoch=epoch, step=step, role="policy")
                train_start = time.perf_counter()
                result = self.areno.train(rows, self.loss_fn, mini_bs=mini_bs, gradient_accumulation_steps=None)
                if isinstance(result, dict):
                    result["policy_train_wall_time_s"] = time.perf_counter() - train_start
                    result["classify_microbatches"] = len(rows) // mini_bs
                record_dashboard_state(self.areno, stage="train_end", epoch=epoch, step=step, role="policy")
                self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                step += 1
                if self._maybe_save(step):
                    saved_step = step
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
            if self.config.max_steps is not None and step >= self.config.max_steps:
                break
        if self.config.save_path is not None and saved_step != step:
            self._save(step)

    def _maybe_save(self, step: int) -> bool:
        if self.config.save_path is None or step % self.config.save_interval != 0:
            return False
        self._save(step)
        return True

    def _save(self, step: int) -> None:
        # The backbone is written in HF layout; the engine adds score_head.safetensors.
        ckpt_path = str(Path(self.config.save_path) / f"step_{step:06d}")
        self.logger.info("step=%d stage=save_checkpoint_start path=%s", step, ckpt_path)
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("step=%d stage=save_checkpoint_end path=%s", step, saved_path)


def encode_questions(dataset, tokenizer, *, max_seq_len: int) -> tuple[list[EncodedQuestion], int]:
    """Tokenize every row; drop questions whose longest path exceeds `max_seq_len`."""

    questions = []
    skipped = 0
    for index in range(len(dataset)):
        question = encode_question(dataset[index], tokenizer)
        if max(len(leaf) for leaf in question.leaves) > max_seq_len:
            skipped += 1
            continue
        questions.append(question)
    return questions, skipped


def encode_question(record: Any, tokenizer) -> EncodedQuestion:
    """Normalize one dataset row into candidate token paths and a target."""

    record = dict(record)
    target = [float(value) for value in record["target"]]
    if "candidate_tokens" in record:
        leaves = [[int(token) for token in leaf] for leaf in record["candidate_tokens"]]
    elif "prompt" in record and "candidates" in record:
        prefix = _encode(tokenizer, str(record["prompt"]))
        leaves = [prefix + _encode(tokenizer, str(candidate)) for candidate in record["candidates"]]
    else:
        raise ValueError("classify rows need `prompt` + `candidates` or `candidate_tokens`, plus `target`")
    if len(leaves) < 2:
        raise ValueError("classify questions need at least two candidates")
    if len(target) != len(leaves):
        raise ValueError(f"target has {len(target)} entries for {len(leaves)} candidates")
    if any(not math.isfinite(value) or value < 0.0 for value in target) or abs(math.fsum(target) - 1.0) > 1e-4:
        raise ValueError("classify target must be a probability distribution")
    if any(len(leaf) < 1 for leaf in leaves):
        raise ValueError("classify candidate paths must be non-empty")
    return EncodedQuestion(leaves=leaves, target=target)


def build_step_rows(
    questions: list[EncodedQuestion],
    *,
    dp_size: int,
    microbatch_tokens: int,
    pad_token_id: int,
) -> tuple[list, int]:
    """Lay out one optimizer step; returns `(rows, mini_bs)` for `Trainer.train`."""

    if not questions:
        raise ValueError("classify step needs at least one question")
    bins: list[list[int]] = []
    bin_costs: list[int] = []
    for index in sorted(range(len(questions)), key=lambda item: questions[item].cost, reverse=True):
        cost = questions[index].cost
        slot = next((b for b, used in enumerate(bin_costs) if used + cost <= microbatch_tokens), None)
        if slot is None:
            # An oversize question still gets a bin of its own.
            bins.append([])
            bin_costs.append(0)
            slot = len(bins) - 1
        bins[slot].append(index)
        bin_costs[slot] += cost
    while len(bins) % dp_size:
        bins.append([])
    num_microbatches = len(bins) // dp_size
    # Each rank's microbatch loss is summed over its questions with this
    # weight; the engine divides by microbatch count and DP averages, so the
    # step gradient is the mean over all questions.
    weight = dp_size * num_microbatches / len(questions)
    bin_rows = [_bin_rows(questions, members, weight, pad_token_id) for members in bins]
    rows_per_bin = max(len(rows) for rows in bin_rows)
    padding = _padding_row(pad_token_id, weight)
    rows = []
    for micro in range(num_microbatches):
        group = bin_rows[micro * dp_size : (micro + 1) * dp_size]
        for position in range(rows_per_bin):
            for rank_rows in group:
                rows.append(rank_rows[position] if position < len(rank_rows) else padding)
    return rows, rows_per_bin * dp_size


def _bin_rows(questions: list[EncodedQuestion], members: list[int], weight: float, pad_token_id: int) -> list:
    rows = []
    for question_id in members:
        question = questions[question_id]
        for leaf, probability in zip(question.leaves, question.target, strict=True):
            rows.append(
                _train_row(
                    leaf,
                    pad_token_id,
                    {"group": float(question_id), "target": probability, "weight": weight},
                )
            )
    return rows


def _padding_row(pad_token_id: int, weight: float):
    return _train_row([pad_token_id, pad_token_id], pad_token_id, {"group": -1.0, "target": 0.0, "weight": weight})


def _train_row(tokens: list[int], pad_token_id: int, labels: dict[str, float]):
    # No token-level loss: every position is "prompt"; RL fields stay zero.
    zeros = [0.0] * len(tokens)
    return areno.api.TrainSequence(
        tokens=list(tokens),
        prompt_mask=[True] * len(tokens),
        logprobs=zeros,
        advantages=zeros,
        eos_token_id=pad_token_id,
        sequence_labels=labels,
    )


def _encode(tokenizer, text: str) -> list[int]:
    try:
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(token) for token in tokenizer.encode(text)]


__all__ = ["ClassifyTrainer", "EncodedQuestion", "build_step_rows", "encode_question", "encode_questions"]
