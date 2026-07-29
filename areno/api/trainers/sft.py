"""Supervised fine-tuning trainer.

SFT reuses the backend's generic `train(batch, loss_fn)` path. The trainer only
turns dataset rows into `TrainSequence` objects with prompt positions masked
out, then the loss optimizes next-token likelihood on target tokens.

Each step follows the same backend contract as the policy-only RL trainer:
    1. Convert dataset rows into token sequences and prompt/target masks.
    2. Build `TrainSequence` rows whose dummy logprobs/advantages only keep the
       tensor packing path shape-compatible with RL batches.
    3. Hand the batch to `Trainer.train(...)`; `sft_loss_fn` ignores RL fields
       and trains on non-prompt next-token positions.
No rollout, reward function, or weight sync is needed because SFT consumes
teacher-forced examples directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import areno.api
from areno.api.dashboard import record_dashboard_state
from areno.api.data import OverlengthPolicy, OverlengthReason
from areno.api.data_utils import prompt_response_to_tokens_and_mask
from areno.api.overlength import decide_overlength, truncate_sft_response
from areno.api.tokenizer import configure_chat_template_enable_thinking


class SFTTrainer:
    """Dataset-to-next-token-loss loop for supervised fine-tuning.

    This mirrors `PolicyOnlyTrainer`'s lifecycle but removes the RL-only stages:
    there is no rollout policy, no reward normalization, and no old-policy
    logprob. The backend still sees `TrainSequence` so optimizer, packing, and
    checkpoint behavior stay shared with GSPO/GRPO/PPO.
    """

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        del reward_fn
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
        configure_chat_template_enable_thinking(tokenizer, getattr(self.config, "chat_template_enable_thinking", None))
        step = 0
        for epoch in range(self.config.epochs):
            self.logger.info("epoch=%d stage=epoch_start", epoch)
            record_dashboard_state(self.areno, stage="epoch_start", epoch=epoch, step=step, role="policy")
            for train_batch in self._iter_train_batches(
                tokenizer,
                max_prompt_tokens=self.config.max_prompt_tokens,
                max_new_tokens=self.config.max_new_tokens,
            ):
                if not train_batch:
                    continue
                self.logger.info(
                    "epoch=%d step=%d role=policy stage=train_start rows=%d", epoch, step, len(train_batch)
                )
                record_dashboard_state(self.areno, stage="train_start", epoch=epoch, step=step, role="policy")
                train_start = time.perf_counter()
                # The backend computes next-token logprobs for the supplied
                # labels; `sft_loss_fn` selects only response/target positions
                # using the prompt mask produced below.
                result = self.areno.train(
                    train_batch,
                    self.loss_fn,
                    mini_bs=self.config.mini_bs,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                )
                train_time_s = time.perf_counter() - train_start
                if isinstance(result, dict):
                    result["policy_train_wall_time_s"] = train_time_s
                self.logger.info("epoch=%d step=%d role=policy stage=train_end rows=%d", epoch, step, len(train_batch))
                record_dashboard_state(self.areno, stage="train_end", epoch=epoch, step=step, role="policy")
                self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                self._maybe_save(epoch, step)
                step += 1
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    self.logger.info("epoch=%d step=%d stage=max_steps_reached", epoch, step)
                    record_dashboard_state(self.areno, stage="max_steps_reached", epoch=epoch, step=step, role="policy")
                    return
            self.logger.info("epoch=%d stage=epoch_end", epoch)
            record_dashboard_state(self.areno, stage="epoch_end", epoch=epoch, step=step, role="policy")

    def _iter_train_batches(self, tokenizer, *, max_prompt_tokens: int, max_new_tokens: int):
        # Dataset rows are converted lazily so large HF datasets do not need an
        # up-front tokenized copy. Rows that are empty, all-prompt, or exceed
        # the configured prompt or supervised-response budgets are dropped or
        # truncated according to the configured overlength policy.
        policy = getattr(self.config, "overlength_policy", "reject")
        batch = []
        skipped_empty = 0
        overlength_counters: dict[str, int] = {}
        accepted = 0
        total_rows = len(self.dataset)
        for index in range(total_rows):
            # `overlength_counters` is mutated as a side effect when a row is
            # dropped/truncated for being over budget; an empty/invalid row
            # leaves it untouched so we can tell the two drop reasons apart.
            before = sum(overlength_counters.values())
            seq = _record_to_train_sequence(
                self.dataset[index],
                tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                policy=policy,
                counters=overlength_counters,
            )
            if seq is None:
                if sum(overlength_counters.values()) == before:
                    skipped_empty += 1
                continue
            accepted += 1
            batch.append(seq)
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        if skipped_empty or overlength_counters:
            parts = [f"skipped_empty={skipped_empty}"]
            for key, count in sorted(overlength_counters.items()):
                parts.append(f"{key}={count}")
            self.logger.info("stage=sft_dataset_filter %s", " ".join(parts))
        if accepted == 0:
            raise ValueError(
                "SFT dataset produced no valid training rows after filtering: "
                f"scanned {total_rows} row(s), skipped {skipped_empty} as empty, "
                f"overlength {sum(overlength_counters.values())}, or all-prompt examples. "
                "Check dataset quality, --max-prompt-tokens, and --max-new-tokens."
            )
        if batch:
            yield batch

    def _maybe_save(self, epoch: int, step: int) -> None:
        # Keep the same step-based checkpoint cadence as the RL trainers.
        if self.config.save_path is None or (step + 1) % self.config.save_interval != 0:
            return
        ckpt_path = str(Path(self.config.save_path) / f"step_{step + 1:06d}")
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_start path=%s", epoch, step, ckpt_path)
        record_dashboard_state(self.areno, stage="save_checkpoint_start", epoch=epoch, step=step, role="policy")
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_end path=%s", epoch, step, saved_path)
        record_dashboard_state(self.areno, stage="save_checkpoint_end", epoch=epoch, step=step, role="policy")


def _record_to_train_sequence(
    record: Any,
    tokenizer,
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    policy: str = "reject",
    counters: dict[str, int] | None = None,
):
    """Normalize one loader-produced SFT row into backend training format.

    `prompt_mask=True` means "do not train this source token"; the backend loss
    is next-token aligned, so the loss function later uses positions after the
    prompt prefix. RL-only fields are filled with zeros to satisfy the shared
    `TrainSequence` packing contract.

    Returns a ``TrainSequence`` or ``None``. When a row is dropped because it
    exceeds a budget, the per-reason, per-action count is recorded into
    ``counters`` (as ``{f"{reason}/{policy}": count}``) so callers can surface
    per-reason diagnostics; empty/invalid rows return ``None`` without touching
    ``counters``. Under ``warn`` the original sequence is kept and counted;
    under ``truncate`` the response is cut to ``max_new_tokens`` with a trailing
    EOS preserved.
    """

    def _count(reason_value: str) -> None:
        if counters is not None:
            key = f"{reason_value}/{policy}"
            counters[key] = counters.get(key, 0) + 1

    record = dict(record)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if "prompt" not in record or "response" not in record:
        raise ValueError(
            "SFT dataset loader must return rows with `prompt` and `response`; "
            "normalize raw dataset fields in --dataset-loader-fn"
        )
    if record["prompt"] is None or record["response"] is None:
        return None
    prompt = str(record["prompt"])
    response = str(record["response"])
    if not response:
        return None
    tokens, prompt_mask = prompt_response_to_tokens_and_mask(prompt, response, tokenizer, eos_token_id)

    if len(tokens) < 2:
        return None
    prompt_tokens = prompt_mask.count(True)
    response_tokens = prompt_mask[1:].count(False)
    if response_tokens == 0:
        return None

    overlength_policy = OverlengthPolicy(policy)
    decision = decide_overlength(
        prompt_len=prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
        response_len=response_tokens,
        max_new_tokens=max_new_tokens,
        policy=overlength_policy,
    )

    # Within budget: keep as-is (no overlength reason to count).
    if not decision.truncated and decision.reason.value in ("within_budget", "exact_limit"):
        return _build_train_sequence(tokens, prompt_mask, eos_token_id)

    reason_value = decision.reason.value

    # Prompt overlength: SFT prompts are plain strings with no chat-turn
    # boundary to cut on, so truncate degrades to reject for this path.
    if decision.reason is OverlengthReason.SINGLE_MESSAGE_OVERSIZED:
        if overlength_policy is OverlengthPolicy.WARN:
            _count(reason_value)
            return _build_train_sequence(tokens, prompt_mask, eos_token_id)
        _count(reason_value)
        return None

    # Response overlength.
    if overlength_policy is OverlengthPolicy.REJECT:
        _count(reason_value)
        return None
    if overlength_policy is OverlengthPolicy.WARN:
        _count(reason_value)
        return _build_train_sequence(tokens, prompt_mask, eos_token_id)

    # TRUNCATE: cut the response at max_new_tokens, preserve trailing EOS.
    prompt_ids = tokens[:prompt_tokens]
    response_ids = tokens[prompt_tokens:]
    cut_tokens, cut_mask, _truncated = truncate_sft_response(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=(eos_token_id,) if eos_token_id is not None else None,
    )
    if len(cut_tokens) < 2 or cut_mask[1:].count(False) == 0:
        # Trimming produced an un-trainable sequence; reject instead.
        _count(reason_value)
        return None
    _count(reason_value)
    return _build_train_sequence(cut_tokens, cut_mask, eos_token_id)


def _build_train_sequence(tokens: list[int], prompt_mask: list[bool], eos_token_id: int):
    """Wrap token/mask rows into a TrainSequence with zeroed RL-only fields."""

    zeros = [0.0] * len(tokens)
    # Dummy rollout fields keep the backend packer shared with RL trainers.
    return areno.api.TrainSequence(
        prompt_mask=prompt_mask,
        tokens=tokens,
        logprobs=zeros,
        advantages=zeros,
        eos_token_id=eos_token_id,
    )


__all__ = ["SFTTrainer"]
