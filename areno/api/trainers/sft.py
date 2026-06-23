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
from areno.api.data_utils import apply_chat_template, prompt_response_to_tokens_and_mask


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
        step = 0
        for epoch in range(self.config.epochs):
            self.logger.info("epoch=%d stage=epoch_start", epoch)
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
                self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                self._maybe_save(epoch, step)
                step += 1
            self.logger.info("epoch=%d stage=epoch_end", epoch)

    def _iter_train_batches(self, tokenizer, *, max_prompt_tokens: int, max_new_tokens: int):
        # Dataset rows are converted lazily so large HF datasets do not need an
        # up-front tokenized copy. Rows that are empty, all-prompt, or exceed
        # the configured prompt or supervised-response budgets are dropped.
        batch = []
        skipped = 0
        accepted = 0
        total_rows = len(self.dataset)
        for index in range(total_rows):
            # Normalize each supported row schema into one TrainSequence.
            seq = _record_to_train_sequence(
                self.dataset[index],
                tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
            if seq is None:
                skipped += 1
                continue
            accepted += 1
            batch.append(seq)
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        if skipped:
            self.logger.info("stage=sft_dataset_filter skipped_long_or_empty=%d", skipped)
        if accepted == 0:
            raise ValueError(
                "SFT dataset produced no valid training rows after filtering: "
                f"scanned {total_rows} row(s), skipped {skipped} as empty, over-budget, or all-prompt examples. "
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
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_end path=%s", epoch, step, saved_path)


def _record_to_train_sequence(record: Any, tokenizer, *, max_prompt_tokens: int, max_new_tokens: int):
    """Normalize common SFT schemas into the backend training row format.

    `prompt_mask=True` means "do not train this source token"; the backend loss
    is next-token aligned, so the loss function later uses positions after the
    prompt prefix. RL-only fields are filled with zeros to satisfy the shared
    `TrainSequence` packing contract.
    """

    record = dict(record)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if isinstance(record.get("messages"), list):
        # Chat rows train only assistant turns.
        tokens, prompt_mask = _messages_to_tokens_and_mask(record["messages"], tokenizer)
    elif "prompt" in record and "response" in record:
        # Prompt/response style rows train on the response suffix.
        tokens, prompt_mask = prompt_response_to_tokens_and_mask(
            record["prompt"], record["response"], tokenizer, eos_token_id
        )
    elif isinstance(record.get("text"), str):
        # Plain text rows train on every token after the first context token.
        tokens = tokenizer.encode(record["text"], add_special_tokens=True)
        prompt_mask = [True] + [False] * max(0, len(tokens) - 1)
    else:
        raise ValueError("SFT dataset row must contain `messages`, `prompt`/`response`, or `text`")

    prompt_tokens = sum(1 for item in prompt_mask if item)
    response_tokens = sum(1 for item in prompt_mask[1:] if not item)
    if len(tokens) < 2 or prompt_tokens > max_prompt_tokens or response_tokens > max_new_tokens or response_tokens == 0:
        return None
    zeros = [0.0] * len(tokens)
    # Dummy rollout fields keep the backend packer shared with RL trainers.
    return areno.api.TrainSequence(
        prompt_mask=prompt_mask,
        tokens=tokens,
        logprobs=zeros,
        advantages=zeros,
        eos_token_id=eos_token_id,
    )


def _messages_to_tokens_and_mask(messages: list[dict[str, Any]], tokenizer) -> tuple[list[int], list[bool]]:
    # Chat SFT should only optimize assistant turns. We apply the chat template
    # incrementally and mark the token delta introduced by each assistant
    # message as target tokens; user/system/tool deltas remain prompt-masked.
    tokens: list[int] = []
    prompt_mask: list[bool] = []
    previous: list[int] = []
    for end in range(1, len(messages) + 1):
        current = apply_chat_template(tokenizer, messages[:end])
        delta = current[len(previous) :] if current[: len(previous)] == previous else current
        is_target = str(messages[end - 1].get("role", "")).lower() == "assistant"
        tokens.extend(delta)
        prompt_mask.extend([not is_target] * len(delta))
        previous = current
    return tokens, prompt_mask


__all__ = ["SFTTrainer"]
