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
from areno.api.data_utils import prompt_response_to_tokens_and_mask
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

    @property
    def _eval_enabled(self) -> bool:
        return self.config.eval_dataset_path is not None

    def _load_eval_dataset(self) -> list:
        """Lazily load the evaluation dataset, caching on first call."""
        if hasattr(self, "_eval_dataset"):
            return self._eval_dataset
        from datasets import load_dataset, load_from_disk

        from areno.cli.train import _load_dataset_for_training

        self._eval_dataset = _load_dataset_for_training(
            self.config.eval_dataset_path,
            dataset_loader_fn=getattr(self.config, "dataset_loader_fn", None),
            model_hub=self.config.model_hub,
            load_dataset=load_dataset,
            load_from_disk=load_from_disk,
        )
        return self._eval_dataset

    def fit(self) -> None:
        self.areno.init()
        try:
            self._fit_initialized()
        finally:
            self.areno.close()

    def _run_eval(self, step: int) -> None:
        """Execute one evaluation pass: load data, tokenize, batch, evaluate, record.

        Metrics are aggregated across eval batches: loss is weighted by
        ``sft_target_tokens``, sample counts are summed, and the final dict is
        written via ``MetricsRecorder.record_eval_step`` under the ``eval/``
        TensorBoard namespace.
        """
        import time

        eval_dataset = self._load_eval_dataset()
        if len(eval_dataset) == 0:
            raise ValueError("eval dataset is empty")

        tokenizer = self.areno.get_tokenizer()
        max_prompt_tokens = self.config.max_prompt_tokens
        max_new_tokens = self.config.max_new_tokens

        total_weighted_loss = 0.0
        total_target_tokens = 0
        total_samples = 0
        eval_start = time.perf_counter()

        eval_batch: list = []
        num_batches = 0
        for record in eval_dataset:
            seq = _record_to_train_sequence(
                record,
                tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
            if seq is None:
                continue
            eval_batch.append(seq)
            if len(eval_batch) >= self.config.batch_size:
                result = self.areno.evaluate(
                    eval_batch,
                    self.loss_fn,
                    mini_bs=self.config.mini_bs,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                )
                batch_tokens = result.get("sft_target_tokens", len(eval_batch))
                total_weighted_loss += result.get("sft_loss", 0.0) * batch_tokens
                total_target_tokens += batch_tokens
                total_samples += len(eval_batch)
                num_batches += 1
                eval_batch = []
                if self.config.eval_batches > 0 and num_batches >= self.config.eval_batches:
                    break

        # Process any remaining incomplete batch.
        if eval_batch and (self.config.eval_batches == 0 or num_batches < self.config.eval_batches):
            result = self.areno.evaluate(
                eval_batch,
                self.loss_fn,
                mini_bs=self.config.mini_bs,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
            batch_tokens = result.get("sft_target_tokens", len(eval_batch))
            total_weighted_loss += result.get("sft_loss", 0.0) * batch_tokens
            total_target_tokens += batch_tokens
            total_samples += len(eval_batch)
            num_batches += 1

        if total_samples == 0:
            raise ValueError("eval dataset produced no valid rows")

        final_metrics: dict[str, float] = {}
        if total_target_tokens > 0:
            final_metrics["sft_loss"] = total_weighted_loss / total_target_tokens
        else:
            final_metrics["sft_loss"] = 0.0
        final_metrics["sft_target_tokens"] = float(total_target_tokens)
        final_metrics["sample_count"] = float(total_samples)
        final_metrics["duration_s"] = time.perf_counter() - eval_start

        if self.areno._metrics is not None:
            self.areno._metrics.record_eval_step(step=step, eval_result=final_metrics)

        self.logger.info(
            "step=%d stage=eval_end eval_loss=%.4f sample_count=%d duration_s=%.3f",
            step,
            final_metrics["sft_loss"],
            total_samples,
            final_metrics["duration_s"],
        )
        record_dashboard_state(self.areno, stage="eval_end", step=step)

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
                # Interval evaluation: triggered every eval_interval training steps.
                if self._eval_enabled and self.config.eval_interval > 0 and step % self.config.eval_interval == 0:
                    self._run_eval(step)
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    self.logger.info("epoch=%d step=%d stage=max_steps_reached", epoch, step)
                    record_dashboard_state(self.areno, stage="max_steps_reached", epoch=epoch, step=step, role="policy")
                    # End-of-training evaluation triggered before exiting at max_steps.
                    # Skip when the interval eval already ran at this step.
                    if self._eval_enabled and not (
                        self.config.eval_interval > 0 and step % self.config.eval_interval == 0
                    ):
                        self._run_eval(step)
                    return
            # End-of-epoch evaluation: triggers regardless of interval alignment.
            if self._eval_enabled:
                self._run_eval(step)
            self.logger.info("epoch=%d stage=epoch_end", epoch)
            record_dashboard_state(self.areno, stage="epoch_end", epoch=epoch, step=step, role="policy")

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
        record_dashboard_state(self.areno, stage="save_checkpoint_start", epoch=epoch, step=step, role="policy")
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_end path=%s", epoch, step, saved_path)
        record_dashboard_state(self.areno, stage="save_checkpoint_end", epoch=epoch, step=step, role="policy")


def _record_to_train_sequence(record: Any, tokenizer, *, max_prompt_tokens: int, max_new_tokens: int):
    """Normalize one loader-produced SFT row into backend training format.

    `prompt_mask=True` means "do not train this source token"; the backend loss
    is next-token aligned, so the loss function later uses positions after the
    prompt prefix. RL-only fields are filled with zeros to satisfy the shared
    `TrainSequence` packing contract.
    """

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
    if prompt_tokens > max_prompt_tokens or response_tokens > max_new_tokens or response_tokens == 0:
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


__all__ = ["SFTTrainer"]
