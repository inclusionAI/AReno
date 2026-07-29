"""Direct Preference Optimization trainer.

DPO consumes offline preference pairs instead of sampling rollouts:
    1. Convert each dataset row into a chosen/rejected token pair.
    2. Score both rows with a frozen reference policy role.
    3. Train the policy with `dpo_loss_fn`, which compares chosen-vs-rejected
       sequence logprob margins against the reference margins.

Rows are handed to the backend as consecutive `[chosen, rejected]` pairs so the
loss can stay backend-agnostic and recover pairs from row order.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any

import areno.api
from areno.api.dashboard import record_dashboard_state
from areno.api.data_utils import apply_chat_template, encode_prompt_value, response_to_tokens_and_mask
from areno.api.roles import ModelRole
from areno.api.tokenizer import configure_chat_template_enable_thinking


class DPOTrainer:
    """Offline preference trainer using one frozen reference policy."""

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        del reward_fn
        # Each DPO pair is represented as two adjacent rows. Keeping microbatch
        # size even prevents the backend from splitting chosen/rejected rows.
        if int(config.mini_bs) % 2 != 0:
            raise ValueError("DPO requires --mini-bs to be even so chosen/rejected pairs stay together")
        self.config = config
        self.areno = instance
        self.dataset = dataset
        # DPO beta is an algorithm hyperparameter, so bind it once here and
        # leave the backend-facing loss signature as loss_fn(data_pack, logprobs).
        self.loss_fn = partial(loss_fn, beta=config.dpo_beta)
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.roles = {
            # DPO only needs a frozen reference policy; no rollout, reward, or
            # critic roles are involved.
            "ref": ModelRole("ref", config.ref_ckpt or config.ckpt, trainable=False),
        }

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
        self._ensure_roles()
        try:
            self._fit_initialized()
        finally:
            self.areno.close()

    def _ensure_roles(self) -> None:
        for role in self.roles.values():
            self.logger.info("role=%s stage=init_start trainable=%s path=%s", role.name, role.trainable, role.path)
        self.areno.ensure_roles(self.roles)
        for role in self.roles.values():
            self.logger.info("role=%s stage=init_end trainable=%s", role.name, role.trainable)

    def _run_eval(self, step: int) -> None:
        """Execute one DPO evaluation pass: load data, tokenize pairs, pre-compute
        reference logprobs, batch evaluate with ``dpo_loss_fn``, and record
        aggregated metrics under the ``eval/`` TensorBoard namespace.

        Each pair (chosen/rejected) is converted to two ``TrainSequence`` rows.
        Before calling ``trainer.evaluate()``, the frozen reference policy
        scores all token rows so ``dpo_loss_fn`` can compare chosen-vs-rejected
        logprob margins against the reference margins.
        """
        import time

        eval_dataset = self._load_eval_dataset()
        if len(eval_dataset) == 0:
            raise ValueError("eval dataset is empty")

        tokenizer = self.areno.get_tokenizer()
        max_seq_len = self.config.max_prompt_tokens + self.config.max_new_tokens

        total_loss = 0.0
        total_accuracy_sum = 0.0
        total_margin_sum = 0.0
        total_pairs = 0
        eval_start = time.perf_counter()

        eval_pairs: list = []
        num_batches = 0
        for record in eval_dataset:
            pair = _record_to_train_pair(record, tokenizer, max_seq_len=max_seq_len)
            if pair is None:
                continue
            # pair is [chosen_seq, rejected_seq]
            eval_pairs.append(pair)
            if len(eval_pairs) >= self.config.batch_size:
                # Flatten pairs into [chosen, rejected, chosen, rejected, ...].
                eval_batch = []
                for p in eval_pairs:
                    eval_batch.extend(p)

                # Pre-compute reference logprobs for all sequences.
                tokens = [seq.tokens for seq in eval_batch]
                ref_logprob_rows = self.areno.score_logprobs(
                    "ref", tokens, microbatch_size=self.config.score_micro_bs
                )
                for seq, ref_lp in zip(eval_batch, ref_logprob_rows):
                    seq.ref_logprobs = [float(v) for v in ref_lp]
                    seq.values = [0.0] * len(seq.tokens)
                    seq.returns = [0.0] * len(seq.tokens)

                result = self.areno.evaluate(
                    eval_batch,
                    self.loss_fn,
                    mini_bs=self.config.mini_bs,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                )
                total_loss += result.get("dpo_loss", 0.0)
                total_accuracy_sum += result.get("dpo_accuracy", 0.0)
                total_margin_sum += result.get("dpo_margin", 0.0)
                total_pairs += len(eval_pairs)
                num_batches += 1
                eval_pairs = []
                if self.config.eval_batches > 0 and num_batches >= self.config.eval_batches:
                    break

        # Process any remaining incomplete batch.
        if eval_pairs and (self.config.eval_batches == 0 or num_batches < self.config.eval_batches):
            eval_batch = []
            for p in eval_pairs:
                eval_batch.extend(p)

            tokens = [seq.tokens for seq in eval_batch]
            ref_logprob_rows = self.areno.score_logprobs(
                "ref", tokens, microbatch_size=self.config.score_micro_bs
            )
            for seq, ref_lp in zip(eval_batch, ref_logprob_rows):
                seq.ref_logprobs = [float(v) for v in ref_lp]
                seq.values = [0.0] * len(seq.tokens)
                seq.returns = [0.0] * len(seq.tokens)

            result = self.areno.evaluate(
                eval_batch,
                self.loss_fn,
                mini_bs=self.config.mini_bs,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
            total_loss += result.get("dpo_loss", 0.0)
            total_accuracy_sum += result.get("dpo_accuracy", 0.0)
            total_margin_sum += result.get("dpo_margin", 0.0)
            total_pairs += len(eval_pairs)
            num_batches += 1

        if total_pairs == 0:
            raise ValueError("eval dataset produced no valid pairs")

        final_metrics: dict[str, float] = {}
        if num_batches > 0:
            final_metrics["dpo_loss"] = total_loss / num_batches
            final_metrics["dpo_accuracy"] = total_accuracy_sum / num_batches
            final_metrics["dpo_margin"] = total_margin_sum / num_batches
        else:
            final_metrics["dpo_loss"] = 0.0
            final_metrics["dpo_accuracy"] = 0.0
            final_metrics["dpo_margin"] = 0.0
        final_metrics["sample_count"] = float(total_pairs * 2)
        final_metrics["duration_s"] = time.perf_counter() - eval_start

        if self.areno._metrics is not None:
            self.areno._metrics.record_eval_step(step=step, eval_result=final_metrics)

        self.logger.info(
            "step=%d stage=eval_end eval_loss=%.4f accuracy=%.4f margin=%.4f sample_count=%d duration_s=%.3f",
            step,
            final_metrics["dpo_loss"],
            final_metrics["dpo_accuracy"],
            final_metrics["dpo_margin"],
            int(final_metrics["sample_count"]),
            final_metrics["duration_s"],
        )
        record_dashboard_state(self.areno, stage="eval_end", step=step)

    def _fit_initialized(self) -> None:
        tokenizer = self.areno.get_tokenizer()
        configure_chat_template_enable_thinking(tokenizer, getattr(self.config, "chat_template_enable_thinking", None))
        step = 0
        max_seq_len = self.config.max_prompt_tokens + self.config.max_new_tokens
        for epoch in range(self.config.epochs):
            self.logger.info("epoch=%d stage=epoch_start", epoch)
            record_dashboard_state(self.areno, stage="epoch_start", epoch=epoch, step=step, role="policy")
            for train_batch in self._iter_train_batches(tokenizer, max_seq_len=max_seq_len):
                if not train_batch:
                    continue
                self.logger.info(
                    "epoch=%d step=%d role=ref stage=logprob_score_start rows=%d", epoch, step, len(train_batch)
                )
                record_dashboard_state(self.areno, stage="logprob_score_start", epoch=epoch, step=step, role="ref")
                ref_start = time.perf_counter()
                # Score the exact chosen/rejected token rows under the frozen
                # reference before the actor update.
                ref_logprob_rows = self.areno.score_logprobs(
                    "ref", [seq.tokens for seq in train_batch], microbatch_size=self.config.score_micro_bs
                )
                ref_time_s = time.perf_counter() - ref_start
                self.logger.info(
                    "epoch=%d step=%d role=ref stage=logprob_score_end rows=%d", epoch, step, len(train_batch)
                )
                record_dashboard_state(self.areno, stage="logprob_score_end", epoch=epoch, step=step, role="ref")
                for seq, ref_logprobs in zip(train_batch, ref_logprob_rows, strict=True):
                    if len(ref_logprobs) != len(seq.tokens):
                        raise ValueError("reference role returned misaligned logprobs")
                    # Reuse TrainSequence.ref_logprobs so the existing backend
                    # packer carries reference scores to dpo_loss_fn.
                    seq.ref_logprobs = [float(value) for value in ref_logprobs]
                    # The areno packed path currently emits packed
                    # ref_logprobs only when the full PPO optional-field bundle
                    # is present. DPO does not use values/returns, but filling
                    # zero vectors preserves the existing packer contract.
                    seq.values = [0.0] * len(seq.tokens)
                    seq.returns = [0.0] * len(seq.tokens)

                self.logger.info(
                    "epoch=%d step=%d role=policy stage=train_start pairs=%d", epoch, step, len(train_batch) // 2
                )
                record_dashboard_state(self.areno, stage="train_start", epoch=epoch, step=step, role="policy")
                train_start = time.perf_counter()
                # The train batch rows are [chosen, rejected, ...]; dpo_loss_fn
                # recovers pairs by row order inside each even-sized microbatch.
                result = self.areno.train(
                    train_batch,
                    self.loss_fn,
                    mini_bs=self.config.mini_bs,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                )
                train_time_s = time.perf_counter() - train_start
                if isinstance(result, dict):
                    result["ref_logprob_forward_time_s"] = ref_time_s
                    result["policy_train_wall_time_s"] = train_time_s
                self.logger.info(
                    "epoch=%d step=%d role=policy stage=train_end pairs=%d", epoch, step, len(train_batch) // 2
                )
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
                    if self._eval_enabled and not (self.config.eval_interval > 0 and step % self.config.eval_interval == 0):
                        self._run_eval(step)
                    return
            # End-of-epoch evaluation: triggers regardless of interval alignment.
            if self._eval_enabled:
                self._run_eval(step)
            self.logger.info("epoch=%d stage=epoch_end", epoch)
            record_dashboard_state(self.areno, stage="epoch_end", epoch=epoch, step=step, role="policy")

    def _iter_train_batches(self, tokenizer, *, max_seq_len: int):
        # `batch_size` counts preference pairs; the emitted train batch has two
        # rows per pair and always preserves chosen/rejected adjacency.
        batch = []
        skipped = 0
        for index in range(len(self.dataset)):
            pair = _record_to_train_pair(self.dataset[index], tokenizer, max_seq_len=max_seq_len)
            if pair is None:
                skipped += 1
                continue
            batch.extend(pair)
            if len(batch) >= self.config.batch_size * 2:
                yield batch
                batch = []
        if skipped:
            self.logger.info("stage=dpo_dataset_filter skipped_invalid_or_long=%d", skipped)
        if batch:
            yield batch

    def _maybe_save(self, epoch: int, step: int) -> None:
        if self.config.save_path is None or (step + 1) % self.config.save_interval != 0:
            return
        ckpt_path = str(Path(self.config.save_path) / f"step_{step + 1:06d}")
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_start path=%s", epoch, step, ckpt_path)
        record_dashboard_state(self.areno, stage="save_checkpoint_start", epoch=epoch, step=step, role="policy")
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_end path=%s", epoch, step, saved_path)
        record_dashboard_state(self.areno, stage="save_checkpoint_end", epoch=epoch, step=step, role="policy")


def _record_to_train_pair(record: Any, tokenizer, *, max_seq_len: int):
    record = dict(record)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if "chosen" not in record or "rejected" not in record:
        raise ValueError("DPO dataset row must contain `chosen` and `rejected`")
    chosen, rejected = record["chosen"], record["rejected"]

    if isinstance(chosen, list) and isinstance(rejected, list):
        # Preference datasets sometimes store full chosen/rejected chats. The
        # shared prefix is treated as prompt context; only divergent suffixes
        # contribute to the DPO objective.
        chosen_tokens = apply_chat_template(tokenizer, chosen)
        rejected_tokens = apply_chat_template(tokenizer, rejected)
        prefix_len = _common_prefix_len(chosen_tokens, rejected_tokens)
        chosen_mask = [True] * prefix_len + [False] * (len(chosen_tokens) - prefix_len)
        rejected_mask = [True] * prefix_len + [False] * (len(rejected_tokens) - prefix_len)
    else:
        # Prompt/response preference rows share one encoded prompt and differ
        # only in the chosen/rejected response suffix.
        if "prompt" not in record:
            raise ValueError("DPO prompt/response rows must contain `prompt`")
        prompt = record["prompt"]
        prompt_ids = encode_prompt_value(tokenizer, prompt)
        chosen_tokens, chosen_mask = response_to_tokens_and_mask(prompt_ids, str(chosen), tokenizer, eos_token_id)
        rejected_tokens, rejected_mask = response_to_tokens_and_mask(prompt_ids, str(rejected), tokenizer, eos_token_id)

    chosen_seq = _make_sequence(chosen_tokens, chosen_mask, eos_token_id, max_seq_len)
    rejected_seq = _make_sequence(rejected_tokens, rejected_mask, eos_token_id, max_seq_len)
    if chosen_seq is None or rejected_seq is None:
        return None
    return [chosen_seq, rejected_seq]


def _make_sequence(tokens: list[int], prompt_mask: list[bool], eos_token_id: int, max_seq_len: int):
    # Drop examples that cannot produce a next-token loss or exceed the shared
    # max sequence budget.
    if len(tokens) < 2 or len(tokens) > max_seq_len or not any(not item for item in prompt_mask[1:]):
        return None
    zeros = [0.0] * len(tokens)
    # Dummy rollout fields keep the TrainSequence shape contract shared with
    # RL trainers; DPO only consumes tokens, prompt_mask, and ref_logprobs.
    return areno.api.TrainSequence(
        prompt_mask=prompt_mask,
        tokens=tokens,
        logprobs=zeros,
        advantages=zeros,
        eos_token_id=eos_token_id,
    )


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    prefix_len = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        prefix_len += 1
    return prefix_len


__all__ = ["DPOTrainer"]
