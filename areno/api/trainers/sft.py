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
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import areno.api
from areno.api.dashboard import record_dashboard_state
from areno.api.data import DATASET_MIX_METADATA_KEY
from areno.api.data_utils import prompt_response_to_tokens_and_mask
from areno.api.dataset_mix_artifacts import write_dataset_mix_plan
from areno.api.multimodal import (
    encode_multimodal_prompt,
    expand_image_tokens,
    image_token_counts_from_features,
    mrope_position_ids_from_image_grid,
    record_has_image,
)
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
        processor = self.areno.get_processor()
        configure_chat_template_enable_thinking(tokenizer, getattr(self.config, "chat_template_enable_thinking", None))
        configure_chat_template_enable_thinking(processor, getattr(self.config, "chat_template_enable_thinking", None))
        step = 0
        for epoch in range(self.config.epochs):
            set_epoch = getattr(self.dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            mix_summary = getattr(self.dataset, "summary", None)
            if callable(mix_summary):
                resolved_mix_summary = mix_summary()
                if "schedule_hash" in resolved_mix_summary:
                    write_dataset_mix_plan(
                        resolved_mix_summary,
                        getattr(self.config, "metrics_log_dir", None),
                    )
                self.logger.info("epoch=%d stage=dataset_mix_plan dataset_mix=%s", epoch, resolved_mix_summary)
                mix_source_names = [
                    source["name"] for source in resolved_mix_summary.get("sources", []) if "name" in source
                ]
            else:
                mix_source_names = []
            mix_progress = {
                "scheduled": Counter(),
                "filtered": Counter(),
                "trained": Counter(),
                "target_tokens": Counter(),
            }
            self.logger.info("epoch=%d stage=epoch_start", epoch)
            record_dashboard_state(self.areno, stage="epoch_start", epoch=epoch, step=step, role="policy")
            for train_batch, batch_mix_counts, batch_mix_target_tokens in self._iter_train_batches(
                tokenizer,
                processor,
                max_prompt_tokens=self.config.max_prompt_tokens,
                max_new_tokens=self.config.max_new_tokens,
                mix_progress=mix_progress,
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
                if batch_mix_counts:
                    mix_progress["trained"].update(batch_mix_counts)
                    mix_progress["target_tokens"].update(batch_mix_target_tokens)
                    self.logger.info(
                        "epoch=%d step=%d stage=dataset_mix_progress dataset_mix=%s",
                        epoch,
                        step,
                        _dataset_mix_progress(mix_progress, mix_source_names),
                    )
                self._maybe_save(epoch, step)
                step += 1
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    self.logger.info("epoch=%d step=%d stage=max_steps_reached", epoch, step)
                    record_dashboard_state(self.areno, stage="max_steps_reached", epoch=epoch, step=step, role="policy")
                    return
            if mix_source_names:
                self.logger.info(
                    "epoch=%d stage=dataset_mix_epoch_end dataset_mix=%s",
                    epoch,
                    _dataset_mix_progress(mix_progress, mix_source_names),
                )
            self.logger.info("epoch=%d stage=epoch_end", epoch)
            record_dashboard_state(self.areno, stage="epoch_end", epoch=epoch, step=step, role="policy")

    def _iter_train_batches(
        self,
        tokenizer,
        processor,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        mix_progress: dict[str, Counter[str]] | None = None,
    ):
        # Dataset rows are converted lazily so large HF datasets do not need an
        # up-front tokenized copy. Rows that are empty, all-prompt, or exceed
        # the configured prompt or supervised-response budgets are dropped.
        batch = []
        batch_mix_counts: Counter[str] = Counter()
        batch_mix_target_tokens: Counter[str] = Counter()
        skipped = 0
        accepted = 0
        total_rows = len(self.dataset)
        for index in range(total_rows):
            # Normalize each supported row schema into one TrainSequence.
            record = self.dataset[index]
            source_name = _dataset_mix_source_name(record)
            if source_name is not None and mix_progress is not None:
                mix_progress["scheduled"][source_name] += 1
            seq = _record_to_train_sequence(
                record,
                tokenizer,
                processor,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
            if seq is None:
                skipped += 1
                if source_name is not None and mix_progress is not None:
                    mix_progress["filtered"][source_name] += 1
                continue
            accepted += 1
            batch.append(seq)
            if source_name is not None:
                batch_mix_counts[source_name] += 1
                batch_mix_target_tokens[source_name] += _sft_target_token_count(seq)
            if len(batch) >= self.config.batch_size:
                yield batch, batch_mix_counts, batch_mix_target_tokens
                batch = []
                batch_mix_counts = Counter()
                batch_mix_target_tokens = Counter()
        if skipped:
            self.logger.info("stage=sft_dataset_filter skipped_long_or_empty=%d", skipped)
        if accepted == 0:
            raise ValueError(
                "SFT dataset produced no valid training rows after filtering: "
                f"scanned {total_rows} row(s), skipped {skipped} as empty, over-budget, or all-prompt examples. "
                "Check dataset quality, --max-prompt-tokens, and --max-new-tokens."
            )
        if batch:
            yield batch, batch_mix_counts, batch_mix_target_tokens

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


def _record_to_train_sequence(record: Any, tokenizer, processor=None, *, max_prompt_tokens: int, max_new_tokens: int):
    """Normalize one loader-produced SFT row into backend training format.

    `prompt_mask=True` means "do not train this source token"; the backend loss
    is next-token aligned, so the loss function later uses positions after the
    prompt prefix. RL-only fields are filled with zeros to satisfy the shared
    `TrainSequence` packing contract.
    """

    record = dict(record)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if record_has_image(record):
        if "response" not in record:
            raise ValueError("SFT image rows must contain `response`")
        if record["response"] is None:
            return None
        response = str(record["response"])
        if not response:
            return None
        prompt_tokens, features = encode_multimodal_prompt(tokenizer, processor, record)
        try:
            response_tokens = [int(token) for token in tokenizer.encode(response, add_special_tokens=False)]
        except TypeError:
            response_tokens = [int(token) for token in tokenizer.encode(response)]
        response_tokens.append(eos_token_id)
        tokens = prompt_tokens + response_tokens
        prompt_mask = [True] * len(prompt_tokens) + [False] * len(response_tokens)
        prompt_token_count = len(prompt_tokens)
        if prompt_token_count > max_prompt_tokens or len(response_tokens) > max_new_tokens:
            return None
        zeros = [0.0] * len(tokens)
        return areno.api.TrainSequence(
            prompt_mask=prompt_mask,
            tokens=tokens,
            logprobs=zeros,
            advantages=zeros,
            features=features,
            eos_token_id=eos_token_id,
        )
    if "tokens" in record and "prompt_mask" in record:
        tokens = [int(token) for token in record["tokens"]]
        prompt_mask = [bool(item) for item in record["prompt_mask"]]
        if len(tokens) != len(prompt_mask):
            raise ValueError("SFT encoded row `tokens` and `prompt_mask` must have the same length")
        loss_mask = [bool(item) for item in record.get("loss_mask", [])]
        if loss_mask and len(loss_mask) != len(tokens):
            raise ValueError("SFT encoded row `loss_mask` must be empty or have the same length as `tokens`")
        features = record.get("features")
        image_counts = image_token_counts_from_features(features if isinstance(features, dict) else None)
        if image_counts:
            image_token_id = features.get("image_token_id") if isinstance(features, dict) else None
            if image_token_id is None:
                raise ValueError("SFT multimodal encoded rows require features.image_token_id")
            aligned = {"prompt_mask": prompt_mask}
            if loss_mask:
                aligned["loss_mask"] = loss_mask
            tokens, expanded = expand_image_tokens(
                tokens,
                image_token_id=int(image_token_id),
                image_token_counts=image_counts,
                aligned_sequences=aligned,
            )
            prompt_mask = [bool(item) for item in expanded["prompt_mask"]]
            loss_mask = [bool(item) for item in expanded.get("loss_mask", [])]
            mrope_position_ids = mrope_position_ids_from_image_grid(
                tokens,
                image_token_id=int(image_token_id),
                features=features,
            )
            if mrope_position_ids is not None:
                features = dict(features)
                features["mrope_position_ids"] = mrope_position_ids
        if len(tokens) < 2:
            return None
        prompt_tokens = prompt_mask.count(True)
        response_tokens = prompt_mask[1:].count(False)
        if prompt_tokens > max_prompt_tokens or response_tokens > max_new_tokens or response_tokens == 0:
            return None
        zeros = [0.0] * len(tokens)
        return areno.api.TrainSequence(
            prompt_mask=prompt_mask,
            loss_mask=loss_mask,
            tokens=tokens,
            logprobs=zeros,
            advantages=zeros,
            features=features,
            eos_token_id=int(record.get("eos_token_id", eos_token_id)),
        )
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


def _dataset_mix_source_name(record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return None
    metadata = record.get(DATASET_MIX_METADATA_KEY)
    if not isinstance(metadata, Mapping):
        return None
    source_name = metadata.get("source")
    return source_name if isinstance(source_name, str) else None


def _sft_target_token_count(sequence: areno.api.TrainSequence) -> int:
    """Count positions that contribute to SFT loss after next-token alignment."""

    prompt_mask = sequence.prompt_mask[1:]
    if sequence.loss_mask:
        return sum(
            not is_prompt and is_enabled
            for is_prompt, is_enabled in zip(prompt_mask, sequence.loss_mask[1:], strict=True)
        )
    return prompt_mask.count(False)


def _dataset_mix_progress(progress: dict[str, Counter[str]], source_names: list[str]) -> dict[str, Any]:
    rows_scheduled = sum(progress["scheduled"].values())
    rows_filtered = sum(progress["filtered"].values())
    rows_trained = sum(progress["trained"].values())
    target_tokens = sum(progress["target_tokens"].values())
    return {
        "rows_scheduled": rows_scheduled,
        "rows_filtered": rows_filtered,
        "rows_trained": rows_trained,
        "target_tokens_trained": target_tokens,
        "sources": [
            {
                "name": name,
                "rows_scheduled": progress["scheduled"][name],
                "rows_filtered": progress["filtered"][name],
                "rows_trained": progress["trained"][name],
                "target_tokens_trained": progress["target_tokens"][name],
                "observed_sample_proportion": progress["trained"][name] / rows_trained if rows_trained else 0.0,
                "observed_token_proportion": (
                    progress["target_tokens"][name] / target_tokens if target_tokens else 0.0
                ),
            }
            for name in sorted(source_names)
        ],
    }


__all__ = ["SFTTrainer"]
