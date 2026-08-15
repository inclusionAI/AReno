"""Dynamic-sampling trainer for experimental DAPO."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

import numpy as np

from areno.api.dashboard import record_dashboard_state
from areno.api.data import PromptItem
from areno.api.models import RolloutResult
from areno.api.tokenizer import configure_chat_template_enable_thinking
from areno.api.trainers.policy_only import PolicyOnlyTrainer


@dataclass(slots=True)
class _DAPOPromptGroup:
    """One prompt's rollout samples and rewards before/after shaping."""

    item: PromptItem
    rollout_result: RolloutResult
    raw_rewards: list[float]
    shaped_rewards: list[float]

    @property
    def informative(self) -> bool:
        """Return whether raw verifier rewards produce a non-zero advantage."""

        return _has_reward_variance(self.raw_rewards)


@dataclass(slots=True)
class _DAPOCollection:
    """Candidate-generation outcome for one intended optimizer batch."""

    groups: list[_DAPOPromptGroup]
    target_groups: int
    gen_batches: int = 0
    generated_groups: int = 0
    filtered_groups: int = 0
    discarded_qualified_groups: int = 0

    @property
    def complete(self) -> bool:
        return len(self.groups) >= self.target_groups


class DAPOTrainer(PolicyOnlyTrainer):
    """Policy trainer implementing DAPO's dynamic sampling controller."""

    def _fit_initialized(self) -> None:
        import areno.api

        if self._agentic_enabled():
            raise ValueError("DAPO does not support agentic rollout")
        if not callable(self.reward_fn):
            raise ValueError("DAPO requires a callable reward_fn")
        tokenizer = self.areno.get_tokenizer()
        processor = self.areno.get_processor()
        configure_chat_template_enable_thinking(tokenizer, self.config.chat_template_enable_thinking)
        configure_chat_template_enable_thinking(processor, self.config.chat_template_enable_thinking)
        sampling_params = areno.api.SamplingParams(
            greedy=self.config.greedy,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            max_context_len=self.config.max_context_len,
            max_prompt_len=self.config.max_prompt_tokens,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )

        step = 0
        for epoch in range(self.config.epochs):
            self.logger.info("epoch=%d stage=epoch_start", epoch)
            record_dashboard_state(self.areno, stage="epoch_start", epoch=epoch, step=step, role="policy")
            prompt_batches = iter(
                self.areno.load_prompt_batches(
                    self.dataset,
                    batch_size=self.config.resolved_gen_batch_size(),
                    max_prompt_tokens=self.config.max_prompt_tokens,
                )
            )
            while True:
                collection = self._collect_qualified_groups(
                    tokenizer,
                    sampling_params,
                    prompt_batches,
                    epoch=epoch,
                    step=step,
                )
                if not collection.complete:
                    if collection.gen_batches:
                        self.logger.warning(
                            "epoch=%d dropped incomplete DAPO batch qualified_groups=%d target_groups=%d",
                            epoch,
                            len(collection.groups),
                            self.config.batch_size,
                        )
                        self.areno.finish_step()
                    break

                train_batch, raw_rewards, shaped_rewards, rollout_logprobs = self._materialize_scored_groups(
                    tokenizer, collection.groups
                )
                train_stats = self._collection_metrics(collection, raw_rewards, shaped_rewards)
                if rollout_logprobs:
                    self.logger.info(
                        "epoch=%d step=%d metric=rollout_logprob_mean value=%.6f",
                        epoch,
                        step,
                        float(np.mean(rollout_logprobs)),
                    )
                accumulation_steps = self._resolved_gradient_accumulation_steps()
                if (
                    _optimizer_update_count(
                        len(train_batch),
                        mini_bs=self.config.mini_bs,
                        gradient_accumulation_steps=accumulation_steps,
                    )
                    <= 1
                ):
                    self.logger.warning(
                        "epoch=%d step=%d DAPO train batch has one optimizer update; "
                        "importance ratios will remain near one",
                        epoch,
                        step,
                    )

                self.logger.info("epoch=%d step=%d role=policy stage=train_start", epoch, step)
                record_dashboard_state(self.areno, stage="train_start", epoch=epoch, step=step, role="policy")
                train_start = time.perf_counter()
                result = self.areno.train(
                    train_batch,
                    self.loss_fn,
                    mini_bs=self.config.mini_bs,
                    gradient_accumulation_steps=accumulation_steps,
                )
                train_time_s = time.perf_counter() - train_start
                if not isinstance(result, dict):
                    result = {}
                result.update(train_stats)
                result["policy_train_wall_time_s"] = train_time_s
                result = self._augment_train_stats(result)
                self.logger.info("epoch=%d step=%d role=policy stage=train_end", epoch, step)
                record_dashboard_state(self.areno, stage="train_end", epoch=epoch, step=step, role="policy")
                self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                self._maybe_save(epoch, step)

                step += 1
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    self.logger.info("epoch=%d step=%d stage=max_steps_reached", epoch, step)
                    record_dashboard_state(
                        self.areno,
                        stage="max_steps_reached",
                        epoch=epoch,
                        step=step,
                        role="policy",
                    )
                    return
            self.logger.info("epoch=%d stage=epoch_end", epoch)
            record_dashboard_state(self.areno, stage="epoch_end", epoch=epoch, step=step, role="policy")

        if step == 0:
            raise RuntimeError(
                "DAPO dynamic sampling produced no complete training batch; "
                "reduce --batch-size, increase the dataset, or inspect reward variance"
            )

    def _collect_qualified_groups(
        self,
        tokenizer,
        sampling_params,
        prompt_batches,
        *,
        epoch: int,
        step: int,
    ) -> _DAPOCollection:
        collection = _DAPOCollection(groups=[], target_groups=self.config.batch_size)
        while len(collection.groups) < self.config.batch_size:
            try:
                prompt_batch = next(prompt_batches)
            except StopIteration:
                return collection

            collection.gen_batches += 1
            groups = self._generate_candidate_groups(
                tokenizer,
                sampling_params,
                prompt_batch,
                epoch=epoch,
                step=step,
            )
            collection.generated_groups += len(groups)
            for group in groups:
                if not group.informative:
                    collection.filtered_groups += 1
                    continue
                if len(collection.groups) < self.config.batch_size:
                    collection.groups.append(group)
                else:
                    collection.discarded_qualified_groups += 1

            self.logger.info(
                "epoch=%d step=%d stage=dynamic_sampling gen_batches=%d generated_groups=%d "
                "qualified_groups=%d filtered_groups=%d",
                epoch,
                step,
                collection.gen_batches,
                collection.generated_groups,
                len(collection.groups),
                collection.filtered_groups,
            )
            if len(collection.groups) >= self.config.batch_size:
                return collection
            if collection.gen_batches >= self.config.dapo_max_num_gen_batches:
                raise RuntimeError(
                    "DAPO dynamic sampling reached dapo_max_num_gen_batches="
                    f"{self.config.dapo_max_num_gen_batches}: generated_groups={collection.generated_groups} "
                    f"qualified_groups={len(collection.groups)} filtered_groups={collection.filtered_groups}; "
                    "increase --dapo-max-num-gen-batches or inspect reward variance"
                )
        return collection

    def _generate_candidate_groups(self, tokenizer, sampling_params, prompt_batch, *, epoch: int, step: int):
        self.logger.info("epoch=%d step=%d role=policy stage=rollout_start", epoch, step)
        record_dashboard_state(self.areno, stage="rollout_start", epoch=epoch, step=step, role="policy")
        rollout_results = asyncio.run(self._run_prompt_rollout(sampling_params, prompt_batch))
        self.logger.info("epoch=%d step=%d role=policy stage=rollout_end", epoch, step)
        record_dashboard_state(self.areno, stage="rollout_end", epoch=epoch, step=step, role="policy")
        self._record_sample_completions(tokenizer, epoch, step, prompt_batch, rollout_results)
        return [
            self._score_prompt_group(tokenizer, item, result, prompt_index=prompt_index)
            for prompt_index, (item, result) in enumerate(zip(prompt_batch.items, rollout_results, strict=True))
        ]

    def _score_prompt_group(self, tokenizer, item, result, *, prompt_index: int) -> _DAPOPromptGroup:
        from areno.api.rewards import make_reward_record

        raw_rewards = []
        shaped_rewards = []
        prefix_len = len(item.input_tokens)
        for sample_index, sequence in enumerate(result.sequences):
            if len(sequence.resp_tokens) != len(sequence.resp_logprobs):
                raise ValueError("DAPO rollout response tokens and logprobs must have equal length")
            if any(not math.isfinite(logprob) for logprob in sequence.resp_logprobs):
                raise ValueError("DAPO rollout must contain finite rollout logprobs")
            completion = tokenizer.decode(sequence.resp_tokens)
            reward = float(
                self.reward_fn(
                    make_reward_record(
                        prompt=item.prompt,
                        completion=completion,
                        source_record=item.record,
                        answer=item.solutions,
                        tokens=item.input_tokens + sequence.resp_tokens,
                        logprobs=[0.0] * prefix_len + sequence.resp_logprobs,
                        loss_mask=[False] * prefix_len + [True] * len(sequence.resp_tokens),
                        metadata={"prompt_index": prompt_index, "sample_index": sample_index},
                    )
                )
            )
            if not math.isfinite(reward):
                raise ValueError("DAPO reward_fn must return finite rewards")
            penalty = _overlong_penalty(
                len(sequence.resp_tokens),
                max_response_length=self.config.max_new_tokens,
                buffer_length=self.config.dapo_overlong_buffer_len,
                factor=self.config.dapo_overlong_penalty_factor,
            )
            raw_rewards.append(reward)
            shaped_rewards.append(reward + penalty)
        return _DAPOPromptGroup(
            item=item,
            rollout_result=result,
            raw_rewards=raw_rewards,
            shaped_rewards=shaped_rewards,
        )

    def _materialize_scored_groups(self, tokenizer, groups: list[_DAPOPromptGroup]):
        import areno.api
        from areno.api.rewards import compute_group_advantages

        train_batch = []
        raw_rewards_all = []
        shaped_rewards_all = []
        rollout_logprobs = []
        for group in groups:
            sequences = group.rollout_result.sequences
            if len(sequences) != len(group.raw_rewards):
                raise ValueError("DAPO rollout sequence and raw reward counts must match")
            if len(sequences) != len(group.shaped_rewards):
                raise ValueError("DAPO rollout sequence and reward counts must match")
            advantages = compute_group_advantages(group.shaped_rewards)
            prefix_len = len(group.item.input_tokens)
            raw_rewards_all.extend(group.raw_rewards)
            shaped_rewards_all.extend(group.shaped_rewards)
            for sequence, advantage, reward in zip(
                sequences,
                advantages,
                group.shaped_rewards,
                strict=True,
            ):
                response_len = len(sequence.resp_tokens)
                rollout_logprobs.extend(sequence.resp_logprobs)
                train_batch.append(
                    areno.api.TrainSequence(
                        prompt_mask=[True] * prefix_len + [False] * response_len,
                        tokens=group.item.input_tokens + sequence.resp_tokens,
                        logprobs=[0.0] * prefix_len + sequence.resp_logprobs,
                        advantages=[0.0] * prefix_len + [advantage] * response_len,
                        features=group.item.record.get("features"),
                        reward=reward,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                )
        return train_batch, raw_rewards_all, shaped_rewards_all, rollout_logprobs

    def _collection_metrics(self, collection, raw_rewards, shaped_rewards) -> dict[str, float]:
        penalties = [shaped - raw for raw, shaped in zip(raw_rewards, shaped_rewards, strict=True)]
        return {
            "dapo_gen_batches": float(collection.gen_batches),
            "dapo_generated_groups": float(collection.generated_groups),
            "dapo_qualified_groups": float(len(collection.groups)),
            "dapo_filtered_groups": float(collection.filtered_groups),
            "dapo_discarded_qualified_groups": float(collection.discarded_qualified_groups),
            "dapo_sampling_efficiency": float(len(collection.groups)) / max(collection.generated_groups, 1),
            "dapo_raw_reward_mean": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
            "dapo_shaped_reward_mean": float(np.mean(shaped_rewards)) if shaped_rewards else 0.0,
            "dapo_overlong_penalty_mean": float(np.mean(penalties)) if penalties else 0.0,
        }

    def _resolved_gradient_accumulation_steps(self) -> int:
        """Default to per-pack updates so stored rollout ratios can evolve."""

        if self.config.gradient_accumulation_steps is not None:
            return max(int(self.config.gradient_accumulation_steps), 1)
        return 1


def _has_reward_variance(rewards: list[float]) -> bool:
    """Return whether a prompt group yields a non-zero relative advantage."""

    if len(rewards) < 2:
        return False
    rewards_array = np.asarray(rewards, dtype=np.float32)
    return bool(rewards_array.max() > rewards_array.min())


def _overlong_penalty(
    response_length: int,
    *,
    max_response_length: int,
    buffer_length: int,
    factor: float,
) -> float:
    """Return DAPO's bounded linear penalty near the response-length limit."""

    if buffer_length <= 0 or factor <= 0:
        return 0.0
    safe_length = max_response_length - buffer_length
    excess = max(int(response_length) - safe_length, 0)
    return -min(float(excess) / float(buffer_length), 1.0) * float(factor)


def _optimizer_update_count(
    sequence_count: int,
    *,
    mini_bs: int,
    gradient_accumulation_steps: int,
) -> int:
    pack_count = math.ceil(sequence_count / max(int(mini_bs), 1))
    return math.ceil(pack_count / max(int(gradient_accumulation_steps), 1))
