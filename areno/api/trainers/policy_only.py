"""Policy-only RL training loop (GSPO/GRPO).

Each step performs the standard rollout/reward/train cycle:
    1. rollout_batch() returns `n_samples` completions per prompt.
    2. The reward function scores every completion against its prompt record.
    3. Group-relative advantages are computed within each prompt and broadcast
       to every response token (prompt positions are masked to zero).
    4. A `TrainSequence` is built per (prompt, sample) pair and handed to the
       backend's `train()`, which runs the caller-provided loss.
PPOTrainer subclasses this class and overrides only the batch assembly and
role-management hooks; this is why the helpers are designed to be small.
"""

from __future__ import annotations

import logging
import os
import time
import asyncio
from pathlib import Path

import numpy as np


class PolicyOnlyTrainer:
    """Rollout-reward-train loop for policy-only RL algorithms.

    This covers GSPO/GRPO-style training where the only model role is the
    trainable policy. Rollout logprobs returned by the backend are treated as
    old policy logprobs, rewards are supplied by a Python reward function, and
    advantages are normalized within each prompt group.
    """

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        self.config = config
        self.areno = instance
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.loss_fn = loss_fn
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._agent_run_fn = None

    def fit(self) -> None:
        self.areno.init()
        try:
            self._fit_initialized()
        finally:
            self.areno.close()

    def _fit_initialized(self) -> None:
        import areno.api

        tokenizer = self.areno.get_tokenizer()
        sampling_params = areno.api.SamplingParams(
            greedy=self.config.greedy,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            max_prompt_len=self.config.max_prompt_tokens,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )

        step = 0
        for epoch in range(self.config.epochs):
            self.logger.info("epoch=%d stage=epoch_start", epoch)
            for prompt_batch in self.areno.load_prompt_batches(
                self.dataset,
                batch_size=self.config.batch_size,
                max_prompt_tokens=self.config.max_prompt_tokens,
            ):
                role = self._policy_role_name()
                self.logger.info("epoch=%d step=%d role=%s stage=rollout_start", epoch, step, role)
                if self._agentic_enabled():
                    agent_batch = asyncio.run(self._run_agentic_rollout(sampling_params, prompt_batch))
                    self.logger.info("epoch=%d step=%d role=%s stage=rollout_end", epoch, step, role)
                    self._log_agentic_sample_completions(epoch, step, agent_batch)
                    train_batch, rewards_all, rollout_logprobs = self._materialize_agentic_train_batch(tokenizer, prompt_batch, agent_batch)
                else:
                    # 1) Sample n_samples completions per prompt; ordering
                    #    matches `prompt_batch.items` so we can zip downstream.
                    rollout_results = asyncio.run(self._run_prompt_rollout(sampling_params, prompt_batch))
                    self.logger.info("epoch=%d step=%d role=%s stage=rollout_end", epoch, step, role)
                    self._log_sample_completions(tokenizer, epoch, step, prompt_batch, rollout_results)

                    # 2+3) Score rewards and broadcast group-normalised
                    #      advantages down to per-token tensors.
                    train_batch, rewards_all, rollout_logprobs = self._materialize_train_batch(tokenizer, prompt_batch, rollout_results)

                if rewards_all:
                    self.logger.info("epoch=%d step=%d metric=reward_mean value=%.6f", epoch, step, float(np.mean(rewards_all)))
                if rollout_logprobs:
                    self.logger.info(
                        "epoch=%d step=%d metric=rollout_logprob_mean value=%.6f",
                        epoch,
                        step,
                        float(np.mean(rollout_logprobs)),
                    )

                if train_batch:
                    # PPO uses this hook to skip actor updates during the
                    # critic-only warmup window; GSPO/GRPO always train.
                    if not self._should_train_policy(step):
                        result = self._augment_train_stats({"actor_train_skipped": 1.0})
                        self.logger.info("epoch=%d step=%d role=%s stage=train_skip", epoch, step, role)
                        self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                        self.areno.finish_step()
                        step += 1
                        continue
                    self.logger.info("epoch=%d step=%d role=%s stage=train_start", epoch, step, role)
                    train_start = time.perf_counter()
                    # 4) The actual gradient step happens inside the backend.
                    result = self.areno.train(
                        train_batch,
                        self.loss_fn,
                        mini_bs=self.config.mini_bs,
                        gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    )
                    train_time_s = time.perf_counter() - train_start
                    if isinstance(result, dict):
                        result[f"{role}_train_wall_time_s"] = train_time_s
                    result = self._augment_train_stats(result)
                    self.logger.info("epoch=%d step=%d role=%s stage=train_end", epoch, step, role)
                    self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                    self._maybe_save(epoch, step)
                step += 1
            self.logger.info("epoch=%d stage=epoch_end", epoch)

    def _policy_role_name(self) -> str:
        # GSPO/GRPO have a single trainable model called "policy"; PPO
        # overrides this to "actor" so logs distinguish between actor/critic.
        return "policy"

    def _should_train_policy(self, step: int) -> bool:
        # PPO overrides this to defer actor updates during critic warmup.
        del step
        return True

    def _augment_train_stats(self, result):
        # Hook for PPO to attach role-specific stats (critic loss, KL,
        # reference forward-time, ...) before they reach the metric recorder.
        return result

    def _agentic_enabled(self) -> bool:
        return bool(getattr(self.config, "agent_fn", None))

    def _loss_mask_policy(self):
        from areno.api.agentic import LossMaskPolicy

        return LossMaskPolicy(
            tool_results=bool(getattr(self.config, "train_tool_results", False)),
        )

    def _get_agent_run_fn(self):
        from areno.api.agentic import load_agent_run_fn

        if self._agent_run_fn is None:
            self._agent_run_fn = load_agent_run_fn(self.config.agent_fn)
        return self._agent_run_fn

    async def _run_prompt_rollout(self, sampling_params, prompt_batch):
        async with self.areno.rollout_session(
            sampling_params=sampling_params,
            max_running_prompts=self.config.resolved_max_running_prompts(),
            proxy=False,
        ):
            prompt_tokens = [item.input_tokens for item in prompt_batch.items]
            return self.areno.rollout_token_batch(prompt_tokens, self.config.n_samples, sampling_params)

    async def _run_agentic_rollout(self, sampling_params, prompt_batch):
        from areno.api.agentic import AgentBatch, maybe_await

        agent_batch = AgentBatch.from_prompt_batch(prompt_batch, n_samples=self.config.n_samples)
        self.logger.info(
            "agentic rollout batch prompts=%d n_samples=%d expected_requests=%d max_running_prompts=%d",
            len(agent_batch.records),
            agent_batch.n_samples,
            len(agent_batch),
            self.config.resolved_max_running_prompts(),
        )
        async with self.areno.rollout_session(
            sampling_params=sampling_params,
            loss_mask_policy=self._loss_mask_policy(),
            max_running_prompts=self.config.resolved_max_running_prompts(),
            timeout_s=self.config.agent_timeout_s,
        ) as ctx:
            ctx.attach_batch(agent_batch)
            await maybe_await(self._get_agent_run_fn()(ctx, agent_batch))
            ctx.finish_requests()
            return await ctx.get_train_batch(reward_fn=self.reward_fn)

    def _materialize_agentic_train_batch(self, tokenizer, prompt_batch, agent_batch):
        """Assemble TrainSequence rows from an agentic rollout batch."""

        import areno.api
        from areno.api.rewards import compute_group_advantages

        del prompt_batch
        if agent_batch.rewards is None:
            raise ValueError("agentic policy training requires a reward_fn")
        train_batch = []
        rewards_all = [float(reward) for reward in agent_batch.rewards]
        rollout_logprobs = []
        grouped: dict[int, list[int]] = {}
        for row_idx, record in enumerate(agent_batch.reward_records):
            prompt_index = int(record.metadata.get("prompt_index", row_idx))
            grouped.setdefault(prompt_index, []).append(row_idx)
        advantages_by_row: dict[int, float] = {}
        for row_indices in grouped.values():
            group_rewards = [rewards_all[row_idx] for row_idx in row_indices]
            for row_idx, advantage in zip(row_indices, compute_group_advantages(group_rewards), strict=True):
                advantages_by_row[row_idx] = float(advantage)
        for row_idx, (tokens, response_mask, loss_mask, logprobs, reward) in enumerate(
            zip(
                agent_batch.token_rows,
                agent_batch.response_masks,
                agent_batch.loss_masks,
                agent_batch.rollout_logprobs,
                rewards_all,
                strict=True,
            )
        ):
            if len(tokens) != len(response_mask) or len(tokens) != len(loss_mask) or len(tokens) != len(logprobs):
                raise ValueError("agentic train batch has misaligned token/mask/logprob rows")
            prompt_mask = [not item for item in response_mask]
            advantage = advantages_by_row.get(row_idx, 0.0)
            advantages = [advantage if is_loss else 0.0 for is_loss in loss_mask]
            rollout_logprobs.extend(lp for lp, is_loss in zip(logprobs, loss_mask, strict=True) if is_loss)
            train_batch.append(
                areno.api.TrainSequence(
                    prompt_mask=prompt_mask,
                    loss_mask=loss_mask,
                    tokens=tokens,
                    logprobs=logprobs,
                    advantages=advantages,
                    reward=float(reward),
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
        return train_batch, rewards_all, rollout_logprobs

    def _log_sample_completions(self, tokenizer, epoch: int, step: int, prompt_batch, rollout_results) -> None:
        # Diagnostics knob: setting ARENO_LOG_COMPLETIONS=N dumps up to N
        # decoded completions per step so reward debugging is easier.
        limit = int(os.getenv("ARENO_LOG_COMPLETIONS", "0"))
        if limit <= 0:
            return
        logged = 0
        for prompt_idx, (item, result) in enumerate(zip(prompt_batch.items, rollout_results, strict=True)):
            for sample_idx, seq in enumerate(result.sequences):
                self.logger.info(
                    "epoch=%d step=%d prompt_idx=%d sample_idx=%d prompt=%r completion=%r tokens=%s",
                    epoch,
                    step,
                    prompt_idx,
                    sample_idx,
                    item.prompt,
                    tokenizer.decode(seq.resp_tokens),
                    seq.resp_tokens[:64],
                )
                logged += 1
                if logged >= limit:
                    return

    def _log_agentic_sample_completions(self, epoch: int, step: int, agent_batch) -> None:
        # Match non-agentic rollout diagnostics so reward/debug workflows do
        # not depend on rollout mode.
        limit = int(os.getenv("ARENO_LOG_COMPLETIONS", "0"))
        if limit <= 0:
            return
        for logged, record in enumerate(agent_batch.reward_records):
            prompt_idx = int(record.metadata.get("prompt_index", -1))
            sample_idx = int(record.metadata.get("sample_index", -1))
            loss_mask = agent_batch.loss_masks[logged]
            token_row = agent_batch.token_rows[logged]
            first_loss_idx = next((idx for idx, enabled in enumerate(loss_mask) if enabled), -1)
            self.logger.info(
                "epoch=%d step=%d prompt_idx=%d sample_idx=%d prompt=%r rendered_completion=%r final_answer=%r tool_calls=%s tool_results=%s loss_mask_true=%d/%d first_loss_idx=%d loss_mask=%s tokens=%s",
                epoch,
                step,
                prompt_idx,
                sample_idx,
                record.prompt,
                record.rendered_completion,
                record.final_answer,
                record.tool_calls,
                record.tool_results[:4],
                sum(1 for enabled in loss_mask if enabled),
                len(loss_mask),
                first_loss_idx,
                loss_mask[:64],
                token_row[:64],
            )
            if logged + 1 >= limit:
                return

    def _materialize_train_batch(self, tokenizer, prompt_batch, rollout_results):
        """Assemble TrainSequence rows for one rollout batch.

        Steps:
            1. Decode each completion and score it with `reward_fn`.
            2. Standardise rewards within each prompt group to get advantages
               (`compute_group_advantages`); this is the GRPO/GSPO baseline.
            3. Stitch each prompt prefix with its response tokens and copy the
               group-level advantage onto every response position; prompt
               positions carry zero advantage and zero logprob.
        """

        import areno.api
        from areno.api.rewards import compute_group_advantages

        train_batch = []
        rewards_all = []
        rollout_logprobs = []
        for item, result in zip(prompt_batch.items, rollout_results, strict=True):
            prefix_len = len(item.input_tokens)
            completions = [tokenizer.decode(seq.resp_tokens) for seq in result.sequences]
            rewards = self.reward_fn(item.record, completions)
            if len(rewards) != len(completions):
                raise ValueError(f"reward_fn returned {len(rewards)} rewards for {len(completions)} completions")
            rewards_all += rewards
            # Group-relative advantage: A_i = (r_i - mean(r))/std(r); shared by
            # every response token of sample i.
            advantages = compute_group_advantages(rewards)
            for seq, advantage, reward in zip(result.sequences, advantages, rewards, strict=True):
                resp_len = len(seq.resp_tokens)
                rollout_logprobs += seq.resp_logprobs
                train_batch.append(
                    areno.api.TrainSequence(
                        # Prompt positions are masked (1=prompt, 0=response).
                        prompt_mask=[1] * prefix_len + [0] * resp_len,
                        tokens=item.input_tokens + seq.resp_tokens,
                        # Rollout logprobs play the role of "old logprobs"; the
                        # zero prefix keeps tensor lengths aligned with tokens.
                        logprobs=[0.0] * prefix_len + seq.resp_logprobs,
                        advantages=[0.0] * prefix_len + [advantage] * resp_len,
                        reward=reward,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                )
        return train_batch, rewards_all, rollout_logprobs

    def _maybe_save(self, epoch: int, step: int) -> None:
        # Checkpoint cadence is "save_interval" steps; `step + 1` mirrors the
        # usual convention that step 99 saves at the end of the 100th update.
        if self.config.save_path is None or (step + 1) % self.config.save_interval != 0:
            return
        ckpt_path = str(Path(self.config.save_path) / f"step_{step + 1:06d}")
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_start path=%s", epoch, step, ckpt_path)
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_end path=%s", epoch, step, saved_path)
