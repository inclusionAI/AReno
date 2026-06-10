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
                # 1) Sample n_samples completions per prompt; ordering matches
                #    `prompt_batch.items` so we can zip them downstream.
                prompt_tokens = [item.input_tokens for item in prompt_batch.items]
                rollout_results = self.areno.rollout_token_batch(prompt_tokens, self.config.n_samples, sampling_params)
                self.logger.info("epoch=%d step=%d role=%s stage=rollout_end", epoch, step, role)
                self._log_sample_completions(tokenizer, epoch, step, prompt_batch, rollout_results)

                # 2+3) Score rewards and broadcast group-normalised advantages
                #      down to per-token tensors via `_materialize_train_batch`.
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
