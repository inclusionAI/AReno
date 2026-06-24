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

import asyncio
import logging
import os
import time
from pathlib import Path

import numpy as np

from areno.api.tokenizer import configure_chat_template_enable_thinking


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
        configure_chat_template_enable_thinking(tokenizer, self.config.chat_template_enable_thinking)
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
                    train_batch, rewards_all, rollout_logprobs = self._materialize_agentic_train_batch(
                        tokenizer, prompt_batch, agent_batch
                    )
                else:
                    # 1) Sample n_samples completions per prompt; ordering
                    #    matches `prompt_batch.items` so we can zip downstream.
                    rollout_results = asyncio.run(self._run_prompt_rollout(sampling_params, prompt_batch))
                    self.logger.info("epoch=%d step=%d role=%s stage=rollout_end", epoch, step, role)
                    self._log_sample_completions(tokenizer, epoch, step, prompt_batch, rollout_results)

                    # 2+3) Score rewards and broadcast group-normalised
                    #      advantages down to per-token tensors.
                    train_batch, rewards_all, rollout_logprobs = self._materialize_train_batch(
                        tokenizer, prompt_batch, rollout_results
                    )

                if rewards_all:
                    self.logger.info(
                        "epoch=%d step=%d metric=reward_mean value=%.6f", epoch, step, float(np.mean(rewards_all))
                    )
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
            return await self.areno.rollout_token_batch_async(prompt_tokens, self.config.n_samples, sampling_params)

    async def _run_agentic_rollout(self, sampling_params, prompt_batch):
        from areno.api.agentic import AgentBatch, AgentTrainBatch, maybe_await

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
            await ctx.sync_rollout_session_async()
            trajectories = await maybe_await(self._get_agent_run_fn()(ctx, agent_batch))
            if trajectories is None:
                raise RuntimeError("agent run function must return explicit trajectories")
            samples = []
            for turn in self._agent_trajectory_turns(ctx, trajectories):
                sample = ctx._sample_from_trajectory_turn(turn)
                existing = self._find_agent_sample(samples, sample.item)
                if existing is None:
                    samples.append(sample)
                else:
                    ctx._append_sample_response(existing, sample)
            samples, filtered_count, filter_diagnostics = self._filter_overlong_agent_samples(
                ctx, samples, sampling_params
            )
            expected = len(agent_batch)
            if len(samples) + filtered_count != expected:
                raise RuntimeError(
                    f"agent rollout produced {len(samples)} trajectories and filtered {filtered_count}, expected {expected}"
                )
            if not samples:
                raise RuntimeError(
                    f"all {filtered_count} agent trajectories exceeded the configured context length; "
                    f"{self._format_agent_filter_diagnostics(filter_diagnostics)}"
                )
            reward_records = [ctx.reward_record(sample) for sample in samples]
            rewards = [float(self.reward_fn(record)) for record in reward_records]
            rows = ctx._train_rows_from_samples(samples)
            tool_call_count = sum(len(record.tool_calls) for record in reward_records)
            tool_result_count = sum(len(record.tool_results) for record in reward_records)
            message_count = sum(len(record.messages) for record in reward_records)
            self.logger.info(
                "agentic train batch built samples=%d tokens=%d messages=%d tool_calls=%d tool_results=%d",
                len(samples),
                rows.total_tokens,
                message_count,
                tool_call_count,
                tool_result_count,
            )
            return AgentTrainBatch(
                token_rows=rows.token_rows,
                response_masks=rows.response_masks,
                loss_masks=rows.loss_masks,
                rollout_logprobs=rows.rollout_logprobs,
                rewards=rewards,
                records=[sample.item.record for sample in samples],
                reward_records=reward_records,
            )

    def _filter_overlong_agent_samples(self, ctx, samples, sampling_params):
        max_context_len = self._agent_model_context_len()
        if max_context_len is None:
            return samples, 0, {}
        kept = []
        filtered_details = []
        all_details = []
        for sample in samples:
            rows = ctx._train_rows_from_samples([sample])
            token_len = len(rows.token_rows[0]) if rows.token_rows else 0
            detail = self._agent_sample_filter_detail(sample, token_len)
            all_details.append(detail)
            if token_len <= max_context_len:
                kept.append(sample)
                continue
            filtered_details.append(detail)
        diagnostics = self._agent_filter_diagnostics(
            all_details,
            filtered_details,
            max_context_len=max_context_len,
            kept_count=len(kept),
        )
        if filtered_details:
            self.logger.warning("agentic trajectory filtered: %s", self._format_agent_filter_diagnostics(diagnostics))
        return kept, len(filtered_details), diagnostics

    def _agent_sample_filter_detail(self, sample, token_len):
        tool_result_count = sum(1 for message in sample.messages if message.get("role") == "tool")
        assistant_count = sum(1 for message in sample.messages if message.get("role") == "assistant")
        return {
            "prompt_idx": sample.item.prompt_index,
            "sample_idx": sample.item.sample_index,
            "tokens": int(token_len),
            "messages": len(sample.messages),
            "assistant_messages": assistant_count,
            "tool_results": tool_result_count,
            "response_tokens": len(sample.response_tokens),
            "trace_events": len(sample.trace),
            "prompt": str(sample.item.prompt).replace("\n", "\\n")[:120],
        }

    def _agent_filter_diagnostics(self, all_details, filtered_details, *, max_context_len, kept_count):
        token_lengths = sorted(detail["tokens"] for detail in all_details)
        return {
            "max_context_len": int(max_context_len),
            "total": len(all_details),
            "kept": int(kept_count),
            "filtered": len(filtered_details),
            "min_tokens": token_lengths[0] if token_lengths else 0,
            "p50_tokens": self._percentile_value(token_lengths, 0.50),
            "p90_tokens": self._percentile_value(token_lengths, 0.90),
            "max_tokens": token_lengths[-1] if token_lengths else 0,
            "top": sorted(filtered_details, key=lambda item: item["tokens"], reverse=True)[:5],
        }

    def _format_agent_filter_diagnostics(self, diagnostics):
        if not diagnostics:
            return "no context-length diagnostics available"
        top = "; ".join(
            "prompt_idx={prompt_idx} sample_idx={sample_idx} tokens={tokens} messages={messages} "
            "assistant_messages={assistant_messages} tool_results={tool_results} response_tokens={response_tokens} "
            "trace_events={trace_events} prompt='{prompt}'".format(**detail)
            for detail in diagnostics.get("top", [])
        )
        return (
            "max_context_len={max_context_len} total={total} kept={kept} filtered={filtered} "
            "tokens[min/p50/p90/max]={min_tokens}/{p50_tokens}/{p90_tokens}/{max_tokens} top=[{top}]"
        ).format(
            max_context_len=diagnostics["max_context_len"],
            total=diagnostics["total"],
            kept=diagnostics["kept"],
            filtered=diagnostics["filtered"],
            min_tokens=diagnostics["min_tokens"],
            p50_tokens=diagnostics["p50_tokens"],
            p90_tokens=diagnostics["p90_tokens"],
            max_tokens=diagnostics["max_tokens"],
            top=top,
        )

    def _percentile_value(self, sorted_values, fraction):
        if not sorted_values:
            return 0
        index = min(int(round((len(sorted_values) - 1) * fraction)), len(sorted_values) - 1)
        return int(sorted_values[index])

    def _agent_model_context_len(self):
        try:
            value = self.areno.model_context_len()
        except (AttributeError, RuntimeError):
            return None
        if value is None:
            return None
        return int(value)

    def _agent_trajectory_turns(self, ctx, trajectories):
        from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

        del ctx
        if trajectories is None:
            return
        if isinstance(trajectories, AgentTrajectoryTurn):
            yield trajectories
            return
        if isinstance(trajectories, AgentTrajectory):
            yield from trajectories.turns
            return
        for trajectory in trajectories:
            if isinstance(trajectory, AgentTrajectoryTurn):
                yield trajectory
            elif isinstance(trajectory, AgentTrajectory):
                yield from trajectory.turns
            else:
                yield from trajectory

    def _find_agent_sample(self, samples, item):
        if item.prompt_index < 0 or item.sample_index < 0:
            return None
        key = (item.prompt_index, item.sample_index)
        for sample in samples:
            if (sample.item.prompt_index, sample.item.sample_index) == key:
                return sample
        return None

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
                    "epoch=%d step=%d prompt_idx=%d sample_idx=%d prompt=%r decoded_prompt=%r completion=%r "
                    "prompt_tokens=%s response_tokens=%s",
                    epoch,
                    step,
                    prompt_idx,
                    sample_idx,
                    item.prompt,
                    tokenizer.decode(item.input_tokens),
                    tokenizer.decode(seq.resp_tokens),
                    item.input_tokens[:64],
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
                "epoch=%d step=%d prompt_idx=%d sample_idx=%d prompt=%r messages=%s final_answer=%r tool_calls=%s tool_results=%s loss_mask_true=%d/%d first_loss_idx=%d loss_mask=%s tokens=%s",
                epoch,
                step,
                prompt_idx,
                sample_idx,
                record.prompt,
                record.messages,
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
        from areno.api.rewards import compute_group_advantages, make_reward_record

        train_batch = []
        rewards_all = []
        rollout_logprobs = []
        for item_idx, (item, result) in enumerate(zip(prompt_batch.items, rollout_results, strict=True)):
            prefix_len = len(item.input_tokens)
            completions = [tokenizer.decode(seq.resp_tokens) for seq in result.sequences]
            rewards = [
                float(
                    self.reward_fn(
                        make_reward_record(
                            prompt=item.prompt,
                            completion=completion,
                            source_record=item.record,
                            answer=item.solutions,
                            tokens=item.input_tokens + seq.resp_tokens,
                            logprobs=[0.0] * prefix_len + seq.resp_logprobs,
                            loss_mask=[False] * prefix_len + [True] * len(seq.resp_tokens),
                            metadata={"prompt_index": item_idx, "sample_index": sample_idx},
                        )
                    )
                )
                for sample_idx, (completion, seq) in enumerate(zip(completions, result.sequences, strict=True))
            ]
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
