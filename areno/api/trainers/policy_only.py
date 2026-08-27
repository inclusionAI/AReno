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
import copy
import json
import logging
import os
import time
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from areno.api.dashboard import record_dashboard_state
from areno.api.tokenizer import configure_chat_template_enable_thinking


class _LogprobStats:
    """Amortized (sum, count) over response logprobs for per-step logging.

    The training loop only consumes the per-step mean of rollout logprobs for
    a log line, so materializing one Python float per response token just to
    average it is pure CPU overhead on long-CoT workloads. This keeps the same
    metric with O(1) memory.
    """

    __slots__ = ("_sum", "_count")

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def add(self, values: Any) -> None:
        # Sized inputs use the C-speed builtin sum; generator inputs (agentic
        # loss-mask filters) fall back to a Python loop.
        if hasattr(values, "__len__") and not isinstance(values, (str, bytes)):
            self._sum += float(sum(values))
            self._count += len(values)
            return
        total = 0.0
        count = 0
        for value in values:
            total += float(value)
            count += 1
        self._sum += total
        self._count += count

    @property
    def mean(self) -> float | None:
        return self._sum / self._count if self._count else None

    def __bool__(self) -> bool:
        return self._count > 0


def _batch_decode(tokenizer: Any, token_lists: list[list[int]]) -> list[str]:
    """Decode every completion with one batched tokenizer call when supported.

    Falls back to per-completion ``tokenizer.decode`` for tokenizers without a
    ``batch_decode`` entry point so the call site stays tokenizer-agnostic.
    """

    batch_decode = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decode):
        return list(batch_decode(token_lists))
    return [tokenizer.decode(tokens) for tokens in token_lists]


def _rollout_logprob_mean(value: Any) -> float | None:
    """Return the mean of a rollout-logprob accumulator (list or stats).

    ``PPOTrainer`` overrides the batch assembly and still returns a plain
    Python list of logprobs; ``_LogprobStats`` is returned by the policy-only
    path. Both feed the same per-step metric line.
    """

    if value is None:
        return None
    if isinstance(value, _LogprobStats):
        return value.mean
    if value:
        return float(np.mean(value))
    return None


def _dashboard_safe_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Convert a dataset value into bounded JSON data for rollout samples."""

    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if "base64" in key.lower() and len(value) > 256:
            return f"<base64 data: {len(value)} characters>"
        return value if len(value) <= 20_000 else value[:20_000] + "... <truncated>"
    if depth >= 8:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {
            str(item_key): _dashboard_safe_value(item_value, key=str(item_key), depth=depth + 1)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list | tuple):
        items = list(value)
        converted = [_dashboard_safe_value(item, key=key, depth=depth + 1) for item in items[:200]]
        if len(items) > 200:
            converted.append(f"<{len(items) - 200} more items>")
        return converted
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": list(shape), "dtype": str(dtype)}
    rendered = str(value)
    return rendered if len(rendered) <= 2_000 else rendered[:2_000] + "... <truncated>"


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
        processor = self.areno.get_processor()
        configure_chat_template_enable_thinking(tokenizer, getattr(self.config, "chat_template_enable_thinking", None))
        configure_chat_template_enable_thinking(processor, getattr(self.config, "chat_template_enable_thinking", None))
        sampling_params = areno.api.SamplingParams(
            greedy=self.config.greedy,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            max_context_len=getattr(self.config, "max_context_len", None),
            max_prompt_len=self.config.max_prompt_tokens,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )

        step = 0
        for epoch in range(self.config.epochs):
            self.logger.info("epoch=%d stage=epoch_start", epoch)
            record_dashboard_state(
                self.areno, stage="epoch_start", epoch=epoch, step=step, role=self._policy_role_name()
            )
            for prompt_batch in self.areno.load_prompt_batches(
                self.dataset,
                batch_size=self.config.batch_size,
                max_prompt_tokens=self.config.max_prompt_tokens,
            ):
                role = self._policy_role_name()
                self.logger.info("epoch=%d step=%d role=%s stage=rollout_start", epoch, step, role)
                record_dashboard_state(self.areno, stage="rollout_start", epoch=epoch, step=step, role=role)
                self._dashboard_epoch = epoch
                self._dashboard_step = step
                if self._agentic_enabled():
                    agent_batch = asyncio.run(self._run_agentic_rollout(sampling_params, prompt_batch))
                    self.logger.info("epoch=%d step=%d role=%s stage=rollout_end", epoch, step, role)
                    record_dashboard_state(self.areno, stage="rollout_end", epoch=epoch, step=step, role=role)
                    self._log_agentic_sample_completions(epoch, step, agent_batch)
                    train_batch, rewards_all, rollout_logprobs = self._materialize_agentic_train_batch(
                        tokenizer, prompt_batch, agent_batch
                    )
                else:
                    # 1) Sample n_samples completions per prompt; ordering
                    #    matches `prompt_batch.items` so we can zip downstream.
                    rollout_results = asyncio.run(self._run_prompt_rollout(sampling_params, prompt_batch))
                    self.logger.info("epoch=%d step=%d role=%s stage=rollout_end", epoch, step, role)
                    record_dashboard_state(self.areno, stage="rollout_end", epoch=epoch, step=step, role=role)
                    self._record_sample_completions(tokenizer, epoch, step, prompt_batch, rollout_results)

                    # 2+3) Score rewards and broadcast group-normalised
                    #      advantages down to per-token tensors.
                    train_batch, rewards_all, rollout_logprobs = self._materialize_train_batch(
                        tokenizer, prompt_batch, rollout_results
                    )

                if rewards_all:
                    self.logger.info(
                        "epoch=%d step=%d metric=reward_mean value=%.6f", epoch, step, float(np.mean(rewards_all))
                    )
                rollout_logprob_mean = _rollout_logprob_mean(rollout_logprobs)
                if rollout_logprob_mean is not None:
                    self.logger.info(
                        "epoch=%d step=%d metric=rollout_logprob_mean value=%.6f",
                        epoch,
                        step,
                        rollout_logprob_mean,
                    )

                if train_batch:
                    # PPO uses this hook to skip actor updates during the
                    # critic-only warmup window; GSPO/GRPO always train.
                    if not self._should_train_policy(step):
                        result = self._augment_train_stats({"actor_train_skipped": 1.0})
                        self.logger.info("epoch=%d step=%d role=%s stage=train_skip", epoch, step, role)
                        record_dashboard_state(self.areno, stage="train_skip", epoch=epoch, step=step, role=role)
                        self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                        self.areno.finish_step()
                        step += 1
                        if self.config.max_steps is not None and step >= self.config.max_steps:
                            self.logger.info("epoch=%d step=%d stage=max_steps_reached", epoch, step)
                            record_dashboard_state(
                                self.areno, stage="max_steps_reached", epoch=epoch, step=step, role=role
                            )
                            return
                        continue
                    self.logger.info("epoch=%d step=%d role=%s stage=train_start", epoch, step, role)
                    record_dashboard_state(self.areno, stage="train_start", epoch=epoch, step=step, role=role)
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
                    record_dashboard_state(self.areno, stage="train_end", epoch=epoch, step=step, role=role)
                    self.logger.info("epoch=%d step=%d train_stats=%s", epoch, step, result)
                    self._maybe_save(epoch, step)
                step += 1
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    self.logger.info("epoch=%d step=%d stage=max_steps_reached", epoch, step)
                    record_dashboard_state(self.areno, stage="max_steps_reached", epoch=epoch, step=step, role=role)
                    return
            self.logger.info("epoch=%d stage=epoch_end", epoch)
            record_dashboard_state(self.areno, stage="epoch_end", epoch=epoch, step=step, role=self._policy_role_name())

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
            prompt_features = [item.record.get("features") for item in prompt_batch.items]
            if not any(feature is not None for feature in prompt_features):
                prompt_features = None
            return await self.areno.rollout_token_batch_async(
                prompt_tokens,
                self.config.n_samples,
                sampling_params,
                prompt_features=prompt_features,
            )

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
            agent_filtered_count = self._agent_trajectory_invalid_count(trajectories)
            samples = []
            turn_samples_by_item = {}
            proxy_filtered_items = set()
            for turn in self._agent_trajectory_turns(ctx, trajectories):
                item_key = (turn.item.prompt_index, turn.item.sample_index)
                if getattr(turn, "filtered", False):
                    # The proxy returns a length response without running the
                    # model when a later agent request exceeds the context
                    # limit. Keep any earlier complete turns for this item;
                    # there are no tokens, logprobs, or routes to append.
                    proxy_filtered_items.add(item_key)
                    continue
                sample = ctx._sample_from_trajectory_turn(turn)
                turn_samples_by_item.setdefault(item_key, []).append(sample)
                existing = self._find_agent_sample(samples, sample.item)
                if existing is None:
                    # Rewarding needs one aggregate trajectory, while training
                    # must preserve each model call's exact causal context.
                    # Keep the aggregate independent from the per-turn row.
                    samples.append(copy.deepcopy(sample))
                else:
                    ctx._append_sample_response(existing, sample)
            sampled_items = {(sample.item.prompt_index, sample.item.sample_index) for sample in samples}
            filtered_without_sample = proxy_filtered_items - sampled_items
            agent_filtered_count += len(filtered_without_sample)
            if proxy_filtered_items:
                self.logger.warning(
                    "agentic rollout stopped at proxy context limit trajectories=%d without_prior_turn=%d",
                    len(proxy_filtered_items),
                    len(filtered_without_sample),
                )
            samples, filtered_count, filter_diagnostics = self._filter_overlong_agent_samples(
                ctx, samples, sampling_params, turn_samples_by_item=turn_samples_by_item
            )
            expected = len(agent_batch)
            if len(samples) + filtered_count + agent_filtered_count != expected:
                raise RuntimeError(
                    f"agent rollout produced {len(samples)} trajectories, filtered {filtered_count} overlong and "
                    f"{agent_filtered_count} invalid, expected {expected}"
                )
            if not samples:
                raise RuntimeError(
                    f"all agent trajectories were filtered ({filtered_count} overlong, "
                    f"{agent_filtered_count} invalid); "
                    f"{self._format_agent_filter_diagnostics(filter_diagnostics)}"
                )
            if agent_filtered_count:
                self.logger.warning(
                    "agentic rollout filtered invalid samples=%d valid_samples=%d",
                    agent_filtered_count,
                    len(samples),
                )
            reward_records = [ctx.reward_record(sample) for sample in samples]
            rewards = self._score_reward_records(reward_records)
            train_samples = []
            row_reward_indices = []
            for reward_index, sample in enumerate(samples):
                item_key = (sample.item.prompt_index, sample.item.sample_index)
                item_turns = turn_samples_by_item.get(item_key)
                if not item_turns:
                    raise RuntimeError("agentic trajectory has no trainable model turns")
                train_samples.extend(item_turns)
                row_reward_indices.extend([reward_index] * len(item_turns))
            rows = ctx._train_rows_from_samples(train_samples)
            tool_call_count = sum(len(record.tool_calls) for record in reward_records)
            tool_result_count = sum(len(record.tool_results) for record in reward_records)
            message_count = sum(len(record.messages) for record in reward_records)
            self.logger.info(
                "agentic train batch built samples=%d train_rows=%d tokens=%d messages=%d "
                "tool_calls=%d tool_results=%d",
                len(samples),
                len(train_samples),
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
                features=rows.features,
                rewards=rewards,
                records=[sample.item.record for sample in samples],
                reward_records=reward_records,
                routed_experts=rows.routed_experts,
                row_reward_indices=row_reward_indices,
            )

    def _filter_overlong_agent_samples(self, ctx, samples, sampling_params, *, turn_samples_by_item=None):
        del sampling_params
        max_context_len = self._agent_model_context_len()
        if max_context_len is None:
            return samples, 0, {}
        kept = []
        filtered_details = []
        all_details = []
        for sample in samples:
            item_key = (sample.item.prompt_index, sample.item.sample_index)
            train_samples = (turn_samples_by_item or {}).get(item_key, [sample])
            rows = ctx._train_rows_from_samples(train_samples)
            token_len = max((len(row) for row in rows.token_rows), default=0)
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
        limits = []
        config = getattr(self, "config", None)
        config_limit = getattr(config, "max_context_len", None)
        if config_limit is not None:
            limits.append(int(config_limit))
        try:
            value = self.areno.model_context_len()
        except (AttributeError, RuntimeError):
            value = None
        if value is not None:
            limits.append(int(value))
        if not limits:
            return None
        return min(limits)

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

    def _agent_trajectory_invalid_count(self, trajectories):
        from areno.api.agentic import AgentTrajectory

        if isinstance(trajectories, AgentTrajectory):
            return len(trajectories.invalid_items)
        if isinstance(trajectories, list | tuple):
            return sum(
                len(trajectory.invalid_items) for trajectory in trajectories if isinstance(trajectory, AgentTrajectory)
            )
        return 0

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
        row_reward_indices = getattr(agent_batch, "row_reward_indices", None)
        if row_reward_indices is None:
            row_reward_indices = list(range(len(agent_batch.token_rows)))
        if len(row_reward_indices) != len(agent_batch.token_rows):
            raise ValueError("agentic row_reward_indices must align with training rows")
        if any(index < 0 or index >= len(rewards_all) for index in row_reward_indices):
            raise ValueError("agentic row_reward_indices contains an invalid reward index")
        if len(agent_batch.reward_records) != len(rewards_all):
            raise ValueError("agentic rewards must align with trajectory reward records")
        logprob_stats = _LogprobStats()
        grouped: dict[int, list[int]] = {}
        for row_idx, record in enumerate(agent_batch.reward_records):
            prompt_index = int(record.metadata.get("prompt_index", row_idx))
            grouped.setdefault(prompt_index, []).append(row_idx)
        advantages_by_reward: dict[int, float] = {}
        for row_indices in grouped.values():
            group_rewards = [rewards_all[row_idx] for row_idx in row_indices]
            for row_idx, advantage in zip(row_indices, compute_group_advantages(group_rewards), strict=True):
                advantages_by_reward[row_idx] = float(advantage)
        row_features = getattr(agent_batch, "features", [None] * len(agent_batch.token_rows))
        row_routes = getattr(agent_batch, "routed_experts", None) or [None] * len(agent_batch.token_rows)
        for row_idx, (tokens, response_mask, loss_mask, logprobs, features, routed_experts) in enumerate(
            zip(
                agent_batch.token_rows,
                agent_batch.response_masks,
                agent_batch.loss_masks,
                agent_batch.rollout_logprobs,
                row_features,
                row_routes,
                strict=True,
            )
        ):
            if len(tokens) != len(response_mask) or len(tokens) != len(loss_mask) or len(tokens) != len(logprobs):
                raise ValueError("agentic train batch has misaligned token/mask/logprob rows")
            prompt_len = _agentic_prompt_len(response_mask)
            reward_index = row_reward_indices[row_idx]
            reward = rewards_all[reward_index]
            advantage = advantages_by_reward.get(reward_index, 0.0)
            effective_loss_mask = loss_mask if any(not item for item in loss_mask[prompt_len:]) else []
            if effective_loss_mask:
                logprob_stats.add(lp for lp, is_loss in zip(logprobs, effective_loss_mask, strict=True) if is_loss)
            else:
                logprob_stats.add(islice(logprobs, prompt_len, None))
            train_batch.append(
                areno.api.TrainSequence.model_construct(
                    prompt_mask=[],
                    loss_mask=effective_loss_mask,
                    tokens=tokens,
                    logprobs=logprobs,
                    advantages=[],
                    prompt_len=prompt_len,
                    scalar_advantage=advantage,
                    features=features,
                    reward=float(reward),
                    eos_token_id=tokenizer.eos_token_id,
                    returns=[],
                    values=[],
                    ref_logprobs=[],
                    routed_experts=routed_experts,
                )
            )
        return train_batch, rewards_all, logprob_stats

    def _record_sample_completions(self, tokenizer, epoch: int, step: int, prompt_batch, rollout_results) -> None:
        # Diagnostics knob: setting ARENO_LOG_COMPLETIONS=N records up to N
        # decoded completions per step in the metrics directory.
        limit = int(os.getenv("ARENO_LOG_COMPLETIONS", "0"))
        if limit <= 0:
            return
        logged = 0
        for prompt_idx, (item, result) in enumerate(zip(prompt_batch.items, rollout_results, strict=True)):
            for sample_idx, seq in enumerate(result.sequences):
                self._emit_completion_sample(
                    {
                        "kind": "rollout",
                        "epoch": epoch,
                        "step": step,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "prompt": item.prompt,
                        "decoded_prompt": tokenizer.decode(item.input_tokens),
                        "completion": tokenizer.decode(seq.resp_tokens),
                        "source_record": _dashboard_safe_value(item.record),
                        "prompt_tokens": item.input_tokens[:64],
                        "response_tokens": seq.resp_tokens[:64],
                    }
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
            prompt_messages = (
                record.messages[:-1]
                if record.messages and record.messages[-1].get("role") == "assistant"
                else record.messages
            )
            self._emit_completion_sample(
                {
                    "kind": "agentic",
                    "epoch": epoch,
                    "step": step,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": record.prompt,
                    "prompt_messages": _dashboard_safe_value(prompt_messages),
                    "messages": _dashboard_safe_value(record.messages),
                    "source_record": _dashboard_safe_value(record.source_record),
                    "completion": record.completion,
                    "rendered_completion": record.rendered_completion,
                    "final_answer": record.final_answer,
                    "tool_calls": _dashboard_safe_value(record.tool_calls),
                    "tool_results": _dashboard_safe_value(record.tool_results[:4]),
                    "loss_mask_true": sum(1 for enabled in loss_mask if enabled),
                    "loss_mask_total": len(loss_mask),
                    "first_loss_idx": first_loss_idx,
                    "loss_mask": loss_mask[:64],
                    "tokens": token_row[:64],
                }
            )
            if logged + 1 >= limit:
                return

    def _emit_completion_sample(self, sample: dict) -> None:
        """Log an opted-in completion and persist it with rollout metrics."""

        self.logger.info("rollout_completion=%s", json.dumps(sample, ensure_ascii=False, default=str))
        self.areno.record_rollout_sample(sample)

    def _materialize_train_batch(self, tokenizer, prompt_batch, rollout_results):
        """Assemble TrainSequence rows for one rollout batch.

        Steps:
            1. Decode each completion and score it with `reward_fn`.
            2. Standardise rewards within each prompt group to get advantages
               (`compute_batch_group_advantages`); this is the GRPO/GSPO baseline.
            3. Stitch each prompt prefix with its response tokens and copy the
               group-level advantage onto every response position; prompt
               positions carry zero advantage and zero logprob.

        The assembly stays on the CPU side (the main loop remains CPU-only) but
        avoids per-token Python churn: completions are decoded with one batched
        tokenizer call, per-sample prefix/response lists are built once and
        reused for both the reward record and the ``TrainSequence``, advantages
        are computed in one vectorized pass over the batch, rollout-logprob
        statistics are amortized instead of materializing one float per
        response token, and the structured prompt/advantage layout
        (``prompt_len`` + ``scalar_advantage``) lets the pack helpers derive
        the prompt mask and per-token advantage tensors with vectorized fast
        paths instead of per-token lists.
        """

        import areno.api
        from areno.api.advantages import compute_batch_group_advantages
        from areno.api.rewards import make_reward_record

        # One batched decode for the whole rollout batch instead of one
        # `tokenizer.decode` round trip per completion.
        all_resp_tokens: list[list[int]] = []
        for item, result in zip(prompt_batch.items, rollout_results, strict=True):
            all_resp_tokens.extend(seq.resp_tokens for seq in result.sequences)
        completions = _batch_decode(tokenizer, all_resp_tokens)
        completion_iter = iter(completions)

        # Pass 1: build and score reward records per prompt, collecting the
        # per-sample rows and rewards in prompt-group order.
        pending: list[tuple[Any, Any, list[int], list[float], float]] = []
        group_sizes: list[int] = []
        all_rewards: list[float] = []
        for item_idx, (item, result) in enumerate(zip(prompt_batch.items, rollout_results, strict=True)):
            prefix_len = len(item.input_tokens)
            reward_records = []
            sample_rows = []  # (seq, tokens, logprobs, loss_mask)
            for sample_idx, seq in enumerate(result.sequences):
                completion = next(completion_iter)
                # Build each prefix/response list once and reuse it for the
                # reward record and the TrainSequence below.
                tokens = item.input_tokens + seq.resp_tokens
                logprobs = [0.0] * prefix_len + seq.resp_logprobs
                loss_mask = [False] * prefix_len + [True] * len(seq.resp_tokens)
                reward_records.append(
                    make_reward_record(
                        prompt=item.prompt,
                        completion=completion,
                        source_record=item.record,
                        answer=item.solutions,
                        tokens=tokens,
                        logprobs=logprobs,
                        loss_mask=loss_mask,
                        metadata={"prompt_index": item_idx, "sample_index": sample_idx},
                    )
                )
                sample_rows.append((seq, tokens, logprobs, loss_mask))
            rewards = self._score_reward_records(reward_records)
            # Skip degenerate empty groups so advantage alignment stays intact.
            if rewards:
                group_sizes.append(len(rewards))
                all_rewards.extend(rewards)
            pending.extend(
                (item, seq, tokens, logprobs, reward)
                for (seq, tokens, logprobs, loss_mask), reward in zip(sample_rows, rewards, strict=True)
            )

        # Group-relative advantage: A_i = (r_i - mean(r))/std(r); shared by
        # every response token of sample i. One vectorized pass over the whole
        # batch using the prompt-group boundaries, instead of one numpy call
        # per prompt group.
        advantages = compute_batch_group_advantages(all_rewards, group_sizes)

        train_batch = []
        logprob_stats = _LogprobStats()
        for (item, seq, tokens, logprobs, reward), advantage in zip(pending, advantages, strict=True):
            prefix_len = len(item.input_tokens)
            logprob_stats.add(seq.resp_logprobs)
            train_batch.append(
                areno.api.TrainSequence(
                    tokens=tokens,
                    # Rollout logprobs play the role of "old logprobs"; the
                    # zero prefix keeps tensor lengths aligned with tokens.
                    logprobs=logprobs,
                    # Structured prompt/advantage layout: the pack helpers
                    # derive the prompt mask and per-token advantage tensors
                    # from these scalars with vectorized fast paths, avoiding
                    # per-token list construction for every response position.
                    prompt_len=prefix_len,
                    scalar_advantage=float(advantage),
                    features=item.record.get("features"),
                    reward=reward,
                    eos_token_id=tokenizer.eos_token_id,
                    routed_experts=seq.routed_experts,
                )
            )
        return train_batch, all_rewards, logprob_stats

    def _score_reward_records(self, records):
        from areno.api.rewards import score_reward_records

        return score_reward_records(self.reward_fn, records)

    def _maybe_save(self, epoch: int, step: int) -> None:
        # Checkpoint cadence is "save_interval" steps; `step + 1` mirrors the
        # usual convention that step 99 saves at the end of the 100th update.
        if self.config.save_path is None or (step + 1) % self.config.save_interval != 0:
            return
        ckpt_path = str(Path(self.config.save_path) / f"step_{step + 1:06d}")
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_start path=%s", epoch, step, ckpt_path)
        record_dashboard_state(
            self.areno, stage="save_checkpoint_start", epoch=epoch, step=step, role=self._policy_role_name()
        )
        saved_path = self.areno.save_checkpoint(ckpt_path)
        self.logger.info("epoch=%d step=%d stage=save_checkpoint_end path=%s", epoch, step, saved_path)
        record_dashboard_state(
            self.areno, stage="save_checkpoint_end", epoch=epoch, step=step, role=self._policy_role_name()
        )


def _agentic_prompt_len(response_mask: list[bool]) -> int:
    for idx, is_response in enumerate(response_mask):
        if is_response:
            return idx
    return len(response_mask)
