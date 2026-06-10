"""Full actor-critic PPO trainer.

Inherits the rollout/train scaffolding from `PolicyOnlyTrainer` and overrides
batch assembly to weave in three auxiliary backend roles:
    * ref    - frozen reference policy, supplies log-probs for the KL penalty.
    * critic - trainable value head, supplies per-token values for GAE and is
               updated once per step before the actor.
    * reward - optional reward model used when no Python reward_fn is supplied.

Per step the trainer:
    1. Scores rewards either via reward_fn or the reward role.
    2. Forwards the ref / actor / critic models over every rollout token.
    3. Aligns critic values with target tokens (predict-next semantics).
    4. Computes GAE per response, normalises advantages across the batch.
    5. Trains the critic, then runs the actor PPO update with the supplied
       loss function (typically ``ppo_loss_fn`` partially-applied with the PPO
       clipping / KL hyperparameters from config).
"""

from __future__ import annotations

import logging
import time
from functools import partial

import numpy as np

import areno.api
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.api.advantages import compute_gae
from areno.api.roles import MissingRoleCapability, ModelRole


logger = logging.getLogger(__name__)


class PPOTrainer(PolicyOnlyTrainer):
    """PPO trainer skeleton with role-owned memory choreography.

    PPO needs actor rollout, ref logprob scoring, reward scoring, critic value
    estimation, and actor/critic updates. This class keeps those roles behind
    the trainer boundary so future colocated offload/onload logic can be added
    without exposing memory APIs to algorithm scripts.
    """

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        super().__init__(config, instance=instance, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)
        # Partially apply PPO knobs so the trainer can pass `loss_fn(data_pack, logp)`
        # without re-specifying clipping/KL configuration each step.
        self.loss_fn = partial(
            loss_fn,
            clip_eps=config.clip_eps,
            clip_ratio_c=config.clip_ratio_c,
            use_kl_loss=config.use_kl_loss,
            kl_loss_coef=config.kl_loss_coef,
            kl_loss_type=config.kl_loss_type,
        )
        # Holding pen for per-step auxiliary stats (critic loss, role forward
        # times, summary stats of ref/old/value tensors). `_augment_train_stats`
        # merges it into the dict the metrics recorder receives.
        self._last_ppo_stats: dict[str, float] = {}
        # Each role can have its own checkpoint; fall back to the actor's
        # checkpoint when nothing role-specific is configured so a basic PPO
        # run with three copies of the same model just works.
        actor_ckpt = config.ckpt
        ref_ckpt = config.ref_ckpt or config.ckpt
        critic_ckpt = config.critic_ckpt or config.ckpt
        self.roles = {
            "actor": ModelRole("actor", actor_ckpt, trainable=True),
            "ref": ModelRole("ref", ref_ckpt, trainable=False),
            "critic": ModelRole("critic", critic_ckpt, trainable=True, optimizer_lr=float(config.critic_lr)),
        }
        if config.reward_ckpt:
            self.roles["reward"] = ModelRole("reward", config.reward_ckpt, trainable=False)

    def _policy_role_name(self) -> str:
        return "actor"

    def _should_train_policy(self, step: int) -> bool:
        # During warmup the actor is frozen and only the critic learns, so its
        # value estimates are usable when the actor starts updating.
        return step >= int(self.config.critic_warmup_steps)

    def _materialize_train_batch(self, tokenizer, prompt_batch, rollout_results):
        self._last_ppo_stats = {}
        train_batch = []
        rewards_all = []
        rollout_logprobs = []
        ref_logprobs_all = []
        critic_values_all = []
        returns_all = []
        advantages_all = []
        # Collect (full_token_row, metadata) for every prompt/sample pair so
        # we can issue a single batched forward to each role.
        token_rows: list[list[int]] = []
        row_meta = []
        completions_by_prompt = []
        for item, result in zip(prompt_batch.items, rollout_results, strict=True):
            prefix_len = len(item.input_tokens)
            completions = [tokenizer.decode(seq.resp_tokens) for seq in result.sequences]
            completions_by_prompt.append((item, completions))
            for seq in result.sequences:
                tokens = item.input_tokens + seq.resp_tokens
                token_rows.append(tokens)
                row_meta.append((item, seq, prefix_len, len(seq.resp_tokens)))

        # Reward scoring: either Python reward_fn or a backend-owned reward
        # role. Both produce one float per (prompt, sample) in row order.
        if self.reward_fn is not None:
            reward_start = time.perf_counter()
            reward_iter = []
            for item, completions in completions_by_prompt:
                rewards = self.reward_fn(item.record, completions)
                if len(rewards) != len(completions):
                    raise ValueError(f"reward_fn returned {len(rewards)} rewards for {len(completions)} completions")
                reward_iter.extend(float(reward) for reward in rewards)
            rewards_all = reward_iter
            self._last_ppo_stats["reward_score_time_s"] = time.perf_counter() - reward_start
        else:
            self.logger.info("role=reward stage=score_start rows=%d", len(token_rows))
            reward_start = time.perf_counter()
            raw_rewards = [float(reward) for reward in self._score_rewards(token_rows)]
            rewards_all = raw_rewards
            self._last_ppo_stats.update(_summary_stats("reward_model_raw_reward", raw_rewards))
            self._last_ppo_stats["reward_score_time_s"] = time.perf_counter() - reward_start
            self.logger.info("role=reward stage=score_end rows=%d", len(token_rows))

        # Forward ref/actor/critic over every row in a single batched call per
        # role so the backend can amortise activation memory and kernel launch
        # cost across all sequences.
        self.logger.info("role=ref stage=logprob_score_start rows=%d", len(token_rows))
        ref_start = time.perf_counter()
        ref_logprob_rows = self._score_logprobs("ref", token_rows)
        self._last_ppo_stats["ref_logprob_forward_time_s"] = time.perf_counter() - ref_start
        self.logger.info("role=ref stage=logprob_score_end rows=%d", len(token_rows))
        self.logger.info("role=actor stage=old_logprob_score_start rows=%d", len(token_rows))
        actor_logprob_start = time.perf_counter()
        # Even though we already have rollout logprobs, PPO needs an actor
        # forward pass at the same parameters used by the upcoming update to
        # form the "old logprobs" baseline in the importance ratio.
        old_logprob_rows = self._score_logprobs("actor", token_rows)
        self._last_ppo_stats["actor_old_logprob_forward_time_s"] = time.perf_counter() - actor_logprob_start
        self.logger.info("role=actor stage=old_logprob_score_end rows=%d", len(token_rows))
        self.logger.info("role=critic stage=value_score_start rows=%d", len(token_rows))
        critic_value_start = time.perf_counter()
        value_rows = self._score_values("critic", token_rows)
        self._last_ppo_stats["critic_value_forward_time_s"] = time.perf_counter() - critic_value_start
        self.logger.info("role=critic stage=value_score_end rows=%d", len(token_rows))

        old_logprobs_all = []
        logp_diff_all = []
        for (item, seq, prefix_len, resp_len), reward, ref_logprobs, old_logprobs, values in zip(
            row_meta,
            rewards_all,
            ref_logprob_rows,
            old_logprob_rows,
            value_rows,
            strict=True,
        ):
            if resp_len < 1:
                continue
            rollout_logprobs += seq.resp_logprobs
            # Slice each role output to the response window. The actor/ref are
            # per-position predictions of the corresponding token, so the
            # response window starts at `prefix_len`.
            action_old_logprobs = old_logprobs[prefix_len : prefix_len + resp_len]
            if len(action_old_logprobs) != resp_len:
                raise ValueError("actor returned misaligned old logprobs")
            old_logprobs_all.extend(action_old_logprobs)
            # Drift between the rollout policy and the actor at scoring time.
            logp_diff_all.extend(
                float(old_logprob) - float(rollout_logprob)
                for old_logprob, rollout_logprob in zip(action_old_logprobs, seq.resp_logprobs, strict=True)
            )
            # Reward shaping: only the terminal token receives the scalar
            # reward; intermediate tokens get zero. GAE will smear the credit
            # across the response via the bootstrap.
            token_rewards = [0.0 for _ in range(resp_len)]
            token_rewards[-1] = float(reward)
            # Critic outputs values for input tokens; the value used at step t
            # of the response corresponds to the model's reading position
            # `prefix_len - 1 + t` (one step ahead of the response position).
            action_values = values[prefix_len - 1 : prefix_len + resp_len - 1]
            if len(action_values) != resp_len:
                raise ValueError("critic returned misaligned token values")
            # GAE turns (token_rewards, values) -> (advantages, returns).
            advantages, returns = self._gae_for_response(token_rewards, action_values)
            full_advantages = [0.0] * prefix_len + advantages
            full_returns = [0.0] * prefix_len + returns
            full_values = [0.0] * prefix_len + action_values
            ref_logprobs_all.extend(ref_logprobs[prefix_len : prefix_len + resp_len])
            critic_values_all.extend(action_values)
            returns_all.extend(returns)
            advantages_all.extend(advantages)
            train_batch.append(
                areno.api.TrainSequence(
                    prompt_mask=[1] * prefix_len + [0] * resp_len,
                    tokens=item.input_tokens + seq.resp_tokens,
                    # Use the actor-scored old logprobs (NOT the rollout
                    # logprobs) so the PPO ratio is consistent with the
                    # gradient computed by the optimizer.
                    logprobs=[0.0] * prefix_len + action_old_logprobs,
                    advantages=full_advantages,
                    returns=full_returns,
                    values=full_values,
                    ref_logprobs=ref_logprobs,
                    reward=float(reward),
                    eos_token_id=tokenizer.eos_token_id,
                )
            )

        if train_batch:
            # Train the critic first so the value loss reflects the current
            # batch before the actor consumes the advantages.
            self.logger.info("role=critic stage=train_start rows=%d", len(train_batch))
            critic_train_start = time.perf_counter()
            critic_stats = self._train_values(train_batch)
            self._last_ppo_stats["critic_train_time_s"] = time.perf_counter() - critic_train_start
            self.logger.info("role=critic stage=train_end rows=%d", len(train_batch))
            if critic_stats:
                self._last_ppo_stats.update({key: float(value) for key, value in critic_stats.items()})
                self.logger.info("role=critic metric=value_loss value=%.6f", float(critic_stats["critic_value_loss"]))
            # Summary stats for tensorboard: mean/std/min/max of each interim
            # quantity helps detect when something (advantages, returns,
            # logprob drift) suddenly explodes.
            self._last_ppo_stats.update(_summary_stats("ref_logprob", ref_logprobs_all))
            self._last_ppo_stats.update(_summary_stats("old_logprob", old_logprobs_all))
            self._last_ppo_stats.update(_summary_stats("old_rollout_logprob_diff", logp_diff_all))
            self._last_ppo_stats.update(_summary_stats("critic_value", critic_values_all))
            self._last_ppo_stats.update(_summary_stats("return", returns_all))
            self._last_ppo_stats.update(_summary_stats("gae_advantage", advantages_all))
            # Standardise advantages across the batch (zero mean, unit std)
            # to stabilise gradients regardless of reward scale.
            _normalize_response_advantages(train_batch)
            normalized_advantages = [
                adv
                for seq in train_batch
                for adv, is_prompt in zip(seq.advantages, seq.prompt_mask, strict=True)
                if not is_prompt
            ]
            self._last_ppo_stats.update(_summary_stats("normalized_advantage", normalized_advantages))
        return train_batch, rewards_all, rollout_logprobs

    def _ensure_roles(self) -> None:
        # Surfaces a structured log per role so initialisation order is easy
        # to follow when something fails mid-load.
        for role in self.roles.values():
            self.logger.info(
                "role=%s stage=init_start trainable=%s path=%s",
                role.name,
                role.trainable,
                role.path,
            )
        self.areno.ensure_roles(self.roles)
        for role in self.roles.values():
            self.logger.info("role=%s stage=init_end trainable=%s", role.name, role.trainable)

    def fit(self) -> None:
        # Override the base `fit` so role initialisation happens after the
        # backend is up but before the first rollout/train cycle.
        self.areno.init()
        self._ensure_roles()
        try:
            self._fit_initialized()
        finally:
            self.areno.close()

    def _score_logprobs(self, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        return self.areno.score_logprobs(role, token_rows)

    def _score_values(self, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        return self.areno.score_values(role, token_rows)

    def _score_rewards(self, token_rows: list[list[int]]) -> list[float]:
        return self.areno.score_rewards("reward", token_rows)

    def _train_values(self, train_batch) -> dict[str, float]:
        return self.areno.train_values(
            "critic",
            train_batch,
            self.config.mini_bs,
            self.config.gradient_accumulation_steps,
            cliprange_value=self.config.value_clip_eps,
            value_loss_coef=self.config.value_loss_coef,
        )

    def _augment_train_stats(self, result):
        # Merge the per-step PPO diagnostics into the dict the metric recorder
        # consumes so they show up alongside the actor loss.
        if isinstance(result, dict):
            result.update(self._last_ppo_stats)
        return result

    def _gae_for_response(self, token_rewards: list[float], values: list[float]) -> tuple[list[float], list[float]]:
        # Per-sequence GAE: with gamma=1 and zero intermediate rewards, the
        # tail advantage propagates the terminal reward back through the
        # value baseline.
        advantages, returns = compute_gae(token_rewards, values, gamma=self.config.gamma, lam=self.config.lam)
        return _float_list(advantages), _float_list(returns)


def _pad_token_id(tokenizer) -> int:
    value = tokenizer.pad_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(value)


def _float_list(values) -> list[float]:
    # Accept torch tensors as well as plain iterables; the cugae path may
    # return GPU tensors that need to be moved back to the host first.
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().tolist()
    return [float(value) for value in values]


def _summary_stats(name: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{name}_mean": float(arr.mean()),
        f"{name}_std": float(arr.std()),
        f"{name}_min": float(arr.min()),
        f"{name}_max": float(arr.max()),
    }


def _normalize_response_advantages(train_batch) -> None:
    """In-place batch-wide advantage standardisation (mean=0, std=1).

    Only response tokens contribute to the statistics (prompt advantages are
    zero by construction). The rewrite mutates each `TrainSequence.advantages`
    in place so the backend sees the normalised tensor without an extra copy.
    """

    values = []
    for seq in train_batch:
        values.extend(adv for adv, is_prompt in zip(seq.advantages, seq.prompt_mask, strict=True) if not is_prompt)
    if not values:
        return
    mean = float(np.mean(values))
    std = float(np.std(values))
    # Guard against degenerate batches where every advantage equals the mean.
    scale = std if std > 1e-6 else 1.0
    for seq in train_batch:
        seq.advantages = [
            0.0 if is_prompt else (adv - mean) / scale
            for adv, is_prompt in zip(seq.advantages, seq.prompt_mask, strict=True)
        ]


__all__ = ["PPOTrainer", "MissingRoleCapability"]
