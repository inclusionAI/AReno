"""Native MLX implementations of AReno policy losses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from areno.api.backend.common import TrainMetric

MlxLoss = Callable[[dict[str, Any], Any], tuple[Any, dict[str, Any]]]


def mlx_loss(name: str, batch: dict[str, Any], logprobs: Any, **kwargs: object):
    """Dispatch a registered algorithm loss without importing Torch formulas."""

    try:
        fn = _LOSSES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_LOSSES))
        raise NotImplementedError(f"MLX backend does not implement loss {name!r}; supported: {supported}") from exc
    return fn(batch, logprobs, **kwargs)


def sft_loss(batch, logprobs, **_):
    import mlx.core as mx

    mask = batch["response_mask"]
    count = mx.maximum(mask.sum(), mx.array(1.0))
    logprob_sum = (logprobs * mask).sum()
    total_target_tokens = batch.get("_sft_total_target_tokens")
    if total_target_tokens is None:
        loss = -logprob_sum / count
    else:
        denominator = mx.maximum(total_target_tokens, mx.array(1.0))
        loss = -(logprob_sum / denominator) * batch.get("_sft_grad_scale", mx.array(1.0))
    return loss, {
        "sft_loss": loss,
        "sft_target_tokens": count,
        "sft_logprob_mean": -loss,
    }


def grpo_loss(batch, logprobs, *, clip_eps: float = 0.2, **_):
    import mlx.core as mx

    mask = batch["response_mask"]
    count = mx.maximum(mask.sum(), mx.array(1.0))
    advantages = batch["advantages"]
    ratio = mx.exp(logprobs - mx.stop_gradient(logprobs))
    clipped = mx.clip(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    loss = (-mx.minimum(ratio * advantages, clipped * advantages) * mask).sum() / count
    return loss, _policy_stats(batch, logprobs, ratio, loss)


def gspo_loss(batch, logprobs, *, clip_eps: float = 3e-4, **_):
    import mlx.core as mx

    mask = batch["response_mask"]
    lengths = mx.maximum(mask.sum(axis=-1), mx.array(1.0))
    seq_log_ratio = ((logprobs - mx.stop_gradient(logprobs)) * mask).sum(axis=-1) / lengths
    ratio = mx.exp(seq_log_ratio)
    clipped = mx.clip(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    seq_advantage = (batch["advantages"] * mask).sum(axis=-1) / lengths
    loss = -mx.minimum(ratio * seq_advantage, clipped * seq_advantage).mean()
    stats = _policy_stats(batch, logprobs, mx.ones_like(logprobs), loss)
    stats[TrainMetric.RATIO_MEAN] = ratio.mean()
    stats[TrainMetric.RATIO_STD] = mx.sqrt(mx.maximum(((ratio - ratio.mean()) ** 2).mean(), mx.array(0.0)))
    stats["advantage_mean"] = seq_advantage.mean()
    return loss, stats


def dpo_loss(batch, logprobs, *, beta: float = 0.1, label_smoothing: float = 0.0, **_):
    import mlx.core as mx

    if int(logprobs.shape[0]) % 2:
        raise ValueError("DPO requires an even number of sequences per microbatch")
    mask = batch["response_mask"]
    policy = (logprobs * mask).sum(axis=-1)
    reference = (batch["ref_logprobs"] * mask).sum(axis=-1)
    logits = float(beta) * ((policy[0::2] - policy[1::2]) - (reference[0::2] - reference[1::2]))
    smoothing = float(label_smoothing)
    losses = -(1.0 - smoothing) * mx.logaddexp(mx.array(0.0), -logits) - smoothing * mx.logaddexp(mx.array(0.0), logits)
    loss = -losses.mean()
    chosen_rewards = float(beta) * (policy[0::2] - reference[0::2])
    rejected_rewards = float(beta) * (policy[1::2] - reference[1::2])
    return loss, {
        "dpo_loss": loss,
        "total_loss": loss,
        "dpo_accuracy": (logits > 0).astype(mx.float32).mean(),
        "dpo_margin": logits.mean(),
        "dpo_reward_margin": (chosen_rewards - rejected_rewards).mean(),
        "dpo_chosen_reward": chosen_rewards.mean(),
        "dpo_rejected_reward": rejected_rewards.mean(),
        "dpo_response_len": mx.maximum(mask.sum(axis=-1), mx.array(1.0)).mean(),
    }


def ppo_loss(
    batch,
    logprobs,
    *,
    clip_eps: float = 0.2,
    clip_ratio_c: float = 3.0,
    use_kl_loss: bool = False,
    kl_loss_coef: float = 0.001,
    kl_loss_type: str = "low_var_kl",
    **_,
):
    import mlx.core as mx

    mask = batch["response_mask"]
    count = mx.maximum(mask.sum(), mx.array(1.0))
    old = batch["old_logprobs"]
    advantages = batch["advantages"]
    ref = batch.get("ref_logprobs", old)
    log_ratio = mx.clip(logprobs - old, -20.0, 20.0)
    ratio = mx.exp(log_ratio)
    clipped_ratio = mx.clip(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    losses1 = -ratio * advantages
    losses2 = -clipped_ratio * advantages
    clipped_losses = mx.maximum(losses1, losses2)
    dual_losses = -advantages * float(clip_ratio_c)
    policy_losses = mx.where(advantages < 0.0, mx.minimum(dual_losses, clipped_losses), clipped_losses)

    def masked_mean(value):
        return (value * mask).sum() / count

    policy_loss = masked_mean(policy_losses)
    kl = masked_mean(_kl_penalty(logprobs, ref, kl_loss_type))
    total_loss = policy_loss + (float(kl_loss_coef) * kl if use_kl_loss else 0.0)
    valid_ratio_mean = masked_mean(ratio)
    ratio_variance = masked_mean((ratio - valid_ratio_mean) ** 2)
    return total_loss, {
        "policy_loss": policy_loss,
        "kl_loss": kl,
        "kl_coef": mx.array(float(kl_loss_coef) if use_kl_loss else 0.0),
        "pg_clipfrac": masked_mean((losses2 > losses1).astype(mx.float32)),
        "pg_clipfrac_lower": masked_mean(((clipped_losses > dual_losses) & (advantages < 0.0)).astype(mx.float32)),
        "ppo_kl": masked_mean(-log_ratio),
        "total_loss": total_loss,
        TrainMetric.RATIO_MEAN: valid_ratio_mean,
        TrainMetric.RATIO_STD: mx.sqrt(mx.maximum(ratio_variance, mx.array(0.0))),
        "advantage_mean": masked_mean(advantages),
    }


def _kl_penalty(logprobs, ref_logprobs, kind: str):
    import mlx.core as mx

    difference = logprobs - ref_logprobs
    if kind in {"kl", "k1"}:
        return difference
    if kind == "abs":
        return mx.abs(difference)
    if kind in {"mse", "k2"}:
        return 0.5 * difference**2
    if kind in {"low_var_kl", "k3"}:
        reverse = mx.clip(-difference, -20.0, 20.0)
        return mx.clip(mx.exp(reverse) - reverse - 1.0, -10.0, 10.0)
    raise NotImplementedError(f"unsupported PPO kl_loss_type: {kind}")


def _policy_stats(batch, logprobs, ratio, loss):
    import mlx.core as mx

    mask = batch["response_mask"]
    count = mx.maximum(mask.sum(), mx.array(1.0))
    old = batch["old_logprobs"]
    diff = old - mx.stop_gradient(logprobs)

    def masked_mean(value):
        return (value * mask).sum() / count

    return {
        "policy_loss": loss,
        "total_loss": loss,
        TrainMetric.RATIO_MEAN: masked_mean(ratio),
        TrainMetric.RATIO_STD: mx.sqrt(mx.maximum(masked_mean((ratio - masked_mean(ratio)) ** 2), mx.array(0.0))),
        "advantage_mean": masked_mean(batch["advantages"]),
        "response_len": mx.maximum(mask.sum(axis=-1), mx.array(1.0)).mean(),
        TrainMetric.ROLLOUT_LOGPROBS_MEAN: masked_mean(old),
        TrainMetric.TRAIN_LOGPROBS_MEAN: masked_mean(mx.stop_gradient(logprobs)),
        TrainMetric.LOGP_DIFF_MEAN: masked_mean(diff),
        TrainMetric.LOGP_ABS_DIFF_MEAN: masked_mean(mx.abs(diff)),
    }


_LOSSES: dict[str, MlxLoss] = {
    "sft": sft_loss,
    "dpo": dpo_loss,
    "grpo": grpo_loss,
    "gspo": gspo_loss,
    "ppo": ppo_loss,
}

__all__ = ["mlx_loss"]
