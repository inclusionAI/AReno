"""Off-policy-correct GRPO/GSPO losses for asynchronous rollout pipelines.

The synchronous objectives in :mod:`areno.api.backend.cuda.losses` assume
rollout and training share one policy version, so their clipped ratio is
computed against the fresh forward logprobs and collapses to one. Samples
produced by an asynchronous rollout worker can lag the trained policy by
several versions; these variants importance-weight each token (or sequence)
against the rollout logprobs carried in ``packed_logprobs`` and clip that
real ratio, matching the PPO ratio construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areno.api.backend.common import LOGP_METRIC_WEIGHT, TrainMetric
from areno.api.backend.cuda.losses import masked_mean, response_layout, sequence_sum

if TYPE_CHECKING:
    import torch


def async_grpo_loss_fn(data_pack, logprobs, *, clip_eps: float = 0.2):
    """Token-level clipped policy-gradient loss with importance correction."""

    import torch

    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_sequences=True)
    log_ratio = torch.clamp(logprobs - layout.old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    unclipped = ratio * layout.advantages
    clipped_objective = clipped * layout.advantages
    loss = (-torch.min(unclipped, clipped_objective) * layout.response_mask).sum() / layout.valid_count
    valid_ratio = ratio[layout.response_mask.bool()]
    difference = layout.old_logprobs - logprobs.detach()
    return loss, {
        "policy_loss": loss.detach(),
        "total_loss": loss.detach(),
        "pg_clipfrac": masked_mean((clipped_objective < unclipped).to(dtype=torch.float32), layout).detach(),
        TrainMetric.RATIO_MEAN: valid_ratio.mean().detach() if valid_ratio.numel() else ratio.mean().detach(),
        TrainMetric.RATIO_STD: valid_ratio.std().detach()
        if valid_ratio.numel() > 1
        else torch.zeros((), device=logprobs.device),
        "advantage_mean": masked_mean(layout.advantages, layout).detach(),
        "response_len": layout.response_len.mean().detach(),
        LOGP_METRIC_WEIGHT: layout.valid_count.detach(),
        TrainMetric.ROLLOUT_LOGPROBS_MEAN: masked_mean(layout.old_logprobs, layout).detach(),
        TrainMetric.TRAIN_LOGPROBS_MEAN: masked_mean(logprobs.detach(), layout).detach(),
        TrainMetric.LOGP_DIFF_MEAN: masked_mean(difference, layout).detach(),
        TrainMetric.LOGP_ABS_DIFF_MEAN: masked_mean(difference.abs(), layout).detach(),
    }


def async_gspo_loss_fn(data_pack, logprobs, *, clip_eps: float = 3e-4):
    """Sequence-level clipped policy-gradient loss with importance correction."""

    import torch

    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_sequences=True)
    seq_log_ratio = torch.clamp(
        sequence_sum(logprobs - layout.old_logprobs, layout) / layout.response_len,
        min=-20.0,
        max=20.0,
    )
    seq_ratio = torch.exp(seq_log_ratio)
    clipped = torch.clamp(seq_ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    seq_advantage = sequence_sum(layout.advantages, layout) / layout.response_len
    unclipped = seq_ratio * seq_advantage
    clipped_objective = clipped * seq_advantage
    loss = -torch.min(unclipped, clipped_objective).mean()
    difference = layout.old_logprobs - logprobs.detach()
    return loss, {
        "policy_loss": loss.detach(),
        "total_loss": loss.detach(),
        "pg_clipfrac": (clipped_objective < unclipped).to(dtype=torch.float32).mean().detach(),
        TrainMetric.RATIO_MEAN: seq_ratio.mean().detach(),
        TrainMetric.RATIO_STD: seq_ratio.std(unbiased=False).detach(),
        "advantage_mean": seq_advantage.mean().detach(),
        "response_len": layout.response_len.mean().detach(),
        LOGP_METRIC_WEIGHT: layout.valid_count.detach(),
        TrainMetric.ROLLOUT_LOGPROBS_MEAN: masked_mean(layout.old_logprobs, layout).detach(),
        TrainMetric.TRAIN_LOGPROBS_MEAN: masked_mean(logprobs.detach(), layout).detach(),
        TrainMetric.LOGP_DIFF_MEAN: masked_mean(difference, layout).detach(),
        TrainMetric.LOGP_ABS_DIFF_MEAN: masked_mean(difference.abs(), layout).detach(),
    }


__all__ = ["async_grpo_loss_fn", "async_gspo_loss_fn"]
