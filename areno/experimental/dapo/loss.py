"""DAPO token-level clipped policy-gradient loss."""

from __future__ import annotations

from areno.api.backend.common import LOGP_METRIC_WEIGHT, TrainMetric
from areno.api.backend.cuda.losses import masked_mean, response_layout


def dapo_loss_fn(
    data_pack,
    logprobs,
    *,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
):
    """Compute DAPO's asymmetric token-level policy objective.

    Unlike the current GRPO surrogate, DAPO uses rollout logprobs as the old
    policy so that the importance-sampling ratio can move away from one across
    optimizer updates. The backend may attach response-token normalization
    metadata when multiple packs contribute to one optimizer step.
    """

    import torch

    clip_eps_low = float(clip_eps_low)
    clip_eps_high = float(clip_eps_high)
    layout = response_layout(
        data_pack,
        logprobs,
        need_old_logprobs=True,
        need_advantages=True,
        need_sequences=True,
    )

    token_log_ratio = torch.clamp(logprobs - layout.old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(token_log_ratio)
    clipped_ratio = torch.clamp(ratio, min=1.0 - clip_eps_low, max=1.0 + clip_eps_high)
    unclipped_loss = -ratio * layout.advantages
    clipped_loss = -clipped_ratio * layout.advantages
    active_clip = clipped_loss > unclipped_loss
    per_token_loss = torch.maximum(unclipped_loss, clipped_loss) * layout.response_mask

    total_response_tokens = data_pack.get("_response_token_total")
    grad_scale = data_pack.get("_response_token_grad_scale")
    if total_response_tokens is not None and grad_scale is not None:
        policy_loss = per_token_loss.sum() * float(grad_scale) / max(float(total_response_tokens), 1.0)
    else:
        policy_loss = per_token_loss.sum() / layout.valid_count

    response_mask = layout.response_mask.bool()
    valid_ratio = ratio[response_mask]
    logp_diff = layout.old_logprobs - logprobs.detach()
    lower_clip = active_clip & (ratio < 1.0 - clip_eps_low)
    upper_clip = active_clip & (ratio > 1.0 + clip_eps_high)
    stats = {
        "policy_loss": policy_loss.detach(),
        "total_loss": policy_loss.detach(),
        TrainMetric.RATIO_MEAN: valid_ratio.mean().detach() if valid_ratio.numel() else ratio.mean().detach(),
        TrainMetric.RATIO_STD: (
            valid_ratio.std().detach() if valid_ratio.numel() > 1 else torch.zeros((), device=logprobs.device)
        ),
        "pg_clipfrac": masked_mean(active_clip.to(dtype=logprobs.dtype), layout).detach(),
        "pg_clipfrac_lower": masked_mean(lower_clip.to(dtype=logprobs.dtype), layout).detach(),
        "pg_clipfrac_upper": masked_mean(upper_clip.to(dtype=logprobs.dtype), layout).detach(),
        "advantage_mean": masked_mean(layout.advantages, layout).detach(),
        "response_len": layout.response_len.mean().detach(),
        LOGP_METRIC_WEIGHT: layout.valid_count.detach(),
        TrainMetric.ROLLOUT_LOGPROBS_MEAN: masked_mean(layout.old_logprobs, layout).detach(),
        TrainMetric.TRAIN_LOGPROBS_MEAN: masked_mean(logprobs.detach(), layout).detach(),
        TrainMetric.LOGP_DIFF_MEAN: masked_mean(logp_diff, layout).detach(),
        TrainMetric.LOGP_ABS_DIFF_MEAN: masked_mean(logp_diff.abs(), layout).detach(),
    }
    return policy_loss, stats


setattr(dapo_loss_fn, "_areno_loss_reduction", "response_token_mean")
