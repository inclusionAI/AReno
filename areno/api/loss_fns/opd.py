"""Response-token on-policy distillation objective.

The student samples a response with pi_old; a frozen teacher scores *those
same tokens*. The detached teacher-vs-rollout log-probability difference is
the sampled training signal, while the current student supplies the gradient.

For valid response positions t:

    advantage_t = stopgrad(log p_teacher(y_t|s_t) - log p_old(y_t|s_t))
    ratio_t     = exp(clamp(log p_current(y_t|s_t) - log p_old(y_t|s_t)))
    loss        = -opd_coef * mean_t(ratio_t * advantage_t)

The existing ``ref_logprobs`` tensor is used for teacher scores. Both padded
and packed action layouts are supported through the shared response_layout.
"""

from __future__ import annotations

import math

import torch

from areno.api.loss_fns.layout import masked_mean, response_layout


def opd_loss_fn(
    data_pack: dict,
    logprobs: torch.Tensor,
    *,
    opd_coef: float = 1.0,
    max_log_ratio: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the importance-weighted OPD loss and detached diagnostics.

    ``logprobs`` are the *current trainable student's* log-probabilities of
    the realized next tokens, shaped (B, T-1) for padded inputs or (N,) for
    packed actions. ``data_pack`` contains saved rollout logprobs and teacher
    scores (under the existing ``ref_logprobs`` key).

    Prompt, padding, and loss_mask-excluded positions do not participate.
    The log-ratio clamp is a numerical safeguard, not a PPO clipped objective:
    gradients at values outside the clamp range are zero. The default bound
    is fixed for this baseline, without adding another CLI option.
    """
    opd_coef = float(opd_coef)
    max_log_ratio = float(max_log_ratio)
    if not math.isfinite(opd_coef) or opd_coef <= 0:
        raise ValueError("opd_coef must be finite and positive")
    if not math.isfinite(max_log_ratio) or not 0 < max_log_ratio <= 20.0:
        raise ValueError("max_log_ratio must be finite and in (0, 20]")

    packed = "packed_response_mask" in data_pack
    old_key = "packed_logprobs" if packed else "logprobs"
    if old_key not in data_pack:
        raise ValueError(f"OPD requires saved rollout logprobs ({old_key})")
    ref_key = "packed_ref_logprobs" if packed else "ref_logprobs"
    if ref_key not in data_pack:
        raise ValueError(f"OPD requires teacher scores ({ref_key})")

    layout = response_layout(
        data_pack,
        logprobs,
        need_old_logprobs=True,
        need_ref_logprobs=True,
    )
    old = layout.old_logprobs
    teacher = layout.ref_logprobs
    assert old is not None and teacher is not None

    if any(t.shape != logprobs.shape for t in (old, teacher, layout.response_mask)):
        raise ValueError("OPD current, rollout, teacher and mask shapes must match")

    response_mask = layout.response_mask.bool()
    num_tokens = response_mask.sum()
    if not bool(num_tokens.item()):
        raise ValueError("OPD loss requires at least one unmasked response token")

    # Use the mask BEFORE computing differences so arbitrary padded values,
    # including NaN/Inf, never contaminate the scalar objective.
    current = torch.where(response_mask, logprobs, torch.zeros_like(logprobs))
    old = torch.where(response_mask, old.detach(), torch.zeros_like(old))
    teacher = torch.where(response_mask, teacher.detach(), torch.zeros_like(teacher))
    for name, values in (("student", current), ("rollout", old), ("teacher", teacher)):
        if not bool(torch.isfinite(values[response_mask]).all().item()):
            raise ValueError(f"OPD {name} contains non-finite response logprobs")

    advantage = (teacher - old).detach()
    raw_log_ratio = current - old
    safe_log_ratio = raw_log_ratio.clamp(-max_log_ratio, max_log_ratio)
    ratio = safe_log_ratio.exp()
    policy_loss = -opd_coef * masked_mean(ratio * advantage, layout)

    valid_ratio = ratio[response_mask]
    stats = {
        "opd_loss": policy_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "total_loss": policy_loss.detach(),
        "opd_response_tokens": num_tokens.detach(),
        "opd_advantage_mean": masked_mean(advantage, layout).detach(),
        "advantage_mean": masked_mean(advantage, layout).detach(),
        "ratio_mean": valid_ratio.mean().detach(),
        "ratio_std": valid_ratio.std(unbiased=False).detach(),
        "opd_ratio_clamped_fraction": masked_mean(
            ((raw_log_ratio > max_log_ratio) | (raw_log_ratio < -max_log_ratio)).to(logprobs.dtype),
            layout,
        ).detach(),
        "teacher_logprobs_mean": masked_mean(teacher, layout).detach(),
        "rollout_logprobs_mean": masked_mean(old, layout).detach(),
        "train_logprobs_mean": masked_mean(current.detach(), layout).detach(),
        "logp_diff_mean": masked_mean(old - current.detach(), layout).detach(),
    }
    return policy_loss, stats
