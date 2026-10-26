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

from areno.api.backend.cuda.losses import masked_mean, response_layout


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

    # Upstream's response_layout only accepts canonical packed training data.
    # OPD also supports rectangular (B, T) batches in standalone use and tests.
    # Adapt those to the same action-axis fields without changing shared losses.
    if packed:
        layout_pack = data_pack
    else:
        prompt_mask = data_pack.get("prompt_mask")
        if not isinstance(prompt_mask, torch.Tensor) or prompt_mask.ndim != 2:
            raise ValueError("OPD padded batches require a 2D prompt_mask")
        response_mask = ~prompt_mask[:, 1:].to(dtype=torch.bool)
        loss_mask = data_pack.get("loss_mask")
        if loss_mask is not None:
            if not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != prompt_mask.shape:
                raise ValueError("OPD loss_mask shapes must match prompt_mask")
            response_mask = response_mask & loss_mask[:, 1:].to(dtype=torch.bool)
        layout_pack = {
            "packed_response_mask": response_mask,
            "packed_logprobs": data_pack["logprobs"][:, 1:],
            "packed_ref_logprobs": data_pack["ref_logprobs"][:, 1:],
        }

    layout = response_layout(
        layout_pack,
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
    # Optional backend annotations correct microbatch weighting when batches
    # contain different numbers of response tokens. By default this remains
    # the original local token-mean loss (standalone use is unchanged).
    weighted_sum = (ratio * advantage * response_mask.to(logprobs.dtype)).sum()
    total_group_tokens = data_pack.get("_opd_total_response_tokens")
    if total_group_tokens is None:
        policy_loss = -opd_coef * weighted_sum / num_tokens.to(logprobs.dtype)
    else:
        total_group_tokens = int(total_group_tokens)
        group_size = int(data_pack.get("_opd_grad_scale", 1))
        if total_group_tokens <= 0 or group_size <= 0:
            raise ValueError("OPD accumulation normalization must be positive")
        if total_group_tokens < int(num_tokens.item()):
            raise ValueError("OPD global response token count smaller than local count")
        policy_loss = (
            -opd_coef * weighted_sum * group_size / total_group_tokens
        )

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
