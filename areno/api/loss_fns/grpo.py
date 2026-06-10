"""GRPO token-level clipped policy gradient loss.

GRPO is essentially per-token PPO with group-relative advantages already
baked into `advantages`. Both packed and padded branches compute the same
objective: r_t = exp(log pi_theta(a_t) - log pi_old(a_t)), and the loss is
the mean over response tokens of `-min(r_t * A_t, clip(r_t) * A_t)`.
"""

from areno.api.loss_fns.layout import masked_mean, response_layout


def grpo_loss_fn(data_pack, logprobs, *, clip_eps: float = 0.2):
    """Token-level clipped policy-gradient loss used by GRPO.

    GRPO applies PPO-style clipping to each response token and masks prompt
    tokens out of the objective. The packed and padded branches implement the
    same math for different backend batch layouts.
    """

    import torch

    clip_eps = float(clip_eps)
    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_sequences=True)

    # `logprobs - logprobs.detach()` keeps the gradient path while ratio value
    # stays at 1.0, matching the existing GRPO surrogate semantics.
    token_log_ratio = logprobs - logprobs.detach()
    ratio = torch.exp(token_log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    per_token_loss = -torch.min(ratio * layout.advantages, clipped_ratio * layout.advantages) * layout.response_mask
    policy_loss = per_token_loss.sum() / layout.valid_count

    valid_ratio = ratio[layout.response_mask.bool()]
    logp_diff = layout.old_logprobs - logprobs.detach()
    stats = {
        "policy_loss": policy_loss.detach(),
        "total_loss": policy_loss.detach(),
        "ratio_mean": valid_ratio.mean().detach() if valid_ratio.numel() else ratio.mean().detach(),
        "ratio_std": valid_ratio.std().detach() if valid_ratio.numel() > 1 else torch.zeros((), device=logprobs.device),
        "advantage_mean": masked_mean(layout.advantages, layout).detach(),
        "response_len": layout.response_len.mean().detach(),
        "rollout_logprobs_mean": masked_mean(layout.old_logprobs, layout).detach(),
        "train_logprobs_mean": masked_mean(logprobs.detach(), layout).detach(),
        "logp_diff_mean": masked_mean(logp_diff, layout).detach(),
        "logp_abs_diff_mean": masked_mean(logp_diff.abs(), layout).detach(),
    }
    return policy_loss, stats
