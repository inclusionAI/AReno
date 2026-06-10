"""PPO actor loss aligned with verl's vanilla implementation.

Compared to the GSPO/GRPO losses, this variant:
    * uses real old_logprobs supplied by the trainer (not the surrogate
      `logprobs - logprobs.detach()`), so the ratio measures the divergence
      between the actor at the start of the step and the actor mid-update;
    * adds a "dual clip" lower bound `-A * clip_ratio_c` for negative
      advantages (Ye et al. 2020) which keeps the loss from running away when
      a very large negative advantage meets a tiny ratio;
    * optionally mixes a KL-to-reference penalty, with several KL estimator
      flavours implemented in `_kl_penalty`.
"""

from areno.api.loss_fns.layout import masked_mean, response_layout


def ppo_loss_fn(
    data_pack,
    logprobs,
    *,
    clip_eps: float = 0.2,
    clip_ratio_c: float = 3.0,
    use_kl_loss: bool = False,
    kl_loss_coef: float = 0.001,
    kl_loss_type: str = "low_var_kl",
):
    """PPO clipped actor loss aligned with verl's vanilla policy loss."""

    import torch

    clip_eps = float(clip_eps)
    clip_ratio_c = float(clip_ratio_c)
    kl_loss_coef = float(kl_loss_coef)

    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_ref_logprobs=True)
    # When no reference is provided fall back to old logprobs so the KL term
    # becomes zero in expectation and the same code path can be used.
    ref_values = layout.old_logprobs if layout.ref_logprobs is None else layout.ref_logprobs

    # `negative_approx_kl = log pi_theta(a) - log pi_old(a)` is also the log
    # importance ratio; clamping prevents inf when ratios blow up.
    negative_approx_kl = torch.clamp(logprobs - layout.old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    # Standard PPO clipped surrogate; the objective takes the worse loss once
    # the ratio leaves the trust region.
    pg_losses1 = -ratio * layout.advantages
    pg_losses2 = -clipped_ratio * layout.advantages
    clip_pg_losses = torch.maximum(pg_losses1, pg_losses2)
    # Dual clipping lower-bounds negative-advantage losses.
    pg_losses3 = -layout.advantages * clip_ratio_c
    pg_losses = torch.where(layout.advantages < 0.0, torch.minimum(pg_losses3, clip_pg_losses), clip_pg_losses)
    policy_loss = masked_mean(pg_losses, layout)
    kl = masked_mean(_kl_penalty(logprobs, ref_values, kl_loss_type), layout)
    total_loss = policy_loss + (kl_loss_coef * kl if use_kl_loss else 0.0)
    ppo_kl = masked_mean(-negative_approx_kl, layout)
    pg_clipfrac = masked_mean((pg_losses2 > pg_losses1).float(), layout)
    pg_clipfrac_lower = masked_mean(((clip_pg_losses > pg_losses3) & (layout.advantages < 0.0)).float(), layout)

    valid_ratio = ratio[layout.response_mask.bool()]
    return total_loss, {
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl.detach(),
        "kl_coef": torch.tensor(kl_loss_coef if use_kl_loss else 0.0, device=logprobs.device),
        "pg_clipfrac": pg_clipfrac.detach(),
        "pg_clipfrac_lower": pg_clipfrac_lower.detach(),
        "ppo_kl": ppo_kl.detach(),
        "total_loss": total_loss.detach(),
        "ratio_mean": valid_ratio.mean().detach() if valid_ratio.numel() else ratio.mean().detach(),
        "ratio_std": valid_ratio.std().detach() if valid_ratio.numel() > 1 else torch.zeros((), device=logprobs.device),
        "advantage_mean": masked_mean(layout.advantages, layout).detach(),
    }


def _kl_penalty(logprob, ref_logprob, kl_type: str):
    """Token-wise KL approximation between policy and reference distributions.

    Implemented estimators (using d = log pi_theta - log pi_ref):
        kl/k1        -> d                              (unbiased, high variance)
        abs          -> |d|                            (sometimes used for safety)
        mse/k2       -> 0.5 * d^2                      (Gaussian-like proxy)
        low_var_kl/k3 -> exp(-d) - (-d) - 1           (Schulman's K3 estimator,
                                                      unbiased, low variance,
                                                      always >= 0)
    """

    import torch

    if kl_type in ("kl", "k1"):
        return logprob - ref_logprob
    if kl_type == "abs":
        return (logprob - ref_logprob).abs()
    if kl_type in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_type in ("low_var_kl", "k3"):
        # K3 estimator: KL(pi || ref) ~= E[r - 1 - log r] with r = pi/ref.
        # Clamping bounds the exponential to avoid Inf/NaN explosions.
        kl = torch.clamp(ref_logprob - logprob, min=-20.0, max=20.0)
        return torch.clamp(torch.exp(kl) - kl - 1.0, min=-10.0, max=10.0)
    raise NotImplementedError(f"unsupported PPO kl_loss_type: {kl_type}")
