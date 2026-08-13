"""Dr. GRPO advantage and policy-loss mathematics."""

from __future__ import annotations

import numpy as np

from areno.api.loss_fns.layout import masked_mean, response_layout


def compute_drgrpo_advantages(rewards: list[float]) -> list[float]:
    """Center rewards within one prompt group without variance scaling."""

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    return (rewards_arr - rewards_arr.mean()).tolist()


def drgrpo_loss_fn(data_pack, logprobs, *, clip_eps: float = 0.2, max_completion_length: int):
    """Compute the fixed-normalizer token policy loss proposed by Dr. GRPO."""

    import torch

    clip_eps = float(clip_eps)
    max_completion_length = int(max_completion_length)
    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive")

    layout = response_layout(
        data_pack,
        logprobs,
        need_old_logprobs=True,
        need_advantages=True,
        need_sequences=True,
    )
    token_log_ratio = logprobs - logprobs.detach()
    ratio = torch.exp(token_log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    per_token_loss = -torch.min(ratio * layout.advantages, clipped_ratio * layout.advantages)
    masked_token_loss = per_token_loss * layout.response_mask

    sequence_total = data_pack.get("_fixed_sequence_total")
    if sequence_total is None:
        if layout.packed:
            sequence_total = int(layout.num_sequences or 0)
        else:
            sequence_total = int(layout.response_mask.shape[0])
    sequence_total = max(int(sequence_total), 1)
    grad_scale = float(data_pack.get("_fixed_sequence_grad_scale", 1))
    policy_loss = masked_token_loss.sum() * grad_scale / (sequence_total * max_completion_length)

    valid_ratio = ratio[layout.response_mask.bool()]
    logp_diff = layout.old_logprobs - logprobs.detach()
    stats = {
        "policy_loss": policy_loss.detach(),
        "total_loss": policy_loss.detach(),
        "ratio_mean": valid_ratio.mean().detach() if valid_ratio.numel() else ratio.mean().detach(),
        "ratio_std": (
            valid_ratio.std().detach() if valid_ratio.numel() > 1 else torch.zeros((), device=logprobs.device)
        ),
        "advantage_mean": masked_mean(layout.advantages, layout).detach(),
        "response_len": layout.response_len.mean().detach(),
        "rollout_logprobs_mean": masked_mean(layout.old_logprobs, layout).detach(),
        "train_logprobs_mean": masked_mean(logprobs.detach(), layout).detach(),
        "logp_diff_mean": masked_mean(logp_diff, layout).detach(),
        "logp_abs_diff_mean": masked_mean(logp_diff.abs(), layout).detach(),
    }
    return policy_loss, stats


__all__ = ["compute_drgrpo_advantages", "drgrpo_loss_fn"]
