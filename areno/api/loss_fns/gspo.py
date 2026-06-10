"""GSPO sequence-level clipped policy gradient loss.

The packed branch consumes the variable-length layout produced by areno
(tokens for all sequences concatenated into 1D tensors, plus `packed_seq_ids`
that map each token to its sequence). The padded branch handles right-padded
rectangular tensors used by other backends. Both branches implement the same
math: form a per-sequence importance ratio by averaging the token log-ratio
over the sequence length, clip it, and minimise `-min(r * A, clip(r) * A)`.
"""

from areno.api.loss_fns.layout import masked_mean, response_layout, sequence_sum


def gspo_loss_fn(data_pack, logprobs, *, clip_eps: float = 3e-4):
    """Sequence-level clipped policy-gradient loss used by GSPO.

    GSPO forms one ratio per response sequence by averaging token log-ratios,
    then applies PPO-style clipping at the sequence level. The packed branch is
    used by areno for variable-length batches; the padded branch is kept for
    backends that materialize rectangular tensors.
    """

    import torch

    clip_eps = float(clip_eps)
    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_sequences=True)

    # `logprobs - logprobs.detach()` is the differentiable surrogate for the
    # importance ratio: value 1 at current params, gradient = d logp / d theta.
    token_log_ratio = logprobs - logprobs.detach()
    # Length normalisation gives the geometric-mean sequence ratio.
    seq_log_ratio = sequence_sum(token_log_ratio, layout) / layout.response_len
    seq_ratio = torch.exp(seq_log_ratio)

    # PPO-style clipping is applied once per sequence for GSPO.
    clipped_seq_ratio = torch.clamp(seq_ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    seq_advantage = sequence_sum(layout.advantages, layout) / layout.response_len

    per_seq_policy_loss = -torch.min(
        seq_ratio * seq_advantage,
        clipped_seq_ratio * seq_advantage,
    )
    policy_loss = per_seq_policy_loss.mean()

    train_values = logprobs.detach()
    rollout_values = layout.old_logprobs
    # logp_diff measures rollout-vs-train policy drift; useful as an
    # off-policy diagnostic since GSPO assumes only mild drift per step.
    logp_diff = rollout_values - train_values

    stats = {
        "policy_loss": policy_loss.detach(),
        "total_loss": policy_loss.detach(),
        "ratio_mean": seq_ratio.mean().detach(),
        "ratio_std": seq_ratio.std(unbiased=False).detach(),
        "advantage_mean": seq_advantage.mean().detach(),
        "response_len": layout.response_len.mean().detach(),
        "rollout_logprobs_mean": masked_mean(rollout_values, layout).detach(),
        "train_logprobs_mean": masked_mean(train_values, layout).detach(),
        "logp_diff_mean": masked_mean(logp_diff, layout).detach(),
        "logp_abs_diff_mean": masked_mean(logp_diff.abs(), layout).detach(),
    }

    return policy_loss, stats
