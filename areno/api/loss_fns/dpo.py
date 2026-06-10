"""Direct Preference Optimization loss over chosen/rejected pairs.

The DPO trainer materializes every preference row as two consecutive
`TrainSequence` rows: chosen first, rejected second. The backend computes
current-policy next-token logprobs, while the trainer pre-fills `ref_logprobs`
with scores from a frozen reference role. This loss sums response-token
logprobs per sequence, forms pairwise log-ratio margins, and applies the DPO
logistic objective.
"""

from __future__ import annotations

from areno.api.loss_fns.layout import response_layout, sequence_sum


def dpo_loss_fn(data_pack, logprobs, *, beta: float = 0.1, label_smoothing: float = 0.0):
    """DPO pairwise loss.

    Assumes row order is `[chosen_0, rejected_0, chosen_1, rejected_1, ...]`.
    `mini_bs` must therefore be even so the backend never splits a pair across
    microbatches.
    """

    beta = float(beta)
    label_smoothing = float(label_smoothing)

    layout = response_layout(data_pack, logprobs, need_ref_logprobs=True, need_sequences=True)
    num_sequences = int(layout.num_sequences) if layout.packed else int(logprobs.shape[0])
    if num_sequences % 2 != 0:
        raise ValueError("DPO requires an even number of sequences per microbatch")

    # The trainer pre-scores the frozen reference role and stores it in the
    # shared PPO ref_logprobs field so the backend packer can carry it.
    ref_logprobs = layout.ref_logprobs.to(dtype=logprobs.dtype)
    # Convert per-token scores to one scalar logprob per sequence.
    policy_seq_logps = sequence_sum(logprobs, layout)
    ref_seq_logps = sequence_sum(ref_logprobs, layout)
    response_lens = layout.response_len.to(dtype=logprobs.dtype)
    return _dpo_from_sequence_logps(policy_seq_logps, ref_seq_logps, response_lens, beta, label_smoothing)


def _dpo_from_sequence_logps(policy_seq_logps, ref_seq_logps, response_lens, beta: float, label_smoothing: float):
    import torch
    import torch.nn.functional as F

    chosen_policy = policy_seq_logps[0::2]
    rejected_policy = policy_seq_logps[1::2]
    chosen_ref = ref_seq_logps[0::2]
    rejected_ref = ref_seq_logps[1::2]

    # DPO compares how much more the policy prefers chosen over rejected
    # relative to the frozen reference policy.
    policy_logratios = chosen_policy - rejected_policy
    ref_logratios = chosen_ref - rejected_ref
    logits = float(beta) * (policy_logratios - ref_logratios)
    # label_smoothing keeps a small amount of probability mass on the rejected
    # response, matching the conservative DPO variant.
    losses = -(1.0 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
    loss = losses.mean()

    # Detached reward-style diagnostics from the DPO paper.
    chosen_rewards = float(beta) * (chosen_policy - chosen_ref).detach()
    rejected_rewards = float(beta) * (rejected_policy - rejected_ref).detach()
    reward_margin = chosen_rewards - rejected_rewards
    return loss, {
        "dpo_loss": loss.detach(),
        "total_loss": loss.detach(),
        "dpo_accuracy": (logits > 0).to(dtype=torch.float32).mean().detach(),
        "dpo_margin": logits.mean().detach(),
        "dpo_reward_margin": reward_margin.mean().detach(),
        "dpo_chosen_reward": chosen_rewards.mean().detach(),
        "dpo_rejected_reward": rejected_rewards.mean().detach(),
        "dpo_response_len": response_lens.clamp(min=1).mean().detach(),
    }


__all__ = ["dpo_loss_fn"]
