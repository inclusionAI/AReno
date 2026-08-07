"""Identity Preference Optimization loss over chosen/rejected pairs.

IPO uses the same offline preference-pair layout as DPO: every chosen response
is immediately followed by its rejected response. The backend computes
current-policy next-token logprobs, while the trainer pre-fills `ref_logprobs`
with scores from a frozen reference policy.

This implementation follows the sampled sequence-level IPO objective from
Eq. 17 of the IPO paper. Response-token logprobs are summed per sequence, and
the policy-vs-reference preference margin is regressed toward 1 / (2 * beta).
Here `beta` corresponds to the regularization parameter denoted by tau in the
paper.
"""

from __future__ import annotations

from areno.api.loss_fns.layout import response_layout, sequence_sum


def ipo_loss_fn(data_pack, logprobs, *, beta: float = 0.1):
    """IPO pairwise loss using sequence-summed log-probabilities.

    Assumes row order is `[chosen_0, rejected_0, chosen_1, rejected_1, ...]`.
    `mini_bs` must therefore be even so the backend never splits a pair across
    microbatches.
    """

    beta = float(beta)
    if beta <= 0:
        raise ValueError("IPO beta must be positive")

    layout = response_layout(
        data_pack,
        logprobs,
        need_ref_logprobs=True,
        need_sequences=True,
    )

    num_sequences = int(layout.num_sequences) if layout.packed else int(logprobs.shape[0])
    if num_sequences % 2 != 0:
        raise ValueError("IPO requires an even number of sequences per microbatch")

    ref_logprobs = layout.ref_logprobs.to(dtype=logprobs.dtype)

    # IPO Eq. 17 operates on sequence log-probabilities. We therefore sum
    # response-token logprobs exactly as the existing DPO loss does.
    policy_seq_logps = sequence_sum(logprobs, layout)
    ref_seq_logps = sequence_sum(ref_logprobs, layout)

    response_lens = layout.response_len.to(dtype=logprobs.dtype)

    return _ipo_from_sequence_logps(
        policy_seq_logps,
        ref_seq_logps,
        response_lens,
        beta,
    )


def _ipo_from_sequence_logps(
    policy_seq_logps,
    ref_seq_logps,
    response_lens,
    beta: float,
):
    chosen_policy = policy_seq_logps[0::2]
    rejected_policy = policy_seq_logps[1::2]
    chosen_ref = ref_seq_logps[0::2]
    rejected_ref = ref_seq_logps[1::2]

    # Difference between the chosen-vs-rejected preference of the current
    # policy and the same preference under the frozen reference policy.
    policy_logratios = chosen_policy - rejected_policy
    ref_logratios = chosen_ref - rejected_ref
    delta = policy_logratios - ref_logratios

    # Sampled IPO objective (Eq. 17). `beta` here is tau in the paper.
    target = 1.0 / (2.0 * float(beta))
    target_error = delta - target
    losses = target_error.square()
    loss = losses.mean()

    return loss, {
        "ipo_loss": loss.detach(),
        "total_loss": loss.detach(),
        "ipo_delta": delta.mean().detach(),
        "ipo_target_error": target_error.abs().mean().detach(),
        "ipo_response_len": response_lens.clamp(min=1).mean().detach(),
    }


__all__ = ["ipo_loss_fn"]
