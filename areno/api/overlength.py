"""Unified overlength-sample handling across training modes.

Issue #216 asks AReno to replace its scattered, reject-only "sample exceeds the
token budget" checks with one explicit contract that supports three policies
(``reject``, ``warn``, ``truncate``) for SFT, DPO pairs, and agentic
trajectories. This module owns the policy decision and the per-mode safe
truncation helpers; the dataclasses live in :mod:`areno.api.data` and the
metric counters flow through :mod:`areno.api.metrics`.

Phase 1 introduced the contract and ``decide_overlength``. Phase 2 added the
SFT safe-truncation helper. Phase 3 adds the DPO pair helper that keeps
chosen/rejected comparable; the agentic helper arrives in a later phase.
"""

from __future__ import annotations

from areno.api.data import OverlengthDecision, OverlengthPolicy, OverlengthReason

__all__ = ["decide_overlength", "truncate_sft_response", "truncate_dpo_pair"]


def decide_overlength(
    *,
    prompt_len: int,
    max_prompt_tokens: int,
    response_len: int | None = None,
    max_new_tokens: int | None = None,
    policy: OverlengthPolicy = OverlengthPolicy.REJECT,
) -> OverlengthDecision:
    """Classify one sample against the token budgets under ``policy``.

    The decision is pure and deterministic: identical inputs always yield the
    same :class:`OverlengthDecision`. ``response_len``/``max_new_tokens`` are
    optional because the RL prompt-side path only inspects the prompt budget.

    A prompt that exactly matches ``max_prompt_tokens`` is *not* overlength;
    callers rely on this boundary so ``max_prompt_tokens`` is an inclusive cap.
    """

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive when provided")
    if not isinstance(policy, OverlengthPolicy):
        raise ValueError(f"policy must be an OverlengthPolicy, got {type(policy).__name__}")

    over_by = 0
    reason: OverlengthReason | None = None

    if prompt_len > max_prompt_tokens:
        # A prompt that already exceeds the budget cannot be safely truncated
        # when it is a single base-prompt string (no chat-turn boundary to cut
        # on). Later phases refine this for chat-message prompts; for now it is
        # reported as a single oversized message.
        reason = OverlengthReason.SINGLE_MESSAGE_OVERSIZED
        over_by = prompt_len - max_prompt_tokens
    elif response_len is not None and max_new_tokens is not None and response_len > max_new_tokens:
        reason = OverlengthReason.RESPONSE_TOO_LONG
        over_by = response_len - max_new_tokens

    if reason is None:
        # Within budget. `EXACT_LIMIT` marks the inclusive boundary (length ==
        # cap, not exceeding it); otherwise the sample simply fit, reported as
        # WITHIN_BUDGET. Neither case triggers an action.
        within_reason = (
            OverlengthReason.EXACT_LIMIT if prompt_len == max_prompt_tokens else OverlengthReason.WITHIN_BUDGET
        )
        return OverlengthDecision(action=policy, reason=within_reason, truncated=False, detail=None)

    truncated = policy is OverlengthPolicy.TRUNCATE
    return OverlengthDecision(
        action=policy,
        reason=reason,
        truncated=truncated,
        detail={"over_by_tokens": over_by},
    )


def truncate_sft_response(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...] | None = None,
) -> tuple[list[int], list[bool], bool]:
    """Cut an SFT response to ``max_new_tokens`` at a token boundary, keeping EOS.

    Operates on already-tokenized ids so the result is deterministic and never
    re-pieces text. If the response already fits, it is returned unchanged with
    ``truncated=False``. When the cut drops the trailing EOS, the first available
    EOS id (from ``eos_token_ids``) is re-appended so the sequence still
    terminates correctly; if no EOS ids are supplied the cut stands as-is.

    Returns ``(tokens, prompt_mask, truncated)`` where ``prompt_mask`` masks the
    prompt prefix (``True``) and leaves the response trainable (``False``).
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    if len(response_ids) <= max_new_tokens:
        tokens = list(prompt_ids) + list(response_ids)
        mask = [True] * len(prompt_ids) + [False] * len(response_ids)
        return tokens, mask, False

    cut = list(response_ids[:max_new_tokens])
    # Re-append an EOS if the cut removed it. We only append when there is at
    # least one response token to train on after the prompt; otherwise the
    # sequence cannot produce a next-token loss and the caller should reject it.
    if cut and eos_token_ids and cut[-1] not in eos_token_ids:
        cut.append(eos_token_ids[0])
    tokens = list(prompt_ids) + cut
    mask = [True] * len(prompt_ids) + [False] * len(cut)
    return tokens, mask, True


def truncate_dpo_pair(
    *,
    chosen_tokens: list[int],
    chosen_mask: list[bool],
    rejected_tokens: list[int],
    rejected_mask: list[bool],
    prefix_len: int,
    max_seq_len: int,
    eos_token_ids: tuple[int, ...] | None = None,
) -> tuple[list[int], list[bool], list[int], list[bool], bool] | None:
    """Truncate a DPO chosen/rejected pair to a common budget, keeping it comparable.

    The shared prefix (first ``prefix_len`` tokens) is never touched, so both
    branches keep the same prompt context. Each divergent suffix is cut to
    ``budget = max_seq_len - prefix_len`` at a token boundary; if the cut drops
    the trailing EOS, the first available EOS id is re-appended. Both sides use
    the *same* budget, so after truncation chosen and rejected are still aligned
    on the prefix and directly comparable for the DPO margin loss.

    Returns ``(chosen_tokens, chosen_mask, rejected_tokens, rejected_mask,
    truncated)`` on success, or ``None`` when either side cannot fit even at the
    minimum (prefix + EOS) — the caller should then reject the whole pair rather
    than keep an incomparable single side. When neither side exceeds
    ``max_seq_len`` the pair is returned unchanged with ``truncated=False``.
    """

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if prefix_len < 0 or prefix_len > min(len(chosen_tokens), len(rejected_tokens)):
        raise ValueError("prefix_len must fit within both token sequences")

    # Fast path: both sides already fit.
    if len(chosen_tokens) <= max_seq_len and len(rejected_tokens) <= max_seq_len:
        return (
            list(chosen_tokens),
            list(chosen_mask),
            list(rejected_tokens),
            list(rejected_mask),
            False,
        )

    eos_id = eos_token_ids[0] if eos_token_ids else None

    def _cut_side(tokens: list[int], mask: list[bool]) -> tuple[list[int], list[bool]] | None:
        if len(tokens) <= max_seq_len:
            return list(tokens), list(mask)
        # Keep the prefix, trim the divergent suffix to the budget.
        suffix = list(tokens[prefix_len:max_seq_len])
        # Re-append EOS if the cut removed it. The cut already lands at
        # max_seq_len, so appending would exceed it; instead replace the last
        # cut token with EOS so the total stays within budget. When there is no
        # suffix room beyond the prefix, the side is untrainable -> return None.
        if eos_id is not None and (not suffix or suffix[-1] != eos_id):
            if len(suffix) >= 1:
                suffix[-1] = eos_id
            else:
                # No suffix room beyond prefix; this side is un-trainable.
                return None
        new_tokens = list(tokens[:prefix_len]) + suffix
        new_mask = list(mask[:prefix_len]) + [False] * len(suffix)
        if len(new_tokens) > max_seq_len:
            return None
        return new_tokens, new_mask

    cut_chosen = _cut_side(chosen_tokens, chosen_mask)
    cut_rejected = _cut_side(rejected_tokens, rejected_mask)
    if cut_chosen is None or cut_rejected is None:
        # Either side cannot fit even at minimum; reject the whole pair to keep
        # chosen/rejected comparable (never keep only one side).
        return None
    if len(cut_chosen[0]) < 2 or len(cut_rejected[0]) < 2:
        return None
    return cut_chosen[0], cut_chosen[1], cut_rejected[0], cut_rejected[1], True
