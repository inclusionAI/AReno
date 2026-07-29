"""Effective trainable token counting (issue #227).

Computes per-update token statistics using the same masks consumed by the
loss function.  The four emitted metrics are:

* ``total_input_tokens`` — all valid tokens in the batch (excluding padding).
* ``masked_tokens`` — tokens excluded from loss (prompt positions).
* ``effective_loss_tokens`` — tokens that contribute to the loss gradient.
* ``mean_effective_length`` — average effective (response) tokens per sequence.

The module is pure Python and operates on plain lists/ints so it can be
unit-tested without torch or GPU.  The trainer integration calls
:func:`compute_token_counts` after packing and before the loss, reading the
same ``prompt_mask`` / ``loss_mask`` / ``packed_response_mask`` fields that
the loss function uses.

Public API:

* :func:`compute_token_counts` — main entry point; returns
  :class:`TokenCounts`.
* :class:`TokenCounts` — dataclass with ``to_dict()``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenCounts:
    """Per-update token statistics.

    Attributes:
        total_input_tokens: All valid tokens in the batch (excluding padding).
        masked_tokens: Tokens excluded from loss (prompt positions).
        effective_loss_tokens: Tokens that contribute to the loss gradient.
        mean_effective_length: Average effective (response) tokens per sequence.
        num_sequences: Number of sequences in the batch.
    """

    total_input_tokens: int
    masked_tokens: int
    effective_loss_tokens: int
    mean_effective_length: float
    num_sequences: int

    def to_dict(self) -> dict[str, float]:
        """Return a metrics dict suitable for merging into train stats."""

        return {
            "total_input_tokens": float(self.total_input_tokens),
            "masked_tokens": float(self.masked_tokens),
            "effective_loss_tokens": float(self.effective_loss_tokens),
            "mean_effective_length": float(self.mean_effective_length),
        }


def compute_token_counts_from_packed(
    response_mask: list[bool],
    num_sequences: int,
) -> TokenCounts:
    """Compute token counts from a packed (varlen) response mask.

    In packed layout, the response mask is a flat tensor where each entry
    corresponds to an action position (token at position t predicts token t+1).
    Each sequence of length L contributes L-1 actions.

    Args:
        response_mask: Flat mask (True = response token contributing to loss).
        num_sequences: Number of sequences in the packed batch.

    Returns:
        A :class:`TokenCounts`.
    """

    n_seqs = max(num_sequences, 1)
    effective = sum(1 for m in response_mask if m)
    total_actions = len(response_mask)
    # total_input_tokens = sum of sequence lengths = actions + num_sequences
    # (each seq of length L contributes L-1 actions; sum(L_i) = sum(L_i-1) + n).
    total_tokens = total_actions + n_seqs
    masked = total_tokens - effective
    mean_eff = effective / n_seqs if n_seqs > 0 else 0.0
    return TokenCounts(
        total_input_tokens=total_tokens,
        masked_tokens=masked,
        effective_loss_tokens=effective,
        mean_effective_length=mean_eff,
        num_sequences=n_seqs,
    )


def compute_token_counts_from_padded(
    lengths: list[int],
    prompt_mask_rows: list[list[bool]],
    loss_mask_rows: list[list[bool]] | None = None,
) -> TokenCounts:
    """Compute token counts from padded (rectangular) masks.

    In padded layout, each sequence has a length and a per-position prompt
    mask. Actions are positions 1..length-1 (token at position 0 has no
    action because there's no previous token to predict it from).

    ``total_input_tokens`` = sum of lengths (valid tokens, excluding padding).
    For each action position (1..length-1):
        - If prompt_mask[pos] is True → masked (doesn't contribute to loss).
        - If loss_mask exists and loss_mask[pos] is False → masked.
        - Otherwise → effective (contributes to loss).
    ``masked_tokens`` = total action positions - effective.
    Note: position 0 of each sequence is not an action position, so it's
    neither masked nor effective. ``total_input_tokens`` includes it, but
    ``masked_tokens + effective_loss_tokens`` = total_input_tokens - num_sequences.

    Args:
        lengths: Valid token count per sequence (excludes padding).
        prompt_mask_rows: Per-sequence prompt mask (True = prompt token).
        loss_mask_rows: Per-sequence loss mask (True = contributes to loss).
            If None, only prompt_mask is used (all non-prompt = effective).

    Returns:
        A :class:`TokenCounts`.
    """

    if not lengths:
        raise ValueError("lengths must be non-empty for padded layout")
    if prompt_mask_rows is None:
        raise ValueError("prompt_mask_rows required for padded layout")

    total_tokens = 0
    effective = 0
    n_seqs = len(lengths)

    for i, length in enumerate(lengths):
        if length <= 0:
            continue
        total_tokens += length
        pm = prompt_mask_rows[i] if i < len(prompt_mask_rows) else []
        lm = loss_mask_rows[i] if loss_mask_rows is not None and i < len(loss_mask_rows) else None
        # Actions are positions 1..length-1.
        for pos in range(1, length):
            is_prompt = bool(pm[pos]) if pos < len(pm) else False
            is_loss = bool(lm[pos]) if lm is not None and pos < len(lm) else True
            if not is_prompt and is_loss:
                effective += 1

    # masked = total_actions - effective = (total_tokens - n_seqs) - effective
    total_actions = max(total_tokens - n_seqs, 0)
    masked = max(total_actions - effective, 0)
    mean_eff = effective / n_seqs if n_seqs > 0 else 0.0
    return TokenCounts(
        total_input_tokens=total_tokens,
        masked_tokens=masked,
        effective_loss_tokens=effective,
        mean_effective_length=mean_eff,
        num_sequences=n_seqs,
    )


def compute_token_counts(
    lengths: list[int],
    prompt_mask_rows: list[list[bool]] | None = None,
    loss_mask_rows: list[list[bool]] | None = None,
    packed_response_mask: list[bool] | None = None,
    num_sequences: int | None = None,
) -> TokenCounts:
    """Compute effective trainable token statistics from loss masks.

    Dispatches to packed or padded layout based on which arguments are provided.

    For **packed** layout, pass ``packed_response_mask`` and ``num_sequences``.
    For **padded** layout, pass ``lengths``, ``prompt_mask_rows``, and
    optionally ``loss_mask_rows``.

    Args:
        lengths: Valid token count per sequence (excludes padding).
        prompt_mask_rows: Per-sequence prompt mask (True = prompt token).
        loss_mask_rows: Per-sequence loss mask (True = contributes to loss).
        packed_response_mask: Flat response mask for packed layout.
        num_sequences: Number of sequences for packed layout.

    Returns:
        A :class:`TokenCounts`.

    Raises:
        ValueError: If neither packed nor padded inputs are provided, or
            if lengths are empty for padded layout.
    """

    if packed_response_mask is not None:
        return compute_token_counts_from_packed(
            response_mask=packed_response_mask,
            num_sequences=num_sequences if num_sequences is not None else 1,
        )
    return compute_token_counts_from_padded(
        lengths=lengths,
        prompt_mask_rows=prompt_mask_rows,
        loss_mask_rows=loss_mask_rows,
    )
