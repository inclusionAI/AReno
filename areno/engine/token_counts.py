"""Effective trainable token counting (issue #227).

Computes per-update token statistics using the same masks consumed by the
loss function.  The four emitted metrics are:

* ``total_input_tokens`` — all tokens in the batch (including padding).
* ``masked_tokens`` — tokens excluded from loss (prompt positions + padding).
* ``effective_loss_tokens`` — tokens that contribute to the loss gradient.
* ``mean_effective_length`` — effective tokens per sequence (response length).

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
        total_input_tokens: All tokens in the batch (valid positions only,
            excluding padding).
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


def compute_token_counts(
    lengths: list[int],
    prompt_mask_rows: list[list[bool]] | None = None,
    loss_mask_rows: list[list[bool]] | None = None,
    packed_response_mask: list[bool] | None = None,
    num_sequences: int | None = None,
) -> TokenCounts:
    """Compute effective trainable token statistics from loss masks.

    Supports both packed (varlen) and padded (rectangular) layouts.

    For **packed** layout, pass ``packed_response_mask`` and ``num_sequences``.
    For **padded** layout, pass ``lengths``, ``prompt_mask_rows``, and
    optionally ``loss_mask_rows``.

    Args:
        lengths: Valid token count per sequence (excludes padding).
        prompt_mask_rows: Per-sequence prompt mask (True = prompt token).
            Required for padded layout; ignored for packed layout.
        loss_mask_rows: Per-sequence loss mask (True = contributes to loss).
            Optional; if None, only prompt_mask is used.
        packed_response_mask: Flat response mask for packed layout
            (True = response token that contributes to loss).
        num_sequences: Number of sequences for packed layout.

    Returns:
        A :class:`TokenCounts` with the four metrics.

    Raises:
        ValueError: If neither packed nor padded inputs are provided, or
            if lengths are empty for padded layout.
    """

    # --- Packed layout ---
    if packed_response_mask is not None:
        n_seqs = num_sequences if num_sequences is not None and num_sequences > 0 else 1
        effective = sum(1 for m in packed_response_mask if m)
        total_actions = len(packed_response_mask)
        # In packed layout, total_input_tokens = actions + num_sequences
        # (each sequence has length L, contributing L-1 actions; total tokens
        # = sum of lengths = actions + num_sequences).
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

    # --- Padded layout ---
    if not lengths:
        raise ValueError("lengths must be non-empty for padded layout")
    if prompt_mask_rows is None:
        raise ValueError("prompt_mask_rows required for padded layout")

    total_tokens = 0
    masked = 0
    effective = 0
    n_seqs = len(lengths)

    for i, length in enumerate(lengths):
        if length <= 0:
            continue
        total_tokens += length
        pm = prompt_mask_rows[i] if i < len(prompt_mask_rows) else []
        lm = loss_mask_rows[i] if loss_mask_rows is not None and i < len(loss_mask_rows) else None
        # Actions are positions 1..length-1 (token at position 0 has no action).
        for pos in range(1, length):
            is_prompt = bool(pm[pos]) if pos < len(pm) else False
            is_loss = bool(lm[pos]) if lm is not None and pos < len(lm) else True
            if is_prompt or not is_loss:
                masked += 1
            else:
                effective += 1

    mean_eff = effective / n_seqs if n_seqs > 0 else 0.0
    return TokenCounts(
        total_input_tokens=total_tokens,
        masked_tokens=masked,
        effective_loss_tokens=effective,
        mean_effective_length=mean_eff,
        num_sequences=n_seqs,
    )
