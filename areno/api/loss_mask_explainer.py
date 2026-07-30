"""Human-readable loss-mask explainer.

Consumes packer output — token sequences, effective loss masks, and per-span
role annotations — and produces a structured ``LossMaskExplanation`` that maps
each contiguous token span back to its role (prompt, response, assistant text,
tool call, tool result) and reports whether it contributes to training loss.

The explainer does **not** re-implement mask rules; it only interprets the masks
and span metadata already produced by the SFT or Agentic packer.
"""

from __future__ import annotations

from typing import Any

from areno.api.data import LossMaskExplanation, LossSpan


def explain_loss_mask(
    tokens: list[int],
    loss_mask: list[bool],
    spans: list[LossSpan],
    *,
    tokenizer=None,
    show_text: bool = False,
) -> LossMaskExplanation:
    """Build a human-readable + structured loss-mask report from packer output.

    Parameters
    ----------
    tokens
        Packer-produced token sequence.
    loss_mask
        Effective loss mask (``True`` = token contributes to loss).

        * SFT: ``[not m for m in prompt_mask]``
        * Agentic: ``AgentTrainBatch.loss_masks[i]``
    spans
        Packer-produced per-span role annotations.

        * SFT: ``spans_from_prompt_mask(prompt_mask)``
        * Agentic: ``AgentTrainBatch.spans[i]``
    tokenizer
        Optional tokenizer; required when ``show_text=True``.
    show_text
        Whether to decode and include span text in the explanation.

    Returns
    -------
    LossMaskExplanation
        Structured report with spans, token counts, per-role summary, and
        optional text preview.

    Raises
    ------
    ValueError
        If ``tokens`` and ``loss_mask`` lengths differ, or if spans have gaps
        or overlaps after truncation clipping.
    """

    n = len(tokens)
    if n != len(loss_mask):
        raise ValueError(f"tokens and loss_mask length mismatch: tokens={n}, loss_mask={len(loss_mask)}")

    # Clip spans to the actual token range (handles right-truncated sequences).
    clipped: list[LossSpan] = []
    for span in spans:
        if span.start >= n:
            continue
        end = min(span.end, n)
        if end <= span.start:
            continue
        clipped.append(LossSpan(role=span.role, start=span.start, end=end, loss=span.loss, turn=span.turn))

    # Validate coverage: spans should cover [0, n) without gaps or overlaps.
    pos = 0
    for span in clipped:
        if span.start > pos:
            raise ValueError(
                f"span gap: expected coverage at position {pos}, but next span '{span.role}' starts at {span.start}"
            )
        if span.start < pos:
            raise ValueError(
                f"span overlap: span '{span.role}' starts at {span.start} but previous span ended at {pos}"
            )
        pos = span.end
    if pos < n:
        raise ValueError(f"span coverage incomplete: spans end at {pos} but tokens has {n} positions")

    # Compute statistics.
    loss_tokens = sum(1 for m in loss_mask if m)
    summary: list[dict[str, Any]] = []
    role_stats: dict[str, dict[str, int]] = {}
    for span in clipped:
        token_count = span.end - span.start
        span_loss_tokens = token_count if span.loss else 0
        if span.role not in role_stats:
            role_stats[span.role] = {"token_count": 0, "loss_tokens": 0}
        role_stats[span.role]["token_count"] += token_count
        role_stats[span.role]["loss_tokens"] += span_loss_tokens
    for role, stats in role_stats.items():
        summary.append({"role": role, **stats})

    # Optional text preview.
    text_preview: dict[int, str] | None = None
    if show_text and tokenizer is not None:
        text_preview = {}
        for idx, span in enumerate(clipped):
            span_tokens = tokens[span.start : span.end]
            try:
                text_preview[idx] = tokenizer.decode(span_tokens)
            except Exception:
                text_preview[idx] = "<decode-error>"

    return LossMaskExplanation(
        spans=clipped,
        total_tokens=n,
        loss_tokens=loss_tokens,
        summary=summary,
        text_preview=text_preview,
    )


__all__ = ["explain_loss_mask"]
