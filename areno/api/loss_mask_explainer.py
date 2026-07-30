"""Human-readable loss-mask explainer for SFT and agentic training.

The trainer's packer produces a token-level ``loss_mask`` that decides which
positions contribute to the gradient.  This module consumes that output —
token ids, the loss mask, and the original message list — and maps each
contiguous masked/unmasked region back to its conversational role, turn
index, and text preview.  It does **not** reimplement mask rules; it reads
the packer's actual output so the explanation always matches training
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from areno.api.tokenizer import apply_chat_template_with_options


@dataclass(slots=True)
class MaskSpan:
    """One conversational span with its loss-mask summary.

    ``text_preview`` is truncated to 50 characters unless the caller passes
    ``show_full_text=True``.  ``mask_ratio`` is ``trainable_count /
    token_count`` and is 0 when the entire span is masked out.
    """

    role: str
    turn_index: int
    text_preview: str
    token_count: int
    trainable_count: int
    mask_ratio: float
    is_trainable: bool


@dataclass(slots=True)
class LossMaskReport:
    """Aggregated loss-mask report across all conversational spans.

    Use :meth:`to_dict` for structured (JSON) output and
    :meth:`to_human_readable` for terminal display.
    """

    model_name: str
    total_tokens: int
    trainable_tokens: int
    overall_mask_ratio: float
    spans: list[MaskSpan] = field(default_factory=list)
    show_full_text: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for programmatic consumption."""

        return {
            "model_name": self.model_name,
            "total_tokens": self.total_tokens,
            "trainable_tokens": self.trainable_tokens,
            "overall_mask_ratio": self.overall_mask_ratio,
            "show_full_text": self.show_full_text,
            "spans": [
                {
                    "role": s.role,
                    "turn_index": s.turn_index,
                    "text_preview": s.text_preview,
                    "token_count": s.token_count,
                    "trainable_count": s.trainable_count,
                    "mask_ratio": s.mask_ratio,
                    "is_trainable": s.is_trainable,
                }
                for s in self.spans
            ],
        }

    def to_human_readable(self) -> str:
        """Return a terminal-friendly table for CLI display."""

        lines: list[str] = []
        lines.append(f"Loss Mask Report: {self.model_name}")
        lines.append(
            f"Total tokens: {self.total_tokens}, "
            f"trainable: {self.trainable_tokens} "
            f"({self.overall_mask_ratio:.1%})"
        )
        lines.append("-" * 70)
        lines.append(
            f"{'Role':<12} {'Turn':>4} {'Tokens':>7} {'Train':>7} "
            f"{'Ratio':>6}  Text"
        )
        lines.append("-" * 70)
        for s in self.spans:
            preview = s.text_preview
            if not self.show_full_text and len(preview) > 50:
                preview = preview[:47] + "..."
            lines.append(
                f"{s.role:<12} {s.turn_index:>4} {s.token_count:>7} "
                f"{s.trainable_count:>7} {s.mask_ratio:>5.0%}  {preview}"
            )
        return "\n".join(lines)


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode a token-id slice to text, tolerating tokenizer quirks."""

    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def _find_span_boundaries(
    tokenizer: Any, messages: list[dict[str, Any]]
) -> list[tuple[int, int, str, int]]:
    """Return ``(start, end, role, turn_index)`` for each message.

    Renders messages incrementally to find where each message's tokens
    start and end in the full tokenised conversation.  ``turn_index`` is
    the position of the message in the original list.
    """

    boundaries: list[tuple[int, int, str, int]] = []
    prev_len = 0

    for i, msg in enumerate(messages):
        partial = messages[: i + 1]
        try:
            token_ids = apply_chat_template_with_options(
                tokenizer, partial, tokenize=True
            )
        except Exception:
            # If the template cannot render this prefix, we cannot determine
            # the boundary; skip this message.
            continue

        # Normalise to a plain list of ints.
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        if hasattr(token_ids, "ids"):
            token_ids = token_ids.ids
        if not isinstance(token_ids, (list, tuple)):
            token_ids = list(token_ids)

        cur_len = len(token_ids)
        role = msg.get("role", "unknown")
        boundaries.append((prev_len, cur_len, role, i))
        prev_len = cur_len

    return boundaries


class LossMaskExplainer:
    """Map packer-produced loss masks back to conversational structure."""

    @staticmethod
    def explain(
        model_name: str,
        tokenizer: Any,
        token_ids: list[int],
        loss_mask: list[bool],
        messages: list[dict[str, Any]],
        *,
        show_full_text: bool = False,
    ) -> LossMaskReport:
        """Consume packer output and produce a :class:`LossMaskReport`.

        ``token_ids`` and ``loss_mask`` are the packer's actual output for
        one training sequence.  ``messages`` is the original OpenAI-style
        message list that produced the sequence.  The explainer does not
        recompute the mask — it only reads and interprets it.
        """

        total_tokens = len(token_ids)
        trainable_tokens = sum(1 for m in loss_mask if m)
        overall_ratio = trainable_tokens / total_tokens if total_tokens else 0.0

        spans: list[MaskSpan] = []
        boundaries = _find_span_boundaries(tokenizer, messages)

        for start, end, role, turn_idx in boundaries:
            # Clamp to the actual token sequence length (the sequence may
            # have been truncated by the packer).
            span_start = min(start, total_tokens)
            span_end = min(end, total_tokens)
            if span_start >= span_end:
                continue

            span_tokens = list(token_ids[span_start:span_end])
            span_mask = list(loss_mask[span_start:span_end])
            token_count = len(span_tokens)
            trainable_count = sum(1 for m in span_mask if m)
            ratio = trainable_count / token_count if token_count else 0.0

            text = _decode_tokens(tokenizer, span_tokens)
            if show_full_text:
                preview = text
            else:
                preview = text[:50]

            spans.append(MaskSpan(
                role=role,
                turn_index=turn_idx,
                text_preview=preview,
                token_count=token_count,
                trainable_count=trainable_count,
                mask_ratio=ratio,
                is_trainable=ratio > 0,
            ))

        # If the token sequence extends beyond the last message boundary
        # (e.g. appended EOS or padding), report it as a trailing span.
        if boundaries and boundaries[-1][1] < total_tokens:
            trailing_start = boundaries[-1][1]
            trailing_tokens = list(token_ids[trailing_start:])
            trailing_mask = list(loss_mask[trailing_start:])
            tcount = len(trailing_tokens)
            ttrainable = sum(1 for m in trailing_mask if m)
            text = _decode_tokens(tokenizer, trailing_tokens)
            spans.append(MaskSpan(
                role="trailing",
                turn_index=len(messages),
                text_preview=text[:50] if not show_full_text else text,
                token_count=tcount,
                trainable_count=ttrainable,
                mask_ratio=ttrainable / tcount if tcount else 0.0,
                is_trainable=ttrainable > 0,
            ))

        return LossMaskReport(
            model_name=model_name,
            total_tokens=total_tokens,
            trainable_tokens=trainable_tokens,
            overall_mask_ratio=overall_ratio,
            spans=spans,
            show_full_text=show_full_text,
        )