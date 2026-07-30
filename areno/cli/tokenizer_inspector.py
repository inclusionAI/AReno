"""Tokenizer alignment inspector for AReno (#219).

Renders token IDs, token pieces, special-token markers, EOS placement,
role labels, and loss-mask spans side by side for plain prompts, chat
messages, and tool calls — without modifying the actual tokenizer path.

This module intentionally avoids importing from ``areno.api`` at module
load time to prevent a torch dependency chain. All tokenizer interaction
uses the tokenizer's own public methods (encode, decode, apply_chat_template).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Default loss-mask policy matching LossMaskPolicy defaults.
_DEFAULT_NO_LOSS_ROLES = frozenset({"system", "user", "tool", "prompt", "generation_prompt"})


@dataclass(frozen=True)
class TokenAlignment:
    """Alignment information for a single token."""

    index: int
    token_id: int
    token_piece: str
    is_special: bool
    is_eos: bool
    is_unknown: bool
    role: str
    in_loss: bool


@dataclass(frozen=True)
class AlignmentReport:
    """Full alignment report for one input."""

    input_type: str
    tokens: list[TokenAlignment] = field(default_factory=list)
    num_tokens: int = 0
    num_special: int = 0
    num_unknown: int = 0
    eos_positions: list[int] = field(default_factory=list)
    round_trip_lossless: bool = True
    warnings: list[str] = field(default_factory=list)


def inspect_plain_prompt(
    tokenizer: Any,
    text: str,
    *,
    model_path: str | None = None,
    max_tokens: int | None = None,
) -> AlignmentReport:
    """Inspect tokenization of a plain text prompt.

    Encodes the text through the tokenizer's ``encode`` method (applying
    chat template when available), then decodes each token individually.
    """

    token_ids = _encode_text(tokenizer, text)
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    eos_ids = _get_eos_ids(tokenizer, model_path)
    special_ids = _get_special_ids(tokenizer)
    unk_id = _get_unk_id(tokenizer)
    tokens = _build_alignments(tokenizer, token_ids, special_ids, eos_ids, unk_id, role="prompt")
    return _finalize_report("plain", tokens, tokenizer, token_ids, max_tokens)


def inspect_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    model_path: str | None = None,
    add_generation_prompt: bool = True,
) -> AlignmentReport:
    """Inspect tokenization of chat messages with role and loss-mask labels.

    Uses ``apply_chat_template`` for encoding. Role labels are determined
    by incrementally encoding message prefixes to find boundaries.
    """

    token_ids = _encode_chat(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    eos_ids = _get_eos_ids(tokenizer, model_path)
    special_ids = _get_special_ids(tokenizer)
    unk_id = _get_unk_id(tokenizer)
    roles = _assign_roles(tokenizer, token_ids, messages, add_generation_prompt)
    tokens = []
    for i, tid in enumerate(token_ids):
        piece = _decode_single(tokenizer, tid)
        role = roles[i] if i < len(roles) else "unknown"
        tokens.append(TokenAlignment(
            index=i, token_id=tid, token_piece=piece,
            is_special=tid in special_ids, is_eos=tid in eos_ids,
            is_unknown=(unk_id is not None and tid == unk_id),
            role=role, in_loss=role not in _DEFAULT_NO_LOSS_ROLES,
        ))
    return _finalize_report("chat", tokens, tokenizer, token_ids, None)


def inspect_tool_call(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    model_path: str | None = None,
) -> AlignmentReport:
    """Inspect tokenization of messages with tool definitions."""

    token_ids = _encode_chat_with_tools(tokenizer, messages, tools)
    eos_ids = _get_eos_ids(tokenizer, model_path)
    special_ids = _get_special_ids(tokenizer)
    unk_id = _get_unk_id(tokenizer)
    roles = _assign_roles(tokenizer, token_ids, messages, True)
    tokens = []
    for i, tid in enumerate(token_ids):
        piece = _decode_single(tokenizer, tid)
        role = roles[i] if i < len(roles) else "unknown"
        tokens.append(TokenAlignment(
            index=i, token_id=tid, token_piece=piece,
            is_special=tid in special_ids, is_eos=tid in eos_ids,
            is_unknown=(unk_id is not None and tid == unk_id),
            role=role, in_loss=role not in _DEFAULT_NO_LOSS_ROLES,
        ))
    return _finalize_report("tool_call", tokens, tokenizer, token_ids, None)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_alignment_text(report: AlignmentReport) -> str:
    """Render an :class:`AlignmentReport` as a human-readable table."""

    lines = [
        f"Tokenizer alignment report ({report.input_type})",
        f"  tokens={report.num_tokens}  special={report.num_special}  "
        f"unknown={report.num_unknown}  eos_positions={report.eos_positions}",
        f"  round_trip_lossless={report.round_trip_lossless}",
        "",
    ]
    if report.warnings:
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")
        lines.append("")
    lines.append(f"{'Idx':>4}  {'ID':>8}  {'Piece':<20}  {'Spec':>4}  {'EOS':>3}  {'Unk':>3}  {'Role':<16}  {'Loss':>4}")
    lines.append("-" * 80)
    for t in report.tokens:
        piece_repr = repr(t.token_piece)[:20]
        lines.append(
            f"{t.index:>4}  {t.token_id:>8}  {piece_repr:<20}  "
            f"{'Y' if t.is_special else 'N':>4}  "
            f"{'Y' if t.is_eos else 'N':>3}  "
            f"{'Y' if t.is_unknown else 'N':>3}  "
            f"{t.role:<16}  "
            f"{'Y' if t.in_loss else 'N':>4}"
        )
    return "\n".join(lines)


def alignment_to_json(report: AlignmentReport) -> str:
    """Serialise an :class:`AlignmentReport` to JSON."""

    return json.dumps(asdict(report), indent=2, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Internal helpers — all use only the tokenizer's own methods, no areno.api imports
# ---------------------------------------------------------------------------


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """Encode text, applying chat template if the tokenizer has one."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            result = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            return _normalize_ids(result)
        except Exception:
            pass
    return _normalize_ids(tokenizer.encode(text))


def _encode_chat(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True) -> list[int]:
    """Encode chat messages using the tokenizer's chat template."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            result = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            )
            return _normalize_ids(result)
        except TypeError:
            # Some tokenizers don't accept add_generation_prompt.
            result = tokenizer.apply_chat_template(messages, tokenize=True)
            return _normalize_ids(result)
    # Fallback: simple text concatenation.
    text = "\n".join(str(m.get("content", "")) for m in messages)
    return _normalize_ids(tokenizer.encode(text))


def _encode_chat_with_tools(
    tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[int]:
    """Encode chat messages with tool definitions."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            result = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, tools=tools,
            )
            return _normalize_ids(result)
        except TypeError:
            pass
    return _encode_chat(tokenizer, messages)


def _normalize_ids(value: Any) -> list[int]:
    """Convert tokenizer output to a plain list of ints."""
    if hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, list | tuple):
        if value and hasattr(value[0], "ids"):
            if len(value) == 1:
                return _normalize_ids(value[0])
        if value and isinstance(value[0], list | tuple):
            if len(value) == 1:
                return _normalize_ids(value[0])
        return [int(tid) for tid in value]
    raise TypeError(f"expected token ids, got {type(value).__name__}")


def _build_alignments(
    tokenizer: Any,
    token_ids: list[int],
    special_ids: set[int],
    eos_ids: set[int],
    unk_id: int | None,
    *,
    role: str = "prompt",
) -> list[TokenAlignment]:
    """Build per-token alignment entries for a flat token list."""
    tokens = []
    for i, tid in enumerate(token_ids):
        piece = _decode_single(tokenizer, tid)
        tokens.append(TokenAlignment(
            index=i, token_id=tid, token_piece=piece,
            is_special=tid in special_ids, is_eos=tid in eos_ids,
            is_unknown=(unk_id is not None and tid == unk_id),
            role=role, in_loss=False,
        ))
    return tokens


def _finalize_report(
    input_type: str,
    tokens: list[TokenAlignment],
    tokenizer: Any,
    token_ids: list[int],
    max_tokens: int | None,
) -> AlignmentReport:
    """Assemble the final report with statistics and warnings."""
    num_special = sum(1 for t in tokens if t.is_special)
    num_unknown = sum(1 for t in tokens if t.is_unknown)
    eos_positions = [t.index for t in tokens if t.is_eos]
    warnings: list[str] = []
    lossless = _check_round_trip(tokenizer, token_ids)
    if not lossless:
        warnings.append("encode→decode round trip is not lossless; some information is lost.")
    if max_tokens is not None:
        warnings.append(f"output truncated to {max_tokens} tokens.")
    if num_unknown > 0:
        warnings.append(f"{num_unknown} unknown token(s) detected.")
    return AlignmentReport(
        input_type=input_type, tokens=tokens, num_tokens=len(tokens),
        num_special=num_special, num_unknown=num_unknown,
        eos_positions=eos_positions, round_trip_lossless=lossless, warnings=warnings,
    )


def _decode_single(tokenizer: Any, token_id: int) -> str:
    """Decode a single token ID to its text piece."""
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        # Some tokenizers don't accept skip_special_tokens as a kwarg.
        try:
            return tokenizer.decode([token_id])
        except Exception:
            return f"<id:{token_id}>"
    except Exception:
        return f"<id:{token_id}>"


def _check_round_trip(tokenizer: Any, token_ids: list[int]) -> bool:
    """Check if encode(decode(token_ids)) == token_ids."""
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        re_encoded = _normalize_ids(tokenizer.encode(decoded))
        return re_encoded == token_ids
    except Exception:
        return False


def _assign_roles(
    tokenizer: Any,
    token_ids: list[int],
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> list[str]:
    """Assign role labels by incrementally encoding message prefixes."""
    if not getattr(tokenizer, "chat_template", None):
        return ["prompt"] * len(token_ids)

    roles: list[str] = ["unknown"] * len(token_ids)
    prev_len = 0
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        try:
            partial = _encode_chat(tokenizer, messages[: i + 1], add_generation_prompt=False)
            curr_len = len(partial)
        except Exception:
            curr_len = prev_len
        for idx in range(prev_len, min(curr_len, len(token_ids))):
            roles[idx] = role
        prev_len = curr_len

    # Trailing generation_prompt tokens.
    if add_generation_prompt:
        for idx in range(prev_len, len(token_ids)):
            roles[idx] = "generation_prompt"

    # Any remaining unassigned default to "prompt".
    for i in range(len(roles)):
        if roles[i] == "unknown":
            roles[i] = "prompt"

    return roles


def _get_special_ids(tokenizer: Any) -> set[int]:
    """Return the set of special token IDs from the tokenizer."""
    ids: set[int] = set()
    all_special = getattr(tokenizer, "all_special_ids", None)
    if all_special is not None:
        ids.update(all_special)
    additional = getattr(tokenizer, "additional_special_tokens", None)
    if additional:
        for tok in additional:
            tok_id = _convert_token_to_id(tokenizer, tok)
            if tok_id is not None:
                ids.add(tok_id)
    return ids


def _get_eos_ids(tokenizer: Any, model_path: str | None) -> set[int]:
    """Return the set of EOS token IDs."""
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(eos)
    # Also check config.json for additional EOS IDs.
    if model_path:
        try:
            from pathlib import Path
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
                cfg_eos = config.get("eos_token_id")
                if isinstance(cfg_eos, int):
                    ids.add(cfg_eos)
                elif isinstance(cfg_eos, list):
                    ids.update(cfg_eos)
        except Exception:
            pass
    return ids


def _get_unk_id(tokenizer: Any) -> int | None:
    """Return the unknown token ID, or None."""
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(unk, int):
        return unk
    if isinstance(unk, (list, tuple)) and unk:
        return int(unk[0])
    return None


def _convert_token_to_id(tokenizer: Any, token: str) -> int | None:
    """Convert a token string to its ID, returning None on failure."""
    try:
        result = tokenizer.convert_tokens_to_ids(token)
        if isinstance(result, int):
            return result
    except Exception:
        pass
    return None