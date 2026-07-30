"""Post-loader data contract validation for AReno training modes.

Validates dataset rows after the user loader returns them and before expensive
model/worker initialization. Aggregates a bounded set of errors per mode (SFT,
DPO, online RL, agentic) instead of stopping at the first failure.

Public API:
    validate_dataset(dataset, algo, *, max_errors, agent_fn) -> ValidationResult
    format_validation_result(result) -> str
    format_validation_result_json(result) -> str
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ValidationError:
    """A single field-level validation failure.

    ``index`` is the 0-based row index in the dataset. ``field_path`` uses
    dot/bracket notation (e.g. ``"response"`` or ``"messages[2].content"``).
    """

    index: int
    field_path: str
    message: str
    expected_type: str | None = None
    actual_type: str | None = None
    hint: str | None = None


@dataclass(slots=True)
class ValidationResult:
    """Aggregated validation outcome for one dataset.

    ``mode`` is one of ``"sft"``, ``"dpo"``, ``"online_rl"``, or ``"agentic"``.
    ``errors`` is bounded by the ``max_errors`` parameter passed to
    :func:`validate_dataset`.
    """

    mode: str
    total_rows: int
    valid_rows: int
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------


def _type_name(tp: type | tuple[type, ...]) -> str:
    """Human-readable type name for a type or tuple of types."""
    if isinstance(tp, tuple):
        return " | ".join(t.__name__ for t in tp)
    return tp.__name__


def _check_field(
    row: dict[str, Any],
    index: int,
    field: str,
    *,
    expected_type: type | tuple[type, ...],
    required: bool,
    non_empty: bool,
) -> Iterator[ValidationError]:
    """Yield validation errors for a single field in a row."""
    if field not in row:
        if required:
            yield ValidationError(
                index=index,
                field_path=field,
                message=f"字段缺失",
                expected_type=_type_name(expected_type),
                hint=f"请确保数据加载器为每条记录提供 {field} 字段",
            )
        return

    value = row[field]
    if value is None:
        if required:
            yield ValidationError(
                index=index,
                field_path=field,
                message="字段值为 None",
                expected_type=_type_name(expected_type),
                hint=f"请确保 {field} 字段不为 None",
            )
        return

    if not isinstance(value, expected_type):
        yield ValidationError(
            index=index,
            field_path=field,
            message=f"类型错误",
            expected_type=_type_name(expected_type),
            actual_type=type(value).__name__,
            hint=f"请将 {field} 字段改为 {_type_name(expected_type)} 类型",
        )
        return

    if non_empty and isinstance(value, str | list) and len(value) == 0:
        yield ValidationError(
            index=index,
            field_path=field,
            message="值为空",
            expected_type=_type_name(expected_type),
            hint=f"请确保 {field} 字段非空",
        )


# ---------------------------------------------------------------------------
# Per-mode validators
# ---------------------------------------------------------------------------


def _validate_sft_row(row: dict[str, Any], index: int) -> Iterator[ValidationError]:
    """Validate one SFT row: requires ``prompt`` and ``response`` as non-empty strings."""
    yield from _check_field(row, index, "prompt", expected_type=str, required=True, non_empty=True)
    yield from _check_field(row, index, "response", expected_type=str, required=True, non_empty=True)


def _validate_dpo_row(row: dict[str, Any], index: int) -> Iterator[ValidationError]:
    """Validate one DPO row.

    Two formats are accepted:
      A) ``prompt`` + ``chosen`` + ``rejected`` (prompt/response style)
      B) ``chosen`` + ``rejected`` as full chat message lists

    ``chosen`` and ``rejected`` must have the same type (both str or both list).
    """
    yield from _check_field(row, index, "chosen", expected_type=(str, list), required=True, non_empty=True)
    yield from _check_field(row, index, "rejected", expected_type=(str, list), required=True, non_empty=True)

    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if chosen is not None and rejected is not None and type(chosen) is not type(rejected):
        yield ValidationError(
            index=index,
            field_path="chosen/rejected",
            message="chosen 和 rejected 类型不一致",
            expected_type=type(chosen).__name__,
            actual_type=type(rejected).__name__,
            hint="chosen 和 rejected 必须同为 str 或同为 list",
        )

    # Format A: prompt is optional but must be str or list if present
    if "prompt" in row:
        yield from _check_field(row, index, "prompt", expected_type=(str, list), required=False, non_empty=False)


def _validate_online_rl_row(row: Any, index: int) -> Iterator[ValidationError]:
    """Validate one online RL row: the row itself must be a dict.

    Online RL (GSPO/GRPO/PPO) datasets return flat raw rows consumed by
    reward functions. There is no fixed field schema beyond the row being
    a dict so that reward functions can read task-specific fields.
    """
    if not isinstance(row, dict):
        yield ValidationError(
            index=index,
            field_path="(row)",
            message="行必须是 dict 类型",
            expected_type="dict",
            actual_type=type(row).__name__,
            hint="在线 RL 数据集每行应返回一个 dict，包含 reward 函数所需的字段",
        )


def _validate_agentic_row(row: Any, index: int) -> Iterator[ValidationError]:
    """Validate one agentic row: same as online RL (row must be a dict)."""
    yield from _validate_online_rl_row(row, index)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

_VALIDATORS: dict[str, Any] = {
    "sft": _validate_sft_row,
    "dpo": _validate_dpo_row,
    "gspo": _validate_online_rl_row,
    "grpo": _validate_online_rl_row,
    "ppo": _validate_online_rl_row,
    "agentic": _validate_agentic_row,
}

_MODE_LABELS: dict[str, str] = {
    "sft": "SFT",
    "dpo": "DPO",
    "gspo": "在线 RL (GSPO)",
    "grpo": "在线 RL (GRPO)",
    "ppo": "在线 RL (PPO)",
    "agentic": "Agentic",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_dataset(
    dataset,
    algo: str,
    *,
    max_errors: int = 20,
    agent_fn: str | None = None,
) -> ValidationResult:
    """Validate a dataset against the schema for ``algo``.

    Args:
        dataset: A HuggingFace Dataset, list of dicts, or any iterable of rows.
        algo: Training algorithm name (``"sft"``, ``"dpo"``, ``"gspo"``,
              ``"grpo"``, ``"ppo"``).  When ``agent_fn`` is set the mode
              is treated as ``"agentic"``.
        max_errors: Maximum number of errors to collect before stopping.
        agent_fn: Optional path to an agent function; if set, the mode
                  switches to ``"agentic"``.

    Returns:
        A :class:`ValidationResult` with aggregated errors and warnings.
    """
    mode = "agentic" if agent_fn is not None else algo
    validator = _VALIDATORS.get(mode)
    if validator is None:
        raise ValueError(
            f"不支持的数据校验模式: {mode!r}。"
            f"支持的模式: {', '.join(sorted(_VALIDATORS))}"
        )

    errors: list[ValidationError] = []
    total_rows = 0

    for index, row in enumerate(dataset):
        total_rows += 1
        # Online RL / agentic validators check the raw row directly (it must
        # be a dict).  SFT / DPO validators expect a dict so we coerce
        # non-dict rows (e.g. HuggingFace Dataset items) first.
        if mode in {"gspo", "grpo", "ppo", "agentic"}:
            row_to_check = row
        else:
            row_to_check = dict(row) if not isinstance(row, dict) else row
        for err in validator(row_to_check, index):
            errors.append(err)
            if len(errors) >= max_errors:
                break
        if len(errors) >= max_errors:
            break

    valid_rows = total_rows - len({e.index for e in errors})
    return ValidationResult(
        mode=mode,
        total_rows=total_rows,
        valid_rows=max(valid_rows, 0),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_validation_result(result: ValidationResult) -> str:
    """Format a :class:`ValidationResult` as a human-readable string.

    Args:
        result: The validation result to format.

    Returns:
        A multi-line string suitable for display.
    """
    mode_label = _MODE_LABELS.get(result.mode, result.mode)
    lines = [f"数据集校验结果（{mode_label} 模式）"]
    lines.append(f"  总行数: {result.total_rows}")
    lines.append(f"  有效行: {result.valid_rows}")

    if result.errors:
        lines.append(f"  错误数: {len(result.errors)}（最多显示 {len(result.errors)} 条）")
        lines.append("")
        lines.append("  错误详情:")
        for err in result.errors:
            lines.append(f"    [行 {err.index}] {err.field_path}: {err.message}")
            if err.expected_type:
                type_info = f"，期望类型 {err.expected_type}"
                if err.actual_type:
                    type_info += f"，实际 {err.actual_type}"
                lines.append(f"      {type_info}")
            if err.hint:
                lines.append(f"      提示: {err.hint}")
        lines.append("")
        lines.append("  校验未通过。请修复上述问题后重试。")
    else:
        lines.append("  校验通过。")

    return "\n".join(lines)


def format_validation_result_json(result: ValidationResult) -> str:
    """Format a :class:`ValidationResult` as a JSON string.

    Returns:
        A compact JSON string (single line).
    """
    return json.dumps(
        {
            "mode": result.mode,
            "total_rows": result.total_rows,
            "valid_rows": result.valid_rows,
            "errors": [
                {
                    "index": e.index,
                    "field_path": e.field_path,
                    "message": e.message,
                    "expected_type": e.expected_type,
                    "actual_type": e.actual_type,
                    "hint": e.hint,
                }
                for e in result.errors
            ],
        },
        ensure_ascii=False,
    )