"""Post-loader data contract validation for AReno training modes.

This module validates dataset records produced by user loader functions
against per-mode field contracts *before* expensive model or worker
initialization.  Contracts cover SFT, DPO, online RL (GSPO/GRPO/PPO), and
agentic inputs.

The validator aggregates a bounded set of errors instead of stopping at the
first, so users see all fixable issues in one pass.  Error messages carry
``sample_index``, ``field_path`` (e.g. ``messages[2].role``), ``expected``,
``actual``, and a concrete ``hint`` — without exposing full training samples.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractError:
    """One field-level contract violation found in a dataset record."""

    sample_index: int
    field_path: str
    expected: str
    actual: str
    hint: str


@dataclass(frozen=True, slots=True)
class ContractReport:
    """Aggregated result of validating a dataset against a contract."""

    mode: str
    total_scanned: int
    errors: list[ContractError] = field(default_factory=list)
    warnings: list[ContractError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` when no errors were collected."""

        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON / dashboard consumption."""

        return {
            "mode": self.mode,
            "total_scanned": self.total_scanned,
            "ok": self.ok,
            "errors": [
                {
                    "sample_index": e.sample_index,
                    "field_path": e.field_path,
                    "expected": e.expected,
                    "actual": e.actual,
                    "hint": e.hint,
                }
                for e in self.errors
            ],
            "warnings": [
                {
                    "sample_index": e.sample_index,
                    "field_path": e.field_path,
                    "expected": e.expected,
                    "actual": e.actual,
                    "hint": e.hint,
                }
                for e in self.warnings
            ],
        }


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Declarative specification for one top-level record field."""

    name: str
    required: bool
    expected_type: type | tuple[type, ...]
    element_type: type | tuple[type, ...] | None = None
    min_length: int | None = None
    nullable: bool = False


# ---------------------------------------------------------------------------
# Type-name helpers
# ---------------------------------------------------------------------------

_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _type_name(value: Any) -> str:
    """Return a short human-readable type name for *value*."""

    if value is None:
        return "NoneType"
    return type(value).__name__


def _expected_type_name(spec: FieldSpec) -> str:
    """Return the expected type label shown in error messages."""

    types = spec.expected_type if isinstance(spec.expected_type, tuple) else (spec.expected_type,)
    names = [t.__name__ for t in types]
    base = " | ".join(names) if len(names) > 1 else names[0]
    if spec.element_type is not None:
        elem_types = (
            spec.element_type if isinstance(spec.element_type, tuple) else (spec.element_type,)
        )
        elem_names = [t.__name__ for t in elem_types]
        base += f"[{' | '.join(elem_names)}]"
    return base


# ---------------------------------------------------------------------------
# Per-mode contract specs
# ---------------------------------------------------------------------------

_MESSAGES_FIELD_SPECS: list[FieldSpec] = [
    FieldSpec("messages", required=True, expected_type=list, element_type=dict),
]

_PROMPT_FIELD_SPECS: list[FieldSpec] = [
    FieldSpec("prompt", required=True, expected_type=str),
]

_SFT_PROMPT_RESPONSE_SPECS: list[FieldSpec] = [
    FieldSpec("prompt", required=True, expected_type=str),
    FieldSpec("response", required=True, expected_type=str),
]

_DPO_STR_SPECS: list[FieldSpec] = [
    FieldSpec("prompt", required=True, expected_type=str),
    FieldSpec("chosen", required=True, expected_type=str),
    FieldSpec("rejected", required=True, expected_type=str),
]

_DPO_LIST_SPECS: list[FieldSpec] = [
    FieldSpec("chosen", required=True, expected_type=list, element_type=dict),
    FieldSpec("rejected", required=True, expected_type=list, element_type=dict),
    FieldSpec("prompt", required=False, expected_type=str),
]

_ONLINE_RL_SPECS: list[FieldSpec] = [
    FieldSpec("prompt", required=True, expected_type=str),
    FieldSpec("solutions", required=False, expected_type=list, element_type=str, nullable=True),
]


def _sniff_sft_variant(record: dict[str, Any]) -> str:
    """Detect whether an SFT row uses ``messages`` or ``prompt``+``response``."""

    if isinstance(record.get("messages"), list):
        return "messages"
    return "prompt_response"


def _sniff_dpo_variant(record: dict[str, Any]) -> str:
    """Detect whether a DPO row uses string or chat-list ``chosen``/``rejected``."""

    chosen = record.get("chosen")
    if isinstance(chosen, list):
        return "list"
    return "str"


def _sniff_agentic_variant(record: dict[str, Any]) -> str:
    """Detect whether an agentic row uses ``messages`` or ``prompt``."""

    if isinstance(record.get("messages"), list):
        return "messages"
    return "prompt"


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Full contract for one training mode."""

    mode: str
    fields: list[FieldSpec]
    sniff_variant: Callable[[dict[str, Any]], str] | None = None
    variants: dict[str, list[FieldSpec]] | None = None

    def fields_for_record(self, record: dict[str, Any]) -> list[FieldSpec]:
        """Return the effective field list for *record*, resolving variants."""

        if self.sniff_variant is not None and self.variants is not None:
            variant = self.sniff_variant(record)
            return self.variants[variant]
        return self.fields


def _build_contract_specs() -> dict[str, ContractSpec]:
    """Build the built-in per-mode contract registry."""

    return {
        "sft": ContractSpec(
            mode="sft",
            fields=[],
            sniff_variant=_sniff_sft_variant,
            variants={
                "prompt_response": _SFT_PROMPT_RESPONSE_SPECS,
                "messages": _MESSAGES_FIELD_SPECS,
            },
        ),
        "dpo": ContractSpec(
            mode="dpo",
            fields=[],
            sniff_variant=_sniff_dpo_variant,
            variants={
                "str": _DPO_STR_SPECS,
                "list": _DPO_LIST_SPECS,
            },
        ),
        "online_rl": ContractSpec(
            mode="online_rl",
            fields=_ONLINE_RL_SPECS,
        ),
        "agentic": ContractSpec(
            mode="agentic",
            fields=[],
            sniff_variant=_sniff_agentic_variant,
            variants={
                "prompt": _PROMPT_FIELD_SPECS,
                "messages": _MESSAGES_FIELD_SPECS,
            },
        ),
    }


_CONTRACT_SPECS = _build_contract_specs()


# ---------------------------------------------------------------------------
# Nested message validation
# ---------------------------------------------------------------------------


def _validate_messages(
    record: dict[str, Any], index: int, errors: list[ContractError]
) -> None:
    """Validate the nested structure of an OpenAI-style ``messages`` list."""

    messages = record.get("messages")
    if not isinstance(messages, list):
        return  # top-level type check already handled by FieldSpec
    for i, msg in enumerate(messages):
        prefix = f"messages[{i}]"
        if not isinstance(msg, dict):
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=prefix,
                    expected="dict",
                    actual=_type_name(msg),
                    hint="each message must be a dict with 'role' and 'content' keys",
                )
            )
            continue
        role = msg.get("role")
        if not isinstance(role, str):
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{prefix}.role",
                    expected="str",
                    actual=_type_name(role) if "role" in msg else "missing",
                    hint="add a 'role' field with one of: system, user, assistant, tool",
                )
            )
        elif role not in _VALID_ROLES:
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{prefix}.role",
                    expected=f"one of {sorted(_VALID_ROLES)}",
                    actual=role,
                    hint=f"use a valid role: {', '.join(sorted(_VALID_ROLES))}",
                )
            )
        if "content" not in msg:
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{prefix}.content",
                    expected="str",
                    actual="missing",
                    hint="add a 'content' field (may be empty string for tool-call messages)",
                )
            )
        elif msg["content"] is not None and not isinstance(msg["content"], str):
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{prefix}.content",
                    expected="str",
                    actual=_type_name(msg["content"]),
                    hint="'content' must be a string or null",
                )
            )
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            _validate_tool_calls(tool_calls, index, prefix, errors)


def _validate_tool_calls(
    tool_calls: Any, index: int, prefix: str, errors: list[ContractError]
) -> None:
    """Validate the nested structure of ``tool_calls`` in an assistant message."""

    if not isinstance(tool_calls, list):
        errors.append(
            ContractError(
                sample_index=index,
                field_path=f"{prefix}.tool_calls",
                expected="list[dict]",
                actual=_type_name(tool_calls),
                hint="'tool_calls' must be a list of tool-call dicts",
            )
        )
        return
    for j, call in enumerate(tool_calls):
        call_prefix = f"{prefix}.tool_calls[{j}]"
        if not isinstance(call, dict):
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=call_prefix,
                    expected="dict",
                    actual=_type_name(call),
                    hint="each tool_call must be a dict with a 'function' key",
                )
            )
            continue
        func = call.get("function")
        if not isinstance(func, dict):
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{call_prefix}.function",
                    expected="dict",
                    actual=_type_name(func) if "function" in call else "missing",
                    hint="add a 'function' dict with 'name' and 'arguments' keys",
                )
            )
            continue
        if not isinstance(func.get("name"), str):
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{call_prefix}.function.name",
                    expected="str",
                    actual=_type_name(func.get("name")) if "name" in func else "missing",
                    hint="add a 'name' string identifying the tool",
                )
            )
        if "arguments" not in func:
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{call_prefix}.function.arguments",
                    expected="str | dict",
                    actual="missing",
                    hint="add an 'arguments' field (JSON string or dict)",
                )
            )


# ---------------------------------------------------------------------------
# Core field validation
# ---------------------------------------------------------------------------


def _validate_field(
    record: dict[str, Any],
    spec: FieldSpec,
    index: int,
    errors: list[ContractError],
    warnings: list[ContractError],
) -> None:
    """Validate one field spec against a record, appending to *errors*/*warnings*."""

    value = record.get(spec.name, _MISSING)
    if value is _MISSING:
        if spec.required:
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=spec.name,
                    expected=_expected_type_name(spec),
                    actual="missing",
                    hint=f"add a '{spec.name}' field to the dataset record",
                )
            )
        return
    if value is None:
        if not spec.nullable:
            if spec.required:
                errors.append(
                    ContractError(
                        sample_index=index,
                        field_path=spec.name,
                        expected=_expected_type_name(spec),
                        actual="NoneType",
                        hint=f"'{spec.name}' must not be null",
                    )
                )
            else:
                warnings.append(
                    ContractError(
                        sample_index=index,
                        field_path=spec.name,
                        expected=_expected_type_name(spec),
                        actual="NoneType",
                        hint=f"optional field '{spec.name}' is null; it will be ignored",
                    )
                )
        return
    if not isinstance(value, spec.expected_type):
        errors.append(
            ContractError(
                sample_index=index,
                field_path=spec.name,
                expected=_expected_type_name(spec),
                actual=_type_name(value),
                hint=f"'{spec.name}' must be of type {_expected_type_name(spec)}",
            )
        )
        return
    if spec.element_type is not None and isinstance(value, list):
        _validate_list_elements(value, spec, index, errors)
    if spec.min_length is not None and isinstance(value, (str, list)) and len(value) < spec.min_length:
        errors.append(
            ContractError(
                sample_index=index,
                field_path=spec.name,
                expected=f"{_expected_type_name(spec)} with length >= {spec.min_length}",
                actual=f"length {len(value)}",
                hint=f"'{spec.name}' must have at least {spec.min_length} element(s)",
            )
        )


_MISSING = object()


def _validate_list_elements(
    value: list, spec: FieldSpec, index: int, errors: list[ContractError]
) -> None:
    """Check that every element of a list field has the expected element type."""

    for i, item in enumerate(value):
        if not isinstance(item, spec.element_type):
            errors.append(
                ContractError(
                    sample_index=index,
                    field_path=f"{spec.name}[{i}]",
                    expected=_element_type_name(spec),
                    actual=_type_name(item),
                    hint=f"element {i} of '{spec.name}' has wrong type",
                )
            )


def _element_type_name(spec: FieldSpec) -> str:
    """Return the element type label for list field specs."""

    if spec.element_type is None:
        return "any"
    types = spec.element_type if isinstance(spec.element_type, tuple) else (spec.element_type,)
    names = [t.__name__ for t in types]
    return " | ".join(names)


def _validate_record(
    record: Any,
    spec: ContractSpec,
    index: int,
    errors: list[ContractError],
    warnings: list[ContractError],
) -> None:
    """Validate one dataset record against the contract."""

    if not isinstance(record, dict):
        errors.append(
            ContractError(
                sample_index=index,
                field_path="<root>",
                expected="dict",
                actual=_type_name(record),
                hint="each dataset row must be a dict; normalize rows in --dataset-loader-fn",
            )
        )
        return

    field_specs = spec.fields_for_record(record)
    for fs in field_specs:
        _validate_field(record, fs, index, errors, warnings)

    # Nested message validation for modes that use ``messages``
    if "messages" in record and isinstance(record["messages"], list):
        _validate_messages(record, index, errors)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_CONTRACT_MODES = ("sft", "dpo", "online_rl", "agentic")


def get_contract_spec(mode: str) -> ContractSpec:
    """Return the ``ContractSpec`` registered for *mode*."""

    key = mode.strip().lower()
    if key not in _CONTRACT_SPECS:
        raise ValueError(
            f"unknown contract mode {mode!r}; supported modes: {', '.join(_CONTRACT_MODES)}"
        )
    return _CONTRACT_SPECS[key]


def list_contract_modes() -> tuple[str, ...]:
    """Return the supported contract mode names."""

    return _CONTRACT_MODES


def validate_contract(
    dataset: Iterable[Any],
    *,
    mode: str,
    max_samples: int = 100,
    max_errors: int = 20,
) -> ContractReport:
    """Validate *dataset* records against the *mode* data contract.

    Scans up to ``max_samples`` records and collects up to ``max_errors``
    errors before stopping.  The returned :class:`ContractReport` aggregates
    all found violations with sample indices, field paths, expected/actual
    types, and concrete fix hints — without exposing original values.
    """

    spec = get_contract_spec(mode)
    errors: list[ContractError] = []
    warnings: list[ContractError] = []
    scanned = 0
    truncated = False

    for index, record in enumerate(dataset):
        if scanned >= max_samples:
            break
        scanned += 1
        pre_error_count = len(errors)
        _validate_record(record, spec, index, errors, warnings)
        if len(errors) >= max_errors and len(errors) > pre_error_count:
            truncated = True
            break

    if truncated:
        warnings.append(
            ContractError(
                sample_index=-1,
                field_path="<report>",
                expected=f"<= {max_errors} errors",
                actual=f">= {len(errors)} errors",
                hint=f"error list truncated at {max_errors}; fix the shown errors and re-run to see more",
            )
        )

    return ContractReport(
        mode=spec.mode,
        total_scanned=scanned,
        errors=errors,
        warnings=warnings,
    )


__all__ = [
    "ContractError",
    "ContractReport",
    "ContractSpec",
    "FieldSpec",
    "get_contract_spec",
    "list_contract_modes",
    "validate_contract",
]
