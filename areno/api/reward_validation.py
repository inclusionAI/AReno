"""Runtime validation for user-supplied reward hooks.

All validation logic is concentrated here to keep :mod:`areno.api.rewards`
focused on reward data structures and loading.  The single public entry
point is :func:`validate_and_wrap_reward_fn`, called by
:func:`areno.api.rewards.load_reward_fn` to wrap every loaded reward
function with input/output checks.
"""

from __future__ import annotations

import inspect
import math
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from areno.api.rewards import RewardRecord, make_reward_record

__all__ = ["RewardValidationError", "validate_and_wrap_reward_fn"]


class RewardValidationError(ValueError):
    """Raised when a reward hook produces an invalid output."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_PARAM_ANNOTATIONS = {"RewardRecord", "Any", "Any | None", "Optional[Any]"}

_VALID_RETURN_ANNOTATIONS = {"float", "int", "Any", "float | None", "Optional[float]"}


def _annotation_name(annotation: Any) -> str | None:
    """Return a human-readable name for a type annotation object.

    Returns ``None`` when the parameter / return value has no annotation.
    """
    if annotation is inspect.Parameter.empty:
        return None
    if annotation is None:
        return "None"
    name = getattr(annotation, "__name__", None)
    if name is not None:
        return name
    # typing constructs fall back to str(annotation)
    return str(annotation)


def _check_signature(fn: Callable, hook_name: str) -> None:
    """Inspect *fn*'s signature and warn on suspicious annotations.

    Raises ``TypeError`` only when the function does not accept exactly
    one positional argument.  Annotation mismatches produce warnings so
    that existing hooks without annotations remain compatible.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) != 1:
        raise TypeError(
            f"reward hook '{hook_name}' must accept exactly 1 positional argument, "
            f"got {len(params)}"
        )
    param = params[0]
    if param.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise TypeError(
            f"reward hook '{hook_name}' first parameter must be positional, "
            f"got {param.kind.name}"
        )

    param_ann = _annotation_name(param.annotation)
    if param_ann is not None and param_ann not in _VALID_PARAM_ANNOTATIONS:
        warnings.warn(
            f"reward hook '{hook_name}' parameter is annotated as '{param_ann}', "
            f"expected 'RewardRecord' or 'Any'; the hook will still be called "
            f"with a RewardRecord instance",
            UserWarning,
            stacklevel=3,
        )

    return_ann = _annotation_name(sig.return_annotation)
    if return_ann is not None and return_ann not in _VALID_RETURN_ANNOTATIONS:
        warnings.warn(
            f"reward hook '{hook_name}' return type is annotated as '{return_ann}', "
            f"expected 'float' or 'int'; the return value will be validated at runtime",
            UserWarning,
            stacklevel=3,
        )


def _validate_scalar_output(value: Any, *, hook_name: str, prompt_preview: str,
                            sample_idx: int | str = "unknown") -> float:
    """Validate a single reward output and return it as ``float``.

    Raises :class:`RewardValidationError` for invalid types or non-finite
    values.
    """
    ctx = f"sample_index: {sample_idx}; prompt: {prompt_preview}"
    # --- torch tensor / numpy scalar (duck-typed via .item()) ---------------
    if hasattr(value, "item") and callable(value.item):
        # Reject multi-dimensional tensors
        ndim = getattr(value, "ndim", None)
        if ndim is not None and ndim > 0:
            raise RewardValidationError(
                f"reward hook '{hook_name}' returned a {ndim}-d tensor, "
                f"expected a scalar; {ctx}"
            )
        try:
            value = float(value.item())
        except (TypeError, ValueError) as exc:
            raise RewardValidationError(
                f"reward hook '{hook_name}' returned a tensor that cannot be "
                f"converted to float; {ctx}"
            ) from exc
    elif value is None:
        raise RewardValidationError(
            f"reward hook '{hook_name}' returned None; {ctx}"
        )
    elif isinstance(value, list):
        raise RewardValidationError(
            f"reward hook '{hook_name}' returned a list of length {len(value)}, "
            f"expected a scalar; reward_fn must return one float per call; {ctx}"
        )
    elif isinstance(value, bool):
        # bool is a subclass of int; accept and convert
        value = float(value)
    elif isinstance(value, (int, float)):
        value = float(value)
    else:
        raise RewardValidationError(
            f"reward hook '{hook_name}' returned non-numeric value of type "
            f"{type(value).__name__}; {ctx}"
        )

    if not math.isfinite(value):
        raise RewardValidationError(
            f"reward hook '{hook_name}' returned non-finite value: {value}; {ctx}"
        )

    return value


def _truncate(text: str, limit: int = 100) -> str:
    """Truncate *text* to *limit* characters for error messages."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _wrap(fn: Callable, *, hook_name: str) -> Callable[[RewardRecord], float]:
    """Return a validated wrapper around *fn*."""

    def validated_reward_fn(record: RewardRecord) -> float:
        prompt_preview = _truncate(getattr(record, "prompt", ""))
        metadata = getattr(record, "metadata", {}) or {}
        sample_idx = metadata.get("sample_index", "unknown")
        try:
            result = fn(record)
        except Exception as exc:
            raise RewardValidationError(
                f"reward hook '{hook_name}' raised {type(exc).__name__}: {exc}; "
                f"sample_index: {sample_idx}; prompt: {prompt_preview}"
            ) from exc
        return _validate_scalar_output(
            result, hook_name=hook_name, prompt_preview=prompt_preview,
            sample_idx=sample_idx,
        )

    return validated_reward_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_wrap_reward_fn(
    fn: Callable,
    module_path: Path,
) -> Callable[[RewardRecord], float]:
    """Validate *fn*'s signature, optionally dry-run, then return a wrapper.

    This is the single entry point called by
    :func:`areno.api.rewards.load_reward_fn`.

    Parameters
    ----------
    fn
        The raw ``reward_fn`` callable extracted from the user's module.
    module_path
        Resolved path to the user's reward file, used for the hook name
        and error messages.

    Returns
    -------
    A new callable with the same ``RewardRecord -> float`` contract but
    with runtime output validation applied on every call.
    """
    hook_name = module_path.stem

    # Validation is opt-in (default off) to preserve existing behavior.
    # Set ARENO_REWARD_VALIDATION=1 to enable signature check, dry-run,
    # and per-call output validation.
    if os.environ.get("ARENO_REWARD_VALIDATION", "0") != "1":
        return fn

    # 1. Signature check (warnings only, never fatal).
    _check_signature(fn, hook_name)

    # 2. Dry-run with a minimal RewardRecord (can be disabled).
    #    Many valid reward hooks require specific field values (e.g. record.answer)
    #    and will naturally raise on a minimal mock.  We treat dry-run *call*
    #    failures as warnings, not errors — only an invalid return value from
    #    a successful dry-run is a hard error.
    if os.environ.get("ARENO_REWARD_VALIDATION_DRY_RUN", "1") != "0":
        mock_record = make_reward_record(
            prompt="What is 2+2?",
            completion="4",
            source_record={"answer": "4", "question": "What is 2+2?"},
            answer=["4"],
        )
        try:
            dry_result = fn(mock_record)
        except Exception as exc:
            warnings.warn(
                f"reward hook '{hook_name}' dry-run raised {type(exc).__name__}: "
                f"{exc}; this may be expected if the hook requires specific field "
                f"values (e.g. record.answer); the hook will still be loaded",
                UserWarning,
                stacklevel=2,
            )
        else:
            _validate_scalar_output(
                dry_result, hook_name=hook_name, prompt_preview="",
                sample_idx="dry-run",
            )

    # 3. Return a wrapper that validates every subsequent call.
    return _wrap(fn, hook_name=hook_name)
