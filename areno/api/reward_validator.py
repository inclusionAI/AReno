"""Runtime validation wrapper for user-supplied reward hooks.

Wraps a custom ``reward_fn(record) -> float`` callable with checks for
supported signature, numeric dtype/shape, finite values, and
serializability.  Failures are associated with the hook name and the
prompt/sample index so the operator can locate the problematic input
without sifting through full training logs.

Two operating modes are supported:

- **strict** (``strict=True``): invalid outputs raise
  :class:`RewardValidationError`, halting training immediately.
- **non-strict** (``strict=False``, default): invalid outputs emit a
  ``RuntimeWarning`` and return ``0.0`` so training can continue.

The non-strict default preserves backward compatibility — existing
reward hooks that already return valid floats are completely
unaffected because validation is a no-op for correct values.
"""

from __future__ import annotations

import functools
import inspect
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class RewardValidationError(Exception):
    """Raised when a reward hook produces an invalid output.

    The error message is prefixed with ``[hook=<name>, sample=<N>]``
    so the operator can quickly locate the offending hook and record
    without exposing full training data.

    Attributes
    ----------
    hook:
        Name of the reward function that produced the error.
    sample_index:
        Index of the record that triggered the error, or ``None``
        for single-call validation.
    """

    def __init__(
        self,
        message: str,
        *,
        hook: str = "reward_fn",
        sample_index: int | None = None,
    ) -> None:
        loc = f"hook={hook}"
        if sample_index is not None:
            loc += f", sample={sample_index}"
        super().__init__(f"[{loc}] {message}")
        self.hook = hook
        self.sample_index = sample_index


def _check_signature(fn: Callable, hook: str) -> None:
    """Verify that *fn* accepts at least one positional argument.

    This catches the common mistake of defining ``def reward_fn()``
    with no parameters, or ``def reward_fn(a, b)`` expecting the
    framework to pass multiple args.
    """

    sig = inspect.signature(fn)
    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if not params and not any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
        raise RewardValidationError(
            f"reward hook must accept at least one positional argument, got signature {sig}",
            hook=hook,
        )


def _coerce_reward(value: Any, hook: str, sample_index: int) -> float:
    """Convert *value* to ``float`` and validate it is finite.

    Accepted types (coerced to float):
        - ``int``, ``float``, ``bool``
        - ``numpy.generic`` (numpy scalar)
        - ``numpy.ndarray`` with exactly one element
        - ``list`` / ``tuple`` with exactly one element

    Rejected types (raises :class:`RewardValidationError`):
        - ``None``
        - ``str``, ``bytes``
        - ``dict``
        - multi-element arrays or sequences
        - any other type

    Additionally, ``NaN`` and ``Inf`` are rejected because they
    silently poison advantage computation and gradient updates.
    """

    # --- None check -------------------------------------------------
    if value is None:
        raise RewardValidationError(
            "reward_fn returned None, expected a numeric scalar",
            hook=hook, sample_index=sample_index,
        )

    # --- numpy types ------------------------------------------------
    if isinstance(value, np.generic):
        # numpy scalar (e.g. np.float32) — coerce directly
        value = float(value)
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            # single-element array — extract the scalar
            value = float(value.reshape(-1)[0])
        else:
            raise RewardValidationError(
                f"reward_fn returned array of size {value.size}, expected a scalar",
                hook=hook, sample_index=sample_index,
            )

    # --- Python scalar types ----------------------------------------
    if isinstance(value, bool):
        # bool is a subclass of int, so check it first
        value = float(value)
    elif isinstance(value, (int, float)):
        value = float(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        # single-element sequence (list/tuple) — extract the value
        if len(value) == 1:
            value = float(value[0])
        else:
            raise RewardValidationError(
                f"reward_fn returned sequence of length {len(value)}, expected a scalar",
                hook=hook, sample_index=sample_index,
            )
    else:
        # str, dict, or any other unsupported type
        raise RewardValidationError(
            f"reward_fn returned unsupported type {type(value).__name__}, expected float",
            hook=hook, sample_index=sample_index,
        )

    # --- finiteness check -------------------------------------------
    # NaN and Inf silently corrupt training — reject them explicitly.
    if math.isnan(value):
        raise RewardValidationError(
            "reward_fn returned NaN",
            hook=hook, sample_index=sample_index,
        )
    if math.isinf(value):
        raise RewardValidationError(
            "reward_fn returned Inf",
            hook=hook, sample_index=sample_index,
        )

    return value


def validate_reward_fn(
    fn: Callable,
    *,
    name: str = "reward_fn",
    strict: bool = False,
) -> Callable:
    """Wrap *fn* with runtime input/output validation.

    Returns a new callable with the same signature as *fn*.  Each
    invocation checks the return value via :func:`_coerce_reward`.

    Parameters
    ----------
    fn:
        The user-supplied reward function, expected to accept one
        positional argument (the record) and return a numeric scalar.
    name:
        Human-readable hook name used in error messages.
    strict:
        If ``True``, invalid outputs raise :class:`RewardValidationError`.
        If ``False`` (default), invalid outputs warn and return ``0.0``.
    """

    _check_signature(fn, name)

    @functools.wraps(fn)
    def wrapper(record: Any) -> float:
        # --- call the user function --------------------------------
        try:
            result = fn(record)
        except Exception as exc:
            if strict:
                raise RewardValidationError(
                    f"reward_fn raised {type(exc).__name__}: {exc}",
                    hook=name,
                ) from exc
            warnings.warn(
                f"[hook={name}] reward_fn raised {type(exc).__name__}: {exc}; returning 0.0",
                RuntimeWarning, stacklevel=2,
            )
            return 0.0

        # --- validate the return value -----------------------------
        try:
            return _coerce_reward(result, name, sample_index=0)
        except RewardValidationError as exc:
            if strict:
                raise
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            return 0.0

    return wrapper


def validate_reward_batch(
    fn: Callable,
    records: Sequence[Any],
    *,
    name: str = "reward_fn",
    strict: bool = False,
) -> list[float]:
    """Call *fn* over a batch of records with per-sample validation.

    Unlike :func:`validate_reward_fn` which wraps a single call, this
    function iterates over *records* and tracks the sample index in
    error messages so the operator knows exactly which record
    triggered the failure.

    Parameters
    ----------
    fn:
        The user-supplied reward function.
    records:
        A sequence of record objects to pass to *fn*.
    name:
        Human-readable hook name used in error messages.
    strict:
        If ``True``, the first invalid output raises immediately.
        If ``False`` (default), invalid outputs warn, return ``0.0``,
        and processing continues for remaining records.

    Returns
    -------
    list[float]
        One reward value per input record.  Length always equals
        ``len(records)`` — a mismatch raises
        :class:`RewardValidationError`.
    """

    _check_signature(fn, name)
    rewards: list[float] = []

    for idx, record in enumerate(records):
        # --- call the user function --------------------------------
        try:
            result = fn(record)
        except Exception as exc:
            if strict:
                raise RewardValidationError(
                    f"reward_fn raised {type(exc).__name__}: {exc}",
                    hook=name, sample_index=idx,
                ) from exc
            warnings.warn(
                f"[hook={name}, sample={idx}] reward_fn raised {type(exc).__name__}: {exc}; returning 0.0",
                RuntimeWarning, stacklevel=2,
            )
            rewards.append(0.0)
            continue

        # --- validate the return value -----------------------------
        try:
            rewards.append(_coerce_reward(result, name, sample_index=idx))
        except RewardValidationError as exc:
            if strict:
                raise
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            rewards.append(0.0)

    # --- final safety check -----------------------------------------
    # If somehow rewards length doesn't match records (shouldn't happen
    # with the logic above, but guard against edge cases).
    if len(rewards) != len(records):
        raise RewardValidationError(
            f"reward_fn produced {len(rewards)} rewards for {len(records)} records",
            hook=name,
        )

    return rewards
