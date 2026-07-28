"""Optional dashboard integration helpers and reward-component analysis.

The first helper, :func:`record_dashboard_state`, keeps the trainer loops
decoupled from the dashboard feature. The rest of this module is a focused,
read-only analysis layer for multi-component reward artifacts: a pure
:class:`RewardComponentAnalyzer` plus a loader for the local
``reward_components.<pid>.jsonl`` artifact format. It deliberately does not
touch the reward-function contract, trainer config, or trainer call sites —
the artifact "producer" is a separate concern.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any


def record_dashboard_state(instance: Any, **kwargs: Any) -> None:
    """Record dashboard state when the backend supports it.

    Unit-test fakes and third-party backend-like objects do not need to
    implement dashboard reporting. Keeping this optional avoids coupling the
    core trainer loops to the dashboard feature.
    """

    recorder = getattr(instance, "record_dashboard_state", None)
    if recorder is not None:
        recorder(**kwargs)


# Small numerical guard so a zero-variance component never divides by zero.
_EPS = 1e-8

#: Defaults mirror the CLI flags in ``areno/cli/reward_analysis.py``.
DEFAULT_HISTORY_LIMIT = 200
DEFAULT_OUTLIER_Z = 3.0
DISTRIBUTION_BUCKETS = 10
STEP_DRILLDOWN_LIMIT = 200

_REWARD_COMPONENTS_GLOB = "reward_components.*.jsonl"


def _json_number(value: float | None) -> float | None:
    """Pass through finite floats; drop NaN/inf to ``None`` for JSON output."""

    if value is None:
        return None
    return value if math.isfinite(value) else None


class _ComponentAccum:
    """Welford accumulators + bounded history for one reward component.

    The accumulators keep O(1) state per component; only the bounded history
    deque grows (capped at ``history_limit``). Raw per-sample rewards are never
    stored, so large batches aggregate without unbounded memory.
    """

    def __init__(self, name: str, history_limit: int):
        self.name = name
        self._count = 0  # finite observations
        self._mean = 0.0
        self._m2 = 0.0
        self._sum = 0.0
        self._min: float | None = None
        self._max: float | None = None
        self._zero_count = 0
        self._non_finite_count = 0
        self._missing_count = 0
        self._last_finite: float | None = None
        self._history: deque[dict[str, int | float]] = deque(maxlen=history_limit)

    def observe_value(self, step: int, value: float | None) -> None:
        """Fold one observation into the accumulator.

        ``None`` or a non-finite value is recorded but kept out of the
        mean/variance/zero/outlier statistics so they stay honest.
        """

        if value is None:
            self._missing_count += 1
            return
        if not math.isfinite(value):
            self._non_finite_count += 1
            return
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (value - self._mean)
        self._sum += value
        if self._min is None or value < self._min:
            self._min = value
        if self._max is None or value > self._max:
            self._max = value
        if value == 0.0:
            self._zero_count += 1
        self._last_finite = value
        self._history.append({"step": step, "value": value})

    def mark_missing(self) -> None:
        """Record that the component was absent for a step it should have had."""

        self._missing_count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean if self._count else 0.0

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        variance = self._m2 / (self._count - 1)
        return math.sqrt(variance) if variance > 0 else 0.0

    @property
    def finite_sum(self) -> float:
        return self._sum


def _histogram(values: list[float], lo: float | None, hi: float | None) -> dict[str, Any]:
    """Build a fixed-bucket distribution over bounded finite history values."""

    count = len(values)
    if count == 0:
        return {"buckets": [], "count": 0}
    if lo is None:
        lo = min(values)
    if hi is None:
        hi = max(values)
    if lo == hi:
        return {"buckets": [{"lo": lo, "hi": hi, "count": count}], "count": count}
    width = (hi - lo) / DISTRIBUTION_BUCKETS
    counts = [0] * DISTRIBUTION_BUCKETS
    for value in values:
        index = int((value - lo) / width)
        index = 0 if index < 0 else min(index, DISTRIBUTION_BUCKETS - 1)
        counts[index] += 1
    buckets = [
        {"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": counts[i]} for i in range(DISTRIBUTION_BUCKETS)
    ]
    return {"buckets": buckets, "count": count}


class RewardComponentAnalyzer:
    """Aggregate multi-component reward time series into a bounded snapshot.

    Input is a stream of per-step component dicts. Components may appear
    dynamically (a name first seen at step *k* is not "missing" before *k*) and
    individual values may be ``None`` (present-but-unknown) or non-finite
    (``NaN``/``inf``). Missing and non-finite observations are tracked
    separately and never treated as zero, so they cannot bias the mean or the
    zero/outlier fractions.
    """

    def __init__(
        self,
        *,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        outlier_z: float = DEFAULT_OUTLIER_Z,
    ):
        if not isinstance(history_limit, int) or isinstance(history_limit, bool) or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        if not isinstance(outlier_z, (int, float)) or isinstance(outlier_z, bool):
            raise ValueError("outlier_z must be a number")
        if not math.isfinite(float(outlier_z)) or float(outlier_z) <= 0:
            raise ValueError("outlier_z must be a positive finite number")
        self._history_limit = history_limit
        self._outlier_z = float(outlier_z)
        self._accums: dict[str, _ComponentAccum] = {}
        self._total_finite_sum = 0.0
        self._all_component_finite_sum = 0.0
        self._steps: deque[dict[str, Any]] = deque(maxlen=STEP_DRILLDOWN_LIMIT)

    def _accum(self, name: str) -> _ComponentAccum:
        accum = self._accums.get(name)
        if accum is None:
            accum = _ComponentAccum(name, self._history_limit)
            self._accums[name] = accum
        return accum

    def update(
        self,
        step: int,
        components: dict[str, float | None],
        total: float | None = None,
    ) -> None:
        """Fold one step's components into the aggregation.

        ``total`` defaults to the sum of the step's finite component values;
        a non-finite explicit total falls back to that same sum.
        """

        if not isinstance(components, dict):
            raise TypeError("components must be a dict[str, float | None]")
        # A component introduced earlier but absent this step is "missing".
        for name, accum in self._accums.items():
            if name not in components:
                accum.mark_missing()
        step_non_finite: list[str] = []
        step_missing: list[str] = []
        component_sum = 0.0
        for name, value in components.items():
            accum = self._accum(str(name))
            if value is None:
                accum.mark_missing()
                step_missing.append(str(name))
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                accum.mark_missing()
                step_missing.append(str(name))
                continue
            if not math.isfinite(numeric):
                accum.observe_value(step, numeric)
                step_non_finite.append(str(name))
                continue
            accum.observe_value(step, numeric)
            component_sum += numeric
        total_value: float | None
        if total is None:
            total_value = component_sum
        else:
            try:
                total_value = float(total)
            except (TypeError, ValueError):
                total_value = component_sum
            if not math.isfinite(total_value):
                total_value = component_sum
        if math.isfinite(total_value):
            self._total_finite_sum += total_value
        self._all_component_finite_sum += component_sum
        self._steps.append(
            {
                "step": step,
                "total": total_value,
                "components": {str(k): v for k, v in components.items()},
                "non_finite": step_non_finite,
                "missing": step_missing,
            }
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable aggregate snapshot."""

        components: list[dict[str, Any]] = []
        for name, accum in self._accums.items():
            present = accum.count + accum._non_finite_count
            mean = accum.mean
            std = accum.std
            history = list(accum._history)
            distribution = _histogram([h["value"] for h in history], accum._min, accum._max)
            outlier_count = 0
            if std > 0 and accum.count > 0:
                threshold = self._outlier_z
                for item in history:
                    if abs(item["value"] - mean) / (std + _EPS) > threshold:
                        outlier_count += 1
            finite_sum = accum.finite_sum
            weighted = finite_sum / self._total_finite_sum if self._total_finite_sum != 0 else 0.0
            contribution = finite_sum / self._all_component_finite_sum if self._all_component_finite_sum != 0 else 0.0
            components.append(
                {
                    "name": name,
                    "current": accum._last_finite,
                    "count": accum.count,
                    "present_count": present,
                    "missing_count": accum._missing_count,
                    "non_finite_count": accum._non_finite_count,
                    "zero_fraction": accum._zero_count / present if present else 0.0,
                    "non_finite_fraction": accum._non_finite_count / present if present else 0.0,
                    "outlier_fraction": outlier_count / accum.count if accum.count else 0.0,
                    "mean": mean,
                    "std": std,
                    "min": accum._min,
                    "max": accum._max,
                    "weighted_contribution": weighted,
                    "contribution_fraction": contribution,
                    "history": history,
                    "distribution": distribution,
                }
            )
        steps: list[dict[str, Any]] = []
        for record in self._steps:
            steps.append(
                {
                    "step": record["step"],
                    "total": _json_number(record["total"]),
                    "components": {
                        name: _json_number(value) if isinstance(value, float) else value
                        for name, value in record["components"].items()
                    },
                    "non_finite": list(record["non_finite"]),
                    "missing": list(record["missing"]),
                }
            )
        return {
            "components": components,
            "steps": steps,
            "history_limit": self._history_limit,
            "outlier_z": self._outlier_z,
            "total_finite_sum": self._total_finite_sum,
        }


def load_reward_component_steps(
    metrics_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read reward-component rows from a metrics directory.

    Reuses AReno's existing ``{name, value, step}`` JSONL metric convention —
    the same row schema ``areno.dashboard.server`` already ingests generically
    — scoped to ``reward_components.<pid>.jsonl`` files (naming mirrors
    ``rollout_samples.<pid>.jsonl``). Each row names one component for one
    step; rows are grouped by step before analysis.

    Returns ``(steps, errors)``. Each step is ``{"step": int, "components":
    dict[str, float | None]}`` (``total`` is left to the analyzer, which
    defaults it to the sum of a step's finite components). Malformed rows are
    skipped and reported in ``errors`` with the offending file, line number, and
    step/component name only — never prompt or completion text from samples.
    """

    root = Path(metrics_dir)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(root.glob(_REWARD_COMPONENTS_GLOB))
    else:
        return [], [
            {
                "stage": "artifact resolution",
                "message": f"metrics directory not found: {root}",
                "input": str(root),
            }
        ]

    grouped: dict[int, dict[str, float | None]] = {}
    errors: list[dict[str, Any]] = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append({"stage": "artifact read", "file": file.name, "message": str(exc)})
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "stage": "artifact parse",
                        "file": file.name,
                        "line": lineno,
                        "message": f"invalid json: {exc.msg}",
                    }
                )
                continue
            if not isinstance(obj, dict):
                errors.append(
                    {
                        "stage": "artifact parse",
                        "file": file.name,
                        "line": lineno,
                        "message": "expected a json object",
                    }
                )
                continue
            step = obj.get("step")
            name = obj.get("name")
            value = obj.get("value")
            if not isinstance(step, int) or isinstance(step, bool) or not isinstance(name, str) or not name:
                errors.append(
                    {
                        "stage": "artifact parse",
                        "file": file.name,
                        "line": lineno,
                        "step": step if isinstance(step, int) and not isinstance(step, bool) else None,
                        "message": "expected int `step` and non-empty `name`",
                    }
                )
                continue
            if value is None:
                grouped.setdefault(step, {})[name] = None
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(
                    {
                        "stage": "artifact parse",
                        "file": file.name,
                        "line": lineno,
                        "step": step,
                        "component": name,
                        "message": "non-numeric component value treated as missing",
                    }
                )
                grouped.setdefault(step, {})[name] = None
                continue
            grouped.setdefault(step, {})[name] = numeric
    steps = [{"step": step, "components": components} for step, components in sorted(grouped.items())]
    return steps, errors


def analyze_reward_components(
    metrics_dir: str | Path,
    *,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    outlier_z: float = DEFAULT_OUTLIER_Z,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and analyze a metrics directory in one call.

    Returns ``(snapshot, errors)`` so callers (CLI, dashboard route) share one
    validation + aggregation path.
    """

    steps, errors = load_reward_component_steps(metrics_dir)
    analyzer = RewardComponentAnalyzer(history_limit=history_limit, outlier_z=outlier_z)
    for step in steps:
        analyzer.update(step["step"], step["components"])
    return analyzer.snapshot(), errors
