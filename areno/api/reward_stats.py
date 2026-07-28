"""Reward distribution statistics for terminal summarisation.

This module provides pure functions to compute summary statistics (mean,
std, min/max, zero fraction, missing fraction, outlier fraction) for both
total reward and individual named reward components.  The functions are
deliberately free of I/O so they can be unit-tested without any files.

The typical call chain is::

    samples = load_reward_samples(metrics_log_dir)
    report   = compute_component_statistics(samples, outlier_threshold=3.0)
    text     = format_reward_table(report)      # human-readable
    json_str = format_reward_json(report)       # machine-readable
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from io import StringIO

import numpy as np


@dataclass
class RewardStatistics:
    """Summary statistics for a single reward stream."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    zero_fraction: float
    missing_fraction: float
    outlier_fraction: float


@dataclass
class RewardSummaryReport:
    """Aggregated statistics for total reward and named components."""

    total: RewardStatistics
    components: dict[str, RewardStatistics] = field(default_factory=dict)
    sample_count: int = 0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

_MISSING = None  # sentinel: value absent (component did not appear for sample)
_NON_FINITE = {float("inf"), float("-inf")}


def _is_missing(value: float | None) -> bool:
    """Return True when a value is missing (None or NaN)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _is_non_finite(value: float | None) -> bool:
    """Return True when a value is inf or -inf (but not NaN/None)."""
    return value in _NON_FINITE


def compute_reward_statistics(
    values: list[float | None],
    *,
    outlier_threshold: float = 3.0,
) -> RewardStatistics:
    """Compute summary statistics for a list of reward values.

    ``None`` and ``NaN`` entries are treated as *missing* (counted in
    ``missing_fraction``) and excluded from mean/std/min/max.  ``inf`` and
    ``-inf`` are treated as *non-finite* — they are counted in
    ``missing_fraction`` as well (since they are not usable for training)
    but are also tracked separately via the non-finite check so callers can
    distinguish if needed.

    ``outlier_fraction`` counts the fraction of *finite* values whose
    absolute deviation from the mean exceeds ``outlier_threshold * std``.
    """
    total = len(values)
    if total == 0:
        return RewardStatistics(
            count=0, mean=0.0, std=0.0, min=0.0, max=0.0,
            zero_fraction=0.0, missing_fraction=0.0, outlier_fraction=0.0,
        )

    missing_count = sum(1 for v in values if _is_missing(v) or _is_non_finite(v))
    finite_values = np.array(
        [v for v in values if not _is_missing(v) and not _is_non_finite(v)],
        dtype=np.float64,
    )
    finite_count = len(finite_values)

    if finite_count == 0:
        return RewardStatistics(
            count=total, mean=0.0, std=0.0, min=0.0, max=0.0,
            zero_fraction=0.0, missing_fraction=missing_count / total,
            outlier_fraction=0.0,
        )

    mean = float(finite_values.mean())
    std = float(finite_values.std())

    zero_count = int(np.sum(finite_values == 0.0))
    if std > 0:
        outlier_count = int(np.sum(np.abs(finite_values - mean) > outlier_threshold * std))
    else:
        outlier_count = 0

    return RewardStatistics(
        count=total,
        mean=mean,
        std=std,
        min=float(finite_values.min()),
        max=float(finite_values.max()),
        zero_fraction=zero_count / total,
        missing_fraction=missing_count / total,
        outlier_fraction=outlier_count / total,
    )


def compute_component_statistics(
    samples: list[dict],
    *,
    outlier_threshold: float = 3.0,
) -> RewardSummaryReport:
    """Build a :class:`RewardSummaryReport` from raw JSONL sample dicts.

    Each *sample* dict is expected to carry a ``reward`` key (float) and an
    optional ``reward_components`` key (``dict[str, float] | None``).

    Components that appear in some samples but not others are treated as
    *missing* for the samples where they are absent — this distinguishes
    "component not produced" (missing) from "component value is 0" (zero).
    """
    total_rewards: list[float | None] = []
    component_values: dict[str, list[float | None]] = {}

    for sample in samples:
        reward = sample.get("reward")
        if reward is not None and not (isinstance(reward, float) and math.isnan(reward)):
            total_rewards.append(float(reward))
        else:
            total_rewards.append(None)

        components = sample.get("reward_components")
        if components:
            for name, value in components.items():
                component_values.setdefault(name, [None] * len(total_rewards))
                # Backfill previous samples where this component didn't exist.
                while len(component_values[name]) < len(total_rewards) - 1:
                    component_values[name].append(None)
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    component_values[name].append(float(value))
                else:
                    component_values[name].append(None)

    # Ensure all component lists have the same length as total_rewards.
    for name in component_values:
        while len(component_values[name]) < len(total_rewards):
            component_values[name].append(None)

    total_stats = compute_reward_statistics(total_rewards, outlier_threshold=outlier_threshold)
    component_stats: dict[str, RewardStatistics] = {}
    for name, values in component_values.items():
        component_stats[name] = compute_reward_statistics(values, outlier_threshold=outlier_threshold)

    return RewardSummaryReport(
        total=total_stats,
        components=component_stats,
        sample_count=len(samples),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_reward_table(report: RewardSummaryReport, *, use_color: bool = True) -> str:
    """Render a human-readable table from a :class:`RewardSummaryReport`.

    Uses ``rich`` if available; falls back to a plain-text table otherwise.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Reward Distribution Summary", show_lines=False)
        table.add_column("Component", style="cyan" if use_color else "", no_wrap=True)
        table.add_column("Count", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Zero%", justify="right")
        table.add_column("Missing%", justify="right")
        table.add_column("Outlier%", justify="right")

        def _row(label: str, stats: RewardStatistics) -> None:
            table.add_row(
                label,
                str(stats.count),
                f"{stats.mean:.4f}",
                f"{stats.std:.4f}",
                f"{stats.min:.4f}",
                f"{stats.max:.4f}",
                f"{stats.zero_fraction:.2%}",
                f"{stats.missing_fraction:.2%}",
                f"{stats.outlier_fraction:.2%}",
            )

        _row("total", report.total)
        for name in sorted(report.components):
            _row(name, report.components[name])

        console = Console(file=StringIO(), force_terminal=use_color, width=120)
        console.print(table)
        console.print(f"Samples: {report.sample_count}")
        return console.file.getvalue()
    except ImportError:
        return _format_reward_table_plain(report)


def _format_reward_table_plain(report: RewardSummaryReport) -> str:
    """Fallback plain-text table renderer (no rich dependency)."""
    header = (
        f"{'Component':<20} {'Count':>6} {'Mean':>10} {'Std':>10} "
        f"{'Min':>10} {'Max':>10} {'Zero%':>8} {'Missing%':>9} {'Outlier%':>9}"
    )
    lines = [header, "-" * len(header)]

    def _row(label: str, stats: RewardStatistics) -> None:
        lines.append(
            f"{label:<20} {stats.count:>6} {stats.mean:>10.4f} {stats.std:>10.4f} "
            f"{stats.min:>10.4f} {stats.max:>10.4f} {stats.zero_fraction:>7.2%} "
            f"{stats.missing_fraction:>8.2%} {stats.outlier_fraction:>8.2%}"
        )

    _row("total", report.total)
    for name in sorted(report.components):
        _row(name, report.components[name])
    lines.append(f"\nSamples: {report.sample_count}")
    return "\n".join(lines) + "\n"


def format_reward_json(report: RewardSummaryReport) -> str:
    """Render a :class:`RewardSummaryReport` as a JSON string."""
    payload = {
        "total": asdict(report.total),
        "components": {name: asdict(stats) for name, stats in sorted(report.components.items())},
        "sample_count": report.sample_count,
    }
    return json.dumps(payload, indent=2, sort_keys=True)