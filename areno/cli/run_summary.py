"""Structured terminal summary printed when a training run ends.

This module provides ``format_run_summary``, a pure formatting function that
turns run-end data (outcome, duration, metrics, sample counts, errors) into
either a human-readable text block or a JSON string.  Trainers call it from
the ``finally`` block of ``fit()`` so the summary appears on success,
interruption, *and* failure without replacing the original traceback.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO


# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------

class RunSummaryData:
    """Plain container for the data needed to print a run-end summary.

    Trainers populate this object during ``_fit_initialized`` and hand it to
    ``print_run_summary`` in the ``finally`` block of ``fit``.
    """

    __slots__ = (
        "algo",
        "model",
        "outcome",
        "duration_s",
        "final_step",
        "final_epoch",
        "metrics",
        "samples_processed",
        "samples_trained",
        "samples_skipped",
        "errors",
    )

    def __init__(self, *, algo: str = "", model: str = "") -> None:
        self.algo: str = algo
        self.model: str = model
        self.outcome: str = "success"
        self.duration_s: float = 0.0
        self.final_step: int = 0
        self.final_epoch: int = 0
        self.metrics: dict[str, float] = {}
        self.samples_processed: int = 0
        self.samples_trained: int = 0
        self.samples_skipped: int = 0
        self.errors: list[str] = []


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Render a duration in a compact human-readable form."""
    if seconds < 0:
        seconds = 0.0
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {sec:02d}s"


def _format_float(value: float) -> str:
    """Format a float with up to 6 significant digits, trimming trailing zeros."""
    if value == 0:
        return "0"
    s = f"{value:.6g}"
    return s


def _metric_lines(metrics: dict[str, float]) -> list[str]:
    """Turn a metrics dict into aligned ``key: value`` lines."""
    if not metrics:
        return ["  (no metrics recorded)"]
    lines: list[str] = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            lines.append(f"  {key:<24s}  {_format_float(float(value))}")
        else:
            lines.append(f"  {key:<24s}  {value}")
    return lines


def _outcome_label(outcome: str) -> str:
    return {
        "success": "success",
        "interrupted": "interrupted",
        "error": "error",
    }.get(outcome, outcome)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_run_summary(
    data: RunSummaryData,
    *,
    json_output: bool = False,
) -> str:
    """Format a run-end summary as human-readable text or JSON.

    Parameters
    ----------
    data:
        Populated :class:`RunSummaryData` instance.
    json_output:
        When *True*, return a JSON string suitable for machine parsing.
    """
    if json_output:
        return json.dumps(
            {
                "outcome": data.outcome,
                "duration_s": round(data.duration_s, 3),
                "algo": data.algo,
                "model": data.model,
                "final_step": data.final_step,
                "final_epoch": data.final_epoch,
                "metrics": data.metrics,
                "samples": {
                    "processed": data.samples_processed,
                    "trained": data.samples_trained,
                    "skipped": data.samples_skipped,
                },
                "errors": data.errors,
            },
            ensure_ascii=False,
            indent=2,
        )

    # --- human-readable text ---
    lines: list[str] = []
    width = 44
    lines.append("=" * width)
    lines.append("  AReno Training Summary")
    lines.append("-" * width)
    lines.append(f"  Outcome:    {_outcome_label(data.outcome)}")
    lines.append(f"  Duration:   {_format_duration(data.duration_s)}")
    if data.algo:
        lines.append(f"  Algorithm:  {data.algo}")
    if data.model:
        # Truncate long model paths.
        model_display = data.model if len(data.model) <= 28 else "..." + data.model[-25:]
        lines.append(f"  Model:      {model_display}")
    lines.append(
        f"  Steps: {data.final_step}  |  Epochs: {data.final_epoch}"
    )
    lines.append("-" * width)
    lines.append(
        f"  Samples: {data.samples_trained} trained, "
        f"{data.samples_skipped} skipped, "
        f"{data.samples_processed} processed"
    )
    lines.append("-" * width)
    lines.append("  Metrics:")
    lines.extend(_metric_lines(data.metrics))
    if data.errors:
        lines.append("-" * width)
        lines.append("  Errors (bounded):")
        for err in data.errors[:5]:
            err_display = err if len(err) <= 38 else err[:35] + "..."
            lines.append(f"    - {err_display}")
    lines.append("=" * width)
    return "\n".join(lines)


def print_run_summary(
    data: RunSummaryData,
    *,
    enabled: bool = True,
    json_output: bool = False,
    stream: TextIO | None = None,
) -> None:
    """Print the run-end summary to *stream* (defaults to ``sys.stderr``).

    The summary is deliberately written to stderr so it does not pollute
    stdout when the CLI output is piped.

    Parameters
    ----------
    enabled:
        When *False*, do nothing.  This corresponds to ``--no-summary``.
    json_output:
        When *True*, emit JSON instead of human-readable text.
    stream:
        Output stream; defaults to ``sys.stderr``.
    """
    if not enabled:
        return
    text = format_run_summary(data, json_output=json_output)
    out = stream if stream is not None else sys.stderr
    print(text, file=out, flush=True)