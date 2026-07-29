"""CLI command for querying TensorBoard metric history from local run artifacts.

Reads scalar tags and values from TensorBoard event files without starting a
training run or loading a model.  When no metric name is given, lists all
available scalar tags.  When a name is given, prints recent values, min/max,
last value, and a compact sparkline trend.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import click


@click.command(name="metrics", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--log-dir", required=True, help="Directory containing TensorBoard event files.")
@click.option("--name", default=None, help="Metric tag to query. Omit to list available tags.")
@click.option("--limit", default=50, help="Maximum number of recent points to display.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON output.")
def metrics_command(log_dir: str, name: str | None, limit: int, as_json: bool) -> None:
    """Query TensorBoard scalar metric history from a local run directory."""

    log_path = Path(log_dir)
    if not log_path.exists():
        click.echo(f"Error: log directory does not exist: {log_dir}", err=True)
        raise SystemExit(1)

    tags = _collect_scalar_tags(log_path)

    if name is None:
        if not tags:
            click.echo("No TensorBoard scalar tags found in the given directory.")
            return
        if as_json:
            click.echo(json.dumps({"tags": sorted(tags)}, indent=2))
        else:
            click.echo("Available metric tags:")
            for tag in sorted(tags):
                click.echo(f"  {tag}")
        return

    if name not in tags:
        click.echo(f"Error: metric '{name}' not found.", err=True)
        if tags:
            click.echo("Available tags:", err=True)
            for tag in sorted(tags):
                click.echo(f"  {tag}", err=True)
        else:
            click.echo("No scalar tags found in the given directory.", err=True)
        raise SystemExit(1)

    points = _read_scalar_series(log_path, name, limit=limit)
    if not points:
        click.echo(f"Metric '{name}' has no valid data points.")
        return

    summary = _summarize(name, points)
    if as_json:
        click.echo(json.dumps(summary, indent=2, default=str))
    else:
        _print_summary(summary)


def _collect_scalar_tags(log_dir: Path) -> set[str]:
    """Return all scalar tags found across TensorBoard event files in *log_dir*."""

    tags: set[str] = set()
    for accumulator in _iter_accumulators(log_dir):
        try:
            tags.update(accumulator.Tags().get("scalars", []))
        except Exception:
            continue
    return tags


def _read_scalar_series(log_dir: Path, tag: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Read scalar events for *tag*, returning sorted point dicts.

    Each dict has ``step`` (int), ``value`` (float), and ``wall_time`` (float).
    NaN/Inf values are skipped.  Results are sorted by step and truncated to
    *limit* most recent points.
    """

    raw: list[tuple[int, float, float]] = []
    for accumulator in _iter_accumulators(log_dir):
        try:
            events = accumulator.Scalars(tag)
        except Exception:
            continue
        for event in events:
            value = float(event.value)
            if math.isnan(value) or math.isinf(value):
                continue
            raw.append((int(event.step), value, float(event.wall_time)))

    raw.sort(key=lambda row: row[0])
    # Deduplicate by step, keeping the last occurrence.
    by_step: dict[int, tuple[int, float, float]] = {}
    for step, value, wall_time in raw:
        by_step[step] = (step, value, wall_time)
    points = [{"step": step, "value": value, "wall_time": wall_time} for step, value, wall_time in by_step.values()]
    return points[-max(1, min(limit, 10000)) :]


def _summarize(name: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a summary dict for *name* from *points*."""

    values = [float(p["value"]) for p in points]
    steps = [int(p["step"]) for p in points]
    return {
        "name": name,
        "count": len(points),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "min_value": min(values) if values else None,
        "max_value": max(values) if values else None,
        "last_value": values[-1] if values else None,
        "mean_value": sum(values) / len(values) if values else None,
        "sparkline": _sparkline(values, width=40),
        "recent": points[-20:],
    }


def _sparkline(values: list[float], *, width: int = 40) -> str:
    """Render a compact ASCII sparkline for *values*."""

    if not values:
        return ""
    width = max(1, width)
    if len(values) > width and width > 1:
        indices = [round(i * (len(values) - 1) / (width - 1)) for i in range(width)]
        sampled = [values[i] for i in indices]
    else:
        sampled = values
    lo = min(sampled)
    hi = max(sampled)
    if hi == lo:
        return "_" * len(sampled)
    levels = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    chars = []
    for v in sampled:
        normalized = (v - lo) / (hi - lo)
        idx = min(int(normalized * (len(levels) - 1)), len(levels) - 1)
        chars.append(levels[idx])
    return "".join(chars)


def _print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable metric summary."""

    click.echo(f"Metric: {summary['name']}")
    click.echo(f"  Points:    {summary['count']}")
    click.echo(f"  Steps:     {summary['first_step']} -> {summary['last_step']}")
    click.echo(f"  Min:       {_fmt(summary['min_value'])}")
    click.echo(f"  Max:       {_fmt(summary['max_value'])}")
    click.echo(f"  Last:      {_fmt(summary['last_value'])}")
    click.echo(f"  Mean:      {_fmt(summary['mean_value'])}")
    click.echo(f"  Trend:     {summary['sparkline']}")
    recent = summary.get("recent", [])
    if recent:
        click.echo("  Recent:")
        for point in recent:
            step = point["step"]
            value = _fmt(point["value"])
            click.echo(f"    step {step:>6}: {value}")


def _fmt(value: float | None) -> str:
    """Format a numeric value for display."""

    if value is None:
        return "n/a"
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.4e}"
    return f"{value:.6g}"


def _iter_accumulators(log_dir: Path):
    """Yield EventAccumulator instances for each event file in *log_dir*."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return

    event_files = sorted(log_dir.rglob("events.out.tfevents.*"), key=lambda item: item.stat().st_mtime)
    if event_files:
        for event_file in event_files:
            try:
                accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 10000})
                accumulator.Reload()
                yield accumulator
            except Exception:
                continue
    else:
        try:
            accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 10000})
            accumulator.Reload()
            yield accumulator
        except Exception:
            return
