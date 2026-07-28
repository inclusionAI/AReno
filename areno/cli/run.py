"""List and inspect AReno training and serving runs from the terminal."""

from __future__ import annotations

import datetime as dt
import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from areno.dashboard.server import (
    TIME_SEGMENT_ORDER,
    dashboard_state_source,
    pid_is_running,
    registered_job_items,
    rollout_sample_sources,
    run_config_sources,
    tensorboard_event_sources,
    tensorboard_time_segment_name,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Aggregated information for a single AReno run."""

    run_id: str = ""
    kind: str = ""
    name: str = ""
    pid: int | None = None
    status: str = "unknown"
    stage: str = ""
    step: int = 0
    created_at: str = ""
    updated_at: str = ""
    metrics_dir: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    config_text: str = ""
    metrics: list[dict[str, Any]] = field(default_factory=list)
    timeperf: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    returncode: int | None = None
    command: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_status(entry: dict[str, Any]) -> str:
    """Determine the effective status of a registered job."""
    pid = entry.get("pid")
    if pid is not None and pid_is_running(pid):
        return "running"
    rc = entry.get("returncode")
    if rc is not None:
        return "succeeded" if rc == 0 else "failed"
    return "exited"


def _format_timestamp(value: Any) -> str:
    """Convert a float epoch or ISO string to a readable timestamp."""
    if value is None or value == "":
        return ""
    try:
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).replace(microsecond=0).isoformat()
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _summary_from_item(item: dict[str, Any]) -> RunSummary:
    """Build a RunSummary from a registry entry."""
    return RunSummary(
        run_id=item.get("id", ""),
        kind=item.get("kind", ""),
        name=item.get("name", ""),
        pid=item.get("pid"),
        status=_resolve_status(item),
        returncode=item.get("returncode"),
        created_at=_format_timestamp(item.get("created_at")),
        updated_at=_format_timestamp(item.get("updated_at")),
        metrics_dir=item.get("metrics_dir"),
        config=dict(item.get("config") or {}),
        command=list(item.get("command") or []),
    )


def list_runs() -> list[RunSummary]:
    """List all registered runs from the dashboard registry."""
    return [_summary_from_item(item) for item in registered_job_items()]


def get_run(run_id: str) -> RunSummary | None:
    """Get detailed information for a specific run."""
    for item in registered_job_items():
        if item.get("id") == run_id:
            summary = _summary_from_item(item)
            _load_run_metrics(summary)
            return summary
    return None


def _load_run_metrics(summary: RunSummary) -> None:
    """Load metrics, samples, state, and config from the metrics directory."""
    if not summary.metrics_dir:
        return
    path = Path(summary.metrics_dir)
    if not path.exists() or not path.is_dir():
        return
    pid = summary.pid
    _load_dashboard_state(summary, path, pid)
    _load_tensorboard_scalars(summary, path, pid)
    _load_rollout_samples(summary, path, pid)
    _load_run_config(summary, path, pid)


def _load_dashboard_state(summary: RunSummary, path: Path, pid: int | None) -> None:
    """Load live stage/step/status from dashboard_state.{pid}.json."""
    state_file = dashboard_state_source(path, pid)
    if state_file is None:
        return
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    stage = payload.get("stage")
    if isinstance(stage, str) and stage:
        summary.stage = stage
    try:
        summary.step = max(summary.step, int(payload.get("step", summary.step)))
    except (TypeError, ValueError):
        pass
    status = payload.get("status")
    if isinstance(status, str) and summary.status not in {"stopped", "failed", "succeeded", "exited"}:
        summary.status = status
    summary.updated_at = _format_timestamp(payload.get("updated_at")) or summary.updated_at


def _load_tensorboard_scalars(summary: RunSummary, path: Path, pid: int | None) -> None:
    """Load scalar metrics and time performance from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return
    by_step: dict[int, dict[str, float]] = {}
    for event_path in tensorboard_event_sources(path, pid):
        try:
            accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 10000})
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
        except Exception:
            continue
        for tag in tags:
            try:
                events = accumulator.Scalars(tag)[-500:]
            except Exception:
                continue
            for event in events:
                step = int(event.step)
                value = float(event.value)
                if math.isnan(value):
                    continue
                summary.metrics.append({"name": tag, "value": value, "step": step})
                summary.step = max(summary.step, step)
                time_name = tensorboard_time_segment_name(tag)
                if time_name:
                    by_step.setdefault(step, {})[time_name] = value
                if tag in {"train/step_e2e_time_s", "time/total", "time/e2e"}:
                    by_step.setdefault(step, {})["total"] = value
                elif tag == "train/step_rollout_time_s":
                    by_step.setdefault(step, {})["rollout"] = value
                elif tag in {"train/step_train_time_s", "train/policy_train_wall_time_s"}:
                    by_step.setdefault(step, {})["train"] = value
    for step, values in sorted(by_step.items()):
        total = values.pop("total", None)
        if total is None:
            total = sum(v for v in values.values() if v > 0)
        if total <= 0:
            continue
        ordered = sorted(
            [{"name": name, "seconds": max(v, 0.0)} for name, v in values.items() if v > 0],
            key=lambda item: (
                TIME_SEGMENT_ORDER.index(item["name"])
                if item["name"] in TIME_SEGMENT_ORDER
                else len(TIME_SEGMENT_ORDER)
            ),
        )
        accounted = sum(item["seconds"] for item in ordered)
        if total > accounted:
            ordered.append({"name": "other", "seconds": total - accounted})
        summary.timeperf.append({"step": step, "segments": ordered, "total_s": total})


def _load_rollout_samples(summary: RunSummary, path: Path, pid: int | None) -> None:
    """Load rollout samples from JSONL files."""
    for sample_file in rollout_sample_sources(path, pid):
        try:
            lines = sample_file.read_text(encoding="utf-8").splitlines()[-100:]
        except Exception:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            summary.samples.append(item)


def _load_run_config(summary: RunSummary, path: Path, pid: int | None) -> None:
    """Load run configuration from areno_run_config.{pid}.txt/.json."""
    text_file, json_file = run_config_sources(path, pid)
    if text_file.exists():
        try:
            summary.config_text = text_file.read_text(encoding="utf-8")
        except Exception:
            pass
    if not json_file.exists():
        return
    try:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(payload, dict):
        settings = payload.get("settings")
        if isinstance(settings, dict):
            summary.config = settings
        if not summary.config_text and isinstance(payload.get("summary_text"), str):
            summary.config_text = payload["summary_text"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def collect_metric_summaries(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group metrics by name, return count/latest_step/latest_value per group."""
    grouped: dict[str, dict[str, Any]] = {}
    for point in metrics:
        name = str(point.get("name") or "")
        if not name:
            continue
        step = int(point.get("step") or 0)
        value = point.get("value")
        current = grouped.get(name)
        if current is None:
            grouped[name] = {"name": name, "count": 1, "latest_step": step, "latest_value": value}
            continue
        current["count"] += 1
        if step >= int(current.get("latest_step") or 0):
            current["latest_step"] = step
            current["latest_value"] = value
    return sorted(grouped.values(), key=lambda item: item["name"])


def collect_timeperf_summary(timeperf: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Aggregate average time spent in each RL phase across all steps."""
    if not timeperf:
        return []
    totals: dict[str, list[float]] = {}
    for row in timeperf:
        segments = row.get("segments")
        if not isinstance(segments, list):
            continue
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            name = str(seg.get("name", ""))
            if not name:
                continue
            try:
                totals.setdefault(name, []).append(float(seg.get("seconds", 0)))
            except (TypeError, ValueError):
                continue
    result = [(name, sum(values) / len(values)) for name, values in totals.items() if values]
    order = {name: i for i, name in enumerate(TIME_SEGMENT_ORDER)}
    result.sort(key=lambda item: order.get(item[0], len(order)))
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_run_info(run: RunSummary) -> str:
    """Format a RunSummary into a structured terminal-friendly string."""
    lines: list[str] = []

    # Section 1: Basic Info
    lines.append(f"Run: {run.run_id}")
    lines.append(f"  Kind:        {run.kind}")
    lines.append(f"  Name:        {run.name}")
    status_line = f"  Status:      {run.status}"
    if run.returncode is not None:
        status_line += f" (returncode={run.returncode})"
    lines.append(status_line)
    lines.append(f"  Stage:       {run.stage or '-'}")
    lines.append(f"  Step:        {run.step}")
    lines.append(f"  PID:         {run.pid or '-'}")
    lines.append(f"  Created:     {run.created_at or '-'}")
    lines.append(f"  Updated:     {run.updated_at or '-'}")
    lines.append(f"  Metrics Dir: {run.metrics_dir or '-'}")

    # Section 2: Configuration
    if run.config_text:
        lines.append("")
        lines.append("Configuration:")
        lines.append(textwrap.indent(run.config_text, "  "))
    elif run.config:
        lines.append("")
        lines.append("Configuration:")
        for key, val in sorted(run.config.items()):
            lines.append(f"  {key}: {val}")

    # Section 3: Metric Summaries
    summaries = collect_metric_summaries(run.metrics)
    if summaries:
        lines.append("")
        lines.append("Metrics Summary:")
        lines.append(f"  {'Name':<40} {'Latest':>12} {'Step':>8} {'Count':>6}")
        lines.append(f"  {'-' * 40} {'-' * 12} {'-' * 8} {'-' * 6}")
        for m in summaries:
            latest = m.get("latest_value")
            latest_str = f"{latest:>12.6f}" if isinstance(latest, (int, float)) else f"{str(latest):>12}"
            lines.append(
                f"  {m['name']:<40} {latest_str} "
                f"{int(m.get('latest_step') or 0):>8} {int(m.get('count') or 0):>6}"
            )

    # Section 4: Time Performance
    time_summary = collect_timeperf_summary(run.timeperf)
    if time_summary:
        lines.append("")
        lines.append("Time Breakdown (avg per step):")
        for name, seconds in time_summary:
            lines.append(f"  {name:<25} {seconds:>8.2f}s")

    # Section 5: Recent Rollout Samples
    if run.samples:
        lines.append("")
        lines.append(f"Recent Rollout Samples (last {min(len(run.samples), 5)}):")
        for s in run.samples[-5:]:
            reward = s.get("reward", "?")
            resp_len = s.get("response_len", "?")
            lines.append(f"  step={s.get('step', '?')} reward={reward} resp_len={resp_len}")

    return "\n".join(lines)


def format_run_list(runs: list[RunSummary]) -> str:
    """Format a compact table of all runs."""
    if not runs:
        return "No runs found."
    lines: list[str] = []
    header = f"  {'ID':<12} {'Kind':<6} {'Status':<12} {'Step':>6} {'Name':<40} {'Created'}"
    lines.append(header)
    lines.append(f"  {'-' * 12} {'-' * 6} {'-' * 12} {'-' * 6} {'-' * 40} {'-' * 20}")
    for r in runs:
        lines.append(
            f"  {r.run_id:<12} {r.kind:<6} {r.status:<12} {r.step:>6} "
            f"{r.name[:40]:<40} {r.created_at[:19]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group(
    name="run",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def run_command() -> None:
    """List and inspect AReno training and serving runs."""


@run_command.command("list")
def list_command() -> None:
    """List active and recent AReno runs."""
    runs = list_runs()
    if not runs:
        click.echo("No runs found.")
        return
    click.echo(format_run_list(runs))


@run_command.command("info")
@click.argument("run_id")
def info_command(run_id: str) -> None:
    """Show detailed information for one run."""
    run = get_run(run_id)
    if run is None:
        click.echo(f"Run not found: {run_id}", err=True)
        raise click.Abort()
    click.echo(format_run_info(run))