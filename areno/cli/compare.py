"""CLI command for comparing two AReno training runs."""

from __future__ import annotations

import json as _json
import os
from pathlib import Path
from typing import Any

import click

from areno.cli.dashboard_registry import GLOBAL_REGISTRY_FILE


def _load_registered_jobs() -> list[dict[str, Any]]:
    """Load registered jobs from the global registry file."""
    try:
        data = _json.loads(GLOBAL_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    jobs = data.get("jobs") if isinstance(data, dict) else None
    return jobs if isinstance(jobs, list) else []


def _load_state_jobs(state_file: Path) -> list[dict[str, Any]]:
    """Load persisted jobs from the dashboard state file."""
    if not state_file.exists():
        return []
    try:
        data = _json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data.get("jobs", []) if isinstance(data, dict) else []


def _find_job_candidates() -> list[dict[str, Any]]:
    """Collect job candidates from both the registry and the local state file."""
    seen: dict[str, dict[str, Any]] = {}

    for item in _load_registered_jobs():
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("id", item.get("pid", "")))
        if job_id and job_id not in seen:
            seen[job_id] = item

    state_file = Path(os.environ.get("ARENO_DASHBOARD_ROOT", Path.cwd())) / ".areno-dashboard-state.json"
    for item in _load_state_jobs(state_file):
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("id", ""))
        if job_id and job_id not in seen:
            seen[job_id] = item

    return list(seen.values())


def _format_comparison_human(result: dict[str, Any]) -> str:
    """Format comparison result as human-readable text."""
    lines: list[str] = []
    job_a = result.get("job_a", {})
    job_b = result.get("job_b", {})

    lines.append(f"Job A: {job_a.get('name', '?')} (id={job_a.get('id', '?')}, status={job_a.get('status', '?')}, step={job_a.get('step', 0)})")
    lines.append(f"Job B: {job_b.get('name', '?')} (id={job_b.get('id', '?')}, status={job_b.get('status', '?')}, step={job_b.get('step', 0)})")
    lines.append("")

    if not result.get("comparable"):
        lines.append(f"Not comparable: {result.get('reason', 'unknown')}")
        return "\n".join(lines)

    # Config differences.
    config = result.get("config", {})
    different = config.get("different", [])
    identical = config.get("identical", [])
    lines.append(f"Config: {len(different)} changed, {len(identical)} identical")
    if different:
        lines.append("  Changed settings:")
        for item in different:
            va = item.get("value_a")
            vb = item.get("value_b")
            note = item.get("note")
            line = f"    {item['key']}: A={va!r}  B={vb!r}"
            if note:
                line += f"  ({note})"
            lines.append(line)
    lines.append("")

    # Metrics.
    metrics = result.get("metrics", [])
    if metrics:
        lines.append(f"Metrics ({len(metrics)}):")
        for m in metrics:
            va = m.get("value_a")
            vb = m.get("value_b")
            diff = m.get("diff")
            note = m.get("note")
            line = f"    {m['name']}: A={va}  B={vb}"
            if diff is not None:
                line += f"  diff={diff:+}"
            if note:
                line += f"  ({note})"
            lines.append(line)
    else:
        lines.append("Metrics: none")
    lines.append("")

    # Timing.
    timing = result.get("timing", {})
    ta = timing.get("job_a", {})
    tb = timing.get("job_b", {})
    comp = timing.get("comparison", {})
    lines.append("Timing:")
    lines.append(f"    Steps: A={ta.get('total_steps', 0)}  B={tb.get('total_steps', 0)}")
    lines.append(f"    Avg total/step: A={ta.get('avg_total_s', '?')}s  B={tb.get('avg_total_s', '?')}s")
    lines.append(f"    Avg rollout/step: A={ta.get('avg_rollout_s', '?')}s  B={tb.get('avg_rollout_s', '?')}s")
    lines.append(f"    Avg train/step: A={ta.get('avg_train_s', '?')}s  B={tb.get('avg_train_s', '?')}s")
    if comp.get("avg_total_diff_s") is not None:
        lines.append(f"    Step time diff: {comp['avg_total_diff_s']:+}s")
    if comp.get("note"):
        lines.append(f"    Note: {comp['note']}")

    return "\n".join(lines)


@click.command("compare")
@click.option("--job-a", "job_a_id", required=True, help="First job ID to compare.")
@click.option("--job-b", "job_b_id", required=True, help="Second job ID to compare.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["human", "json"]),
    default="human",
    show_default=True,
    help="Output format: human-readable text or structured JSON.",
)
@click.option(
    "--dashboard-url",
    default="http://127.0.0.1:8765",
    show_default=True,
    help="Dashboard server URL (used when available).",
)
def compare_command(job_a_id: str, job_b_id: str, output_format: str, dashboard_url: str) -> None:
    """Compare two AReno training runs side by side.

    Fetches job data from the local dashboard server if running, otherwise
    reads from local artifact files.  Outputs human-readable text by default
    or structured JSON with --format json.
    """

    # Try the dashboard API first.
    result = _try_dashboard_api(dashboard_url, job_a_id, job_b_id)
    if result is not None:
        _emit_result(result, output_format)
        return

    # Fallback: read from local state files and compare directly.
    click.echo("dashboard server not reachable; reading local artifacts...", err=True)

    from areno.dashboard.server import DashboardState

    state = DashboardState()
    try:
        result = state.compare_jobs(job_a_id, job_b_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_result(result, output_format)


def _try_dashboard_api(base_url: str, job_a_id: str, job_b_id: str) -> dict[str, Any] | None:
    """Attempt to fetch comparison from the running dashboard API."""
    import urllib.parse
    import urllib.request

    url = f"{base_url.rstrip('/')}/api/compare?job_a={urllib.parse.quote(job_a_id)}&job_b={urllib.parse.quote(job_b_id)}"
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                return _json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return None


def _emit_result(result: dict[str, Any], output_format: str) -> None:
    """Emit comparison result in the requested format."""
    if output_format == "json":
        click.echo(_json.dumps(result, ensure_ascii=False, indent=2))
    else:
        click.echo(_format_comparison_human(result))