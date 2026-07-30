"""CLI command to show detailed information for a single training run."""

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


def _resolve_job(run_id: str) -> dict[str, Any] | None:
    """Find a job by ID, partial ID, or PID. Returns the raw job dict or None."""
    candidates = _find_job_candidates()

    # Exact ID match.
    for item in candidates:
        if item.get("id") == run_id:
            return item

    # Partial ID match (prefix).
    prefix_matches = [item for item in candidates if str(item.get("id", "")).startswith(run_id)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise click.ClickException(f"Ambiguous run ID '{run_id}' matches: {[m.get('id') for m in prefix_matches]}")

    # PID match.
    for item in candidates:
        if str(item.get("pid", "")) == run_id:
            return item

    return None


def _normalize_dashboard_config(config: dict[str, Any]) -> dict[str, Any]:
    """Dashboard API returns config with a 'sections' key for structured display.
    Extract flat key-value pairs from sections for CLI consumption."""
    if not isinstance(config, dict) or "sections" not in config:
        return config
    if not isinstance(config["sections"], list):
        return config
    flat: dict[str, Any] = {}
    for section in config["sections"]:
        if isinstance(section, dict) and "items" in section:
            for item in section["items"]:
                if isinstance(item, dict) and "key" in item:
                    flat[item["key"]] = item.get("value")
    # Merge any non-section keys.
    for k, v in config.items():
        if k != "sections":
            flat[k] = v
    return flat


def _load_job_details(job_item: dict[str, Any]) -> dict[str, Any]:
    """Load full job details from the dashboard server or local artifacts."""
    # Try the dashboard API first.
    result = _try_dashboard_api("http://127.0.0.1:8765", str(job_item.get("id", "")))
    if result is not None:
        return _build_details_from_dashboard(result, job_item, "http://127.0.0.1:8765")

    # Fallback: read from local state file via DashboardState.
    from areno.dashboard.server import DashboardState

    state = DashboardState()
    job_id = job_item.get("id", "")
    job = state.get_job(job_id)
    if job is None:
        return _job_item_to_details(job_item)

    summaries = state.metric_summaries(job_id)
    return {
        "id": job.id,
        "name": job.name,
        "kind": job.kind,
        "status": job.status,
        "stage": job.stage,
        "step": job.step,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "returncode": job.returncode,
        "config": job.config or {},
        "launch_config": job.launch_config or {},
        "config_text": job.config_text or "",
        "metrics_dir": job.metrics_dir or "",
        "metrics": summaries,
        "timeperf": job.timeperf or [],
        "logs": job.logs[-20:] if job.logs else [],
    }


def _build_details_from_dashboard(result: dict[str, Any], job_item: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Build a details dict from dashboard API response, normalizing config and fetching metrics."""
    config = _normalize_dashboard_config(result.get("config") or result.get("launch") or {})
    launch_config = result.get("launch") or config
    if isinstance(launch_config, dict) and "sections" in launch_config:
        launch_config = config

    metric_summaries = _fetch_metric_summaries(base_url, str(job_item.get("id", "")))

    return {
        "id": result.get("id", job_item.get("id", "?")),
        "name": result.get("name", "?"),
        "kind": result.get("kind", "train"),
        "status": result.get("status", "unknown"),
        "stage": result.get("stage", ""),
        "step": result.get("step", 0),
        "created_at": result.get("created_at", ""),
        "updated_at": result.get("updated_at", ""),
        "returncode": result.get("returncode"),
        "config": config,
        "launch_config": launch_config,
        "config_text": result.get("config_text", ""),
        "metrics_dir": result.get("metrics_dir", ""),
        "metrics": metric_summaries,
        "timeperf": result.get("timeperf", []),
        "logs": result.get("logs", []),
    }


def _fetch_metric_summaries(base_url: str, job_id: str) -> list[dict[str, Any]]:
    """Fetch metric summaries from the dashboard API."""
    import urllib.request

    url = f"{base_url.rstrip('/')}/api/jobs/{job_id}/metrics"
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                data = _json.loads(response.read().decode("utf-8"))
                return data.get("metrics", [])
    except Exception:
        return []
    return []


def _job_item_to_details(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw registry/state item to a details dict."""
    config = item.get("config") or item.get("launch") or {}
    return {
        "id": item.get("id", "?"),
        "name": item.get("name", "unknown"),
        "kind": item.get("kind", "train"),
        "status": item.get("status", "unknown"),
        "stage": item.get("stage", ""),
        "step": item.get("step", 0),
        "created_at": item.get("created_at", ""),
        "updated_at": item.get("updated_at", ""),
        "returncode": item.get("returncode"),
        "config": config,
        "launch_config": item.get("launch", config),
        "config_text": item.get("config_text", ""),
        "metrics_dir": item.get("metrics_dir", ""),
        "metrics": [],
        "timeperf": [],
        "logs": [],
    }


def _try_dashboard_api(base_url: str, job_id: str) -> dict[str, Any] | None:
    """Attempt to fetch job details from the running dashboard API."""
    import urllib.request

    url = f"{base_url.rstrip('/')}/api/jobs/{job_id}"
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                data = _json.loads(response.read().decode("utf-8"))
                return data.get("job", data)
    except Exception:
        return None
    return None


# -- Formatting --------------------------------------------------------------

def _format_human(details: dict[str, Any]) -> str:
    """Format job details as human-readable text."""
    lines: list[str] = []

    # Header.
    lines.append(f"Run: {details.get('name', '?')}  (id={details.get('id', '?')})")
    lines.append(f"Kind:     {details.get('kind', '?')}")
    lines.append(f"Status:   {details.get('status', '?')}" +
                 (f"  (exit code {details.get('returncode')})" if details.get('returncode') is not None else ""))
    stage = details.get("stage", "")
    if stage:
        lines.append(f"Stage:    {stage}")
    lines.append(f"Step:     {details.get('step', 0)}")
    lines.append(f"Created:  {details.get('created_at', '?')}")
    lines.append(f"Updated:  {details.get('updated_at', '?')}")
    if details.get("metrics_dir"):
        lines.append(f"Metrics:  {details['metrics_dir']}")
    lines.append("")

    # Config / key settings.
    config = details.get("launch_config") or details.get("config") or {}
    if config:
        lines.append("Key settings:")
        key_order = [
            "algo", "ckpt", "model_hub", "dataset_path", "dataset_loader_fn",
            "world_size", "tp_size", "batch_size", "mini_bs", "n_samples",
            "max_new_tokens", "max_context_len", "max_running_prompts",
            "lr", "min_lr", "lr_decay_style", "weight_decay", "epochs", "max_steps",
            "reward_fn_path", "agent_fn",
        ]
        shown = 0
        for key in key_order:
            val = config.get(key)
            if val is not None and val != "":
                lines.append(f"  {key:<22} {val}")
                shown += 1
        # Show any remaining keys not in the predefined order (skip potential secrets).
        secret_patterns = {"key", "secret", "token", "password", "credential"}
        for key in sorted(config):
            if key in key_order or config[key] is None or config[key] == "":
                continue
            if any(s in key.lower() for s in secret_patterns):
                continue
            lines.append(f"  {key:<22} {config[key]}")
            shown += 1
        if shown == 0:
            lines.append("  (no settings recorded)")
        lines.append("")

    # Metrics.
    metrics = details.get("metrics", [])
    if metrics:
        lines.append(f"Latest metrics ({len(metrics)}):")
        for m in metrics:
            name = m.get("name", "?")
            value = m.get("latest_value", "?")
            step = m.get("latest_step", "?")
            lines.append(f"  {name:<30} {value}  @step {step}")
        lines.append("")

    # Timing.
    timeperf = details.get("timeperf", [])
    if timeperf:
        lines.append(f"Timing ({len(timeperf)} steps recorded):")
        total_vals = [r.get("total_s", 0) for r in timeperf if r.get("total_s")]
        rollout_vals = [r.get("rollout_s", 0) for r in timeperf if r.get("rollout_s")]
        train_vals = [r.get("train_s", 0) for r in timeperf if r.get("train_s")]
        if total_vals:
            avg_total = sum(total_vals) / len(total_vals)
            lines.append(f"  Avg total / step:    {avg_total:.2f}s")
        if rollout_vals:
            avg_rollout = sum(rollout_vals) / len(rollout_vals)
            lines.append(f"  Avg rollout / step:  {avg_rollout:.2f}s")
        if train_vals:
            avg_train = sum(train_vals) / len(train_vals)
            lines.append(f"  Avg train / step:    {avg_train:.2f}s")
        lines.append("")

    # Last error (for failed runs).
    returncode = details.get("returncode")
    logs = details.get("logs", [])
    if returncode is not None and returncode != 0:
        error_lines = [l for l in logs if "error" in l.lower() or "traceback" in l.lower() or "exception" in l.lower()]
        lines.append(f"Last error (exit code {returncode}):")
        if error_lines:
            for el in error_lines[-5:]:
                lines.append(f"  {el}")
        else:
            lines.append("  (no error lines found in logs)")
        lines.append("")

    # Recent logs (last 10 lines, no full samples).
    if logs:
        lines.append(f"Recent logs (last {min(10, len(logs))} lines):")
        for log_line in logs[-10:]:
            lines.append(f"  {log_line}")

    return "\n".join(lines)


# -- Command -----------------------------------------------------------------

@click.command("show")
@click.argument("run_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["human", "table", "json"]),
    default="human",
    show_default=True,
    help="Output format: human-readable text, compact table, or structured JSON.",
)
@click.option(
    "--dashboard-url",
    default="http://127.0.0.1:8765",
    show_default=True,
    help="Dashboard server URL (used when available).",
)
def show_command(run_id: str, output_format: str, dashboard_url: str) -> None:
    """Show detailed information for a single training run.

    Given a run ID (or partial ID prefix), display model, dataset, algorithm,
    resolved key settings, current stage, latest metrics, and last error.

    \b
    Examples:
      areno show abc123              # human-readable output
      areno show abc123 --format json
      areno show abc123 --format table
    """

    # Try the dashboard API first for full details.
    result = _try_dashboard_api(dashboard_url, run_id)
    if result is not None:
        details = _build_details_from_dashboard(
            result, {"id": run_id}, dashboard_url
        )
    else:
        # Fallback: resolve from local artifacts.
        job_item = _resolve_job(run_id)
        if job_item is None:
            raise click.ClickException(f"Run '{run_id}' not found. Use 'areno runs' to list available runs.")
        details = _load_job_details(job_item)

    if output_format == "json":
        click.echo(_json.dumps(details, ensure_ascii=False, indent=2, default=str))
    elif output_format == "table":
        _emit_table(details)
    else:
        click.echo(_format_human(details))


def _emit_table(details: dict[str, Any]) -> None:
    """Emit a compact key-value table."""
    rows = [
        ("Run ID", details.get("id", "?")),
        ("Name", details.get("name", "?")),
        ("Kind", details.get("kind", "?")),
        ("Status", details.get("status", "?")),
        ("Stage", details.get("stage", "-")),
        ("Step", details.get("step", 0)),
        ("Created", details.get("created_at", "?")),
        ("Updated", details.get("updated_at", "?")),
    ]
    rc = details.get("returncode")
    if rc is not None:
        rows.append(("Exit code", rc))

    config = details.get("launch_config") or details.get("config") or {}
    for key in ["algo", "ckpt", "dataset_path", "world_size", "batch_size", "lr", "max_steps"]:
        val = config.get(key)
        if val is not None and val != "":
            rows.append((f"  {key}", val))

    metrics = details.get("metrics", [])
    for m in metrics[:10]:
        rows.append((f"  metric:{m.get('name', '?')}", f"{m.get('latest_value', '?')} @step {m.get('latest_step', '?')}"))
    if len(metrics) > 10:
        rows.append(("  ...", f"({len(metrics) - 10} more metrics)"))

    width = max(len(k) for k, _ in rows) + 2
    for key, val in rows:
        click.echo(f"{key:<{width}}  {val}")