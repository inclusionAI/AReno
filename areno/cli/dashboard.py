"""Dashboard lifecycle command."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import click


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dashboard_dir() -> Path:
    return _package_root() / "dashboard"


def _runtime_root() -> Path:
    source_root = _repo_root()
    if (source_root / "pyproject.toml").is_file() and (source_root / "dashboard" / "package.json").is_file():
        return source_root
    return Path.cwd()


def _dashboard_server() -> Path:
    return _dashboard_dir() / "server.py"


def _dashboard_index() -> Path:
    return _dashboard_dir() / "dist" / "index.html"


def _pid_file() -> Path:
    return _runtime_root() / ".areno-dashboard.pid"


def _log_file() -> Path:
    return _runtime_root() / ".areno-dashboard.log"


def _read_pid() -> int | None:
    path = _pid_file()
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@click.command("dashboard")
@click.option("--start", "start", is_flag=True, help="Start the dashboard server in the background.")
@click.option("--stop", "stop", is_flag=True, help="Stop the background dashboard server.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Dashboard bind host.")
@click.option("--port", default=8765, show_default=True, type=int, help="Dashboard bind port.")
def dashboard_command(start: bool, stop: bool, host: str, port: int) -> None:
    """Start or stop the low-intrusion AReno dashboard."""

    if start == stop:
        raise click.UsageError("pass exactly one of --start or --stop")
    if stop:
        pid = _read_pid()
        if pid is None:
            click.echo("dashboard is not running")
            return
        if _is_running(pid):
            os.kill(pid, signal.SIGTERM)
        _pid_file().unlink(missing_ok=True)
        click.echo(f"stopped dashboard pid={pid}")
        return

    server = _dashboard_server()
    if not server.exists():
        raise click.ClickException(f"dashboard server not found: {server}")
    _ensure_dashboard_build()
    existing = _read_pid()
    if existing is not None and _is_running(existing):
        click.echo(f"dashboard already running: http://{host}:{port} pid={existing}")
        return
    runtime_root = _runtime_root()
    env = os.environ.copy()
    env["ARENO_DASHBOARD_ROOT"] = str(runtime_root)
    with _log_file().open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "areno.dashboard.server", "--host", host, "--port", str(port)],
            cwd=runtime_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _pid_file().write_text(str(process.pid), encoding="utf-8")
    click.echo(f"dashboard started: http://{host}:{port} pid={process.pid}")


def _ensure_dashboard_build() -> None:
    if _dashboard_index().exists():
        return
    dashboard_dir = _repo_root() / "dashboard"
    if not (dashboard_dir / "package.json").is_file():
        raise click.ClickException("dashboard static assets are missing; reinstall AReno from a complete distribution")
    click.echo("dashboard static build not found; running pnpm --dir dashboard build")
    try:
        subprocess.run(["pnpm", "--dir", "dashboard", "build"], cwd=_repo_root(), check=True)
    except FileNotFoundError as exc:
        raise click.ClickException("pnpm is required to build dashboard static assets") from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"dashboard build failed with exit code {exc.returncode}") from exc


def _load_dashboard_state() -> Any:
    """Import the dashboard state lazily so `areno metrics` does not pull torch."""
    from areno.dashboard.server import STATE

    return STATE


@click.command("metrics")
@click.option("--job", "job_id", required=True, help="Job id whose metrics to export.")
@click.option(
    "--names",
    "names_csv",
    required=True,
    help="Comma-separated metric names to plot together (e.g. train/loss,train/reward).",
)
@click.option("--limit", default=500, show_default=True, type=int, help="Max points per metric (1..5000).")
@click.option("--json", "as_json", is_flag=True, help="Emit structured JSON instead of human-readable text.")
def metrics_command(job_id: str, names_csv: str, limit: int, as_json: bool) -> None:
    """Export one or more training metrics for a job (issue #265 CLI surface).

    Reuses the dashboard's existing metric_series contract. Default output is
    human-readable; --json gives structured output. Validation failures name the
    affected stage and input without exposing training samples.
    """

    # Validate inputs before touching state (cheap, fails fast).
    if not job_id:
        raise click.UsageError("--job is required")
    requested = [name.strip() for name in names_csv.split(",") if name.strip()]
    if not requested:
        raise click.UsageError("--names must list at least one metric name")
    if limit < 1 or limit > 5000:
        raise click.UsageError(f"--limit must be between 1 and 5000, got {limit}")

    state = _load_dashboard_state()
    job = state.get_job(job_id)
    if job is None:
        raise click.ClickException(f"job not found: {job_id}")

    # Reject names that do not exist for this job so a typo is diagnosed, not
    # silently skipped.
    available = {item["name"] for item in state.metric_summaries(job_id)}
    missing = [name for name in requested if name not in available]
    if missing:
        raise click.ClickException(f"unknown metric names for job {job_id}: {', '.join(missing)}")

    collected = []
    for name in requested:
        points = state.metric_series(job_id, name, limit=limit)
        collected.append({"name": name, "point_count": len(points), "points": points})

    if as_json:
        click.echo(
            json.dumps({"job_id": job_id, "limit": limit, "metrics": collected}, ensure_ascii=False, indent=2)
        )
        return

    # Human-readable: one block per metric, step/value aligned.
    click.echo(f"job: {job_id}  limit: {limit}")
    for entry in collected:
        click.echo(f"\n# {entry['name']}  ({entry['point_count']} points)")
        for point in entry["points"]:
            step = point.get("step")
            value = point.get("value")
            click.echo(f"  step {step:>6}  {value}")
