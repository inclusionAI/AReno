"""Dashboard lifecycle command."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import click


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dashboard_server() -> Path:
    return _repo_root() / "dashboard" / "server.py"


def _dashboard_index() -> Path:
    return _repo_root() / "dashboard" / "dist" / "index.html"


def _pid_file() -> Path:
    return _repo_root() / ".areno-dashboard.pid"


def _log_file() -> Path:
    return _repo_root() / ".areno-dashboard.log"


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
    log_handle = _log_file().open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(server), "--host", host, "--port", str(port)],
        cwd=_repo_root(),
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
    if not dashboard_dir.exists():
        raise click.ClickException(f"dashboard source not found: {dashboard_dir}")
    click.echo("dashboard static build not found; running pnpm --dir dashboard build")
    try:
        subprocess.run(["pnpm", "--dir", "dashboard", "build"], cwd=_repo_root(), check=True)
    except FileNotFoundError as exc:
        raise click.ClickException("pnpm is required to build dashboard static assets") from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"dashboard build failed with exit code {exc.returncode}") from exc
