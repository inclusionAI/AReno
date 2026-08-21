"""Command-line entrypoint for watching training progress."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

# =============================================================================
# Configuration
# =============================================================================

# Default polling interval in seconds
DEFAULT_INTERVAL = 1

# Location of AReno runtime state
ARENO_RUNTIME_DIR = Path(os.path.expanduser("~/.areno"))
RUNS_DIR = ARENO_RUNTIME_DIR / "runs"
# Where training writes dashboard_state.{pid}.json by default
TFEVENT_DIR = Path("/tmp/areno/tfevent")


# =============================================================================
# ANSI Colors
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def colorize(text: str, color: str) -> str:
    """Apply color to text."""

    color_code = getattr(Colors, color.upper(), Colors.WHITE)
    return f"{color_code}{text}{Colors.RESET}"


def green(text: str) -> str:
    return colorize(text, "green")


def yellow(text: str) -> str:
    return colorize(text, "yellow")


def red(text: str) -> str:
    return colorize(text, "red")


def cyan(text: str) -> str:
    return colorize(text, "cyan")


def magenta(text: str) -> str:
    return colorize(text, "magenta")


def bold(text: str) -> str:
    return f"{Colors.BOLD}{text}{Colors.RESET}"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class WatchConfig:
    """Configuration for watch command."""

    run_id: str | None
    latest: bool
    interval: int
    json_output: bool
    quiet: bool
    timeout: int
    tail: int | None = None
    fields: list[str] | None = None
    no_header: bool = False


@dataclass
class RunStatus:
    """Parsed status from dashboard state file."""

    pid: int
    stage: str
    status: str
    updated_at: float
    step: int | None = None
    epoch: int | None = None
    role: str | None = None
    # Extended fields (may be added by trainer)
    loss: float | None = None
    reward_mean: float | None = None
    throughput: int | None = None
    total_steps: int | None = None


# =============================================================================
# Status File Reader
# =============================================================================


def find_latest_run_id() -> str | None:
    """Find the most recent run ID from the runs directory or tfevent."""

    # First check runs directory
    if RUNS_DIR.exists():
        run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if run_dirs:
            run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return run_dirs[0].name

    # Fallback: check tfevent directory for dashboard_state files
    if TFEVENT_DIR.exists():
        status_files = list(TFEVENT_DIR.glob("dashboard_state.*.json"))
        if status_files:
            status_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            # Use PID as run identifier
            for f in status_files:
                # filename like: dashboard_state.12345.json
                pid = f.stem.rsplit(".", 1)[-1]
                if pid.isdigit():
                    return f"process_{pid}"
            return "latest"

    return None


def find_status_file(run_id: str) -> Path | None:
    """Find the dashboard state file for a given run ID."""

    run_dir = RUNS_DIR / run_id

    if run_dir.exists():
        # Look for dashboard_state.*.json in the run directory
        status_files = list(run_dir.glob("dashboard_state.*.json"))
        if status_files:
            status_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return status_files[0]

    # Fallback: search tfevent directory
    if TFEVENT_DIR.exists():
        status_files = list(TFEVENT_DIR.glob("dashboard_state.*.json"))
        if status_files:
            status_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return status_files[0]

    # Legacy: directly in runs dir
    for f in RUNS_DIR.glob("dashboard_state.*.json"):
        return f

    return None


def read_status(status_file: Path) -> RunStatus | None:
    """Read and parse a dashboard state file."""

    try:
        content = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return RunStatus(
        pid=content.get("pid", 0),
        stage=content.get("stage", "unknown"),
        status=content.get("status", "unknown"),
        updated_at=content.get("updated_at", 0.0),
        step=content.get("step"),
        epoch=content.get("epoch"),
        role=content.get("role"),
        loss=content.get("loss"),
        reward_mean=content.get("reward_mean"),
        throughput=content.get("throughput"),
        total_steps=content.get("total_steps"),
    )


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_training_active(status: RunStatus) -> bool:
    """Check if the training process is still active."""

    # Check if process is running
    if status.pid and not is_process_running(status.pid):
        return False

    # Check if status indicates completion
    if status.status in ("completed", "error", "stopped"):
        return False

    return True


# =============================================================================
# ETA Calculation
# =============================================================================


def calculate_eta(
    current_step: int | None,
    total_steps: int | None,
    start_time: float,
    current_time: float,
) -> int | None:
    """Calculate estimated time remaining in seconds."""

    if current_step is None or total_steps is None or total_steps == 0:
        return None

    if current_step >= total_steps:
        return 0

    elapsed = current_time - start_time

    if elapsed <= 0 or current_step <= 0:
        return None

    # Steps per second
    rate = current_step / elapsed

    if rate <= 0:
        return None

    remaining_steps = total_steps - current_step

    return int(remaining_steps / rate)


def format_eta(seconds: int | None) -> str:
    """Format ETA in human-readable format."""

    if seconds is None:
        return "N/A"

    if seconds <= 0:
        return "done"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# =============================================================================
# Output Formatters
# =============================================================================


def get_terminal_size() -> tuple[int, int]:
    """Get terminal size (width, height)."""

    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        return 80, 24


def render_tty(status: RunStatus, eta: int | None, elapsed: float) -> str:
    """Render status for TTY (with cursor control)."""

    width, _ = get_terminal_size()
    width = min(width, 80)  # Cap width for consistent display

    # Progress bar
    progress = ""
    if status.step is not None and status.total_steps is not None:
        percent = status.step / max(status.total_steps, 1)
        bar_width = 20
        filled = int(percent * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        progress = f"Step: {status.step}/{status.total_steps}  {bar}  {percent * 100:.1f}%"

    # Metrics (colorized variants are assembled inline below)
    throughput_str = f"Throughput: {status.throughput} tok/s" if status.throughput is not None else "Throughput: N/A"

    # Time since last update
    time_ago = ""
    if status.updated_at > 0:
        time_ago = f"Updated: {int(elapsed)}s ago"

    # Build the display with colors
    title = f"║  {bold('AReno Watch')} - Run status"
    lines = [
        "╔" + "═" * (width - 2) + "╗",
        title + " " * (width - len(title) - 4) + "║",
        "╠" + "═" * (width - 2) + "╣",
    ]

    if progress:
        # Colorize progress bar
        progress_colored = green(progress)
        lines.append(f"║  {progress_colored}" + " " * (width - len(progress_colored) - 4) + "║")

    # Colorize metrics
    loss_str_colored = f"{cyan('Loss:')} {status.loss:.4f}" if status.loss is not None else f"{cyan('Loss:')} N/A"
    reward_str_colored = (
        f"{magenta('Reward:')} {status.reward_mean:.4f}"
        if status.reward_mean is not None
        else f"{magenta('Reward:')} N/A"
    )
    metrics_line = f"║  {loss_str_colored}    {reward_str_colored}"
    lines.append(metrics_line + " " * (width - len(metrics_line) - 4) + "║")

    if throughput_str:
        throughput_str_colored = f"{cyan('Throughput:')} {status.throughput} tok/s"
        throughput_line = f"║  {throughput_str_colored}"
        lines.append(throughput_line + " " * (width - len(throughput_line) - 4) + "║")

    # Colorize stage and ETA based on status
    stage_color = {"rollout": cyan, "reward": yellow, "advantage": magenta, "train": green}.get(status.stage, cyan)
    eta_str_colored = f"{bold('ETA:')} {format_eta(eta)}"
    status_line = f"║  {stage_color('Stage:')} {status.stage}    {eta_str_colored}"
    lines.append(status_line + " " * (width - len(status_line) - 4) + "║")

    if time_ago:
        time_line = f"║  {time_ago}"
        lines.append(time_line + " " * (width - len(time_line) - 4) + "║")

    lines.append("╚" + "═" * (width - 2) + "╝")

    return "\n".join(lines)


# Store lines for tail functionality
_line_buffer: list[str] = []


def render_line(
    status: RunStatus,
    eta: int | None,
    fields: list[str] | None = None,
    tail: int | None = None,
    include_timestamp: bool = True,
) -> str:
    """Render status as a single line (non-TTY)."""

    # Determine which fields to show
    all_fields = {
        "step": status.step is not None,
        "loss": status.loss is not None,
        "reward": status.reward_mean is not None,
        "throughput": status.throughput is not None,
        "eta": eta is not None,
        "stage": True,
        "status": True,
    }

    # Filter by requested fields
    if fields:
        available = set(all_fields.keys())
        requested = set(fields)
        missing = requested - available
        if missing:
            # Just show what we have
            pass
        # Filter to requested fields that are available
        filtered = {k: v for k, v in all_fields.items() if k in requested}
        all_fields = filtered

    parts = []

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"[{timestamp}]")

    if all_fields.get("step") and status.step is not None:
        step_str = f"Step {status.step}"
        if status.total_steps is not None:
            step_str += f"/{status.total_steps}"
        parts.append(step_str)

    if all_fields.get("loss") and status.loss is not None:
        parts.append(f"Loss {status.loss:.4f}")

    if all_fields.get("reward") and status.reward_mean is not None:
        parts.append(f"Reward {status.reward_mean:.4f}")

    if all_fields.get("throughput") and status.throughput is not None:
        parts.append(f"tok/s {status.throughput}")

    if all_fields.get("eta") and eta is not None:
        parts.append(f"ETA {format_eta(eta)}")

    if all_fields.get("stage"):
        parts.append(f"Stage={status.stage}")

    if all_fields.get("status"):
        parts.append(f"Status={status.status}")

    line = " | ".join(parts)

    # Handle tail buffer
    global _line_buffer
    if tail is not None and tail > 0:
        _line_buffer.append(line)
        if len(_line_buffer) > tail:
            _line_buffer.pop(0)
        return _line_buffer[-1] if len(_line_buffer) <= tail else line

    return line


def render_json(status: RunStatus, eta: int | None) -> str:
    """Render status as JSON (JSON Lines format)."""

    data = {
        "step": status.step,
        "total_steps": status.total_steps,
        "loss": status.loss,
        "reward": status.reward_mean,
        "throughput": status.throughput,
        "eta_seconds": eta,
        "stage": status.stage,
        "status": status.status,
        "pid": status.pid,
        "updated_at": status.updated_at,
    }

    return json.dumps(data, ensure_ascii=False)


# =============================================================================
# Signal Handling
# =============================================================================


class GracefulExit:
    """Handle graceful exit on Ctrl+C."""

    def __init__(self):
        self.exit_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.exit_requested = True


# =============================================================================
# Main Watch Logic
# =============================================================================


def run_watch(config: WatchConfig) -> None:
    """Main watch loop."""

    # Resolve run_id
    run_id = config.run_id
    if config.latest:
        run_id = find_latest_run_id()
        if run_id is None:
            click.echo(red("Error: No runs found."), err=True)
            click.echo("\nTo start a training run, use:", err=True)
            click.echo("  " + cyan("areno train --ckpt <model> --dataset-path <dataset> ..."), err=True)
            click.echo("\nTo list existing runs, use:", err=True)
            click.echo("  " + cyan("areno runs"), err=True)
            sys.exit(1)

    if run_id is None:
        click.echo(red("Error: Must specify either --run-id or --latest"), err=True)
        click.echo("\nExamples:", err=True)
        click.echo("  " + cyan("areno watch --latest") + "       # Watch the most recent run", err=True)
        click.echo("  " + cyan("areno watch --run-id my_run") + " # Watch a specific run", err=True)
        click.echo("\nTo see available runs:", err=True)
        click.echo("  " + cyan("areno runs"), err=True)
        sys.exit(4)

    # Find status file
    status_file = find_status_file(run_id)

    if status_file is None:
        click.echo(red(f"Error: Run '{run_id}' not found or no status file available."), err=True)
        click.echo("\nPossible reasons:", err=True)
        click.echo("  1. The run directory doesn't exist under ~/.areno/runs/", err=True)
        click.echo("  2. The training process hasn't written status yet", err=True)
        click.echo("     (make sure a training run is actively running)", err=True)
        click.echo("\nTo see available runs:", err=True)
        click.echo("  " + cyan("areno runs"), err=True)
        sys.exit(1)

    # Initialize
    exit_handler = GracefulExit()
    start_time = time.time()
    is_tty = sys.stdout.isatty()
    last_status: RunStatus | None = None
    first_output = True

    # Show initial message (unless quiet or no_header)
    if not config.quiet:
        click.echo(f"Watching run: {run_id}")
        click.echo(f"Status file: {status_file}")
        click.echo("Press Ctrl+C to stop watching (training will continue)...")
        click.echo("")

    # Main loop
    while not exit_handler.exit_requested:
        current_time = time.time()

        # Check timeout
        if config.timeout > 0 and (current_time - start_time) >= config.timeout:
            click.echo("\nTimeout reached. Stopping watcher.")
            break

        # Read status
        status = read_status(status_file)

        if status is None:
            # File might be temporarily unavailable
            if not config.quiet:
                click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] Status file temporarily unavailable, retrying...")
            time.sleep(config.interval)
            continue

        # Calculate ETA
        eta = calculate_eta(
            status.step,
            status.total_steps,
            start_time,
            current_time,
        )

        # Skip duplicate outputs (only if not using tail mode)
        if config.tail is None and status == last_status:
            time.sleep(config.interval)
            continue

        last_status = status

        # Render output based on mode
        if config.json_output:
            output = render_json(status, eta)
            click.echo(output)
        elif is_tty and not config.quiet:
            # Clear screen and render TTY view
            # Use ANSI escape to go to top and clear
            output = render_tty(status, eta, current_time - start_time)
            click.echo(f"\033[2J\033[H{output}")
        else:
            # For non-TTY or quiet mode, don't include timestamp if using tail
            include_ts = not config.quiet and first_output
            if config.tail and not config.quiet:
                # In tail mode, only show timestamp on first line after header
                include_ts = False

            output = render_line(
                status,
                eta,
                fields=config.fields,
                tail=config.tail,
                include_timestamp=include_ts,
            )
            click.echo(output)

        first_output = False

        # Check if training is complete
        if not check_training_active(status):
            if not config.json_output:
                click.echo("\nTraining completed or stopped.")
            break

        # Sleep before next iteration
        time.sleep(config.interval)


# =============================================================================
# Click CLI Definition
# =============================================================================


@click.command("watch")
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run ID to watch (directory name under ~/.areno/runs/).",
)
@click.option(
    "--latest",
    is_flag=True,
    default=False,
    help="Watch the most recent run.",
)
@click.option(
    "--interval",
    type=int,
    default=DEFAULT_INTERVAL,
    help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL}).",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output in JSON Lines format (one JSON object per line).",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress header/footer messages.",
)
@click.option(
    "--timeout",
    type=int,
    default=0,
    help="Exit after N seconds (0 = unlimited).",
)
@click.option(
    "--tail",
    type=int,
    default=None,
    help="Only show the last N lines (non-TTY mode, for log tailing).",
)
@click.option(
    "--fields",
    type=str,
    default=None,
    help="Comma-separated fields to display (step,loss,reward,throughput,eta,stage,status).",
)
@click.option(
    "--no-header",
    is_flag=True,
    default=False,
    help="Suppress header/footer messages.",
)
def watch_command(
    run_id: str | None,
    latest: bool,
    interval: int,
    json_output: bool,
    quiet: bool,
    timeout: int,
    tail: int | None,
    fields: str | None,
    no_header: bool,
) -> None:
    """Watch training progress in the terminal.

    Displays real-time updates of training metrics including step progress,
    loss, reward, throughput, and estimated time remaining.

    Examples:

        # Watch the most recent training run
        areno watch --latest

        # Watch a specific run
        areno watch --run-id 20240115_143022

        # Output as JSON Lines (useful for logging)
        areno watch --latest --json

        # Refresh every 2 seconds
        areno watch --latest --interval 2

        # Show only last 10 lines
        areno watch --latest --tail 10

        # Show only specific fields
        areno watch --latest --fields step,loss,reward
    """
    # Validate inputs
    if interval < 1:
        click.echo(red("Error: --interval must be at least 1 second."), err=True)
        sys.exit(3)

    if timeout < 0:
        click.echo(red("Error: --timeout must be non-negative."), err=True)
        sys.exit(3)

    if tail is not None and tail < 1:
        click.echo(red("Error: --tail must be at least 1."), err=True)
        sys.exit(3)

    # Parse fields
    field_list = None
    if fields:
        field_list = [f.strip().lower() for f in fields.split(",")]
        valid_fields = {"step", "loss", "reward", "throughput", "eta", "stage", "status"}
        invalid = set(field_list) - valid_fields
        if invalid:
            click.echo(red(f"Error: Invalid fields: {invalid}"), err=True)
            click.echo(f"Valid fields: {', '.join(sorted(valid_fields))}", err=True)
            sys.exit(3)

    config = WatchConfig(
        run_id=run_id,
        latest=latest,
        interval=interval,
        json_output=json_output,
        quiet=quiet or no_header,
        timeout=timeout,
        tail=tail,
        fields=field_list,
        no_header=no_header,
    )

    run_watch(config)


# =============================================================================
# Runs List Command
# =============================================================================


def list_all_runs() -> list[dict]:
    """List all runs with their status information."""

    if not RUNS_DIR.exists():
        return []

    runs = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        run_info = {
            "run_id": run_dir.name,
            "path": str(run_dir),
            "status": "unknown",
            "step": None,
            "stage": None,
            "last_updated": None,
        }

        # Find status files in the run directory
        status_files = list(run_dir.glob("dashboard_state.*.json"))

        if status_files:
            # Sort by modification time
            status_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            latest_status_file = status_files[0]

            try:
                content = json.loads(latest_status_file.read_text(encoding="utf-8"))
                run_info["status"] = content.get("status", "unknown")
                run_info["step"] = content.get("step")
                run_info["stage"] = content.get("stage")
                run_info["last_updated"] = content.get("updated_at")
            except (OSError, json.JSONDecodeError):
                pass

        # Get directory modification time if no status file
        if run_info["last_updated"] is None:
            run_info["last_updated"] = run_dir.stat().st_mtime

        runs.append(run_info)

    # Sort by last_updated, most recent first
    runs.sort(key=lambda r: r["last_updated"] or 0, reverse=True)

    return runs


def format_run_list(runs: list[dict], verbose: bool = False) -> str:
    """Format run list for display."""

    if not runs:
        return "No runs found. Start a training run first with 'areno train'."

    lines = []

    if verbose:
        lines.append(f"{'Run ID':<30} {'Status':<12} {'Step':<10} {'Stage':<15} {'Last Updated'}")
        lines.append("-" * 90)
    else:
        lines.append(f"{'Run ID':<30} {'Status':<12} {'Step':<10}")
        lines.append("-" * 55)

    for run in runs:
        run_id = run["run_id"]
        status = run["status"] or "unknown"
        step = str(run["step"]) if run["step"] is not None else "N/A"

        if verbose:
            stage = run["stage"] or "N/A"
            lastupdated = "N/A"
            if run["last_updated"]:
                dt = datetime.fromtimestamp(run["last_updated"])
                lastupdated = dt.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{run_id:<30} {status:<12} {step:<10} {stage:<15} {lastupdated}")
        else:
            lines.append(f"{run_id:<30} {status:<12} {step:<10}")

    return "\n".join(lines)


@click.command("runs")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed information including stage and last updated time.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output in JSON format.",
)
def runs_command(verbose: bool, json_output: bool) -> None:
    """List all training runs.

    Shows all training runs stored in ~/.areno/runs/ with their current
    status, step progress, and last update time.

    Examples:

        # List all runs
        areno runs

        # Show detailed information
        areno runs --verbose

        # Output as JSON
        areno runs --json
    """
    runs = list_all_runs()

    if json_output:
        click.echo(json.dumps(runs, ensure_ascii=False, indent=2))
    else:
        output = format_run_list(runs, verbose=verbose)
        click.echo(output)
