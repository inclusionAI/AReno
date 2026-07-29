"""``areno timing-summary`` — summarize time spent in each RL training phase.

This command reads a run's local metrics artifacts (TensorBoard event files
written by ``areno.api.metrics`` under ``--metrics-log-dir``) and aggregates
per-phase timing into two views: the latest update and the whole run. It is a
read-only, side-effect-free snapshot — it writes nothing, kills nothing, and
never initializes models or workers.

Background: the training loop already records per-step wall times for each
phase (rollout / reward / train / etc.) as TensorBoard scalars. This command
simply reads them back and reconciles the totals, so users can operate,
compare, and reproduce runs without one-off scripts. See issue #256.
"""

from __future__ import annotations

from pathlib import Path

import click

from areno.dashboard.timeperf import (
    format_json,
    format_table,
    summarize,
)


def _validate_run_dir(run_dir: Path) -> None:
    """Validate the run directory *before* any expensive event reading.

    Mirrors the issue's requirement that inputs be validated prior to heavy
    work and that failures name the offending stage/input. Raises
    ``click.exceptions.Exit`` with a stderr message on any failure.
    """
    if not run_dir.exists():
        raise click.exceptions.Exit(_fail(f"run directory does not exist: {run_dir}"))
    if not run_dir.is_dir():
        raise click.exceptions.Exit(_fail(f"run path is not a directory: {run_dir}"))
    # A valid AReno metrics dir has either TensorBoard event files or a
    # dashboard_state.<pid>.json snapshot. Reject empty/foreign directories
    # up front with a clear message rather than silently emitting an empty run.
    has_events = any(p.name.startswith("events.out.tfevents.") for p in run_dir.rglob("events.out.tfevents.*"))
    has_state = any(run_dir.glob("dashboard_state.*.json"))
    if not has_events and not has_state:
        raise click.exceptions.Exit(
            _fail(f"no AReno metrics found under {run_dir} (expected events.out.tfevents.* or dashboard_state.*.json)")
        )


def _fail(message: str) -> int:
    """Print ``message`` to stderr and return exit code 1."""
    click.echo(f"areno timing-summary: {message}", err=True)
    return 1


@click.command(
    name="timing-summary",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("run_dir", type=click.Path(exists=False))
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON summary instead of a table.")
def timing_summary_command(run_dir: str, as_json: bool) -> None:
    """Summarize time spent in each RL training phase for a run."""

    run_path = Path(run_dir).expanduser()
    _validate_run_dir(run_path)

    try:
        summary = summarize(run_path)
    except RuntimeError as exc:
        # Raised by load_step_segments when the 'tensorboard' reader package is
        # missing. Surface it clearly rather than crashing with a traceback.
        raise click.exceptions.Exit(_fail(str(exc)))
    except Exception as exc:  # pragma: no cover - defensive, unexpected IO/parse
        raise click.exceptions.Exit(_fail(f"failed to read run metrics: {exc}"))

    if as_json:
        click.echo(format_json(summary))
    else:
        click.echo(format_table(summary))
