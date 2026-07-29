"""CLI command for filtering and following AReno run logs.

Usage examples::

    areno logs /tmp/areno/tfevent           # all logs from a metrics dir
    areno logs /tmp/areno/tfevent --tail 50  # last 50 lines
    areno logs /tmp/areno/tfevent --follow   # follow new output (tail -f)
    areno logs /tmp/areno/tfevent --follow --tail 50 --severity error --grep OOM
    areno logs /tmp/areno/tfevent --output json  # structured JSONL output

The command reuses AReno's existing local artifact formats (files under the
metrics log directory) and the dashboard job registry for run-id resolution.
No external database or sandbox is required.
"""

from __future__ import annotations

import re
import sys

import click

from areno.cli.log_filter import (
    VALID_SEVERITIES,
    VALID_STAGES,
    FilterSpec,
    compile_grep,
    matches,
)
from areno.cli.log_formatter import LogFormatter, format_error
from areno.cli.log_reader import LogReader, ReadStats, resolve_run_paths

_VALID_OUTPUTS = ("text", "json")


def _validate_inputs(
    run_id: str,
    tail: int | None,
    follow: bool,
    rank: int | None,
    stage: str | None,
    severity: str | None,
    grep: str | None,
    output: str,
    poll_interval: float,
) -> tuple[str, str, str] | None:
    """Validate all CLI inputs before any file access.

    Returns ``(stage, input_name, message)`` on failure, or ``None`` if
    all inputs are valid.
    """

    if not run_id:
        return ("log_resolve", "run_id", "run-id is required")

    if tail is not None and tail < 0:
        return ("log_filter", "tail", f"Invalid tail count {tail}: expected non-negative integer")

    if rank is not None and rank < 0:
        return ("log_filter", "rank", f"Invalid rank {rank}: expected non-negative integer")

    if stage is not None and stage not in VALID_STAGES:
        valid = ", ".join(sorted(VALID_STAGES))
        return ("log_filter", "stage", f"Invalid stage '{stage}'. Valid values: {valid}")

    if severity is not None and severity not in VALID_SEVERITIES:
        valid = ", ".join(sorted(VALID_SEVERITIES))
        return ("log_filter", "severity", f"Invalid severity '{severity}'. Valid values: {valid}")

    if grep is not None:
        try:
            compile_grep(grep)
        except re.error as exc:
            return ("log_filter", "grep", f"Invalid grep pattern '{grep}': {exc}")

    if output not in _VALID_OUTPUTS:
        valid = ", ".join(_VALID_OUTPUTS)
        return ("log_output", "output", f"Invalid output '{output}'. Valid values: {valid}")

    if poll_interval <= 0:
        return (
            "log_follow",
            "poll_interval",
            f"Invalid poll-interval {poll_interval}: expected positive float",
        )

    return None


@click.command(name="logs")
@click.argument("run_id")
@click.option("--tail", type=int, default=None, help="Show only the last N lines.")
@click.option("--follow", "-f", is_flag=True, help="Follow new log output (like tail -f).")
@click.option("--rank", type=int, default=None, help="Filter by distributed rank.")
@click.option(
    "--stage",
    default=None,
    help="Filter by training stage. Valid: train, eval, rollout, serve.",
)
@click.option(
    "--severity",
    default=None,
    help="Filter by log severity. Valid: debug, info, warn, error.",
)
@click.option("--grep", default=None, help="Filter by text pattern (regex, case-sensitive).")
@click.option(
    "--output",
    default="text",
    show_default=True,
    help="Output format: text or json.",
)
@click.option(
    "--poll-interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between polls in follow mode.",
)
def logs_command(
    run_id: str,
    tail: int | None,
    follow: bool,
    rank: int | None,
    stage: str | None,
    severity: str | None,
    grep: str | None,
    output: str,
    poll_interval: float,
) -> None:
    """Filter and follow AReno run logs from the CLI."""

    # Normalise case for enum-like options.
    if stage is not None:
        stage = stage.lower()
    if severity is not None:
        severity = severity.lower()
    output = output.lower()

    # 1. Validate inputs before any expensive work.
    error = _validate_inputs(
        run_id, tail, follow, rank, stage, severity, grep, output, poll_interval
    )
    if error is not None:
        err_stage, err_input, err_msg = error
        click.echo(format_error(stage=err_stage, input_name=err_input, message=err_msg, output=output), err=True)  # type: ignore[call-arg]
        raise click.exceptions.Exit(1)

    # 2. Resolve run-id to log file paths.
    paths = resolve_run_paths(run_id)
    if not paths:
        click.echo(
            format_error(
                stage="log_resolve",
                input_name="run_id",
                message=f"No log files found for '{run_id}'. "
                "Provide a log file path, a metrics directory, or a dashboard job id.",
                output=output,
            ),
            err=True,
        )
        raise click.exceptions.Exit(1)

    # 3. Build the filter spec.
    text_pattern = compile_grep(grep) if grep else None
    spec = FilterSpec(
        rank=rank,
        stage=stage,
        severity=severity,
        text_pattern=text_pattern,
    )

    # 4. Build the reader.
    # When filters are active alongside --tail, we read the full file,
    # apply filters, then take the last N matching lines.  When no
    # filters are active, --tail limits the raw read for efficiency.
    has_filters = spec.rank is not None or spec.stage is not None or spec.severity is not None or spec.text_pattern is not None
    effective_tail = None if has_filters else tail

    reader = LogReader(paths)
    iterator, stats = reader.read(
        tail=effective_tail,
        follow=follow,
        poll_interval=poll_interval,
    )

    # 5. Build the formatter.
    formatter = LogFormatter(output=output)

    # 6. Stream filtered entries to stdout.
    # When filters + tail are both active, collect matches and emit only the last N.
    try:
        if has_filters and tail is not None and not follow:
            matched: list = []
            for entry in iterator:
                if matches(entry, spec):
                    matched.append(entry)
            for entry in matched[-tail:] if tail > 0 else []:
                click.echo(formatter.format(entry))
        else:
            for entry in iterator:
                if matches(entry, spec):
                    click.echo(formatter.format(entry))
    except KeyboardInterrupt:
        # Ctrl-C during follow mode — print a summary to stderr.
        pass
    finally:
        if follow:
            _print_summary(stats, output)


def _print_summary(stats: ReadStats, output: str) -> None:
    """Print a short summary to stderr after follow mode ends."""

    if output == "json":
        import json

        click.echo(
            json.dumps(
                {
                    "summary": {
                        "lines_read": stats.lines_read,
                        "lines_yielded": stats.lines_yielded,
                        "rotations": stats.rotations,
                        "truncations": stats.truncations,
                    }
                },
                ensure_ascii=False,
            ),
            err=True,
        )
    else:
        click.echo(
            f"\n[areno logs] stopped: {stats.lines_read} lines read, "
            f"{stats.lines_yielded} lines shown, "
            f"{stats.rotations} rotations, {stats.truncations} truncations",
            err=True,
        )