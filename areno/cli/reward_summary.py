"""CLI command to summarise reward distributions in the terminal.

Reads ``reward_metrics.*.jsonl`` files from a metrics log directory and
prints mean, std, min/max, zero fraction, and a configurable outlier
fraction for the total reward and any named reward components.

Usage::

    areno reward-summary --metrics-log-dir /tmp/areno/tfevent
    areno reward-summary --metrics-log-dir /tmp/areno/tfevent --json
    areno reward-summary --metrics-log-dir /tmp/areno/tfevent --outlier-threshold 2.5
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@click.command(
    name="reward-summary",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--metrics-log-dir",
    default="/tmp/areno/tfevent",
    show_default=True,
    help="Directory containing reward_metrics.*.jsonl files.",
)
@click.option(
    "--outlier-threshold",
    type=float,
    default=3.0,
    show_default=True,
    help="Values deviating more than this many std devs from the mean are outliers.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON instead of a table.",
)
@click.option(
    "--step",
    type=int,
    default=None,
    help="Only summarise records from this training step.",
)
def reward_summary_command(metrics_log_dir: str, outlier_threshold: float, as_json: bool, step: int | None) -> None:
    """Summarise reward distributions from local training artifacts."""

    if outlier_threshold <= 0:
        raise click.UsageError("--outlier-threshold must be positive")

    from areno.api.metrics import load_reward_samples
    from areno.api.reward_stats import compute_component_statistics, format_reward_json, format_reward_table

    log_path = Path(metrics_log_dir)
    if not log_path.is_dir():
        raise click.UsageError(f"--metrics-log-dir does not exist: {metrics_log_dir}")

    samples = load_reward_samples(metrics_log_dir, step=step)
    if not samples:
        click.echo("No reward metrics found. Run a training job with --metrics-log-dir to produce them.")
        return

    report = compute_component_statistics(samples, outlier_threshold=outlier_threshold)

    if as_json:
        click.echo(format_reward_json(report))
    else:
        click.echo(format_reward_table(report, use_color=True), nl=False)


if __name__ == "__main__":
    reward_summary_command()