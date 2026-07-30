"""``areno metrics`` -- query metric history from local run artifacts.

Issue #254 adds a read-only CLI that summarizes one metric (last/min/max/
recent/trend) from the ``events.out.tfevents.*`` artifacts a run writes. The
heavy work lives in the light, pure :mod:`areno.api.metric_reader` module; this
file only wires Click options, renders, and turns not-found cases into a clear
error plus the list of available tags.

The command never writes or starts anything -- it only reads artifacts.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

import click

from areno.api import metric_reader
from areno.api.defaults import DEFAULT_METRICS_LOG_DIR


@click.command(name="metrics", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--metrics-dir",
    "metrics_dir",
    default=None,
    help=(
        "Directory holding the run's events.out.tfevents.* artifacts. "
        "Omit to use areno's default metrics log dir."
    ),
)
@click.option(
    "--pid",
    type=int,
    default=None,
    help="Filter event files by the pid suffix in the filename; merge every run when omitted.",
)
@click.option(
    "--name",
    "name",
    default=None,
    help="Metric tag to summarize (e.g. rollout/rewards_mean). Omit to list available tags.",
)
@click.option(
    "--limit",
    default=20,
    show_default=True,
    help="Number of recent values to include in recent/trend.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON object.")
def metrics_command(
    metrics_dir: str | None,
    pid: int | None,
    name: str | None,
    limit: int,
    as_json: bool,
) -> None:
    """Query metric history (last/min/max/recent/trend) from local run artifacts."""
    if metrics_dir is None:
        metrics_dir = DEFAULT_METRICS_LOG_DIR

    # Surface a missing directory early with a clear, located error -- do not let
    # the reader silently degrade an empty path into an empty result.
    directory = Path(metrics_dir)
    if not directory.exists():
        raise click.ClickException(f"metrics dir not found: {metrics_dir} (pid={pid})")

    # Probe tensorboard up front: read_scalar_points silently returns [] when the
    # import fails, which would otherwise look identical to "no metrics found".
    # Raise a clear, located error so a missing dependency is distinguishable from
    # an empty directory. (The dashboard keeps read_scalar_points' graceful
    # degrade-to-empty; this probe is CLI-only.)
    try:
        import tensorboard  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            "tensorboard is not installed; run `pip install tensorboard` to read metrics."
        ) from exc

    points = metric_reader.read_scalar_points(metrics_dir, pid)
    tags = metric_reader.list_available_tags(points)

    # No tag requested: list what's available so the user can pick one.
    if not name:
        _emit_available(tags, metrics_dir, pid, as_json=as_json)
        return

    if name not in tags:
        _emit_available(tags, metrics_dir, pid, as_json=as_json, missing=name)
        raise click.ClickException(f"metric name not found: {name} (in {metrics_dir}, pid={pid})")

    summary = metric_reader.summarize_metric(points, name, recent_n=limit)
    if as_json:
        click.echo(_json.dumps(metric_reader.render_json(summary), indent=2, sort_keys=True))
    else:
        click.echo(metric_reader.render_table(summary))


def _emit_available(
    tags: list[str],
    metrics_dir: str,
    pid: int | None,
    *,
    as_json: bool,
    missing: str | None = None,
) -> None:
    """Print the available metric tags for the chosen run.

    ``missing`` is set when emitting the list as part of a not-found error so
    the user sees both the unknown name they asked for and what they can use.
    """
    if as_json:
        payload: dict[str, Any] = {
            "metrics_dir": metrics_dir,
            "pid": pid,
            "available_tags": tags,
        }
        if missing is not None:
            payload["missing"] = missing
        click.echo(_json.dumps(payload, indent=2, sort_keys=True))
        return

    label = (
        f"metric '{missing}' not found in {metrics_dir} (pid={pid}); available tags"
        if missing is not None
        else f"available metric tags in {metrics_dir} (pid={pid})"
    )
    if not tags:
        click.echo(f"{label}: (none found)")
        return
    click.echo(label)
    for tag in tags:
        click.echo(f"  {tag}")
