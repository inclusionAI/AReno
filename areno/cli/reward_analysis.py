"""``areno reward-analysis`` command: inspect multi-component reward artifacts.

Post-hoc, read-only analysis over a metrics directory's
``reward_components.<pid>.jsonl`` artifact. Inputs are validated before any
work runs; there is no model or worker initialization in this path.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from areno.api.dashboard import analyze_reward_components


def _pct(value: float) -> str:
    """Format a 0..1 fraction as a percentage string."""

    return f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    """Format a nullable float for a fixed-width table cell."""

    if value is None:
        return "-"
    return f"{value:.4g}"


def _render_human(snapshot: dict, metrics_dir: Path, errors: list[dict]) -> None:
    """Print a human-readable component table plus a bounded step drill-down."""

    components = snapshot.get("components", [])
    if not components:
        click.echo(f"No reward component data found in {metrics_dir} (expected reward_components.<pid>.jsonl).")
        return

    header = (
        f"{'component':<16} {'current':>10} {'mean':>10} {'std':>10} "
        f"{'zero%':>7} {'outlier%':>9} {'nf%':>6} {'missing':>8} {'contrib%':>9}"
    )
    click.echo(f"Reward components in {metrics_dir}")
    click.echo(header)
    click.echo("-" * len(header))
    for comp in components:
        click.echo(
            f"{comp['name']:<16} {_num(comp['current']):>10} {_num(comp['mean']):>10} "
            f"{_num(comp['std']):>10} {_pct(comp['zero_fraction']):>7} "
            f"{_pct(comp['outlier_fraction']):>9} {_pct(comp['non_finite_fraction']):>6} "
            f"{comp['missing_count']:>8} {_pct(comp['contribution_fraction']):>9}"
        )

    steps = snapshot.get("steps", [])
    if steps:
        click.echo()
        click.echo(f"Per-step drill-down (last {len(steps)}):")
        for record in steps:
            total = record["total"]
            total_str = _num(total)
            flagged = []
            if record["non_finite"]:
                flagged.append(f"non_finite={','.join(record['non_finite'])}")
            if record["missing"]:
                flagged.append(f"missing={','.join(record['missing'])}")
            flag_str = f" [{', '.join(flagged)}]" if flagged else ""
            click.echo(f"  step {record['step']}: total={total_str}{flag_str}")

    if errors:
        click.echo()
        click.echo(f"{len(errors)} artifact issue(s):")
        for error in errors:
            location = error.get("file", metrics_dir)
            line = error.get("line")
            where = f"{location}:{line}" if line else str(location)
            click.echo(f"  [{error.get('stage', '?')}] {where} - {error.get('message', '')}")


@click.command(
    name="reward-analysis",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--metrics-dir",
    "metrics_dir",
    required=True,
    type=click.Path(exists=False),
    help="Metrics directory containing reward_components.<pid>.jsonl.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON report.")
@click.option("--history", type=int, default=200, show_default=True, help="Bounded per-component history length.")
@click.option(
    "--outlier-z", type=float, default=3.0, show_default=True, help="Z-score threshold for the outlier fraction."
)
def reward_analysis_command(metrics_dir: str, as_json: bool, history: int, outlier_z: float) -> None:
    """Analyze multi-component reward artifacts from a metrics directory."""

    root = Path(metrics_dir)
    if not root.exists():
        raise click.UsageError(f"stage=artifact resolution: path not found (input={root})")
    try:
        snapshot, errors = analyze_reward_components(root, history_limit=history, outlier_z=outlier_z)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if as_json:
        click.echo(json.dumps({"snapshot": snapshot, "errors": errors}, indent=2))
        return
    _render_human(snapshot, root, errors)
