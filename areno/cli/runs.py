"""``areno runs`` -- list active and recent AReno train/serve runs."""

from __future__ import annotations

import json
import time

import click

from areno.cli.dashboard_registry import compute_age, compute_duration, derive_status, format_table, read_registry


@click.command(name="runs", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON list.")
@click.option("--limit", type=int, default=20, show_default=True, help="Show at most N entries.")
@click.option("--all", "show_all", is_flag=True, help="Show all registered entries.")
def runs_command(as_json: bool, limit: int, show_all: bool) -> None:
    """List active and recent AReno train/serve runs."""

    jobs = read_registry()
    if not jobs:
        if as_json:
            click.echo(json.dumps([], indent=2))
        else:
            click.echo("No registered AReno runs found.")
        return

    # Enrich each entry with derived status, age, and duration.
    now = time.time()
    entries: list[dict] = []
    for job in jobs:
        created_at = job.get("created_at")
        entries.append(
            {
                "id": job.get("id", ""),
                "kind": job.get("kind", ""),
                "name": job.get("name", ""),
                "pid": job.get("pid"),
                "status": derive_status(job),
                "created_at": created_at,
                "age": compute_age(created_at, now) if isinstance(created_at, (int, float)) else "-",
                "duration": (
                    compute_duration(created_at, job.get("updated_at"), now)
                    if isinstance(created_at, (int, float))
                    else "-"
                ),
            }
        )

    # Deterministic sort: most-recent first, then by PID.
    entries.sort(key=lambda e: (e.get("created_at") or 0, e.get("pid") or 0), reverse=True)

    if not show_all:
        entries = entries[:limit]

    if as_json:
        click.echo(json.dumps(entries, indent=2, ensure_ascii=False))
        return

    columns = ["STATUS", "KIND", "NAME", "PID", "AGE", "DURATION"]
    rows = []
    for e in entries:
        rows.append(
            [
                e["status"],
                e["kind"],
                e["name"],
                str(e["pid"]) if e["pid"] is not None else "-",
                e["age"],
                e["duration"],
            ]
        )
    click.echo(format_table(columns, rows))
