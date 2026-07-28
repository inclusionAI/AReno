"""CLI subcommands for inspecting reward hooks.

``areno reward inspect`` loads a reward module, scores a small fixture file,
and prints both the human-readable scores and the execution stats. Pass
``--json`` for a machine-readable report that downstream tooling can consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from areno.api.rewards import RewardRecord, call_reward, load_reward


def _load_fixtures(path: str) -> list[RewardRecord]:
    """Load a JSONL file of {prompt, completion, answer} rows into records."""

    records: list[RewardRecord] = []
    line_no = 0
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append(
            RewardRecord(
                prompt=str(obj["prompt"]),
                completion=str(obj["completion"]),
                answer=obj.get("answer"),
                source_record=dict(obj),
            )
        )
    if not records:
        raise click.ClickException(f"no non-empty rows found in {path} (line {line_no})")
    return records


@click.group(name="reward", context_settings={"help_option_names": ["-h", "--help"]})
def reward_command() -> None:
    """Inspect or test reward hooks."""


@reward_command.command(name="inspect", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--path", "path", required=True, help="Path to a Python file defining reward_fn and/or reward_batch.")
@click.option("--fixtures", "fixtures", required=True, help="JSONL file of {prompt, completion, answer} rows.")
@click.option(
    "--batch-size", "batch_size", default=0, help="Slice fixtures into batches of this size (0 = all at once)."
)
@click.option(
    "--scalar-only", "scalar_only", is_flag=True, help="Force the per-example path even if reward_batch exists."
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON report.")
def inspect_command(path: str, fixtures: str, batch_size: int, scalar_only: bool, as_json: bool) -> None:
    """Load a reward module, score fixtures, and print scores + execution stats."""

    bundle = load_reward(path)
    records = _load_fixtures(fixtures)
    prefer_batch = not scalar_only

    if batch_size <= 0:
        batch_size = len(records)

    all_scores: list[float] = []
    all_stats: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        scores, stats = call_reward(bundle, chunk, prefer_batch=prefer_batch)
        all_scores.extend(scores)
        all_stats.append(
            {
                "path": stats.path,
                "wall_time_s": stats.wall_time_s,
                "per_example_time_s": stats.per_example_time_s,
                "count": stats.count,
                "error": stats.error,
            }
        )

    if as_json:
        click.echo(
            json.dumps(
                {
                    "source": bundle.source_path,
                    "has_reward_fn": bundle.reward_fn is not None,
                    "has_reward_batch": bundle.reward_batch is not None,
                    "prefer_batch": prefer_batch,
                    "scores": all_scores,
                    "stats": all_stats,
                },
                indent=2,
            )
        )
        return

    click.echo(f"source: {bundle.source_path}")
    click.echo(
        f"hooks:  reward_fn={'yes' if bundle.reward_fn else 'no'} reward_batch={'yes' if bundle.reward_batch else 'no'}"
    )
    click.echo(f"prefer_batch: {prefer_batch}")
    click.echo("")
    for idx, (record, score) in enumerate(zip(records, all_scores, strict=True)):
        click.echo(f"  [{idx}] prompt={record.prompt[:40]!r} -> score={score:.4f}")
    click.echo("")
    click.echo("stats per chunk:")
    for chunk_stats in all_stats:
        click.echo(
            f"  path={chunk_stats['path']} count={chunk_stats['count']} "
            f"wall={chunk_stats['wall_time_s']:.6f}s "
            f"per_example={chunk_stats['per_example_time_s']:.6f}s"
        )
