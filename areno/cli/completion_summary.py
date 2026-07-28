"""CLI command for summarising completion quality from metrics artifacts.

Reads ``completion_summary.{pid}.jsonl`` (primary) or ``rollout_samples.{pid}.jsonl``
(fallback) from a metrics log directory and reports completion-length
distribution, empty count, length-limit count, filtered count, and generated
tokens per second, distinguishing single-turn and agentic generation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

from areno.api.metrics import compute_percentile


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_DEFAULT_FIELDS = {
    "total_completions": 0,
    "total_generated_tokens": 0,
    "empty_count": 0,
    "length_limit_count": 0,
    "stop_count": -1,
    "tool_calls_count": -1,
    "filtered_count": 0,
    "completion_length_min": 0,
    "completion_length_max": 0,
    "completion_length_mean": 0.0,
    "completion_length_p50": 0,
    "completion_length_p90": 0,
    "completion_lengths": [],
    "rollout_time_s": 0.0,
    "tokens_per_second": 0.0,
}


def _fill_defaults(record: dict) -> dict:
    """Fill missing fields with safe defaults so old records don't crash."""
    filled = dict(_DEFAULT_FIELDS)
    filled.update(record)
    return filled


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return records


def _load_completion_summary_files(log_dir: Path, pid: int | None) -> tuple[list[dict], str]:
    """Load per-step completion quality summaries.

    Returns ``(records, source_type)`` where *source_type* is either
    ``"completion_summary"`` or ``"rollout_samples"`` (fallback).
    """

    if pid is not None:
        candidate = log_dir / f"completion_summary.{pid}.jsonl"
        files = [candidate] if candidate.exists() else []
    else:
        files = sorted(log_dir.glob("completion_summary.*.jsonl"))
    if files:
        records = []
        for f in files:
            records.extend(_fill_defaults(r) for r in _load_jsonl(f))
        return records, "completion_summary"

    # Fallback: aggregate from rollout_samples JSONL.
    if pid is not None:
        candidate = log_dir / f"rollout_samples.{pid}.jsonl"
        files = [candidate] if candidate.exists() else []
    else:
        files = sorted(log_dir.glob("rollout_samples.*.jsonl"))
    if files:
        records = []
        for f in files:
            records.extend(_rollout_samples_to_summaries(_load_jsonl(f)))
        return records, "rollout_samples"

    return [], "none"


def _rollout_samples_to_summaries(samples: list[dict]) -> list[dict]:
    """Aggregate raw rollout_samples entries into per-step summaries.

    This is a *fallback* path for runs that pre-date
    ``completion_summary`` files.  Because ``response_tokens`` is truncated
    to 64 in the sample log and only a limited number of samples are
    recorded per step, the numbers are approximate.
    """

    by_step: dict[tuple[int, int, str], list[dict]] = {}
    for sample in samples:
        kind = sample.get("kind", "rollout")
        epoch = int(sample.get("epoch", 0))
        step = int(sample.get("step", 0))
        key = (epoch, step, kind)
        by_step.setdefault(key, []).append(sample)

    records = []
    for (epoch, step, kind), group in sorted(by_step.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        completion_lengths = []
        empty_count = 0
        length_limit_count = 0
        for s in group:
            # Prefer response_len (new field); fall back to response_tokens
            # length (truncated to 64, hence approximate).
            if "response_len" in s:
                rlen = int(s["response_len"])
            else:
                rlen = len(s.get("response_tokens", []))
            completion_lengths.append(rlen)
            if rlen == 0:
                empty_count += 1
            fr = s.get("finish_reason", "")
            if fr == "length":
                length_limit_count += 1
        sorted_lengths = sorted(completion_lengths)
        records.append(
            {
                "epoch": epoch,
                "step": step,
                "kind": kind,
                "total_completions": len(group),
                "total_generated_tokens": sum(completion_lengths),
                "empty_count": empty_count,
                "length_limit_count": length_limit_count,
                "stop_count": -1,
                "tool_calls_count": -1,
                "filtered_count": 0,
                "completion_length_min": min(completion_lengths) if completion_lengths else 0,
                "completion_length_max": max(completion_lengths) if completion_lengths else 0,
                "completion_length_mean": float(sum(completion_lengths) / len(completion_lengths))
                if completion_lengths
                else 0.0,
                "completion_length_p50": int(compute_percentile(sorted_lengths, 0.50)),
                "completion_length_p90": int(compute_percentile(sorted_lengths, 0.90)),
                "completion_lengths": completion_lengths,
                "rollout_time_s": 0.0,
                "tokens_per_second": 0.0,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_kind(records: list[dict]) -> dict | None:
    """Aggregate all per-step records of one kind into an overall summary."""
    if not records:
        return None
    total_completions = sum(r.get("total_completions", 0) for r in records)
    total_generated_tokens = sum(r.get("total_generated_tokens", 0) for r in records)
    empty_count = sum(r.get("empty_count", 0) for r in records)
    length_limit_count = sum(r.get("length_limit_count", 0) for r in records)
    stop_count = sum(max(r.get("stop_count", -1), 0) for r in records)
    tool_calls_count = sum(max(r.get("tool_calls_count", -1), 0) for r in records)
    filtered_count = sum(r.get("filtered_count", 0) for r in records)

    all_lengths: list[int] = []
    total_rollout_time = 0.0
    for r in records:
        lengths = r.get("completion_lengths", [])
        all_lengths.extend(l for l in lengths if isinstance(l, (int, float)))
        total_rollout_time += float(r.get("rollout_time_s", 0.0))

    sorted_lengths = sorted(all_lengths)
    tokens_per_second = total_generated_tokens / total_rollout_time if total_rollout_time > 0 else 0.0

    return {
        "num_steps": len(records),
        "total_completions": total_completions,
        "total_generated_tokens": total_generated_tokens,
        "empty_count": empty_count,
        "length_limit_count": length_limit_count,
        "stop_count": stop_count,
        "tool_calls_count": tool_calls_count,
        "filtered_count": filtered_count,
        "completion_length": {
            "min": min(sorted_lengths) if sorted_lengths else 0,
            "max": max(sorted_lengths) if sorted_lengths else 0,
            "mean": float(sum(sorted_lengths) / len(sorted_lengths)) if sorted_lengths else 0.0,
            "p50": int(compute_percentile(sorted_lengths, 0.50)),
            "p90": int(compute_percentile(sorted_lengths, 0.90)),
        },
        "tokens_per_second": round(tokens_per_second, 1),
        "total_rollout_time_s": round(total_rollout_time, 3),
    }


def _aggregate_overall(records: list[dict]) -> dict:
    rollout_records = [r for r in records if r.get("kind") == "rollout"]
    agentic_records = [r for r in records if r.get("kind") == "agentic"]
    return {
        "rollout": _aggregate_kind(rollout_records),
        "agentic": _aggregate_kind(agentic_records),
    }


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------


def _format_overall_section(label: str, overall: dict, *, color: bool = True) -> str:
    if overall is None:
        line = f"  {label}: (none)"
        return click.style(line, fg="white", dim=True) if color else line

    lines = [click.style(f"{label}", fg="cyan", bold=True) if color else f"{label}"]
    metrics = [
        ("Steps", overall["num_steps"]),
        ("Total completions", overall["total_completions"]),
        ("Total generated tokens", f"{overall['total_generated_tokens']:,}"),
        ("Empty count", overall["empty_count"]),
        ("Length-limit count", overall["length_limit_count"]),
        ("Stop count", overall["stop_count"] if overall["stop_count"] >= 0 else "N/A"),
        ("Tool-calls count", overall["tool_calls_count"] if overall["tool_calls_count"] >= 0 else "N/A"),
        ("Filtered count", overall["filtered_count"]),
    ]
    cl = overall["completion_length"]
    metrics.append(("Length (min/p50/p90/max)", f"{cl['min']}/{cl['p50']}/{cl['p90']}/{cl['max']}"))
    metrics.append(("Mean length", f"{cl['mean']:.1f}"))
    if overall["tokens_per_second"] > 0:
        metrics.append(("Tokens per second", f"{overall['tokens_per_second']:.1f}"))
    else:
        metrics.append(("Tokens per second", "N/A"))
    for key, value in metrics:
        lines.append(f"  {key:<24} {value}")
    return "\n".join(lines)


def _format_completion_summary_table(records: list[dict], *, color: bool = True) -> str:
    if not records:
        line = "  (no per-step records)"
        return click.style(line, fg="white", dim=True) if color else line

    header = f"  {'Step':<6} {'Kind':<9} {'Comps':<6} {'Empty':<7} {'Len-Lim':<8} {'Filt':<6} {'Mean':<7} {'P50':<5} {'P90':<5} {'Tok/s':<7}"
    sep = f"  {'-' * 6} {'-' * 9} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 5} {'-' * 5} {'-' * 7}"
    lines = [header, sep]
    for r in records:
        tps = f"{r.get('tokens_per_second', 0.0):.1f}" if r.get("tokens_per_second", 0.0) > 0 else "N/A"
        stop = "N/A" if r.get("stop_count", -1) < 0 else r.get("stop_count", 0)
        row = (
            f"  {r.get('step', 0):<6} {r.get('kind', '?'):<9} "
            f"{r.get('total_completions', 0):<6} {r.get('empty_count', 0):<7} "
            f"{r.get('length_limit_count', 0):<8} {r.get('filtered_count', 0):<6} "
            f"{r.get('completion_length_mean', 0.0):<7.1f} "
            f"{r.get('completion_length_p50', 0):<5} "
            f"{r.get('completion_length_p90', 0):<5} {tps:<7}"
        )
        lines.append(row)

    # Truncate to terminal width if necessary.
    term_width = shutil.get_terminal_size((120, 24)).columns
    if term_width < 120:
        return "\n".join(lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command(
    name="completion-summary",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--metrics-log-dir",
    "log_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing AReno metrics artifacts.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON instead of a human-readable table.",
)
@click.option(
    "--pid",
    type=int,
    default=None,
    help="Filter by process ID. If omitted, reads all matching files.",
)
def completion_summary_command(log_dir: Path, as_json: bool, pid: int | None) -> None:
    """Summarise completion quality from a metrics log directory.

    Reports completion-length distribution, empty count, length-limit count,
    filtered count, and generated tokens per second per update, distinguishing
    single-turn (rollout) and agentic generation.

    Primary data source is ``completion_summary.{pid}.jsonl``. If that file is
    absent, the command falls back to ``rollout_samples.{pid}.jsonl`` with
    reduced fidelity (response tokens are truncated to 64 in samples and only
    a limited number of completions are logged per step).
    """

    records, source = _load_completion_summary_files(log_dir, pid)
    if not records:
        raise click.UsageError(
            f"No completion_summary or rollout_samples files found in {log_dir}."
            " Run a training job with AReno first."
        )

    overall = _aggregate_overall(records)

    if as_json:
        output = {
            "overall": overall,
            "per_step": records,
            "source": source,
        }
        click.echo(json.dumps(output, indent=2, sort_keys=True, default=str))
        return

    # Human-readable output.
    rollout_summary = overall.get("rollout")
    agentic_summary = overall.get("agentic")
    num_rollout_steps = rollout_summary["num_steps"] if rollout_summary else 0
    num_agentic_steps = agentic_summary["num_steps"] if agentic_summary else 0

    click.echo()
    click.echo(click.style("Completion Quality Summary", bold=True))
    click.echo(f"  Metrics directory: {log_dir}")
    click.echo(f"  Source: {source}")
    click.echo(f"  Steps: {len(records)} (rollout: {num_rollout_steps}, agentic: {num_agentic_steps})")
    click.echo()

    if rollout_summary:
        click.echo(_format_overall_section("Single-turn (rollout)", rollout_summary, color=True))
        click.echo()
    if agentic_summary:
        click.echo(_format_overall_section("Agentic", agentic_summary, color=True))
        click.echo()

    click.echo("Per-step detail")
    click.echo(_format_completion_summary_table(records, color=True))
    click.echo()