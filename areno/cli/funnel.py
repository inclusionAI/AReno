"""``areno funnel`` -- show a sample-utilization funnel for a finished or running job.

Each training update appends one JSON line to ``sample_funnel.{pid}.jsonl`` under
the metrics log directory (see ``MetricsRecorder.record_funnel``). This command
reads those records, reconciles the per-stage counts, and renders either a
human-readable funnel or a machine-readable JSON report.

The command prints only integer counts and short drop-reason codes -- never the
prompt, completion, messages, or any other sample *content* that may live in
sibling artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from areno.api.defaults import DEFAULT_METRICS_LOG_DIR
from areno.api.funnel import STAGE_ORDER, FunnelCounters, reconcile

# Whitelist of fields carried from each on-disk record into the rendered report.
# Anything outside this set (e.g. a stray ``prompt`` field) is dropped on load
# so sample contents can never reach the terminal.
_RECORD_FIELDS = {"step", "source", "stages", "drop_reasons", "pid"}

# Human-readable stage labels, in funnel order.
_STAGE_LABELS = {
    "loaded": "loaded",
    "contract_valid": "contract-valid",
    "generated": "generated",
    "length_valid": "length-valid",
    "trainable_token_valid": "trainable-token-valid",
    "trained": "trained",
}


@click.command(name="funnel", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--metrics-log-dir",
    "metrics_log_dir",
    default=DEFAULT_METRICS_LOG_DIR,
    show_default=True,
    help="Directory holding sample_funnel.{pid}.jsonl artifacts.",
)
@click.option("--pid", type=int, default=None, help="Select a specific run by process id.")
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON report.")
@click.option(
    "--cumulative",
    "cumulative_only",
    is_flag=True,
    help="Show only the cumulative funnel across all updates (default: per-update and cumulative).",
)
@click.option(
    "--per-update",
    "per_update_only",
    is_flag=True,
    help="Show only the per-update funnel (default: per-update and cumulative).",
)
@click.option("--max-updates", type=int, default=None, help="Show only the last N updates in the per-update view.")
def funnel_command(
    metrics_log_dir: str,
    pid: int | None,
    as_json: bool,
    cumulative_only: bool,
    per_update_only: bool,
    max_updates: int | None,
) -> None:
    """Show a sample-utilization funnel for a run."""

    log_path = Path(metrics_log_dir)
    if not log_path.is_dir():
        raise click.ClickException(f"metrics log directory not found: {metrics_log_dir}")
    records, load_warnings = _load_funnel_records(log_path, pid)
    if not records:
        target = f"pid {pid}" if pid is not None else "any process"
        raise click.ClickException(
            f"no sample_funnel.*.jsonl found for {target} under {metrics_log_dir} "
            "(was the run started with metrics recording enabled?)"
        )
    show_per_update = not cumulative_only
    show_cumulative = not per_update_only
    report = _build_report(records, load_warnings, show_per_update, show_cumulative, max_updates)
    if as_json:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    _print_report(report)


def _load_funnel_records(log_path: Path, pid: int | None) -> tuple[list[dict], list[str]]:
    """Read and validate funnel JSONL records.

    Returns the parsed records (whitelist-filtered, sorted by step) plus a list
    of human-readable load warnings for malformed lines.
    """

    funnel_file = _resolve_funnel_file(log_path, pid)
    if funnel_file is None:
        return [], []
    records: list[dict] = []
    warnings: list[str] = []
    with funnel_file.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"line {line_no}: unparseable JSON ({exc.msg}); skipped")
                continue
            if not isinstance(payload, dict):
                warnings.append(f"line {line_no}: record is not an object; skipped")
                continue
            # Drop any field outside the whitelist so sample contents can never
            # be echoed, even if a future writer accidentally included them.
            records.append({key: payload[key] for key in _RECORD_FIELDS if key in payload})
    records.sort(key=lambda record: record.get("step", 0))
    return records, warnings


def _resolve_funnel_file(log_path: Path, pid: int | None) -> Path | None:
    if pid is not None:
        candidate = log_path / f"sample_funnel.{pid}.jsonl"
        return candidate if candidate.exists() else None
    candidates = sorted(log_path.glob("sample_funnel.*.jsonl"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _records_to_counters(records: list[dict]) -> list[FunnelCounters]:
    """Reconstruct ``FunnelCounters`` from on-disk records for ``reconcile``."""

    counters_list: list[FunnelCounters] = []
    for record in records:
        stages = record.get("stages", {})
        counters_list.append(
            FunnelCounters(
                step=int(record.get("step", 0)),
                source=str(record.get("source", "unknown")),
                loaded=stages.get("loaded"),
                contract_valid=stages.get("contract_valid"),
                generated=stages.get("generated"),
                length_valid=stages.get("length_valid"),
                trainable_token_valid=stages.get("trainable_token_valid"),
                trained=stages.get("trained"),
                drop_reasons={
                    str(stage): [str(reason) for reason in reasons]
                    for stage, reasons in (record.get("drop_reasons") or {}).items()
                },
            )
        )
    return counters_list


def _aggregate_cumulative(counters_list: list[FunnelCounters]) -> dict[str, Any]:
    """Sum every tracked stage across updates; untracked stages stay ``None``."""

    totals: dict[str, int | None] = {stage: None for stage in STAGE_ORDER}
    for counters in counters_list:
        for stage in STAGE_ORDER:
            value = getattr(counters, stage, None)
            if value is None:
                continue
            totals[stage] = value if totals[stage] is None else totals[stage] + value
    source = counters_list[-1].source if counters_list else "unknown"
    cumulative = FunnelCounters(
        step=-1,
        source=source,
        **{stage: totals[stage] for stage in STAGE_ORDER},
    )
    cumulative.drop_reasons = _merge_drop_reasons(counters_list)
    return {
        "source": cumulative.source,
        "stages": totals,
        "drop_reasons": cumulative.drop_reasons,
        "warnings": reconcile(cumulative),
    }


def _merge_drop_reasons(counters_list: list[FunnelCounters]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = {}
    for counters in counters_list:
        for stage, reasons in counters.drop_reasons.items():
            merged.setdefault(stage, set()).update(reasons)
    return {stage: sorted(reasons) for stage, reasons in merged.items()}


def _build_report(
    records: list[dict],
    load_warnings: list[str],
    show_per_update: bool,
    show_cumulative: bool,
    max_updates: int | None,
) -> dict[str, Any]:
    counters_list = _records_to_counters(records)
    per_update: list[dict[str, Any]] = []
    if show_per_update:
        shown = counters_list
        if max_updates is not None and max_updates >= 0:
            shown = counters_list[-max_updates:] if max_updates else []
        for counters in shown:
            per_update.append(
                {
                    "step": counters.step,
                    "source": counters.source,
                    "stages": {stage: getattr(counters, stage) for stage in STAGE_ORDER},
                    "drop_reasons": counters.drop_reasons,
                    "warnings": reconcile(counters),
                }
            )
    cumulative = _aggregate_cumulative(counters_list) if show_cumulative else None
    return {
        "pid": records[-1].get("pid") if records else None,
        "records": len(records),
        "per_update": per_update,
        "cumulative": cumulative,
        "load_warnings": load_warnings,
    }


def _format_count(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _print_report(report: dict[str, Any]) -> None:
    cumulative = report.get("cumulative")
    pid = report.get("pid")
    click.echo(f"AReno sample funnel  (pid={pid}, updates={report.get('records')})")
    click.echo()

    per_update = report.get("per_update") or []
    if per_update:
        click.echo("Per update:")
        for entry in per_update:
            _print_funnel_entry(entry["step"], entry["source"], entry["stages"], entry["drop_reasons"])
            for warning in entry["warnings"]:
                click.echo(f"      warn: {warning}")
        click.echo()

    if cumulative:
        click.echo("Cumulative:")
        _print_funnel_entry(
            "all", cumulative.get("source", "unknown"), cumulative["stages"], cumulative["drop_reasons"]
        )
        for warning in cumulative["warnings"]:
            click.echo(f"      warn: {warning}")
        click.echo()

    load_warnings = report.get("load_warnings") or []
    if load_warnings:
        click.echo("Load warnings:")
        for warning in load_warnings:
            click.echo(f"  warn: {warning}")


def _print_funnel_entry(
    step: Any, source: str, stages: dict[str, int | None], drop_reasons: dict[str, list[str]]
) -> None:
    click.echo(f"  step={step}  source={source}")
    for stage in STAGE_ORDER:
        label = _STAGE_LABELS[stage]
        count = _format_count(stages.get(stage))
        click.echo(f"    {label:<22} {count}")
        for reason in drop_reasons.get(stage, []):
            click.echo(f"        drop: {reason}")
