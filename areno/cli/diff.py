"""``areno diff`` -- compare two AReno training runs."""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import click

from areno.cli.dashboard_registry import (
    compute_duration,
    format_table,
    pid_is_running,
    read_dashboard_state,
    read_registry,
    read_run_config,
    read_tensorboard_scalars,
)


@click.command(name="diff", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("run_a")
@click.argument("run_b")
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable JSON comparison.")
def diff_command(run_a: str, run_b: str, as_json: bool) -> None:
    """Compare two AReno training runs by registry ID."""

    jobs = read_registry()
    by_id = {j.get("id", ""): j for j in jobs}

    entry_a = by_id.get(run_a)
    if entry_a is None:
        raise click.BadParameter(f'run "{run_a}" not found in registry.')

    entry_b = by_id.get(run_b)
    if entry_b is None:
        raise click.BadParameter(f'run "{run_b}" not found in registry.')

    now = time.time()

    # --- Gather data for each side ---
    side_a = _gather(entry_a, now)
    side_b = _gather(entry_b, now)

    # --- Build sections ---
    summary = _build_summary(side_a, side_b)
    config_diff = _build_config_diff(side_a, side_b)
    metrics_comparison = _build_metrics(side_a, side_b)
    timing_comparison = _build_timing(side_a, side_b)
    incomparables = _build_incomparables(side_a, side_b)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "a": side_a["meta"],
                    "b": side_b["meta"],
                    "config_diff": config_diff,
                    "metrics": metrics_comparison,
                    "timing": timing_comparison,
                    "incomparables": incomparables,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    # --- Terminal output ---
    _print_section("Summary")
    click.echo(summary)
    if config_diff:
        _print_section("Config changes")
        _print_rows(config_diff)
    if metrics_comparison:
        _print_section("Final metrics (last step)")
        _print_rows(metrics_comparison)
    if timing_comparison:
        _print_section("Phase timing (mean per step)")
        _print_rows(timing_comparison)
    if incomparables:
        _print_section("Incomparable")
        for note in incomparables:
            click.echo(f"  {note}")


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def _gather(entry: dict[str, Any], now: float) -> dict[str, Any]:
    """Collect all available data for one run into a single dict."""
    pid = entry.get("pid")
    metrics_dir = entry.get("metrics_dir", "")
    state = read_dashboard_state(metrics_dir, pid) if metrics_dir and isinstance(pid, int) else None
    run_config = read_run_config(metrics_dir, pid) if metrics_dir and isinstance(pid, int) else None
    tb_scalars = read_tensorboard_scalars(metrics_dir, pid) if metrics_dir and isinstance(pid, int) else {}

    # Metadata
    alive = pid_is_running(pid) if isinstance(pid, int) else False
    status = (state or {}).get("status", "running" if alive else "exited")
    if alive:
        status_label = f"{status} (active)"
    else:
        status_label = status

    created_at = entry.get("created_at")
    meta = {
        "id": entry.get("id", ""),
        "kind": entry.get("kind", ""),
        "name": entry.get("name", ""),
        "pid": pid,
        "status": status_label,
        "active": alive,
        "duration": (
            compute_duration(created_at, entry.get("updated_at"), now)
            if isinstance(created_at, (int, float))
            else "-"
        ),
        "steps": (state or {}).get("step"),
    }

    # Config (flattened key -> value)
    config: dict[str, Any] = {}
    if run_config:
        for section in (run_config.get("settings") or {}).get("sections", []):
            for item in section.get("items", []):
                config[item["key"]] = item["value"]

    # Throughput (derive from TB scalars)
    throughput = _derive_throughput(tb_scalars)
    if throughput is not None:
        meta["throughput"] = f"{throughput:.0f} tok/s"

    return {
        "meta": meta,
        "config": config,
        "scalars": tb_scalars,
        "entry": entry,
    }


# ---------------------------------------------------------------------------
# Throughput derivation
# ---------------------------------------------------------------------------


def _derive_throughput(scalars: dict[str, list[tuple[int, float]]]) -> float | None:
    """Derive approximate tokens/sec from TensorBoard scalars.

    ``throughput ≈ response_len_mean × num_sequences / rollout_time_mean``
    """
    res_len = _last_value(scalars.get("rollout/response_len_mean", []))
    num_seqs = _last_value(scalars.get("rollout/num_sequences", []))
    rollout_time = _mean_value(scalars.get("time/rollout", []))
    if (
        res_len is not None
        and num_seqs is not None
        and rollout_time is not None
        and rollout_time > 0
    ):
        return res_len * num_seqs / rollout_time
    return None


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_summary(side_a: dict, side_b: dict) -> str:
    ma, mb = side_a["meta"], side_b["meta"]
    columns = ["", f"Run A ({ma['id']})", f"Run B ({mb['id']})"]
    rows = []
    for key, label in [
        ("kind", "Kind"),
        ("name", "Name"),
        ("status", "Status"),
        ("duration", "Duration"),
        ("steps", "Steps"),
        ("throughput", "Throughput"),
    ]:
        va = ma.get(key)
        vb = mb.get(key)
        if va is not None or vb is not None:
            rows.append([label, str(va) if va is not None else "-", str(vb) if vb is not None else "-"])
    if not rows:
        return ""
    return format_table(columns, rows)


def _build_config_diff(side_a: dict, side_b: dict) -> list[tuple[str, str, str]]:
    ca, cb = side_a["config"], side_b["config"]
    all_keys = sorted(set(ca) | set(cb))
    diffs: list[tuple[str, str, str]] = []
    same = 0
    for key in all_keys:
        va = ca.get(key)
        vb = cb.get(key)
        if va == vb:
            same += 1
            continue
        diffs.append((key, str(va) if va is not None else "-", str(vb) if vb is not None else "-"))
    if same:
        diffs.append(("", f"({same} unchanged fields omitted)", ""))
    return diffs


def _build_metrics(side_a: dict, side_b: dict) -> list[tuple[str, str, str]]:
    sa, sb = side_a["scalars"], side_b["scalars"]
    core_tags = ["train/loss", "rollout/rewards_mean", "rollout/accuracy"]
    rows: list[tuple[str, str, str]] = []
    for tag in core_tags:
        va = _last_value(sa.get(tag, []))
        vb = _last_value(sb.get(tag, []))
        if va is not None or vb is not None:
            short = tag.replace("rollout/", "").replace("train/", "")
            rows.append(
                (
                    short,
                    f"{va:.4f}" if va is not None else "-",
                    f"{vb:.4f}" if vb is not None else "-",
                )
            )
    return rows


def _build_timing(side_a: dict, side_b: dict) -> list[tuple[str, str, str]]:
    sa, sb = side_a["scalars"], side_b["scalars"]
    timing_tags = {
        "time/rollout": "rollout",
        "time/reward": "reward",
        "time/train": "train",
    }
    rows: list[tuple[str, str, str]] = []
    for tag, label in timing_tags.items():
        va = _mean_value(sa.get(tag, []))
        vb = _mean_value(sb.get(tag, []))
        if va is not None or vb is not None:
            rows.append(
                (
                    label,
                    f"{va:.1f}s" if va is not None else "-",
                    f"{vb:.1f}s" if vb is not None else "-",
                )
            )
    return rows


def _build_incomparables(side_a: dict, side_b: dict) -> list[str]:
    notes: list[str] = []
    algo_a = side_a["config"].get("algo")
    algo_b = side_b["config"].get("algo")
    if algo_a and algo_b and algo_a != algo_b:
        notes.append(f"Different algorithms ({algo_a} vs {algo_b}) — metrics may not be directly comparable.")
    steps_a = side_a["meta"].get("steps")
    steps_b = side_b["meta"].get("steps")
    if isinstance(steps_a, int) and isinstance(steps_b, int) and steps_a != steps_b:
        notes.append(f"Unequal step counts ({steps_a} vs {steps_b}) — final metrics are from different points.")
    return notes


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def _last_value(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    _, value = pairs[-1]
    return value


def _mean_value(pairs: list[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    values = [v for _, v in pairs]
    try:
        return statistics.mean(values)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    click.echo()
    click.echo(click.style(title, bold=True))


def _print_rows(rows: list[tuple[str, str, str]]) -> None:
    for key, a_val, b_val in rows:
        if key:
            click.echo(f"  {key:<22} {a_val:<18} {b_val}")
        else:
            click.echo(f"  {click.style(a_val, dim=True)}")