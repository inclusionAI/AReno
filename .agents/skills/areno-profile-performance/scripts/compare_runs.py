#!/usr/bin/env python3
"""Compare throughput, phase timing, peak memory, and settings between two AReno runs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Metric keys whose *increase* indicates a regression (higher = slower).
_TIMING_PREFIXES = ("time/", "step_")

# Metric keys whose *increase* indicates an improvement (higher = faster).
_THROUGHPUT_KEYWORDS = ("throughput", "tokens_per_second")


# ---------------------------------------------------------------------------
# Pure functions (importable and testable without any external dependency)
# ---------------------------------------------------------------------------


def percent_change(baseline: float | None, candidate: float | None) -> float | None:
    """Return percentage change from baseline to candidate, or None when undefined."""
    if baseline is None or candidate is None:
        return None
    if baseline == 0:
        return None
    return (candidate - baseline) / abs(baseline) * 100.0


def _is_timing_metric(key: str) -> bool:
    return key.startswith(_TIMING_PREFIXES) and not any(
        kw in key for kw in _THROUGHPUT_KEYWORDS
    )


def _stat_keys() -> tuple[str, ...]:
    return ("mean", "median", "min", "max", "last")


def compare_metric_pair(
    baseline: dict | None, candidate: dict | None
) -> dict:
    """Compare two statistic dictionaries and return per-stat pct_change."""
    result: dict = {"baseline": baseline, "candidate": candidate, "pct_change": {}}
    for stat in _stat_keys():
        b_val = baseline.get(stat) if baseline else None
        c_val = candidate.get(stat) if candidate else None
        result["pct_change"][stat] = percent_change(
            b_val if isinstance(b_val, (int, float)) else None,
            c_val if isinstance(c_val, (int, float)) else None,
        )
    return result


def _flatten_settings(settings: dict | None) -> dict[str, object]:
    """Flatten the nested areno_run_config settings into {key: value}."""
    if not settings or not isinstance(settings, dict):
        return {}
    flat: dict[str, object] = {}
    for section in settings.values():
        if not isinstance(section, dict):
            continue
        for item in section.get("items", []):
            if isinstance(item, dict) and "key" in item:
                flat[str(item["key"])] = item.get("value")
    return flat


def compare_settings(
    baseline_settings: dict | None, candidate_settings: dict | None
) -> dict:
    """Compare flattened settings from two run configs."""
    base = _flatten_settings(baseline_settings)
    cand = _flatten_settings(candidate_settings)
    base_keys = set(base)
    cand_keys = set(cand)
    matched = sorted(base_keys & cand_keys)
    mismatched: list[dict] = []
    for key in matched:
        if base[key] != cand[key]:
            mismatched.append(
                {"key": key, "baseline": base[key], "candidate": cand[key]}
            )
    return {
        "matched": [k for k in matched if base[k] == cand[k]],
        "mismatched": mismatched,
        "only_baseline": sorted(base_keys - cand_keys),
        "only_candidate": sorted(cand_keys - base_keys),
    }


def identify_extremes(comparisons: dict[str, dict]) -> dict:
    """Find the largest improvement and largest regression by pct_change mean.

    For timing metrics (time/*, step_* without throughput keywords), an
    *increase* is a regression and a *decrease* is an improvement. For
    throughput metrics, the opposite holds. The returned pct_change is
    the raw signed value.
    """
    improvements: list[tuple[str, float, float]] = []  # (metric, raw_pct, magnitude)
    regressions: list[tuple[str, float, float]] = []
    for metric, data in comparisons.items():
        pct = data.get("pct_change", {}).get("mean")
        if pct is None:
            continue
        is_timing = _is_timing_metric(metric)
        if is_timing:
            if pct > 0:
                regressions.append((metric, pct, pct))
            elif pct < 0:
                improvements.append((metric, pct, abs(pct)))
        else:
            if pct > 0:
                improvements.append((metric, pct, pct))
            elif pct < 0:
                regressions.append((metric, pct, abs(pct)))
    best_imp = max(improvements, key=lambda x: x[2]) if improvements else None
    worst_reg = max(regressions, key=lambda x: x[2]) if regressions else None
    return {
        "largest_improvement": {"metric": best_imp[0], "pct_change": best_imp[1]}
        if best_imp
        else None,
        "largest_regression": {"metric": worst_reg[0], "pct_change": worst_reg[1]}
        if worst_reg
        else None,
    }


def _extract_peak_mib(monitor: dict | None) -> float | None:
    """Extract peak memory in MiB from a monitor summary dict."""
    if not monitor:
        return None
    peaks: list[float] = []
    gpus = monitor.get("gpus", {})
    if isinstance(gpus, dict):
        for gpu_stats in gpus.values():
            if isinstance(gpu_stats, dict):
                mem = gpu_stats.get("memory_used_mib")
                if isinstance(mem, dict):
                    peak = mem.get("max")
                    if isinstance(peak, (int, float)):
                        peaks.append(float(peak))
    proc_mem = monitor.get("target_process_memory_mib", {})
    if isinstance(proc_mem, dict):
        for proc_stats in proc_mem.values():
            if isinstance(proc_stats, dict):
                peak = proc_stats.get("max")
                if isinstance(peak, (int, float)):
                    peaks.append(float(peak))
    return max(peaks) if peaks else None


def compare_peak_memory(
    baseline_monitor: dict | None, candidate_monitor: dict | None
) -> dict:
    """Compare peak memory between two monitor summaries."""
    base_peak = _extract_peak_mib(baseline_monitor)
    cand_peak = _extract_peak_mib(candidate_monitor)
    return {
        "baseline_peak_mib": base_peak,
        "candidate_peak_mib": cand_peak,
        "pct_change": percent_change(base_peak, cand_peak),
    }


def build_run_summary(status: dict | None, config: dict | None) -> dict:
    """Extract run metadata from dashboard_state and run_config JSON."""
    summary: dict = {
        "status": None,
        "stage": None,
        "step": None,
        "kind": None,
        "pid": None,
    }
    if isinstance(status, dict):
        summary["status"] = status.get("status")
        summary["stage"] = status.get("stage")
        step = status.get("step")
        if isinstance(step, int):
            summary["step"] = step
        pid = status.get("pid")
        if isinstance(pid, int):
            summary["pid"] = pid
    if isinstance(config, dict):
        summary["kind"] = config.get("kind")
        cpid = config.get("pid")
        if isinstance(cpid, int) and summary["pid"] is None:
            summary["pid"] = cpid
    return summary


def build_result(baseline_data: dict, candidate_data: dict) -> dict:
    """Assemble the full comparison result from loaded run artifacts."""
    base_summary = build_run_summary(
        baseline_data.get("status"), baseline_data.get("config")
    )
    cand_summary = build_run_summary(
        candidate_data.get("status"), candidate_data.get("config")
    )
    base_summary["log_dir"] = baseline_data.get("log_dir")
    cand_summary["log_dir"] = candidate_data.get("log_dir")

    base_metrics = {k: v for k, v in (baseline_data.get("time_metrics") or {}).items() if v is not None}
    cand_metrics = {k: v for k, v in (candidate_data.get("time_metrics") or {}).items() if v is not None}

    common_keys = sorted(set(base_metrics) & set(cand_metrics))
    only_base = sorted(set(base_metrics) - set(cand_metrics))
    only_cand = sorted(set(cand_metrics) - set(base_metrics))

    comparisons: dict[str, dict] = {}
    for key in common_keys:
        comparisons[key] = compare_metric_pair(base_metrics[key], cand_metrics[key])

    extremes = identify_extremes(comparisons)

    peak_memory = compare_peak_memory(
        baseline_data.get("monitor_summary"), candidate_data.get("monitor_summary")
    )

    settings_comp = compare_settings(
        baseline_data.get("config", {}).get("settings") if baseline_data.get("config") else None,
        candidate_data.get("config", {}).get("settings") if candidate_data.get("config") else None,
    )

    warnings: list[str] = []
    if base_summary.get("status") == "running":
        warnings.append(
            f"baseline status is 'running'; baseline metrics may be incomplete"
        )
    if cand_summary.get("status") == "running":
        warnings.append(
            f"candidate status is 'running'; candidate metrics may be incomplete"
        )
    if not common_keys and not only_base and not only_cand:
        warnings.append("no common time/throughput metrics found between runs")
    if only_base:
        warnings.append(
            f"metrics only in baseline (excluded from comparison): {', '.join(only_base)}"
        )
    if only_cand:
        warnings.append(
            f"metrics only in candidate (excluded from comparison): {', '.join(only_cand)}"
        )
    if settings_comp["mismatched"]:
        keys = ", ".join(m["key"] for m in settings_comp["mismatched"])
        warnings.append(f"configuration mismatch on keys: {keys}")
    if peak_memory["baseline_peak_mib"] is None and peak_memory["candidate_peak_mib"] is not None:
        warnings.append("baseline peak memory unavailable; candidate peak memory present")
    if peak_memory["candidate_peak_mib"] is None and peak_memory["baseline_peak_mib"] is not None:
        warnings.append("candidate peak memory unavailable; baseline peak memory present")

    return {
        "ok": True,
        "baseline": base_summary,
        "candidate": cand_summary,
        "metrics": comparisons,
        "peak_memory": peak_memory,
        "settings_comparison": settings_comp,
        "extremes": extremes,
        "warnings": warnings,
    }


def format_terminal_report(result: dict) -> str:
    """Generate a human-readable multi-line report string."""
    lines: list[str] = []
    lines.append("=== AReno Run Comparison ===")
    lines.append("")

    base = result.get("baseline", {})
    cand = result.get("candidate", {})

    def _run_line(label: str, info: dict) -> str:
        log_dir = info.get("log_dir") or "?"
        status = info.get("status") or "unknown"
        step = info.get("step")
        step_str = f", step {step}" if step is not None else ""
        return f"{label}: {log_dir}  [{status}{step_str}]"

    lines.append(_run_line("Baseline ", base))
    lines.append(_run_line("Candidate", cand))
    lines.append("")

    # Metrics section
    metrics = result.get("metrics", {})
    if metrics:
        lines.append("--- Throughput & Timing Metrics ---")
        lines.append("")
        header = f"{'Metric':<30s} {'Baseline (mean)':>16s} {'Candidate (mean)':>16s} {'Change':>12s}"
        lines.append(header)
        lines.append("-" * len(header))
        for key in sorted(metrics):
            entry = metrics[key]
            b_mean = entry.get("baseline", {}).get("mean") if entry.get("baseline") else None
            c_mean = entry.get("candidate", {}).get("mean") if entry.get("candidate") else None
            pct = entry.get("pct_change", {}).get("mean")
            b_str = f"{b_mean:.2f}" if isinstance(b_mean, (int, float)) else "N/A"
            c_str = f"{c_mean:.2f}" if isinstance(c_mean, (int, float)) else "N/A"
            if pct is not None:
                sign = "+" if pct >= 0 else ""
                pct_str = f"{sign}{pct:.2f}%"
                is_timing = _is_timing_metric(key)
                if pct > 0:
                    tag = "REGRESSION" if is_timing else "IMPROVEMENT"
                elif pct < 0:
                    tag = "IMPROVEMENT" if is_timing else "REGRESSION"
                else:
                    tag = ""
                pct_str = f"{pct_str} {tag}".strip()
            else:
                pct_str = "N/A"
            metric_name = key[:30]
            lines.append(f"{metric_name:<30s} {b_str:>16s} {c_str:>16s} {pct_str:>12s}")
        lines.append("")

    # Peak memory section
    peak = result.get("peak_memory", {})
    if peak.get("baseline_peak_mib") is not None or peak.get("candidate_peak_mib") is not None:
        lines.append("--- Peak Memory ---")
        lines.append("")
        b_peak = peak.get("baseline_peak_mib")
        c_peak = peak.get("candidate_peak_mib")
        b_str = f"{b_peak:.2f} MiB" if isinstance(b_peak, (int, float)) else "N/A"
        c_str = f"{c_peak:.2f} MiB" if isinstance(c_peak, (int, float)) else "N/A"
        lines.append(f"Baseline peak: {b_str}")
        lines.append(f"Candidate peak: {c_str}")
        pct = peak.get("pct_change")
        if pct is not None:
            sign = "+" if pct >= 0 else ""
            lines.append(f"Change: {sign}{pct:.2f}%")
        else:
            lines.append("Change: N/A")
        lines.append("")

    # Configuration section
    settings = result.get("settings_comparison", {})
    if settings.get("matched") or settings.get("mismatched") or settings.get("only_baseline") or settings.get("only_candidate"):
        lines.append("--- Configuration ---")
        lines.append("")
        if settings.get("matched"):
            lines.append(f"Matched: {', '.join(settings['matched'])}")
        for m in settings.get("mismatched", []):
            lines.append(
                f"MISMATCH: {m['key']} (baseline={m['baseline']}, candidate={m['candidate']})"
            )
        if settings.get("only_baseline"):
            lines.append(f"Only in baseline: {', '.join(settings['only_baseline'])}")
        if settings.get("only_candidate"):
            lines.append(f"Only in candidate: {', '.join(settings['only_candidate'])}")
        lines.append("")

    # Summary section
    extremes = result.get("extremes", {})
    improvements = extremes.get("largest_improvement")
    regressions = extremes.get("largest_regression")
    if improvements or regressions:
        lines.append("--- Summary ---")
        lines.append("")
        if improvements:
            lines.append(
                f"Largest improvement: {improvements['metric']} ({improvements['pct_change']:+.2f}%)"
            )
        if regressions:
            lines.append(
                f"Largest regression:  {regressions['metric']} ({regressions['pct_change']:+.2f}%)"
            )
        lines.append("")

    # Warnings
    warnings = result.get("warnings", [])
    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O layer (deferred imports, file system access)
# ---------------------------------------------------------------------------


def _latest_glob(directory: Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching pattern, or None."""
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_monitor_peak(directory: Path) -> dict | None:
    """Load monitor JSONL files and extract a simplified GPU/process memory summary."""
    jsonl_files = sorted(
        [f for f in directory.glob("*.jsonl") if not f.name.startswith("rollout_samples")],
        key=lambda p: p.name,
    )
    if not jsonl_files:
        return None
    monitor_file = jsonl_files[-1]
    try:
        records = [
            json.loads(line)
            for line in monitor_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return None
    if not records:
        return None
    first = records[0]
    if "gpus" not in first and "processes" not in first:
        return None
    if "gpus" in first:
        return _summarize_gpu_monitor(records)
    if "processes" in first:
        return _summarize_process_monitor(records)
    return None


def _summarize_gpu_monitor(records: list[dict]) -> dict:
    """Extract peak GPU and target-process memory from GPU monitor JSONL."""
    from collections import defaultdict

    gpu_mem: dict[int, list[float]] = defaultdict(list)
    proc_mem: dict[int, list[float]] = defaultdict(list)
    for record in records:
        for gpu in record.get("gpus", []):
            idx = gpu.get("index")
            mem = gpu.get("memory_used_mib")
            if idx is not None and isinstance(mem, (int, float)):
                gpu_mem[idx].append(float(mem))
        for proc in record.get("target_processes", []):
            pid = proc.get("pid")
            mem = proc.get("memory_used_mib")
            if pid is not None and isinstance(mem, (int, float)):
                proc_mem[pid].append(float(mem))
    gpus_out = {}
    for idx, values in sorted(gpu_mem.items()):
        gpus_out[str(idx)] = {"memory_used_mib": {"max": max(values)}}
    proc_out = {}
    for pid, values in sorted(proc_mem.items()):
        proc_out[str(pid)] = {"max": max(values)}
    return {"gpus": gpus_out, "target_process_memory_mib": proc_out}


def _summarize_process_monitor(records: list[dict]) -> dict:
    """Extract peak RSS from process monitor JSONL (as a peek_memory surrogate)."""
    from collections import defaultdict

    rss_values: list[float] = []
    for record in records:
        for proc in record.get("processes", []):
            rss = proc.get("rss_bytes")
            if isinstance(rss, (int, float)):
                rss_values.append(float(rss))
    if not rss_values:
        return {"gpus": {}, "target_process_memory_mib": {}}
    # Convert bytes to MiB for consistency with GPU monitor
    peak_mib = max(rss_values) / (1024 * 1024)
    return {
        "gpus": {},
        "target_process_memory_mib": {"0": {"max": peak_mib}},
    }


def _load_time_metrics(log_dir: Path, drop_first: int) -> dict | None:
    """Load TensorBoard time/* and throughput scalars (deferred import)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    try:
        accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        keys = sorted(accumulator.Tags().get("scalars", []))
        selected = [
            key
            for key in keys
            if key.startswith("time/")
            or "throughput" in key
            or "tokens_per_second" in key
        ]
        summaries = {}
        for key in selected:
            events = accumulator.Scalars(key)[max(drop_first, 0):]
            values = [event.value for event in events]
            if values:
                summaries[key] = {
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                }
        return summaries if summaries else None
    except Exception:
        return None


def load_run_artifacts(log_dir: Path, drop_first: int, load_monitor: bool = True) -> dict:
    """Load all available artifacts from a run directory."""
    data: dict = {"log_dir": str(log_dir)}

    # Dashboard state (glob for *.<pid>.json, then plain .json)
    state_path = _latest_glob(log_dir, "dashboard_state.*.json")
    if state_path is None:
        plain = log_dir / "dashboard_state.json"
        state_path = plain if plain.exists() else None
    data["status"] = _load_json(state_path) if state_path else None

    # Run config (glob for *.<pid>.json, then plain .json)
    config_path = _latest_glob(log_dir, "areno_run_config.*.json")
    if config_path is None:
        plain = log_dir / "areno_run_config.json"
        config_path = plain if plain.exists() else None
    data["config"] = _load_json(config_path) if config_path else None

    # TensorBoard time metrics
    data["time_metrics"] = _load_time_metrics(log_dir, drop_first)

    # Monitor data
    data["monitor_summary"] = _load_monitor_peak(log_dir) if load_monitor else None

    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare throughput, phase timing, peak memory, and settings between two AReno runs."
    )
    parser.add_argument("baseline", type=Path, help="Baseline run metrics_log_dir")
    parser.add_argument("candidate", type=Path, help="Candidate run metrics_log_dir")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--json-only", action="store_true", help="Output only JSON to stdout")
    output_group.add_argument(
        "--terminal-only", action="store_true", help="Output only terminal report to stdout"
    )
    parser.add_argument("--drop-first", type=int, default=1, help="Drop first N warmup steps (default 1)")
    parser.add_argument("--no-monitor", action="store_true", help="Skip monitor JSONL loading")
    args = parser.parse_args()

    # --- Input validation (before any expensive import) ---
    inputs = {"baseline": str(args.baseline), "candidate": str(args.candidate)}
    for label, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not path.exists():
            result = {
                "ok": False,
                "error": f"{label} directory does not exist: {path}",
                "stage": "input_validation",
                "inputs": inputs,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        if not path.is_dir():
            result = {
                "ok": False,
                "error": f"{label} path is not a directory: {path}",
                "stage": "input_validation",
                "inputs": inputs,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1

    # --- Load artifacts ---
    try:
        baseline_data = load_run_artifacts(
            args.baseline, args.drop_first, load_monitor=not args.no_monitor
        )
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"failed to load baseline: {type(exc).__name__}: {exc}",
            "stage": "load_baseline",
            "inputs": inputs,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    try:
        candidate_data = load_run_artifacts(
            args.candidate, args.drop_first, load_monitor=not args.no_monitor
        )
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"failed to load candidate: {type(exc).__name__}: {exc}",
            "stage": "load_candidate",
            "inputs": inputs,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    # --- Build comparison result ---
    result = build_result(baseline_data, candidate_data)

    # --- Output ---
    json_str = json.dumps(result, indent=2, sort_keys=True)
    terminal_str = format_terminal_report(result)

    if args.json_only:
        print(json_str)
    elif args.terminal_only:
        print(terminal_str)
    else:
        # Default: JSON to stdout, terminal report to stderr
        print(json_str)
        print(terminal_str, file=sys.stderr)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())