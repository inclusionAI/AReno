"""Extract observed OPD timing, token counts, and throughput from Stage 6 logs.

Missing metrics remain null; never infer times from a test fixture or make up
GPU throughput. A simulation is explicitly marked as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from verify_run import read_step_stats

TIMING_FIELDS = (
    "step_rollout_time_s",
    "opd_teacher_score_time_s",
    "student_train_wall_time_s",
    "step_train_time_s",
    "step_e2e_time_s",
)
TOKEN_FIELDS = ("opd_response_token_count", "opd_response_tokens")
FIELDS = TIMING_FIELDS + ("opd_response_token_count", "opd_loss", "ratio_mean", "opd_advantage_mean")


def _real_nonnegative(value):
    try:
        v = float(value)
        return v if math.isfinite(v) and v >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def extract_runtime_steps(log_text: str) -> list[dict]:
    stats = read_step_stats(log_text)
    rows: list[dict] = []
    for entry in stats:
        if not isinstance(entry, dict) or "opd_loss" not in entry:
            continue
        row = {field: _real_nonnegative(entry.get(field)) for field in FIELDS}
        loss = entry.get("opd_loss")
        try:
            value = float(loss)
            row["opd_loss"] = value if math.isfinite(value) else None
        except (ValueError, TypeError):
            row["opd_loss"] = None
        if row["opd_response_token_count"] is None:
            row["opd_response_token_count"] = _real_nonnegative(entry.get("opd_response_tokens"))
        if row["opd_loss"] is None or row["opd_response_token_count"] is None:
            raise ValueError("OPD step has missing/nonfinite loss or response token count")
        row["step"] = len(rows) + 1
        rows.append(row)
    if not rows:
        raise ValueError("no OPD train_stats entries found in run.log")
    return rows


def summarize_runtime(rows: list[dict], *, kind: str = "unknown") -> dict:
    if not rows:
        raise ValueError("no runtime rows")
    metrics = {}
    for field in TIMING_FIELDS:
        observed = [row[field] for row in rows if row.get(field) is not None]
        metrics[field] = {
            "observed_steps": len(observed),
            "mean_s": sum(observed) / len(observed) if observed else None,
            "total_observed_s": sum(observed) if observed else None,
        }
    signal = {}
    for field in ("opd_loss", "ratio_mean", "opd_advantage_mean"):
        values = [row[field] for row in rows if row.get(field) is not None]
        signal[field] = {"observed_steps": len(values), "first": values[0] if values else None,
                         "last": values[-1] if values else None, "mean": sum(values) / len(values) if values else None}
    token_total = sum(row["opd_response_token_count"] for row in rows)
    e2e = metrics["step_e2e_time_s"]
    e2e_full = e2e["observed_steps"] == len(rows) and (e2e["total_observed_s"] or 0) > 0
    return {
        "schema_version": 1, "evidence_kind": kind, "steps": len(rows),
        "response_tokens": token_total, "timings": metrics, "training_signal": signal,
        "end_to_end_response_tokens_per_s": token_total / e2e["total_observed_s"] if e2e_full else None,
        "notes": ["Missing timings are not imputed.",
                  "Teacher scoring and training may overlap other step timers; do not sum these columns as disjoint stages.",
                  "Elapsed times include runtime effects of the measured backend only."],
    }


def classify_run(directory: Path) -> str:
    simulation = directory / "simulation_manifest.json"
    if simulation.is_file():
        record = json.loads(simulation.read_text(encoding="utf-8"))
        if record.get("is_real_model") is False:
            return "cpu_fake_backend_simulation"
    manifest = directory / "manifest.json"
    if manifest.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("status") == "verified" and record.get("cuda_devices"):
            return "reported_real_cuda_run"
    return "unverified_run_log"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    log = args.run_dir / "run.log"
    rows = extract_runtime_steps(log.read_text(encoding="utf-8", errors="replace"))
    report = summarize_runtime(rows, kind=classify_run(args.run_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_file = args.output_dir / "runtime_summary.json"
    csv_file = args.output_dir / "runtime_steps.csv"
    if json_file.exists() or csv_file.exists():
        raise SystemExit("Stage 7 runtime results already exist; use a fresh --output-dir")
    json_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
