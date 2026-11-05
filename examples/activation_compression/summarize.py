"""Summarize activation-compression benchmark, SFT and RL runs into JSON tables.

SFT arms are ``ARM=RUN_DIR[,RUN_DIR...]`` directories written by
``sft_validation.py``; RL arms are ``areno train`` run directories with a
``tb`` TensorBoard log. Logs may also be the gzipped ``metrics.jsonl.gz`` and
``scalars.jsonl.gz`` copies under ``examples/activation_compression/results``,
so the published tables can be regenerated from the repository. The first arm
of each kind is the baseline.

Per SFT run the script reports, after two warm-up steps: median and spread of
the step time, the largest forward/backward and optimizer allocator peaks, the
saved-activation accounting (BF16 bytes of the compressible slots against the
bytes stored for them: codes, scales, exact outlier channels and kept slots)
and the step-1 calibration results; plus the final validation loss and its
change against the baseline seed. For RL it reports the per-seed mean reward
over the final ``--window`` steps, the trainer-native
``train/logp_abs_diff_mean`` and step times. Rewards are the math verifier's
0/1 scores, already on a [0, 1] scale.

    python examples/activation_compression/summarize.py --output summary.json \\
        --sft bf16=runs/sft/bf16_s1,runs/sft/bf16_s2 --sft adaptive=runs/sft/actc_s1,runs/sft/actc_s2 \\
        --rl bf16=runs/rl/bf16_s1 --rl adaptive=runs/rl/actc_s1
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path

WARMUP_STEPS = 2
GIB = 2**30
REWARD = "rollout/rewards_mean"
LOGP = "train/logp_abs_diff_mean"
RL_TAGS = (
    REWARD,
    LOGP,
    "train/step_e2e_time_s",
    "train/step_train_time_s",
    "train/train_peak_allocated_bytes",
    "train/actc_stored_fraction",
    "train/actc_saved_bytes",
    "train/actc_stored_bytes",
)
CALIBRATION_KEYS = (
    "actc_baseline_nondeterminism",
    "actc_noise_floor",
    "actc_reference_drift",
    "actc_calibration_error",
    "actc_held_out_error",
    "actc_predicted_error",
    "actc_plan_rounds",
    "actc_plan_stored_fraction",
    "actc_plan_slots_8bit",
    "actc_plan_slots_4bit",
    "actc_plan_slots_2bit",
    "actc_plan_slots_16bit",
    "actc_calibration_passes",
    "actc_calibration_s",
    "actc_calibration_peak_allocated_bytes",
)
BYTE_KEYS = (
    "actc_saved_bytes",
    "actc_stored_bytes",
    "actc_packed_bytes",
    "actc_scale_bytes",
    "actc_outlier_bytes",
    "actc_kept_bytes",
    "actc_workspace_bytes",
)


def _arms(values: list[str]) -> dict[str, list[Path]]:
    arms = {}
    for value in values:
        name, _, runs = value.partition("=")
        arms[name] = [Path(run) for run in runs.split(",") if run]
    return arms


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _stdev(values):
    values = [value for value in values if value is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL log, or its gzipped copy when only that exists."""

    handle = path.open() if path.is_file() else gzip.open(path.with_name(path.name + ".gz"), "rt")
    with handle:
        return [json.loads(line) for line in handle]


def _sft_run(run: Path) -> dict:
    rows = read_jsonl(run / "metrics.jsonl")
    evals = [row for row in rows if row["kind"] == "eval"]
    train = [row for row in rows if row["kind"] == "train"]
    measured = train[WARMUP_STEPS:]
    times = [row["step_train_time_s"] for row in measured]
    result = {
        "run": run.as_posix(),
        "measured_steps": len(measured),
        "median_step_time_s": _median(times),
        "min_step_time_s": min(times, default=None),
        "max_step_time_s": max(times, default=None),
        "median_optimizer_step_time_s": _median(row.get("optimizer_step_time_s") for row in measured),
        "max_train_peak_allocated_gib": max((row["train_peak_allocated_bytes"] for row in measured), default=0) / GIB,
        "max_train_peak_reserved_gib": max((row["train_peak_reserved_bytes"] for row in measured), default=0) / GIB,
        "max_optimizer_peak_allocated_gib": max((row["optimizer_peak_allocated_bytes"] for row in measured), default=0)
        / GIB,
        "max_optimizer_peak_reserved_gib": max((row["optimizer_peak_reserved_bytes"] for row in measured), default=0)
        / GIB,
        "initial_val_loss": evals[0]["val_loss"] if evals else None,
        "final_val_loss": evals[-1]["val_loss"] if evals else None,
        "val_curve": [(row["step"], row["val_loss"]) for row in evals],
    }
    # Per-microbatch means of the saved-activation accounting.
    if measured and "actc_saved_bytes" in measured[0]:
        result["saved_activations_gib"] = {
            key.removeprefix("actc_"): _median(row[key] for row in measured) / GIB for key in BYTE_KEYS
        }
        result["stored_fraction"] = _median(row["actc_stored_fraction"] for row in measured)
        result["plan_misses"] = max(row.get("actc_plan_misses", 0) for row in measured)
    if train and "actc_calibration_error" in train[0]:
        result["calibration"] = {key.removeprefix("actc_"): train[0].get(key) for key in CALIBRATION_KEYS}
        result["calibration_step_time_s"] = train[0]["step_train_time_s"]
    return result


def summarize_sft(arms: dict[str, list[Path]]) -> dict:
    report = {}
    baseline = None
    for arm, runs in arms.items():
        seeds = [_sft_run(run) for run in runs]
        entry = {
            "seeds": seeds,
            "final_val_loss_mean": _mean([seed["final_val_loss"] for seed in seeds]),
            "final_val_loss_std": _stdev([seed["final_val_loss"] for seed in seeds]),
            "median_step_time_s": _median(seed["median_step_time_s"] for seed in seeds),
            "max_train_peak_allocated_gib": max(seed["max_train_peak_allocated_gib"] for seed in seeds),
            "max_train_peak_reserved_gib": max(seed["max_train_peak_reserved_gib"] for seed in seeds),
        }
        if baseline is None:
            baseline = entry
        else:
            if entry["final_val_loss_mean"] is not None and baseline["final_val_loss_mean"] is not None:
                entry["final_val_loss_rel_change"] = (
                    entry["final_val_loss_mean"] / baseline["final_val_loss_mean"] - 1.0
                )
                entry["per_seed_rel_change"] = [
                    seed["final_val_loss"] / base["final_val_loss"] - 1.0
                    for seed, base in zip(seeds, baseline["seeds"], strict=False)
                ]
            entry["train_peak_allocated_change_gib"] = (
                entry["max_train_peak_allocated_gib"] - baseline["max_train_peak_allocated_gib"]
            )
            entry["step_time_rel_change"] = entry["median_step_time_s"] / baseline["median_step_time_s"] - 1.0
        report[arm] = entry
    return report


def _scalars(run: Path) -> dict[str, list[tuple[int, float]]]:
    if not (run / "tb").is_dir():
        rows = read_jsonl(run / "scalars.jsonl")
        return {tag: [(row["step"], row[tag]) for row in rows if tag in row] for tag in RL_TAGS}

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    return {
        tag: sorted((event.step, event.value) for event in accumulator.Scalars(tag)) for tag in RL_TAGS if tag in tags
    }


def summarize_rl(arms: dict[str, list[Path]], window: int) -> dict:
    report = {}
    baseline = None
    for arm, runs in arms.items():
        seeds = []
        for run in runs:
            scalars = _scalars(run)
            rewards = [value for _, value in scalars.get(REWARD, [])]
            seeds.append(
                {
                    "run": run.as_posix(),
                    "steps": len(rewards),
                    "final_window_reward": _mean(rewards[-window:]),
                    "reward_curve": rewards,
                    "logp_abs_diff_mean": _mean([value for _, value in scalars.get(LOGP, [])]),
                    "logp_abs_diff_max": max((value for _, value in scalars.get(LOGP, [])), default=None),
                    "median_step_e2e_time_s": _median(value for _, value in scalars.get("train/step_e2e_time_s", [])),
                    "median_step_train_time_s": _median(
                        value for _, value in scalars.get("train/step_train_time_s", [])[WARMUP_STEPS:]
                    ),
                    "max_train_peak_allocated_gib": max(
                        (value for _, value in scalars.get("train/train_peak_allocated_bytes", [])[WARMUP_STEPS:]),
                        default=0,
                    )
                    / GIB,
                    "median_stored_fraction": _median(
                        value for _, value in scalars.get("train/actc_stored_fraction", [])
                    ),
                }
            )
        entry = {
            "seeds": seeds,
            "final_window_reward_mean": _mean([seed["final_window_reward"] for seed in seeds]),
            "final_window_reward_std": _stdev([seed["final_window_reward"] for seed in seeds]),
            "logp_abs_diff_mean": _mean([seed["logp_abs_diff_mean"] for seed in seeds]),
        }
        if baseline is None:
            baseline = entry["final_window_reward_mean"]
        elif entry["final_window_reward_mean"] is not None:
            entry["final_window_reward_delta_vs_baseline"] = entry["final_window_reward_mean"] - baseline
        report[arm] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft", action="append", default=[], help="ARM=RUN_DIR[,RUN_DIR...]")
    parser.add_argument("--rl", action="append", default=[], help="ARM=RUN_DIR[,RUN_DIR...]")
    parser.add_argument("--window", type=int, default=10, help="Final RL steps averaged per seed.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"warmup_steps": WARMUP_STEPS}
    if args.sft:
        report["sft"] = summarize_sft(_arms(args.sft))
    if args.rl:
        report["rl"] = {"window": args.window, "arms": summarize_rl(_arms(args.rl), args.window)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
