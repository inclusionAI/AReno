"""Summarize GaLore SFT and RL comparison runs into JSON tables.

SFT arms are ``ARM=RUN_DIR[,RUN_DIR...]`` directories written by
``sft_validation.py``; RL arms are ``areno train`` run directories with a
``tb`` TensorBoard log. Logs may also be the gzipped ``metrics.jsonl.gz`` and
``scalars.jsonl.gz`` copies under ``examples/galore/results``, so the published
tables can be regenerated from the repository. The first arm of each kind is
the baseline.

For SFT the script reports per-seed final validation loss and its relative
change, and per arm the median step and optimizer-step times plus allocator
peaks after two warm-up steps, split into steady steps and GaLore projection
refresh steps. For RL it reports the per-seed mean reward over the final
``--window`` steps, the trainer-native ``train/logp_abs_diff_mean`` and step
times. Rewards are the math verifier's 0/1 scores, already on a [0, 1] scale.

    python examples/galore/summarize.py --output summary.json \\
        --sft adamw=runs/sft/adamw-s1,runs/sft/adamw-s2 --sft galore=runs/sft/galore-s1,runs/sft/galore-s2 \\
        --rl adamw=runs/rl/adamw-s1 --rl galore=runs/rl/galore-s1
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
RL_TAGS = (REWARD, LOGP, "train/step_e2e_time_s", "train/optimizer_step_time_s", "train/train_peak_allocated_bytes")


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
    values = list(values)
    return statistics.median(values) if values else None


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL log, or its gzipped copy when only that exists."""

    handle = path.open() if path.is_file() else gzip.open(path.with_name(path.name + ".gz"), "rt")
    with handle:
        return [json.loads(line) for line in handle]


def _sft_run(run: Path) -> dict:
    rows = read_jsonl(run / "metrics.jsonl")
    evals = [row for row in rows if row["kind"] == "eval"]
    measured = [row for row in rows if row["kind"] == "train"][WARMUP_STEPS:]
    refresh = [row for row in measured if row.get("galore_refreshed_matrices", 0) > 0]
    steady = [row for row in measured if row.get("galore_refreshed_matrices", 0) == 0]

    def timing(subset):
        return {
            "steps": len(subset),
            "median_step_time_s": _median(row["step_train_time_s"] for row in subset),
            "median_optimizer_step_time_s": _median(row["optimizer_step_time_s"] for row in subset),
            "max_train_peak_allocated_gib": max((row["train_peak_allocated_bytes"] for row in subset), default=0) / GIB,
            "max_train_peak_reserved_gib": max((row["train_peak_reserved_bytes"] for row in subset), default=0) / GIB,
            "max_optimizer_peak_allocated_gib": max(
                (row["optimizer_peak_allocated_bytes"] for row in subset), default=0
            )
            / GIB,
            "max_optimizer_peak_reserved_gib": max((row["optimizer_peak_reserved_bytes"] for row in subset), default=0)
            / GIB,
            "median_allocated_after_step_gib": (_median(row["allocated_bytes"] for row in subset) or 0) / GIB,
        }

    state = {
        key.removeprefix("galore_").removeprefix("adam8_").removeprefix("adam4_"): value
        for key, value in (measured[-1] if measured else {}).items()
        if key.startswith(("galore_", "adam8_", "adam4_"))
    }
    return {
        "run": run.as_posix(),
        "initial_val_loss": evals[0]["val_loss"] if evals else None,
        "final_val_loss": evals[-1]["val_loss"] if evals else None,
        "val_curve": [(row["step"], row["val_loss"]) for row in evals],
        "steady": timing(steady),
        "refresh": timing(refresh) if refresh else None,
        "optimizer_state": state,
    }


def summarize_sft(arms: dict[str, list[Path]]) -> dict:
    report = {}
    baseline = None
    for arm, runs in arms.items():
        seeds = [_sft_run(run) for run in runs]
        final = _mean([seed["final_val_loss"] for seed in seeds])
        entry = {
            "seeds": seeds,
            "final_val_loss_mean": final,
            "final_val_loss_std": _stdev([seed["final_val_loss"] for seed in seeds]),
            "median_step_time_s": _median(seed["steady"]["median_step_time_s"] for seed in seeds),
            "median_optimizer_step_time_s": _median(seed["steady"]["median_optimizer_step_time_s"] for seed in seeds),
            "max_train_peak_allocated_gib": max(seed["steady"]["max_train_peak_allocated_gib"] for seed in seeds),
            "max_train_peak_reserved_gib": max(seed["steady"]["max_train_peak_reserved_gib"] for seed in seeds),
        }
        if baseline is None:
            baseline = entry
        else:
            entry["final_val_loss_rel_change"] = final / baseline["final_val_loss_mean"] - 1.0
            entry["per_seed_rel_change"] = [
                seed["final_val_loss"] / base["final_val_loss"] - 1.0
                for seed, base in zip(seeds, baseline["seeds"], strict=False)
            ]
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
                    "logp_abs_diff_curve": [value for _, value in scalars.get(LOGP, [])],
                    "median_step_e2e_time_s": _median(value for _, value in scalars.get("train/step_e2e_time_s", [])),
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
        elif baseline is not None and entry["final_window_reward_mean"] is not None:
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
