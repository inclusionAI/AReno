"""Summarize seeded RL runs per KV-cache arm from their TensorBoard logs.

Each argument is ``ARM=RUN_DIR[,RUN_DIR...]`` where every run directory holds a
``tb`` TensorBoard log written by ``areno train``. For each arm the script
reports the per-seed mean reward over the final window, its mean and standard
deviation across seeds, the difference from the first (baseline) arm, and the
trainer-native ``train/logp_abs_diff_mean`` and step times. Rewards are the
math verifier's 0/1 scores, so they are already on a [0, 1] scale.

    python examples/kivi/summarize_rl.py --window 20 --output rl.json \
        auto=runs/s1_auto,runs/s2_auto,runs/s3_auto kivi2=runs/s1_kivi2,runs/s2_kivi2,runs/s3_kivi2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REWARD = "rollout/rewards_mean"
LOGP = "train/logp_abs_diff_mean"
TIMES = ("time/rollout", "time/train", "train/step_e2e_time_s")


def _scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir / "tb"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in (REWARD, LOGP, *TIMES)
        if tag in tags
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(arms: dict[str, list[Path]], window: int) -> dict:
    report = {"window": window, "arms": {}}
    baseline = None
    for arm, runs in arms.items():
        seeds = []
        for run in runs:
            scalars = _scalars(run)
            rewards = [value for _, value in sorted(scalars.get(REWARD, []))]
            seeds.append(
                {
                    "run": str(run),
                    "steps": len(rewards),
                    "final_window_reward": _mean(rewards[-window:]),
                    "mean_reward": _mean(rewards),
                    "reward_curve": rewards,
                    "logp_abs_diff_mean": _mean([value for _, value in scalars.get(LOGP, [])]),
                    **{
                        f"median_{tag}": statistics.median([v for _, v in scalars[tag]])
                        for tag in TIMES
                        if scalars.get(tag)
                    },
                }
            )
        finals = [seed["final_window_reward"] for seed in seeds if seed["final_window_reward"] is not None]
        entry = {
            "seeds": seeds,
            "final_window_reward_mean": _mean(finals),
            "final_window_reward_std": statistics.stdev(finals) if len(finals) > 1 else None,
            "logp_abs_diff_mean": _mean(
                [seed["logp_abs_diff_mean"] for seed in seeds if seed["logp_abs_diff_mean"] is not None]
            ),
        }
        if baseline is None:
            baseline = entry["final_window_reward_mean"]
        elif baseline is not None and entry["final_window_reward_mean"] is not None:
            entry["final_window_reward_delta_vs_baseline"] = entry["final_window_reward_mean"] - baseline
        report["arms"][arm] = entry
    return report


def _plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, (arm, entry) in enumerate(report["arms"].items()):
        for seed_index, seed in enumerate(entry["seeds"]):
            ax.plot(
                seed["reward_curve"], color=f"C{index}", alpha=0.85, linewidth=1, label=arm if seed_index == 0 else None
            )
    ax.set_xlabel("training step")
    ax.set_ylabel(REWARD)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="ARM=RUN_DIR[,RUN_DIR...]; the first arm is the baseline.")
    parser.add_argument("--window", type=int, default=20, help="Final steps averaged per seed.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG of the reward curves.")
    args = parser.parse_args()
    arms = {}
    for spec in args.arms:
        name, _, dirs = spec.partition("=")
        arms[name] = [Path(path) for path in dirs.split(",") if path]
    report = summarize(arms, args.window)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    if args.plot is not None:
        _plot(report, args.plot)
    for arm, entry in report["arms"].items():
        delta = entry.get("final_window_reward_delta_vs_baseline")
        print(
            f"{arm}: final-window reward {entry['final_window_reward_mean']:.4f}"
            + (f" (std {entry['final_window_reward_std']:.4f})" if entry["final_window_reward_std"] is not None else "")
            + (f" delta vs baseline {delta:+.4f}" if delta is not None else " [baseline]")
            + f"; logp_abs_diff_mean {entry['logp_abs_diff_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
