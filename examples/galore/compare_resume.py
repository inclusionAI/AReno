"""Compare an uninterrupted ``areno train`` run with a resumed one.

Both runs come from ``train_seeded.py``: the uninterrupted run saved weights
and optimizer state every ``--save-interval`` steps, and the resumed run
restarted from one of those checkpoints with ``--resume-step``. For every step
both runs trained, the script reports the trainer-native TensorBoard scalars
(loss, reward, ``logp_abs_diff_mean``, gradient norm) and their largest
difference, and whether the first resumed step matches exactly.

CUDA backward passes are not bitwise deterministic, so later steps drift even
without a restart. ``--control`` names a second uninterrupted run with the same
seed; its differences from the first run are the noise floor that resumed
differences should stay within. Runs may be the gzipped scalar logs under
``examples/galore/results``. ``--continuous-final``/``--resumed-final``
additionally compare final checkpoints tensor by tensor (weights and
``Trainer.save_optimizer_state`` files).

    python examples/galore/compare_resume.py --continuous runs/e2e/cont \\
        --resumed runs/e2e/resumed --control runs/e2e/cont2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TAGS = (
    "train/loss",
    "rollout/rewards_mean",
    "train/logp_abs_diff_mean",
    "train/grad_norm",
    "train/galore_refreshed_matrices",
)


def _scalars(run: Path) -> dict[str, dict[int, float]]:
    if not (run / "tb").is_dir():
        from summarize import read_jsonl

        rows = read_jsonl(run / "scalars.jsonl")
        return {tag: {row["step"]: row[tag] for row in rows if tag in row} for tag in TAGS}

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    return {tag: {event.step: event.value for event in accumulator.Scalars(tag)} for tag in TAGS if tag in tags}


def _tensors(path: Path) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file

    tensors = {}
    for file in sorted(path.glob("*.safetensors")):
        tensors.update({f"weights/{name}": value for name, value in load_file(file).items()})
    for file in sorted(path.glob("areno_optimizer_rank*.pt")):
        state = torch.load(file, map_location="cpu", weights_only=True)
        tensors[f"{file.name}/global_step"] = torch.tensor(state["global_step"])
        _flatten(state["state"], f"{file.name}/state", tensors)
    return tensors


def _flatten(value, prefix: str, out: dict) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{prefix}/{key}", out)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _flatten(item, f"{prefix}/{index}", out)


def compare_steps(continuous: Path, resumed: Path, control: Path | None) -> dict:
    reference = _scalars(continuous)
    candidate = _scalars(resumed)
    noise = _scalars(control) if control is not None else {}
    report = {}
    for tag in TAGS:
        shared = sorted(set(reference.get(tag, {})) & set(candidate.get(tag, {})))
        if not shared:
            continue
        entry = {
            "steps": shared,
            "continuous": [reference[tag][step] for step in shared],
            "resumed": [candidate[tag][step] for step in shared],
            "first_step_equal": reference[tag][shared[0]] == candidate[tag][shared[0]],
            "max_abs_diff": max(abs(reference[tag][step] - candidate[tag][step]) for step in shared),
        }
        if tag in noise and all(step in noise[tag] for step in shared):
            entry["control"] = [noise[tag][step] for step in shared]
            entry["control_max_abs_diff"] = max(abs(reference[tag][step] - noise[tag][step]) for step in shared)
        report[tag] = entry
    return report


def compare_checkpoints(left_path: Path, right_path: Path) -> dict:
    left = _tensors(left_path)
    right = _tensors(right_path)
    names = sorted(set(left) & set(right))
    max_diff = 0.0
    mismatched = []
    for name in names:
        a, b = left[name], right[name]
        if a.shape != b.shape:
            mismatched.append(name)
            continue
        diff = (a.double() - b.double()).abs().max().item() if a.numel() else 0.0
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            mismatched.append(name)
    return {
        "compared_tensors": len(names),
        "only_in_first": sorted(set(left) - set(right)),
        "only_in_second": sorted(set(right) - set(left)),
        "bitwise_equal_tensors": len(names) - len(mismatched),
        "max_abs_diff": max_diff,
        "first_mismatches": mismatched[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--continuous-final", default=None)
    parser.add_argument("--resumed-final", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = {"steps": compare_steps(args.continuous, args.resumed, args.control)}
    if args.continuous_final and args.resumed_final:
        report["final_checkpoint"] = compare_checkpoints(
            args.continuous / "ckpt" / args.continuous_final, args.resumed / "ckpt" / args.resumed_final
        )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text[:6000])


if __name__ == "__main__":
    main()
