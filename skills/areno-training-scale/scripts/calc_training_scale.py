#!/usr/bin/env python3
"""Training-scale calculator for AReno.

Computes effective global batch, updates per epoch, total updates, and
approximate processed tokens from dataset size, micro batch, accumulation,
and GPU count.  Also solves accumulation for a target global batch.

Uses correct counting rules for SFT, DPO, and online RL (GSPO/GRPO/PPO):
  - SFT: 1 row = 1 sample
  - DPO: 1 row = 1 chosen/rejected pair
  - Online RL: 1 row = 1 prompt with n_samples rollouts
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Algorithm metadata (mirrors areno.api.algorithms without importing it)
# ---------------------------------------------------------------------------

_OFFLINE_ALGOS = frozenset({"sft", "dpo"})
_ONLINE_RL_ALGOS = frozenset({"gspo", "grpo", "ppo"})
_ALL_ALGOS = _OFFLINE_ALGOS | _ONLINE_RL_ALGOS


def _is_online_rl(algo: str) -> bool:
    return algo in _ONLINE_RL_ALGOS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScaleInput:
    """All parameters the user can provide."""

    algo: str
    dataset_size: int
    mini_bs: int = 16
    gradient_accumulation_steps: int | None = None
    world_size: int = 8
    tp_size: int = 4
    # RL-specific
    batch_size: int | None = None
    n_samples: int | None = None
    # General
    epochs: int = 1
    avg_seq_len: int = 2048
    # Reverse solve
    target_global_batch: int | None = None


@dataclass
class ScaleResult:
    """Everything the calculator outputs."""

    algo: str
    dp_size: int
    global_batch: int
    gradient_accumulation_steps: int
    updates_per_epoch: int
    total_updates: int
    approx_tokens: int
    samples_per_step: int
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_input(inp: ScaleInput) -> None:
    """Raise ValueError on malformed input."""

    algo = inp.algo.strip().lower()
    if algo not in _ALL_ALGOS:
        raise ValueError(
            f"Unknown algorithm {inp.algo!r}; supported: {sorted(_ALL_ALGOS)}"
        )
    if inp.dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if inp.mini_bs <= 0:
        raise ValueError("mini_bs must be positive")
    if inp.gradient_accumulation_steps is not None and inp.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive when provided")
    if inp.world_size <= 0:
        raise ValueError("world_size must be positive")
    if inp.tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if inp.world_size % inp.tp_size != 0:
        raise ValueError(
            f"world_size ({inp.world_size}) must be divisible by tp_size ({inp.tp_size})"
        )
    if _is_online_rl(algo):
        if inp.batch_size is not None and inp.batch_size <= 0:
            raise ValueError("batch_size must be positive for RL algorithms")
        if inp.n_samples is not None and inp.n_samples <= 0:
            raise ValueError("n_samples must be positive for RL algorithms")
    if inp.epochs <= 0:
        raise ValueError("epochs must be positive")
    if inp.avg_seq_len <= 0:
        raise ValueError("avg_seq_len must be positive")
    if inp.target_global_batch is not None and inp.target_global_batch <= 0:
        raise ValueError("target_global_batch must be positive when provided")


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------


def calculate(inp: ScaleInput) -> ScaleResult:
    """Run the training-scale calculation and return a ScaleResult."""

    algo = inp.algo.strip().lower()
    _validate_input(inp)

    dp_size = inp.world_size // inp.tp_size
    warnings: list[str] = []

    if _is_online_rl(algo):
        return _calculate_online_rl(inp, algo, dp_size, warnings)
    return _calculate_offline(inp, algo, dp_size, warnings)


def _calculate_offline(
    inp: ScaleInput, algo: str, dp_size: int, warnings: list[str]
) -> ScaleResult:
    """SFT / DPO: 1 row = 1 sample (or 1 pair for DPO)."""

    # Determine gradient_accumulation_steps
    if inp.target_global_batch is not None:
        # Reverse solve: find grad_accum so that mini_bs * grad_accum * dp_size
        # is as close to target_global_batch as possible.
        grad_accum = _solve_grad_accum(
            inp.target_global_batch, inp.mini_bs, dp_size, warnings
        )
    elif inp.gradient_accumulation_steps is not None:
        grad_accum = inp.gradient_accumulation_steps
    else:
        # Default: 1 (user can override)
        grad_accum = 1

    global_batch = inp.mini_bs * grad_accum * dp_size
    samples_per_step = global_batch  # 1 row = 1 sample for SFT/DPO

    updates_per_epoch = math.ceil(inp.dataset_size / global_batch)
    total_updates = updates_per_epoch * inp.epochs
    approx_tokens = total_updates * samples_per_step * inp.avg_seq_len

    if inp.dataset_size % global_batch != 0:
        remainder = inp.dataset_size % global_batch
        warnings.append(
            f"dataset_size ({inp.dataset_size}) is not divisible by "
            f"global_batch ({global_batch}); last step uses {remainder} samples"
        )

    return ScaleResult(
        algo=algo,
        dp_size=dp_size,
        global_batch=global_batch,
        gradient_accumulation_steps=grad_accum,
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        approx_tokens=approx_tokens,
        samples_per_step=samples_per_step,
        warnings=warnings,
    )


def _calculate_online_rl(
    inp: ScaleInput, algo: str, dp_size: int, warnings: list[str]
) -> ScaleResult:
    """GSPO / GRPO / PPO: 1 row = 1 prompt with n_samples rollouts."""

    batch_size = inp.batch_size if inp.batch_size is not None else 32
    n_samples = inp.n_samples if inp.n_samples is not None else 8

    # In AReno's RL loop, batch_size is the number of prompts per step
    # (already a global quantity).  The training-side micro batch is
    # mini_bs * grad_accum * dp_size, which must equal batch_size * n_samples
    # for a single optimizer step.
    total_sequences = batch_size * n_samples

    if inp.target_global_batch is not None:
        # For RL, target_global_batch refers to the prompt batch size.
        # Solve for gradient_accumulation_steps so that the train micro batch
        # covers the full rollout: mini_bs * grad_accum * dp_size = target * n_samples
        target_sequences = inp.target_global_batch * n_samples
        grad_accum = _solve_grad_accum(
            target_sequences, inp.mini_bs, dp_size, warnings
        )
        batch_size = inp.target_global_batch
        total_sequences = batch_size * n_samples
    elif inp.gradient_accumulation_steps is not None:
        grad_accum = inp.gradient_accumulation_steps
    else:
        # Default: grad_accum covers the full rollout in one optimizer step
        # i.e. mini_bs * grad_accum * dp_size = batch_size * n_samples
        grad_accum = max(1, math.ceil(total_sequences / (inp.mini_bs * dp_size)))

    global_batch = batch_size  # prompt-level global batch
    samples_per_step = total_sequences

    updates_per_epoch = math.ceil(inp.dataset_size / batch_size)
    total_updates = updates_per_epoch * inp.epochs
    approx_tokens = total_updates * samples_per_step * inp.avg_seq_len

    if inp.dataset_size % batch_size != 0:
        remainder = inp.dataset_size % batch_size
        warnings.append(
            f"dataset_size ({inp.dataset_size}) is not divisible by "
            f"batch_size ({batch_size}); last step uses {remainder} prompts"
        )

    # Warn if train micro batch doesn't cover the full rollout
    train_micro_total = inp.mini_bs * grad_accum * dp_size
    if train_micro_total != total_sequences:
        warnings.append(
            f"train micro-batch total ({train_micro_total} = mini_bs({inp.mini_bs}) "
            f"* grad_accum({grad_accum}) * dp_size({dp_size})) does not equal "
            f"total sequences per step ({total_sequences} = batch_size({batch_size}) "
            f"* n_samples({n_samples})); gradient accumulation may not cover "
            f"the full rollout in one optimizer step"
        )

    return ScaleResult(
        algo=algo,
        dp_size=dp_size,
        global_batch=global_batch,
        gradient_accumulation_steps=grad_accum,
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        approx_tokens=approx_tokens,
        samples_per_step=samples_per_step,
        warnings=warnings,
    )


def _solve_grad_accum(
    target: int, mini_bs: int, dp_size: int, warnings: list[str]
) -> int:
    """Find the smallest gradient_accumulation_steps so that
    mini_bs * grad_accum * dp_size >= target.

    Emits a warning if the product does not exactly equal target.
    """

    divisor = mini_bs * dp_size
    if divisor <= 0:
        raise ValueError("mini_bs * dp_size must be positive")
    grad_accum = max(1, math.ceil(target / divisor))
    actual = mini_bs * grad_accum * dp_size
    if actual != target:
        warnings.append(
            f"target ({target}) is not divisible by mini_bs*dp_size ({divisor}); "
            f"closest global_batch = {actual} with gradient_accumulation_steps = {grad_accum}"
        )
    return grad_accum


# ---------------------------------------------------------------------------
# Suggest valid combinations when division is uneven
# ---------------------------------------------------------------------------


def suggest_combinations(
    *,
    target_global_batch: int,
    mini_bs_range: Sequence[int] = (8, 16, 32, 64),
    dp_size_range: Sequence[int] = (1, 2, 4, 8),
) -> list[dict]:
    """Return valid (mini_bs, dp_size, grad_accum) combos that produce
    exactly target_global_batch, sorted by smallest grad_accum first.
    """

    combos: list[dict] = []
    for dp in dp_size_range:
        for mbs in mini_bs_range:
            divisor = mbs * dp
            if target_global_batch % divisor == 0:
                ga = target_global_batch // divisor
                combos.append(
                    {
                        "mini_bs": mbs,
                        "dp_size": dp,
                        "gradient_accumulation_steps": ga,
                        "global_batch": target_global_batch,
                    }
                )
    combos.sort(key=lambda c: (c["gradient_accumulation_steps"], c["dp_size"]))
    return combos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate AReno training scale: global batch, steps, and tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # SFT with 10k rows, 2 GPUs, tp=1
  python calc_training_scale.py --algo sft --dataset-size 10000 \\
      --mini-bs 16 --world-size 2 --tp-size 1 --epochs 3

  # GSPO with 500 prompts, 8 samples each
  python calc_training_scale.py --algo gspo --dataset-size 500 \\
      --batch-size 32 --n-samples 8 --mini-bs 16 \\
      --world-size 8 --tp-size 4 --epochs 1

  # Solve gradient_accumulation for target global batch = 128
  python calc_training_scale.py --algo sft --dataset-size 10000 \\
      --mini-bs 16 --world-size 8 --tp-size 4 \\
      --target-global-batch 128

  # Suggest valid mini_bs/dp_size/grad_accum combos for global batch = 64
  python calc_training_scale.py --suggest --target-global-batch 64
""",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sft",
        choices=sorted(_ALL_ALGOS),
        help="Training algorithm (default: sft).",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Number of rows in the training dataset.",
    )
    parser.add_argument(
        "--mini-bs",
        type=int,
        default=16,
        help="Backend training microbatch size (default: 16).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Optimizer step interval in microbatches; auto if omitted.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="Total device count (default: 8).",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="Tensor parallel size (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prompt batch size per step for RL algorithms (default: 32).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Rollout samples per prompt for RL algorithms (default: 8).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1).",
    )
    parser.add_argument(
        "--avg-seq-len",
        type=int,
        default=2048,
        help="Average sequence length for token estimation (default: 2048).",
    )
    parser.add_argument(
        "--target-global-batch",
        type=int,
        default=None,
        help="Solve gradient_accumulation for this target global batch size.",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Print valid mini_bs/dp_size/grad_accum combos for --target-global-batch.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output result as JSON (default: human-readable table).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --suggest mode: just print combos and exit
    if args.suggest:
        if args.target_global_batch is None:
            parser.error("--suggest requires --target-global-batch")
        combos = suggest_combinations(target_global_batch=args.target_global_batch)
        if not combos:
            print(
                f"No valid combinations found for target_global_batch={args.target_global_batch}"
            )
            sys.exit(1)
        if args.output_json:
            print(json.dumps(combos, indent=2))
        else:
            print(f"Valid combinations for global_batch={args.target_global_batch}:")
            print(f"  {'mini_bs':>8}  {'dp_size':>8}  {'grad_accum':>10}  {'global_batch':>12}")
            for c in combos:
                print(
                    f"  {c['mini_bs']:>8}  {c['dp_size']:>8}  "
                    f"{c['gradient_accumulation_steps']:>10}  {c['global_batch']:>12}"
                )
        return

    # Normal calculation mode
    if args.dataset_size is None:
        parser.error("--dataset-size is required (unless using --suggest)")

    inp = ScaleInput(
        algo=args.algo,
        dataset_size=args.dataset_size,
        mini_bs=args.mini_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=args.world_size,
        tp_size=args.tp_size,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        epochs=args.epochs,
        avg_seq_len=args.avg_seq_len,
        target_global_batch=args.target_global_batch,
    )

    result = calculate(inp)

    if args.output_json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_result(result)


def _print_result(r: ScaleResult) -> None:
    """Human-readable output."""

    algo_label = r.algo.upper()
    if r.algo in _ONLINE_RL_ALGOS:
        algo_label += " (online RL)"
    else:
        algo_label += " (offline)"

    print(f"AReno Training Scale  [{algo_label}]")
    print("=" * 50)
    print(f"  dp_size                  = {r.dp_size}")
    print(f"  global_batch             = {r.global_batch}")
    print(f"  gradient_accumulation    = {r.gradient_accumulation_steps}")
    print(f"  samples_per_step         = {r.samples_per_step}")
    print(f"  updates_per_epoch        = {r.updates_per_epoch}")
    print(f"  epochs                   = (see input)")
    print(f"  total_updates            = {r.total_updates}")
    print(f"  approx_tokens            = {r.approx_tokens:,}")
    if r.warnings:
        print()
        print("  Warnings:")
        for w in r.warnings:
            print(f"    ⚠  {w}")


if __name__ == "__main__":
    main()