#!/usr/bin/env python
"""Reproducible, CPU-only GaLore rank and learning-rate sweep.

The script optimizes two synthetic matrix objectives with the reference
implementation and prints sampled convergence curves plus theoretical FP32
moment-storage tables.  It does not train a model, access a dataset, or measure
peak memory.  Run from the repository root, for example:

    python tools/galore_sweep.py --objective both --steps 60

All random data and initial parameters are generated from local generators.
The SVD projector is frozen after the first gradient, matching the reference
implementation and making repeated runs directly comparable.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

# Make direct ``python tools/galore_sweep.py`` work without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from areno.optim.galore_reference import (  # noqa: E402
    GaLoreAdamWReference,
    adamw_state_numel,
    galore_state_numel,
)


LossAndGrad = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Problem:
    name: str
    initial: torch.Tensor
    loss_and_grad: LossAndGrad


@dataclass
class RunResult:
    objective: str
    lr: float
    rank: int | None
    losses: dict[int, float]

    @property
    def method(self) -> str:
        return "AdamW" if self.rank is None else f"GaLore-r{self.rank}"


def make_quadratic(seed: int, rows: int, cols: int) -> Problem:
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(rows, cols, generator=generator)
    target = torch.randn(rows, cols, generator=generator)
    curvature = 0.5 + torch.rand(rows, cols, generator=generator)
    scale = float(rows * cols)

    def loss_and_grad(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = weight - target
        loss = 0.5 * (curvature * delta.square()).mean()
        gradient = curvature * delta / scale
        return loss, gradient

    return Problem("quadratic", initial, loss_and_grad)


def make_linear_regression(seed: int, rows: int, cols: int, batch_size: int) -> Problem:
    generator = torch.Generator().manual_seed(seed)
    initial = 0.1 * torch.randn(rows, cols, generator=generator)
    features = torch.randn(batch_size, rows, generator=generator)
    true_weight = 0.5 * torch.randn(rows, cols, generator=generator)
    targets = features @ true_weight + 0.01 * torch.randn(batch_size, cols, generator=generator)
    scale = float(batch_size * cols)

    def loss_and_grad(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = features @ weight - targets
        loss = 0.5 * residual.square().mean()
        gradient = features.transpose(0, 1) @ residual / scale
        return loss, gradient

    return Problem("linear", initial, loss_and_grad)


def curve_checkpoints(steps: int) -> list[int]:
    """Return stable, useful checkpoints for both short and long runs."""

    candidates = (0, 1, 5, 10, 20, 40, 60, 80, 100, steps // 2, steps)
    return sorted({step for step in candidates if 0 <= step <= steps})


def run_one(problem: Problem, lr: float, rank: int | None, steps: int, checkpoints: list[int]) -> RunResult:
    parameter = torch.nn.Parameter(problem.initial.clone())
    if rank is None:
        optimizer = torch.optim.AdamW(
            [parameter], lr=lr, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, foreach=False
        )
    else:
        optimizer = GaLoreAdamWReference(
            [parameter], rank=rank, lr=lr, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0
        )

    initial_loss, _ = problem.loss_and_grad(parameter.detach())
    losses = {0: float(initial_loss)}
    for step in range(1, steps + 1):
        _, gradient = problem.loss_and_grad(parameter.detach())
        parameter.grad = gradient
        optimizer.step()
        if step in checkpoints:
            loss, _ = problem.loss_and_grad(parameter.detach())
            losses[step] = float(loss)
    return RunResult(problem.name, lr, rank, losses)


def print_curve_table(problem: Problem, results: list[RunResult], checkpoints: list[int]) -> None:
    print(f"\n## {problem.name} convergence (loss)")
    print("lr method " + " ".join(f"step_{step}" for step in checkpoints))
    for result in results:
        values = " ".join(f"{result.losses[step]:.8e}" for step in checkpoints)
        print(f"{result.lr:g} {result.method} {values}")


def print_final_table(results: list[RunResult], steps: int) -> None:
    print(f"\n## final loss at step {steps}")
    print("objective lr method final_loss")
    for result in results:
        print(f"{result.objective} {result.lr:g} {result.method} {result.losses[steps]:.8e}")


def print_memory_table(shape: tuple[int, int], ranks: list[int]) -> None:
    full_numel = adamw_state_numel(shape)
    print(f"\n## theoretical FP32 moment memory for one {shape[0]}x{shape[1]} matrix")
    print("method rank moment_numel moment_KiB pct_of_AdamW")
    print(f"AdamW full {full_numel} {full_numel * 4 / 1024:.3f} 100.0")
    for rank in ranks:
        moment_numel = galore_state_numel(shape, rank)
        print(
            f"GaLore {rank} {moment_numel} {moment_numel * 4 / 1024:.3f} "
            f"{100 * moment_numel / full_numel:.1f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("quadratic", "linear", "both"), default="both")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=593)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1 or args.rows < 1 or args.cols < 1 or args.batch_size < 1:
        raise SystemExit("steps, rows, cols, and batch-size must be positive")
    if not args.ranks or any(rank <= 0 or rank > min(args.rows, args.cols) for rank in args.ranks):
        raise SystemExit("every rank must be positive and no greater than min(rows, cols)")
    if not args.lrs or any(lr < 0 for lr in args.lrs):
        raise SystemExit("learning rates must be non-negative")
    if len(set(args.ranks)) != len(args.ranks):
        raise SystemExit("ranks must not contain duplicates")

    # Small CPU matrices are faster and more reproducible with one BLAS thread.
    torch.set_num_threads(1)
    checkpoints = curve_checkpoints(args.steps)
    problem_factories = []
    if args.objective in ("quadratic", "both"):
        problem_factories.append(lambda: make_quadratic(args.seed, args.rows, args.cols))
    if args.objective in ("linear", "both"):
        problem_factories.append(
            lambda: make_linear_regression(args.seed + 1, args.rows, args.cols, args.batch_size)
        )

    all_results: list[RunResult] = []
    for factory in problem_factories:
        problem = factory()
        results = []
        for lr in args.lrs:
            results.append(run_one(problem, lr, None, args.steps, checkpoints))
            for rank in args.ranks:
                results.append(run_one(problem, lr, rank, args.steps, checkpoints))
        print_curve_table(problem, results, checkpoints)
        all_results.extend(results)

    print_final_table(all_results, args.steps)
    print_memory_table((args.rows, args.cols), args.ranks)
    print("\nMemory is theoretical persistent FP32 moment storage only; no peak-memory claim is made.")
    print(f"seed={args.seed}; steps={args.steps}; projector=frozen_after_first_gradient; weight_decay=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
