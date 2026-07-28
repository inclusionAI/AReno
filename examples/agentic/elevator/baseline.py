"""First-come-first-served baseline for the elevator-dispatch example.

The issue asks for a comparison against a first-come-first-served (FCFS) policy
on delivered passengers, mean wait, and invalid actions. This module runs the
engine's FCFS policy over a seeded set of buildings and reports those metrics,
reusing :func:`game.play` so the metric channels are identical to a trained
policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = game.DEFAULT_SEED
DEFAULT_MAX_STEPS = game.DEFAULT_MAX_STEPS


def fcfs_baseline(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    max_steps: int = DEFAULT_MAX_STEPS,
    floors: int = game.DEFAULT_FLOORS,
    capacity: int = game.DEFAULT_CAPACITY,
    arrivals_per_building: int = 6,
) -> dict[str, float]:
    """Aggregate FCFS metrics over ``count`` seeded buildings.

    Returns mean ``delivered_passengers``, mean ``mean_wait``, mean
    ``invalid_rate``, and the episode ``count``. Fully reproducible given the
    seed.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    buildings = dataset_generator.generate_records(
        count, seed=seed, floors=floors, capacity=capacity, arrivals_per_building=arrivals_per_building
    )
    delivered: list[float] = []
    waits: list[float] = []
    invalid_rates: list[float] = []
    for building in buildings:
        actions = game.fcfs_actions(building)
        metrics = game.play(building, actions, max_steps=max_steps)
        delivered.append(float(metrics["delivered_passengers"]))
        waits.append(float(metrics["mean_wait"]))
        invalid_rates.append(float(metrics["invalid_rate"]))
    return {
        "count": float(count),
        "mean_delivered": sum(delivered) / count,
        "mean_wait": sum(waits) / count,
        "mean_invalid_rate": sum(invalid_rates) / count,
    }


def _format_stats(stats: dict[str, float]) -> str:
    return (
        f"FCFS baseline over {int(stats['count'])} buildings: "
        f"mean_delivered={stats['mean_delivered']:.2f} "
        f"mean_wait={stats['mean_wait']:.2f} "
        f"mean_invalid_rate={stats['mean_invalid_rate']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report FCFS baseline metrics for the Areno elevator agentic example.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of buildings.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum valid actions per episode.")
    parser.add_argument("--floors", type=int, default=game.DEFAULT_FLOORS, help="Floors per building.")
    parser.add_argument("--capacity", type=int, default=game.DEFAULT_CAPACITY, help="Car capacity.")
    parser.add_argument("--arrivals", type=int, default=6, help="Arrivals per building.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.floors < 2:
        raise ValueError("--floors must be >= 2")
    if args.capacity < 1:
        raise ValueError("--capacity must be >= 1")

    stats = fcfs_baseline(
        args.count,
        seed=args.seed,
        max_steps=args.max_steps,
        floors=args.floors,
        capacity=args.capacity,
        arrivals_per_building=args.arrivals,
    )
    print(_format_stats(stats))


if __name__ == "__main__":
    main()
