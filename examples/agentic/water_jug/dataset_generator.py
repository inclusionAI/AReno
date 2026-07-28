"""Generate reproducible water-jug puzzle datasets as JSONL.

Each line: {"id": ..., "capacities": [...], "initial_state": [0,...], "target": N, "oracle_steps": K}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow importing game.py when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import game


def generate_puzzle(rng: random.Random, max_jugs: int = 3, max_cap: int = 10) -> dict | None:
    """Generate one solvable puzzle.  Returns None if unsolvable (rare)."""
    n = rng.randint(2, max_jugs)
    caps = []
    seen = set()
    for _ in range(n):
        c = rng.randint(2, max_cap)
        while c in seen:
            c = rng.randint(2, max_cap)
        seen.add(c)
        caps.append(c)
    caps.sort()

    target = rng.randint(1, max(caps))
    initial = [0] * n

    dist = game.bfs_distance(caps, initial, target)
    if dist is None or dist < 1:
        return None

    return {
        "capacities": caps,
        "initial_state": initial,
        "target": target,
        "oracle_steps": dist,
    }


def main():
    p = argparse.ArgumentParser(description="Generate water-jug puzzle JSONL")
    p.add_argument("--output", "-o", required=True, help="Output JSONL path")
    p.add_argument("--count", "-n", type=int, default=128, help="Number of puzzles")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    rng = random.Random(args.seed)
    puzzles: list[dict] = []
    total_solvable = 0
    total_steps = 0

    while len(puzzles) < args.count:
        pz = generate_puzzle(rng)
        if pz is None:
            continue
        pz["id"] = f"generated-{len(puzzles):06d}"
        puzzles.append(pz)
        total_solvable += 1
        total_steps += pz["oracle_steps"]

    with open(args.output, "w") as f:
        for pz in puzzles:
            f.write(json.dumps(pz) + "\n")

    avg = total_steps / len(puzzles) if puzzles else 0
    print(f"Generated {len(puzzles)} puzzles -> {args.output}")
    print(f"Solvable: {total_solvable}/{args.count}")
    print(f"Average oracle steps: {avg:.1f}")


if __name__ == "__main__":
    main()
