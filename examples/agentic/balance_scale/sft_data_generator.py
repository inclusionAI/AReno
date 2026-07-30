"""Generate SFT training data for the balance-scale example.

Uses an optimal divide-and-conquer strategy to solve each puzzle, then
produces multi-turn conversation data (system → user → assistant weigh →
tool result → assistant weigh → ... → assistant submit_answer) that can
be used for supervised fine-tuning before RL.

The solver maintains a candidate set of (ball_index, direction) pairs and
splits them into three groups as evenly as possible for each weighing.
After each weighing, it eliminates candidates that are inconsistent with
the result. When only one candidate remains, it submits the answer.

Usage:
    python examples/agentic/balance_scale/sft_data_generator.py \
        --output /tmp/sft_data.jsonl \
        --count 256 \
        --seed 2026 \
        --num-balls-range 3 12
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def solve_puzzle(ball_set: game.BallSet) -> list[dict[str, Any]]:
    """Solve a puzzle using optimal ternary search.

    Returns a list of turns, each being a dict:
      {"action": "weigh", "left": [...], "right": [...], "result": "..."}
      {"action": "submit_answer", "ball_index": N, "direction": "..."}
    """

    num_balls = ball_set.num_balls
    odd = ball_set.odd_ball_index
    direction = ball_set.direction

    # Candidate set: each ball could be heavier or lighter
    candidates = {(i, d) for i in range(num_balls) for d in game.DIRECTIONS}

    # Track "normal" balls (confirmed not odd) for use as reference weights
    normal_balls: set[int] = set()

    turns: list[dict[str, Any]] = []
    weighings_used = 0

    while len(candidates) > 1 and weighings_used < ball_set.max_weighings:
        # If all candidates are the same ball with different directions,
        # we need one more weighing against a normal ball to determine direction
        candidate_balls = {b for b, _ in candidates}
        if len(candidate_balls) == 1 and len(candidates) == 2:
            ball = next(iter(candidate_balls))
            # Find a normal/reference ball
            ref = None
            for nb in sorted(normal_balls):
                if nb != ball:
                    ref = nb
                    break
            if ref is None:
                all_balls = set(range(num_balls))
                non_candidates = sorted(all_balls - candidate_balls)
                if non_candidates:
                    ref = non_candidates[0]
            if ref is not None and weighings_used < ball_set.max_weighings:
                result = game.weigh(ball_set, [ball], [ref], weighings_used=weighings_used)
                weighings_used += 1
                turns.append({
                    "action": "weigh",
                    "left": [ball],
                    "right": [ref],
                    "result": result,
                })
                # Determine direction from result
                if result == "left_heavy":
                    candidates = {(ball, "heavier")}
                elif result == "right_heavy":
                    candidates = {(ball, "lighter")}
                # If balanced, something went wrong — keep both
                continue
        left_balls, right_balls = _best_split(candidates, num_balls, normal_balls, turns)

        if not left_balls or not right_balls:
            break

        result = game.weigh(ball_set, left_balls, right_balls, weighings_used=weighings_used)
        weighings_used += 1

        turns.append({
            "action": "weigh",
            "left": sorted(left_balls),
            "right": sorted(right_balls),
            "result": result,
        })

        # Update normal balls (those confirmed not odd by balanced result)
        if result == "balanced":
            normal_balls.update(left_balls)
            normal_balls.update(right_balls)

        # Eliminate inconsistent candidates
        candidates = _filter_candidates(candidates, left_balls, right_balls, result)

    # Submit the answer
    if candidates:
        ball_idx, dir_val = next(iter(candidates))
    else:
        ball_idx, dir_val = odd, direction

    turns.append({
        "action": "submit_answer",
        "ball_index": ball_idx,
        "direction": dir_val,
    })

    return turns


def _best_split(
    candidates: set[tuple[int, str]],
    num_balls: int,
    normal_balls: set[int],
    prev_turns: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Choose left/right groups to split candidates as evenly as possible.

    Uses normal (confirmed not-odd) balls as reference weights when available.
    Avoids repeating the same weighing as the previous turn.
    """

    candidate_balls = sorted({b for b, _ in candidates})

    if len(candidate_balls) <= 1:
        return [], []

    if len(candidate_balls) == 2:
        # Try to use a normal ball as reference to distinguish the two candidates
        # If candidate 0 is weighed against a known normal ball:
        #   balanced → candidate 0 is not odd → answer is candidate 1
        #   unbalanced → candidate 0 is odd (direction determined by tilt)
        for normal in sorted(normal_balls):
            if normal not in candidate_balls:
                return [candidate_balls[0]], [normal]
        # No confirmed normal ball — use any ball not in candidates
        all_balls = set(range(num_balls))
        non_candidates = sorted(all_balls - set(candidate_balls))
        if non_candidates:
            return [candidate_balls[0]], [non_candidates[0]]
        # All balls are candidates, put one on each side
        return [candidate_balls[0]], [candidate_balls[1]]

    # For 3+ candidate balls, try to split into three groups
    third = max(1, len(candidate_balls) // 3)
    left_balls = candidate_balls[:third]
    right_balls = candidate_balls[third:third * 2]

    # Ensure equal size
    min_len = min(len(left_balls), len(right_balls))
    if min_len == 0:
        half = max(1, len(candidate_balls) // 2)
        left_balls = candidate_balls[:half]
        right_balls = candidate_balls[half:half + half]

    # Trim to equal size
    min_len = min(len(left_balls), len(right_balls))
    left_balls = left_balls[:min_len]
    right_balls = right_balls[:min_len]

    # Avoid repeating the exact same weighing as last turn
    if prev_turns:
        last = prev_turns[-1]
        if last.get("action") == "weigh":
            prev_key = (tuple(sorted(last["left"])), tuple(sorted(last["right"])))
            curr_key = (tuple(sorted(left_balls)), tuple(sorted(right_balls)))
            if prev_key == curr_key or prev_key == (tuple(sorted(right_balls)), tuple(sorted(left_balls))):
                # Same weighing — try using a normal ball instead
                for normal in sorted(normal_balls):
                    if normal not in candidate_balls:
                        return [candidate_balls[0]], [normal]
                # Swap one ball from left with an off-scale candidate
                off_scale = [b for b in candidate_balls if b not in left_balls and b not in right_balls]
                if off_scale:
                    right_balls = [off_scale[0]] * len(right_balls) if len(right_balls) == 1 else right_balls[:1] + [off_scale[0]]

    return left_balls, right_balls


def _filter_candidates(
    candidates: set[tuple[int, str]],
    left: list[int],
    right: list[int],
    result: str,
) -> set[tuple[int, str]]:
    """Eliminate candidates inconsistent with the weighing result.

    For each candidate (ball, direction), simulate what the weighing
    result *would* be if that candidate were the true odd ball, and
    keep only those that match the actual result.
    """

    left_set = set(left)
    right_set = set(right)
    new_candidates: set[tuple[int, str]] = set()

    for ball, dir_val in candidates:
        # Simulate: what would the scale show if this (ball, dir) were the answer?
        if ball not in left_set and ball not in right_set:
            # Ball is off-scale → result would be "balanced"
            simulated = "balanced"
        elif ball in left_set:
            # Ball is on left: heavier → left_heavy, lighter → right_heavy
            simulated = "left_heavy" if dir_val == "heavier" else "right_heavy"
        else:
            # Ball is on right: heavier → right_heavy, lighter → left_heavy
            simulated = "right_heavy" if dir_val == "heavier" else "left_heavy"

        if simulated == result:
            new_candidates.add((ball, dir_val))

    return new_candidates


def build_sft_record(
    ball_set: game.BallSet,
    turns: list[dict[str, Any]],
    system_prompt: str,
) -> dict[str, Any]:
    """Build an SFT training record with multi-turn conversation.

    The conversation follows the format:
      system → user(prompt) → assistant(weigh) → tool(result) →
      assistant(weigh) → tool(result) → ... → assistant(submit_answer)
    """

    user_prompt = game.format_prompt(ball_set)

    messages: list[dict[str, str | Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for i, turn in enumerate(turns):
        if turn["action"] == "weigh":
            # Assistant calls weigh
            assistant_content = f'{{"left": {turn["left"]}, "right": {turn["right"]}}}'
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [{
                    "id": f"weigh_{i}",
                    "type": "function",
                    "function": {
                        "name": "weigh",
                        "arguments": json.dumps({"left": turn["left"], "right": turn["right"]}),
                    },
                }],
            })
            # Tool result
            messages.append({
                "role": "tool",
                "tool_call_id": f"weigh_{i}",
                "name": "weigh",
                "content": json.dumps({"result": turn["result"], "weighings_used": i + 1}),
            })
        elif turn["action"] == "submit_answer":
            assistant_content = f'{{"ball_index": {turn["ball_index"]}, "direction": "{turn["direction"]}"}}'
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [{
                    "id": f"submit_{i}",
                    "type": "function",
                    "function": {
                        "name": "submit_answer",
                        "arguments": json.dumps({
                            "ball_index": turn["ball_index"],
                            "direction": turn["direction"],
                        }),
                    },
                }],
            })

    return {
        "messages": messages,
        "source": {
            "num_balls": ball_set.num_balls,
            "odd_ball_index": ball_set.odd_ball_index,
            "direction": ball_set.direction,
            "max_weighings": ball_set.max_weighings,
        },
        "num_turns": len(turns),
        "num_weighings": sum(1 for t in turns if t["action"] == "weigh"),
    }


def generate_sft_records(
    count: int = 128,
    *,
    seed: int = 2026,
    num_balls: int = 12,
    max_weighings: int = 0,
    num_balls_range: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Generate SFT training records with optimal solutions."""

    puzzles = dataset_generator.generate_records(
        count,
        seed=seed,
        num_balls=num_balls,
        max_weighings=max_weighings,
        num_balls_range=num_balls_range,
    )

    system_prompt = game.format_system_prompt()
    records: list[dict[str, Any]] = []

    for puzzle in puzzles:
        # SFT solver requires at least 3 balls (need a reference ball for direction)
        if puzzle["num_balls"] < 3:
            continue
        ball_set = game.BallSet(
            num_balls=puzzle["num_balls"],
            odd_ball_index=puzzle["odd_ball_index"],
            direction=puzzle["direction"],
            max_weighings=puzzle["max_weighings"],
        )
        turns = solve_puzzle(ball_set)
        record = build_sft_record(ball_set, turns, system_prompt)
        record["id"] = puzzle["id"]
        records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT training data for the balance-scale example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=128, help="Number of puzzles to solve.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--num-balls", type=int, default=12, help="Fixed number of balls.")
    parser.add_argument(
        "--num-balls-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="Random ball count per puzzle in [MIN, MAX].",
    )
    parser.add_argument("--max-weighings", type=int, default=0, help="Max weighings (0 = auto).")
    parser.add_argument(
        "--split", type=float, default=0.0,
        help="Split into train/test sets (test fraction, e.g. 0.33).",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    num_balls_range = None
    if args.num_balls_range is not None:
        lo, hi = args.num_balls_range
        if lo < 2 or hi < lo:
            raise ValueError("--num-balls-range must be MIN>=2 and MAX>=MIN")
        num_balls_range = (lo, hi)

    records = generate_sft_records(
        args.count,
        seed=args.seed,
        num_balls=args.num_balls,
        max_weighings=args.max_weighings,
        num_balls_range=num_balls_range,
    )

    # Print stats
    avg_weighings = sum(r["num_weighings"] for r in records) / len(records)
    print(f"Generated {len(records)} SFT records, avg {avg_weighings:.1f} weighings per puzzle", file=sys.stderr)

    if args.split > 0:
        n_test = max(1, int(len(records) * args.split))
        train, test = records[:-n_test], records[-n_test:]
        if args.output == "-":
            for r in train:
                print(json.dumps(r, ensure_ascii=False))
        else:
            out_path = Path(args.output).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            test_path = out_path.parent / f"{out_path.stem}_test{out_path.suffix}"
            with out_path.open("w", encoding="utf-8") as f:
                for r in train:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with test_path.open("w", encoding="utf-8") as f:
                for r in test:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Train: {out_path} ({len(train)} records)", file=sys.stderr)
            print(f"Test:  {test_path} ({len(test)} records)", file=sys.stderr)
    else:
        if args.output == "-":
            for r in records:
                print(json.dumps(r, ensure_ascii=False))
        else:
            out_path = Path(args.output).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()