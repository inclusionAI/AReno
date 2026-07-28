"""CPU-only baseline and evaluation harness for the 2048 agentic example.

Phase 1 always runs on CPU: play random-action baseline episodes on seeded
boards and report mean episode score, max tile, and invalid-move rate. Phase 2
is optional: if ``--base-url`` points at a served policy, replay the trained
policy's ``choose_moves`` episodes on the same boards/seeds and print the
trained-vs-baseline improvement. No GPU, no external database, no sandbox.

Run from the repository root:

    python examples/agentic/2048/baseline.py --count 64 --seed 2026 --cap 32 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402

logger = logging.getLogger(__name__)

CHOOSE_MOVES_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_moves",
        "description": "Choose the 2048 direction sequence to play from the current board.",
        "parameters": {
            "type": "object",
            "properties": {
                "moves": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "description": "Legal directions to play, in order.",
                    "items": {"type": "string", "enum": ["up", "down", "left", "right"]},
                }
            },
            "required": ["moves"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are an expert 2048 player. "
    "Choose a sequence of legal directions by calling the choose_moves tool. "
    "Order moves to maximize merges and grow toward larger tiles. "
    "Stop once no direction changes the board; do not pad with no-op moves."
)


def evaluate_random(records: list[dict], *, cap: int, trials: int) -> dict[str, float]:
    """Run the random-action baseline across the given records."""

    scores: list[float] = []
    max_tiles: list[int] = []
    invalid_rates: list[float] = []
    for record in records:
        board = game.normalize_board(record["board"])
        summary = game.random_episode(board, seed=int(record["seed"]), cap=cap, trials=trials)
        scores.append(float(summary["score"]))
        max_tiles.append(int(summary["max_tile"]))
        invalid_rates.append(float(summary["invalid_rate"]))
    return _summarize("random", scores, max_tiles, invalid_rates)


def evaluate_policy(records: list[dict], client, *, model: str, cap: int) -> dict[str, float]:
    """Replay trained-policy episodes on the same boards/seeds."""

    scores: list[float] = []
    max_tiles: list[int] = []
    invalid_rates: list[float] = []
    for record in records:
        board = game.normalize_board(record["board"])
        moves = _policy_moves(client, model, board)
        result = game.play_episode(board, moves, seed=int(record["seed"]), cap=cap)
        scores.append(float(result.score))
        max_tiles.append(int(result.max_tile))
        invalid_rates.append(float(result.invalid_rate))
    return _summarize("policy", scores, max_tiles, invalid_rates)


def _summarize(name: str, scores: list[float], max_tiles: list[int], invalid_rates: list[float]) -> dict[str, float]:
    n = len(scores) or 1
    return {
        "agent": name,
        "mean_score": sum(scores) / n,
        "mean_max_tile": sum(max_tiles) / n,
        "mean_invalid_rate": sum(invalid_rates) / n,
        "episodes": len(scores),
    }


def _policy_moves(client, model: str, board: game.Board) -> list[str]:
    """Ask the served policy for one ``choose_moves`` episode and parse it."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game.format_prompt(board)},
        ],
        tools=[CHOOSE_MOVES_TOOL],
        tool_choice={"type": "function", "function": {"name": "choose_moves"}},
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    tool_calls = choices[0].get("message", {}).get("tool_calls", []) if choices else []
    for call in tool_calls:
        if call.get("function", {}).get("name") != "choose_moves":
            continue
        arguments = call.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return []
        return game.parse_moves(arguments)
    return []


def _make_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Policy evaluation requires `openai`. Install it with `pip install openai`.") from exc
    return OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)


def _improvement(random: dict[str, float], policy: dict[str, float]) -> dict[str, float]:
    return {
        "score_delta": policy["mean_score"] - random["mean_score"],
        "max_tile_delta": policy["mean_max_tile"] - random["mean_max_tile"],
        "invalid_rate_delta": policy["mean_invalid_rate"] - random["mean_invalid_rate"],
    }


def _print_human(random: dict[str, float], policy: dict[str, float | None] | None) -> None:
    print("2048 random-action baseline:")
    print(f"  episodes          : {random['episodes']}")
    print(f"  mean episode score: {random['mean_score']:.2f}")
    print(f"  mean max tile     : {random['mean_max_tile']:.2f}")
    print(f"  mean invalid rate : {random['mean_invalid_rate']:.3f}")
    if policy is None:
        return
    imp = _improvement(random, policy)
    print("\nTrained policy:")
    print(f"  episodes          : {policy['episodes']}")
    print(f"  mean episode score: {policy['mean_score']:.2f}")
    print(f"  mean max tile     : {policy['mean_max_tile']:.2f}")
    print(f"  mean invalid rate : {policy['mean_invalid_rate']:.3f}")
    print("\nTrained-vs-baseline improvement:")
    print(f"  score delta        : {imp['score_delta']:+.2f}")
    print(f"  max tile delta     : {imp['max_tile_delta']:+.2f}")
    print(f"  invalid rate delta : {imp['invalid_rate_delta']:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2048 random baseline and optional policy evaluation.")
    parser.add_argument("--boards", default=None, help="Path to a JSONL dataset; generated in-process if omitted.")
    parser.add_argument("--count", type=int, default=64, help="Number of boards when generating in-process.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for board generation / baseline rollouts.")
    parser.add_argument("--cap", type=int, default=game.DEFAULT_EPISODE_CAP, help="Episode length cap.")
    parser.add_argument("--trials", type=int, default=8, help="Random baseline trials per board.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for trained-policy evaluation.")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of human-readable text.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    records = _load_records(args)
    random = evaluate_random(records, cap=args.cap, trials=args.trials)
    policy: dict[str, float] | None = None
    if args.base_url:
        client = _make_client(args)
        policy = evaluate_policy(records, client, model=args.model, cap=args.cap)

    if args.json:
        payload: dict[str, Any] = {"random_baseline": random}
        if policy is not None:
            payload["trained_policy"] = policy
            payload["improvement"] = _improvement(random, policy)
        print(json.dumps(payload, indent=2))
    else:
        _print_human(random, policy)


def _load_records(args: argparse.Namespace) -> list[dict]:
    if args.boards:
        path = Path(args.boards).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"2048 dataset not found: {path}")
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records
    return dataset_generator.generate_records(args.count, seed=args.seed, cap=args.cap, trials=args.trials)


if __name__ == "__main__":
    main()