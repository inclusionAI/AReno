"""Evaluate any OpenAI-compatible LLM (or a served trained checkpoint) at Battleship.

This is the standalone, no-`areno`-dependency entry point for playing the game
with a model over many seeded fleets. Point it at any OpenAI-compatible chat
endpoint -- an `areno serve` instance of a trained checkpoint, vLLM, Ollama,
OpenAI's hosted API, or a DashScope-compatible endpoint:

    # Evaluate a served trained checkpoint
    areno serve --model-path ./runs/battleship/step_100 --port 8000 --world-size 1
    python examples/agentic/battleship/play_llm.py \
        --base-url http://127.0.0.1:8000/v1 --games 50

    # Evaluate an external LLM
    python examples/agentic/battleship/play_llm.py \
        --base-url https://api.openai.com/v1 --api-key "$OPENAI_API_KEY" \
        --model gpt-4o-mini --games 50

The network call (`_llm_step`) is isolated from the game loop (`_play_game`) so
the loop can be unit-tested with a deterministic injected step function.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

# Mirrors the tool contract in run_agent.py / web_ui.py so behavior is identical
# between this evaluator and an actual agentic training rollout.
SYSTEM_PROMPT = (
    "You are playing Battleship. Use the fire(coordinate) tool to sink all ships.\n\n"
    "Rules:\n"
    "- The grid is 8x8, columns numbered 1-8, rows labeled A-H.\n"
    "- Coordinates are given as letter+number, e.g., A1 (top-left), H8 (bottom-right).\n"
    "- The fire tool returns: miss (no ship), hit (ship not sunk), sunk (ship destroyed).\n"
    "- Do not fire at the same coordinate twice.\n"
    "- Do not fire outside the A1-H8 range.\n"
    "- Win by sinking all ships with as few shots as possible.\n"
    "- After each shot, you will see an updated board showing your hits (X), misses (o), and unknown cells (.)."
)

FIRE_TOOL = {
    "type": "function",
    "function": {
        "name": "fire",
        "description": "Fire a shot at a coordinate on the Battleship board.",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinate": {
                    "type": "string",
                    "description": "The coordinate to fire at, e.g., 'A1', 'B7', 'H8'. Uses letter row (A-H) and number column (1-8).",
                    "pattern": "^[A-H][1-8]$",
                },
            },
            "required": ["coordinate"],
            "additionalProperties": False,
        },
    },
}

# A "step" policy: given the running message history and current board state,
# return the next coordinate string (e.g. "C5") or None / an invalid string to
# register an invalid shot. Returning None keeps the loop progressing exactly
# like run_agent's synthetic empty-tool-call fallback.


def _board_prompt(state: game.GameState) -> str:
    """Build the opening user message describing the board the agent sees."""
    return (
        "You are playing Battleship. Sink all hidden ships on the 8x8 grid.\n\n"
        "Legend: X = your hit, o = your miss, . = unknown.\n"
        "Call the fire tool with a coordinate like 'C5'. Do not repeat a cell.\n\n"
        f"Board:\n{game.board_text(state)}\n\nYour shot:"
    )


def _play_game(
    step: Callable[[list[dict], game.GameState], str | None],
    record: dict,
    max_turns: int,
    show_boards: bool = False,
) -> dict:
    """Play one game with the injected step policy. Pure logic, no network.

    Returns per-game metrics matching the shape of `evaluate.evaluate_player`.
    """
    state = game.init_state(record)
    state.seed = record.get("seed")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _board_prompt(state)},
    ]
    invalid_shots = 0

    while not game.is_terminal(state) and state.shots_used < max_turns:
        coord = step(messages, state)
        coord_str = str(coord).upper() if coord else ""
        result = game.fire(state, coord_str)
        status = result.get("status", "invalid")
        if status == "invalid":
            invalid_shots += 1

        # Feed the updated board back into the running history so the model can
        # reason across turns (mirrors run_agent's per-turn board observation).
        messages.append({"role": "assistant", "content": f"I fired {coord_str or 'nothing'}."})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Result of {coord_str or 'your shot'}: {status}.\n"
                    f"Board:\n{game.board_text(state)}\n\nYour next shot:"
                ),
            }
        )

        if show_boards:
            print(f"[seed {state.seed}] fired {coord_str or '-'} -> {status}")
            print(game.board_text(state))

    total_hits = sum(len(s.hits) for s in state.ships)
    sunk_ships = sum(1 for s in state.ships if s.is_sunk)
    return {
        "win": game.is_win(state),
        "completion": total_hits / game.TOTAL_SHIP_CELLS,
        "shots_used": state.shots_used,
        "hits": total_hits,
        "sunk_ships": sunk_ships,
        "invalid_shots": invalid_shots,
        "seed": state.seed,
    }


def _parse_coord_from_response(response: Any) -> str | None:
    """Extract the coordinate from an OpenAI chat completion response."""
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    if not choices:
        return None
    tool_calls = choices[0].get("message", {}).get("tool_calls", []) if isinstance(choices[0], dict) else []
    for call in tool_calls:
        if call.get("function", {}).get("name") != "fire":
            continue
        args = call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        coord = args.get("coordinate") if isinstance(args, dict) else None
        if coord and game.parse_coordinate(str(coord)) is not None:
            return str(coord).upper()
    return None


def _llm_step(client: Any, model: str, messages: list[dict], state: game.GameState) -> str | None:
    """Ask the OpenAI-compatible endpoint for one fire coordinate."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[FIRE_TOOL],
        tool_choice={"type": "function", "function": {"name": "fire"}},
        stream=False,
    )
    return _parse_coord_from_response(response)


def play_model(args: argparse.Namespace) -> dict:
    """Run `args.games` seeded games against the LLM and aggregate metrics."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "play_llm.py requires `openai`. Install it with `pip install openai`."
        ) from exc

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)

    def step(messages: list[dict], state: game.GameState) -> str | None:
        return _llm_step(client, args.model, messages, state)

    results: list[dict] = []
    for i in range(args.games):
        seed = args.seed + i
        record = game.place_fleet(seed)
        result = _play_game(step, record, args.max_turns, show_boards=args.show_boards)
        results.append(result)
        outcome = "win" if result["win"] else "loss"
        print(
            f"game {i + 1}/{args.games} seed={seed} {outcome} "
            f"shots={result['shots_used']} hits={result['hits']} "
            f"sunk={result['sunk_ships']}/{len(game.SHIPS)} invalid={result['invalid_shots']}"
        )

    summary = _summarize(results, args, max_turns=args.max_turns)
    _print_summary(summary)
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
        print(f"\nSaved detailed results to {out_path}")
    return summary


def _summarize(results: list[dict], args: argparse.Namespace, *, max_turns: int) -> dict:
    total = len(results)
    wins = sum(1 for r in results if r["win"])
    completions = [r["completion"] for r in results]
    shots_to_win = [r["shots_used"] for r in results if r["win"]]
    return {
        "model": args.model,
        "base_url": args.base_url,
        "total_games": total,
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "completion_mean": statistics.mean(completions) if completions else 0.0,
        "completion_std": statistics.pstdev(completions) if len(completions) > 1 else 0.0,
        "shots_to_win_mean": statistics.mean(shots_to_win) if shots_to_win else max_turns,
        "invalid_shots_mean": statistics.mean([r["invalid_shots"] for r in results]) if results else 0.0,
    }


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 54)
    print(f"Battleship LLM evaluation: {summary['model']}")
    print("=" * 54)
    print(f"  games           : {summary['total_games']}")
    print(f"  wins            : {summary['wins']}")
    print(f"  win rate        : {summary['win_rate']:.1%}")
    print(f"  completion mean : {summary['completion_mean']:.1%} (std {summary['completion_std']:.1%})")
    print(f"  shots to win    : {summary['shots_to_win_mean']:.1f}")
    print(f"  invalid/game    : {summary['invalid_shots_mean']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an OpenAI-compatible LLM at Battleship.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (e.g. http://127.0.0.1:8000/v1).")
    parser.add_argument("--api-key", default="token", help="API key for the endpoint (default 'token').")
    parser.add_argument("--model", default="policy", help="Model name to pass to the endpoint (default 'policy').")
    parser.add_argument("--games", type=int, default=32, help="Number of seeded games to play (default 32).")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed; game i uses seed + i (default 2026).")
    parser.add_argument("--max-turns", type=int, default=game.MAX_TURNS, help=f"Turn cap per game (default {game.MAX_TURNS}).")
    parser.add_argument("--output", default=None, help="Optional path to write detailed JSON results.")
    parser.add_argument("--show-boards", action="store_true", help="Print the board after each shot.")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be a positive integer")
    play_model(args)


if __name__ == "__main__":
    main()
