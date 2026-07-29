"""Seeded random opponent and self-play evaluation harness for 6x6 Othello.

Pure Python / CPU: no LLM, no network, no database. Provides the acceptance
harness (seeded random opponent, win rate + invalid-move rate) plus a CLI demo
that runs out of the box.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

PolicyFn = "callable[[list[list[str]], str, random.Random], tuple[int, int] | None]"


def random_opponent_pick(board: game.Board, player: str, rng: random.Random) -> tuple[int, int] | None:
    """Return a uniform random legal move, or ``None`` to pass when forced."""

    moves = game.legal_moves(board, player)
    return rng.choice(moves) if moves else None


def greedy_opponent_pick(board: game.Board, player: str, rng: random.Random) -> tuple[int, int] | None:
    """Return a legal move flipping the most discs (ties broken by seed)."""

    moves = game.legal_moves(board, player)
    if not moves:
        return None
    scored = [(len(game.flips_for(board, r, c, player)), r, c) for r, c in moves]
    best = max(score[0] for score in scored)
    choices = [(r, c) for _, r, c in scored if _ == best]
    return rng.choice(choices)


def play_match(
    policy_fn: PolicyFn,
    *,
    policy_side: str = "B",
    seed: int = 2026,
    max_steps: int = 100,
) -> dict:
    """Play one game: ``policy_fn`` vs a seeded random opponent.

    ``policy_fn(board, player, rng)`` returns ``(row, col) | None`` (None = no
    move / pass). Both sides draw from the same seeded RNG so the match is fully
    reproducible. Returns a structured result dict.
    """

    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if seed < 0:
        raise ValueError("--seed must be non-negative")

    rng = random.Random(seed)
    board = game.new_board()
    player = "B"
    passes = 0
    invalid = 0
    steps = 0

    while steps < max_steps and not game.is_terminal(board):
        steps += 1
        is_policy = player == policy_side
        if is_policy:
            move = policy_fn(board, player, rng)
        else:
            move = random_opponent_pick(board, player, rng)
        if move is None:
            # No move chosen -> treat as a pass (only legal when no moves exist).
            if game.has_legal_move(board, player):
                invalid += 1 if is_policy else 0
            passes += 1
            player = game.opponent(player)
            continue
        if move not in game.legal_moves(board, player):
            invalid += 1 if is_policy else 0
            player = game.opponent(player)
            continue
        board = game.apply_move(board, move[0], move[1], player)
        player = game.opponent(player)
        # If the next player has no legal move, they pass automatically.
        if not game.has_legal_move(board, player):
            passes += 1
            player = game.opponent(player)

    result = game.score_board(board)
    return {
        "winner": result["winner"],
        "black": result["black"],
        "white": result["white"],
        "steps": steps,
        "passes": passes,
        "invalid": invalid,
        "policy_side": policy_side,
        "terminal": game.is_terminal(board),
    }


def evaluate(
    policy_fn: PolicyFn,
    n_games: int = 20,
    *,
    seed: int = 2026,
    max_steps: int = 100,
    policy_side: str = "B",
) -> dict:
    """Aggregate win rate and invalid-move rate over ``n_games`` seeded matches."""

    if n_games <= 0:
        raise ValueError("--n-games must be positive")

    wins = 0
    draws = 0
    invalid_total = 0
    games: list[dict] = []
    for i in range(n_games):
        game_seed = seed + i
        match = play_match(policy_fn, policy_side=policy_side, seed=game_seed, max_steps=max_steps)
        games.append(match)
        if match["winner"] == policy_side:
            wins += 1
        elif match["winner"] == "draw":
            draws += 1
        invalid_total += match["invalid"]

    summary = {
        "n_games": n_games,
        "win_rate": wins / n_games,
        "draw_rate": draws / n_games,
        "invalid_move_rate": invalid_total / n_games,
        "policy_side": policy_side,
    }
    return {"summary": summary, "games": games}


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    print("6x6 Othello self-play evaluation")
    print(f"  games          : {summary['n_games']}")
    print(f"  policy side    : {summary['policy_side']}")
    print(f"  win rate       : {summary['win_rate']:.3f}")
    print(f"  draw rate      : {summary['draw_rate']:.3f}")
    print(f"  invalid rate   : {summary['invalid_move_rate']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CPU-only 6x6 Othello self-play evaluation against a seeded random opponent."
    )
    parser.add_argument("--n-games", type=int, default=20, help="Number of matches to play.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per match.")
    parser.add_argument(
        "--policy", choices=["random", "greedy"], default="random", help="Policy (the random opponent is always random)."
    )
    parser.add_argument("--policy-side", choices=["B", "W"], default="B", help="Side the policy plays.")
    parser.add_argument("--output", "-o", default="-", help="Write JSON report to this path, or '-' for stdout.")
    args = parser.parse_args()

    if args.policy == "greedy":
        def policy_fn(board, player, rng):  # noqa: ANN001
            return greedy_opponent_pick(board, player, rng)
    else:
        def policy_fn(board, player, rng):  # noqa: ANN001
            return random_opponent_pick(board, player, rng)

    report = evaluate(policy_fn, args.n_games, seed=args.seed, max_steps=args.max_steps, policy_side=args.policy_side)
    _print_summary(report)
    payload = json.dumps(report, indent=2, default=str)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).expanduser().write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()