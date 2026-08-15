"""Head-to-head comparison of the three Battleship autoplay modes.

Runs `heuristic`, `random`, and (optionally) `llm` on the **same set of seeded
fleets** so the only variable is the decision policy, then prints a side-by-side
aggregate table and optionally writes JSON. Mirrors the per-game metric schema of
`evaluate.evaluate_player` and `play_llm._play_game` (win / completion /
shots_used / hits / sunk_ships / invalid_shots / seed).

The LLM mode is opt-in via `--base-url`; when omitted, the script compares
`heuristic` vs `random` with no network dependency.

    # Offline, no LLM
    python examples/agentic/battleship/compare_modes.py --games 50 -o /tmp/cmp.json

    # Include an LLM endpoint
    python examples/agentic/battleship/compare_modes.py --games 50 \
        --base-url http://127.0.0.1:8000/v1 --model policy -o /tmp/cmp.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# 添加当前目录到 Python 路径，复用同目录的 game/evaluate/play_llm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
import game  # noqa: E402
import play_llm  # noqa: E402


# 计算样本标准差（n-1），与 evaluate._std 口径一致
def _std(values: list[float]) -> float:
    """Sample standard deviation (n-1), matching evaluate._std."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5


def _mean(values: list[float]) -> float:
    """Mean of a list, 0.0 when empty."""
    return sum(values) / len(values) if values else 0.0


# 用任意 Player（choose_shot(state)）作为单局策略：忽略 messages，只看棋盘
def _play_policy_game(player: evaluate.Player, record: dict, max_turns: int, show_boards: bool = False) -> dict:
    """Play one game with a Player-protocol policy; reuses play_llm._play_game."""

    def step(messages: list[dict], state: game.GameState) -> str | None:
        return player.choose_shot(state)

    return play_llm._play_game(step, record, max_turns, show_boards=show_boards)


# LLM 单局策略：复用 play_llm._play_game 的对话循环，但用一个计时的 step
def _llm_step_timed(client: Any, model: str, messages: list[dict], stats: dict) -> str | None:
    """Ask the endpoint for one fire coordinate, accumulating latency + token usage."""
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[play_llm.FIRE_TOOL],
        tool_choice={"type": "function", "function": {"name": "fire"}},
        stream=False,
    )
    stats["latency_ms"] += (time.perf_counter() - t0) * 1000.0
    usage = getattr(response, "usage", None)
    if usage is not None:
        stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    return play_llm._parse_coord_from_response(response)


def _play_llm_game(client: Any, model: str, record: dict, max_turns: int, show_boards: bool = False) -> dict:
    """Play one game with the LLM; attaches latency/token totals to the result."""
    stats = {"latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    def step(messages: list[dict], state: game.GameState) -> str | None:
        return _llm_step_timed(client, model, messages, stats)

    result = play_llm._play_game(step, record, max_turns, show_boards=show_boards)
    result["latency_ms"] = stats["latency_ms"]
    result["prompt_tokens"] = stats["prompt_tokens"]
    result["completion_tokens"] = stats["completion_tokens"]
    return result


# 聚合单模式的多局结果，返回 summary dict（schema 与 evaluate/play_llm 对齐 + LLM 扩展）
def _summarize(mode: str, results: list[dict], max_turns: int) -> dict:
    """Aggregate per-game results into a per-mode summary."""
    total = len(results)
    wins = sum(1 for r in results if r["win"])
    completions = [r["completion"] for r in results]
    shots_to_win = [r["shots_used"] for r in results if r["win"]]
    invalids = [r["invalid_shots"] for r in results]

    summary: dict[str, Any] = {
        "mode": mode,
        "games": total,
        "wins": wins,
        "win_rate": wins / total if total > 0 else 0.0,
        "completion_mean": _mean(completions),
        "completion_std": _std(completions),
        "shots_to_win_mean": _mean(shots_to_win) if shots_to_win else max_turns,
        "shots_to_win_std": _std(shots_to_win),
        "invalid_shots_mean": _mean(invalids),
    }
    if mode == "llm":
        summary["latency_ms_mean"] = _mean([r.get("latency_ms", 0.0) for r in results])
        summary["prompt_tokens_mean"] = _mean([r.get("prompt_tokens", 0) for r in results])
        summary["completion_tokens_mean"] = _mean([r.get("completion_tokens", 0) for r in results])
    return summary


# 打印并列表格：每模式一行
def _print_table(per_mode: dict[str, dict]) -> None:
    """Print a side-by-side aggregate table to stdout."""
    cols = ["mode", "games", "win_rate", "completion", "shots_to_win", "invalid/game"]
    header = f"{cols[0]:>10} {cols[1]:>6} {cols[2]:>9} {cols[3]:>18} {cols[4]:>16} {cols[5]:>13}"
    print("\n" + "=" * len(header))
    print("Battleship autoplay mode comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for mode, data in per_mode.items():
        s = data["summary"]
        comp = f"{s['completion_mean']:.1%} ± {s['completion_std']:.1%}"
        s2w = f"{s['shots_to_win_mean']:.1f} ± {s['shots_to_win_std']:.1f}"
        print(f"{mode:>10} {s['games']:>6} {s['win_rate']:>9.1%} {comp:>18} {s2w:>16} {s['invalid_shots_mean']:>13.2f}")
        if mode == "llm":
            latency = f"{s['latency_ms_mean']:.0f} ms"
            tokens = s["prompt_tokens_mean"] + s["completion_tokens_mean"]
            print(f"{'':>10} {'':>6} {'':>9} {'latency/game':>18} {latency:>16} {'tokens/game':>13}")
            print(f"{'':>10} {'':>6} {'':>9} {'':>18} {'':>16} {tokens:>13.0f}")
    print("=" * len(header))


# 主流程：生成相同 seeds 的 records，逐模式跑，聚合输出
def run(args: argparse.Namespace) -> dict:
    """Run all selected modes over the same seeded fleets; return the full report."""
    max_turns = args.max_turns if args.max_turns is not None else game.MAX_TURNS

    # 三模式共用同一套 seeded fleets（公平对比的控制变量）
    seeds = [args.seed + i for i in range(args.games)]
    records = [game.place_fleet(s) for s in seeds]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "llm" in modes and not args.base_url:
        print("Warning: --base-url not provided; dropping 'llm' mode.", file=sys.stderr)
        modes = [m for m in modes if m != "llm"]
    if not modes:
        raise SystemExit("No modes to run (need at least one of heuristic/random, or --base-url for llm).")

    # 每模式构造一次 player（与 evaluate.evaluate_player 一致：单实例跨局复用）
    players: dict[str, evaluate.Player] = {}
    if "random" in modes:
        players["random"] = evaluate.RandomPlayer()
    if "heuristic" in modes:
        players["heuristic"] = evaluate.HeuristicPlayer(seed=args.heuristic_seed)

    client: Any = None
    if "llm" in modes:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("llm mode requires `openai`: pip install openai") from exc
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)

    per_mode: dict[str, dict] = {}
    for mode in modes:
        results: list[dict] = []
        for i, record in enumerate(records):
            if mode == "llm":
                result = _play_llm_game(client, args.model, record, max_turns, show_boards=args.show_boards)
            else:
                result = _play_policy_game(players[mode], record, max_turns, show_boards=args.show_boards)
            results.append(result)
            outcome = "win" if result["win"] else "loss"
            print(
                f"[{mode}] game {i + 1}/{args.games} seed={seeds[i]} {outcome} "
                f"shots={result['shots_used']} hits={result['hits']} "
                f"sunk={result['sunk_ships']}/{len(game.SHIPS)} invalid={result['invalid_shots']}"
            )
        per_mode[mode] = {"summary": _summarize(mode, results, max_turns), "results": results}

    _print_table(per_mode)

    report = {
        "config": {
            "games": args.games,
            "base_seed": args.seed,
            "max_turns": max_turns,
            "heuristic_seed": args.heuristic_seed,
            "modes": modes,
            "base_url": args.base_url,
            "model": args.model,
            "seeds": seeds,
        },
        "per_mode": per_mode,
    }

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nResults written to {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Battleship autoplay modes head-to-head.")
    parser.add_argument("--games", type=int, default=50, help="Number of seeded games (default 50).")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed; game i uses seed + i (default 2026).")
    parser.add_argument("--max-turns", type=int, default=None, help=f"Turn cap per game (default {game.MAX_TURNS}).")
    parser.add_argument(
        "--modes", default="heuristic,random,llm", help="Comma-separated subset of heuristic,random,llm."
    )
    parser.add_argument("--heuristic-seed", type=int, default=42, help="RNG seed for heuristic player (default 42).")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL; enables llm mode when set.")
    parser.add_argument("--model", default="policy", help="Model name passed to the endpoint (default 'policy').")
    parser.add_argument("--api-key", default="token", help="API key for the endpoint (default 'token').")
    parser.add_argument("--output", "-o", default=None, help="Optional path to write detailed JSON results.")
    parser.add_argument("--show-boards", action="store_true", help="Print the board after each shot.")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be a positive integer")
    run(args)


if __name__ == "__main__":
    main()
