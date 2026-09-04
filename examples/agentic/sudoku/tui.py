"""Interactive terminal UI for the Sudoku environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402
from game import SYSTEM_PROMPT, TOOLS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--difficulty", choices=list(game.DIFFICULTY_EMPTY), default=game.DEFAULT_DIFFICULTY)
    parser.add_argument("--max-actions", type=int, default=game.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--agent", action="store_true", help="Let an OpenAI-compatible model solve the puzzle")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    puzzle = game.generate_puzzle(args.difficulty, seed=args.seed)
    episode = game.SudokuEpisode(puzzle["puzzle"], max_actions=args.max_actions)

    print("\033[1;36mSUDOKU // TERMINAL\033[0m")
    print(f"Difficulty: {args.difficulty}  Max actions: {args.max_actions}\n")

    if args.agent:
        _run_llm(episode, args)
    else:
        _run_human(episode)


def _run_human(episode: game.SudokuEpisode) -> None:
    while not episode.is_done():
        print(game.board_to_text(episode.board))
        print(f"\n[{episode.actions_taken + 1}/{episode.max_actions}] action> ", end="")
        raw = input().strip().lower()
        if not raw:
            continue
        result = _parse_and_execute(raw, episode)
        _render_result(result, raw)
    _print_final(episode)


def _run_llm(episode: game.SudokuEpisode, args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("LLM mode requires `openai`. Install it with `pip install openai`.") from exc

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    record = {"puzzle": episode.original, "difficulty": args.difficulty, "max_actions": args.max_actions}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": game.make_prompt(record)},
    ]

    while not episode.is_done():
        turn_prompt = {
            "role": "user",
            "content": f"Action {episode.actions_taken + 1}/{episode.max_actions}: call one tool now.",
        }
        response = client.chat.completions.create(
            model=args.model,
            messages=[*messages, turn_prompt],
            tools=TOOLS,
            stream=False,
        )
        message = response.choices[0].message
        calls = list(message.tool_calls or [])
        if not calls:
            print("\033[31mMODEL ERROR\033[0m no tool call returned")
            break
        call = calls[0]
        name = call.function.name
        raw_args = call.function.arguments or "{}"
        try:
            args_dict = json.loads(raw_args)
        except json.JSONDecodeError:
            print("\033[31mMODEL ERROR\033[0m invalid JSON arguments")
            break

        result = _execute_named_tool(name, args_dict, episode)
        label = _format_tool_call(name, args_dict)
        print(f"[{episode.actions_taken}/{episode.max_actions}] model> {label}")
        _render_result(result, label)

        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": call.id, "type": call.type, "function": {"name": name, "arguments": raw_args}}
            ],
        }
        messages.extend(
            [
                turn_prompt,
                assistant_message,
                {"role": "tool", "tool_call_id": call.id, "name": name, "content": json.dumps(result)},
            ]
        )

    client.close()
    _print_final(episode)


def _parse_and_execute(raw: str, episode: game.SudokuEpisode) -> dict:
    parts = raw.replace(",", " ").split()
    cmd = parts[0]
    if cmd == "inspect" and len(parts) >= 3:
        row, col = int(parts[1]), int(parts[2])
        return game.inspect_candidates(episode.board, row, col)
    if cmd == "place" and len(parts) >= 4:
        row, col, digit = int(parts[1]), int(parts[2]), int(parts[3])
        return episode.place(row, col, digit)
    if cmd == "undo":
        return episode.undo()
    return {"valid": False, "error": f"unknown command: {raw}"}


def _execute_named_tool(name: str, args: dict, episode: game.SudokuEpisode) -> dict:
    if name == "inspect_candidates":
        return game.inspect_candidates(episode.board, int(args.get("row", -1)), int(args.get("col", -1)))
    if name == "place_digit":
        return episode.place(int(args.get("row", -1)), int(args.get("col", -1)), int(args.get("digit", 0)))
    if name == "undo":
        return episode.undo()
    return {"valid": False, "error": f"unknown tool: {name}"}


def _format_tool_call(name: str, args: dict) -> str:
    if name == "inspect_candidates":
        return f"inspect_candidates({args.get('row')}, {args.get('col')})"
    if name == "place_digit":
        return f"place_digit({args.get('row')}, {args.get('col')}, {args.get('digit')})"
    if name == "undo":
        return "undo()"
    return f"{name}({args})"


def _render_result(result: dict, label: str) -> None:
    if not result.get("valid"):
        print(f"  \033[31m✗ {result.get('error', 'invalid')}\033[0m")
        return
    if "candidates" in result:
        print(f"  \033[33mcandidates: {result['candidates']}\033[0m")
    elif "undid" in result:
        print(f"  \033[34m↩ undid {result['undid']}\033[0m")
    else:
        print(f"  \033[32m✓ ok\033[0m")


def _print_final(episode: game.SudokuEpisode) -> None:
    print()
    print(game.board_to_text(episode.board))
    if episode.is_solved():
        print("\n\033[1;32mSOLVED\033[0m")
    else:
        print(f"\n\033[1;31mNOT SOLVED\033[0m  actions used: {episode.actions_taken}/{episode.max_actions}")


if __name__ == "__main__":
    main()