"""Deterministic single-board demonstration for the Sudoku agentic example.

This is the issue's *minimal example*: a small, deterministic, self-contained
walkthrough that **demonstrates** (not just asserts) the success path **and**
at least one invalid/boundary input, with no external database, no network
service, and no sandbox. It runs purely on CPU.

The walkthrough prints, step by step, for every action:
  1. the action we are about to take (tool name + arguments),
  2. the tool result returned by the environment (JSON),
  3. the visible board after the action.

It deliberately interleaves a legal placement + undo (success path) with
several invalid/boundary inputs (illegal coordinate, conflicting digit,
placement on a given cell, undo at history start), so a reader can follow one
coherent episode instead of reading a tally of pass/fail counters. The episode
finishes by filling the board legally and showing ``is_solved=True`` — judged
from the visible board only; the solution is never read or printed.

Usage (from the repo root, Python 3.10+):

    python3 examples/agentic/sudoku/demo_episode.py
    python3 examples/agentic/sudoku/demo_episode.py --seed 2026

NOTE this demonstrates *environment correctness and usability*. It does not
exercise the LLM agent loop (that needs a model server and is out of scope for
the local minimal example).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sudoku  # noqa: E402

DEFAULT_SEED = 2026


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic Sudoku walkthrough (minimal example).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed puzzle seed (reproducible).")
    args = parser.parse_args()

    env = sudoku.SudokuEnv.generate(difficulty="easy", seed=args.seed)
    log: list[dict[str, Any]] = []

    _banner("SUDOKU AGENTIC EXAMPLE — minimal deterministic walkthrough")
    print(f"difficulty=easy  seed={args.seed}  action_budget={env.action_budget}")
    print("Rows/cols are 1-based (1..9). '.' marks an empty cell. The solution is never shown.\n")
    _print_board(env, label="initial board")

    # ---- success path: inspect a naked single, place it, then undo it -----
    _step(env, log, "inspect_candidates", {"row": 1, "col": 8},
          note="SUCCESS PATH — inspect a cell with a single candidate (a 'naked single').")
    _step(env, log, "place_digit", {"row": 1, "col": 8, "digit": 4},
          note="SUCCESS PATH — place the forced digit legally.")
    _step(env, log, "undo", {},
          note="SUCCESS PATH — undo reverts the last placement; board returns to prior state.")

    # ---- invalid / boundary inputs (interleaved in the same episode) ------
    _step(env, log, "place_digit", {"row": 1, "col": 1, "digit": 4},
          note="INVALID INPUT — place on a given (already-filled) cell. Expect rejection.")
    _step(env, log, "place_digit", {"row": 1, "col": 3, "digit": 1},
          note="INVALID INPUT — place a digit that conflicts with its row/col/box. Expect rejection.")
    _step(env, log, "inspect_candidates", {"row": 10, "col": 1},
          note="BOUNDARY INPUT — row 10 is out of range (rows/cols are 1..9). Expect an error.")
    _step(env, log, "undo", {},
          note="BOUNDARY INPUT — undo at history start (no placement to revert). Expect an error.")

    # ---- finish: fill the board legally and show is_solved ----------------
    _banner("FINISH — complete the board with a deterministic legal solver")
    print("No model is used: a constraint-propagation solver (MRV + bitmask) finds a legal")
    print("completion on a *copy* of the visible board, then we replay it via place_digit.\n")
    _force_solve(env)
    _print_board(env, label="solved board")
    print(f"is_solved={env.is_solved()}   is_terminal={env.is_terminal()}")
    print("(is_solved is decided from the visible board only — filled + no row/col/box conflict.)")
    print(f"actions_used={env.actions_used}  invalid_actions={env.invalid_actions}\n")

    _banner("STRUCTURED SUMMARY (machine-readable)")
    summary = {
        "seed": args.seed,
        "difficulty": "easy",
        "steps": log,
        "final": {
            "is_solved": env.is_solved(),
            "is_terminal": env.is_terminal(),
            "actions_used": env.actions_used,
            "invalid_actions": env.invalid_actions,
            "solution_leaked": False,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if env.is_solved() else 1


# --------------------------------------------------------------------------- #
# step runner
# --------------------------------------------------------------------------- #


def _step(env: sudoku.SudokuEnv, log: list[dict[str, Any]], name: str, args: dict[str, Any], *, note: str) -> None:
    """Run one tool call, print the action/result/board, and append to the log.

    Captures the board *before* the call so the post-step print can highlight
    exactly which cells changed (wrapped in ``[ ]``) plus a one-line delta —
    this makes the dynamic board change visible at a glance instead of hiding
    a single altered digit inside a 9x9 static reprint.
    """

    prev = [row[:] for row in env.puzzle]
    print(f"--- {name}({', '.join(f'{k}={v}' for k, v in args.items())}) ---")
    print(f"  {note}")
    result, error = _call(env, name, args)
    entry: dict[str, Any] = {"action": name, "arguments": args}
    if error is not None:
        print(f"  result: ERROR {error!r}")
        entry["error"] = error
        entry["ok"] = False
    else:
        # Strip the heavy board copy from the printed result for readability.
        shown = {k: v for k, v in result.items() if k not in ("board", "board_text")}
        print(f"  result: {json.dumps(shown, ensure_ascii=False)}")
        entry["result"] = shown
        entry["ok"] = not result.get("invalid_action", False)
    log.append(entry)
    _print_board_delta(env, prev, label="board after")
    print()


def _call(env: sudoku.SudokuEnv, name: str, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Dispatch a tool call. Returns (result, error). Errors are surfaced, not raised."""

    try:
        if name == "inspect_candidates":
            r, c = sudoku.parse_coord([args.get("row"), args.get("col")])
            return env.inspect_candidates(r, c), None
        if name == "place_digit":
            r, c = sudoku.parse_coord([args.get("row"), args.get("col")])
            return env.place_digit(r, c, int(args.get("digit"))), None
        if name == "undo":
            return env.undo(), None
        return None, f"unknown_tool:{name}"
    except sudoku.SudokuError as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, str(exc)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _force_solve(env: sudoku.SudokuEnv) -> None:
    """Fill the env legally via the MRV bitmask solver on a copy, then place."""

    work = [row[:] for row in env.puzzle]
    row_used, col_used, box_used = sudoku._empty_masks()
    sudoku._apply_masks((row_used, col_used, box_used), work)
    _solve(work, row_used, col_used, box_used)
    for r in range(9):
        for c in range(9):
            if env.puzzle[r][c] == sudoku.EMPTY:
                env.place_digit(r, c, work[r][c])


def _solve(board, row_used, col_used, box_used) -> bool:
    best = sudoku._mrv_cell(board, row_used, col_used, box_used)
    if best is None:
        return True
    r, c, cand_mask = best
    if cand_mask == 0:
        return False
    b = sudoku._box_index(r, c)
    while cand_mask:
        bit = cand_mask & -cand_mask
        cand_mask ^= bit
        d = bit.bit_length() - 1
        board[r][c] = d
        row_used[r] |= bit
        col_used[c] |= bit
        box_used[b] |= bit
        if _solve(board, row_used, col_used, box_used):
            return True
        board[r][c] = sudoku.EMPTY
        row_used[r] &= ~bit
        col_used[c] &= ~bit
        box_used[b] &= ~bit
    return False


def _print_board(env: sudoku.SudokuEnv, *, label: str) -> None:
    print(f"  [{label}]")
    for line in env.board_text().splitlines():
        print(f"  {line}")


def _print_board_delta(env: sudoku.SudokuEnv, prev: list[list[int]], *, label: str) -> None:
    """Print the board with changed cells highlighted as ``[v]`` plus a delta line.

    Only cells that differ from ``prev`` are wrapped in brackets, so the eye
    lands immediately on what just moved. A one-line ``Δ`` summary spells out
    each change (place/remove/replace) in 1-based coordinates; "no change"
    covers read-only inspect calls and rejected/errored actions.
    """

    desc, changed = _board_delta_desc(prev, env.puzzle)
    print(f"  [{label}]  Δ {desc}")
    for r in range(9):
        cells = []
        for c in range(9):
            v = env.puzzle[r][c]
            glyph = str(v) if v != sudoku.EMPTY else "."
            cells.append(f"[{glyph}]" if (r, c) in changed else f" {glyph} ")
        row = " | ".join(" ".join(cells[i * 3 : i * 3 + 3]) for i in range(3)).rstrip()
        if r in (3, 6):
            print("  " + "-" * len(row))
        print(f"  {row}")


def _board_delta_desc(prev: list[list[int]], cur: list[list[int]]) -> tuple[str, set[tuple[int, int]]]:
    """Return (human-readable delta, set of changed (r,c)) for prev -> cur."""

    parts: list[str] = []
    changed: set[tuple[int, int]] = set()
    for r in range(9):
        for c in range(9):
            p, n = prev[r][c], cur[r][c]
            if p == n:
                continue
            changed.add((r, c))
            coord = f"(row{r + 1},col{c + 1})"
            if p == sudoku.EMPTY and n != sudoku.EMPTY:
                parts.append(f"+placed {n} at {coord}")
            elif n == sudoku.EMPTY and p != sudoku.EMPTY:
                parts.append(f"-removed {p} at {coord}")
            else:
                parts.append(f"~{p}->{n} at {coord}")
    if not parts:
        return "no board change (read-only or rejected action)", changed
    return ", ".join(parts), changed


def _banner(text: str) -> None:
    bar = "=" * 72
    print(bar)
    print(text)
    print(bar)


if __name__ == "__main__":
    sys.exit(main())