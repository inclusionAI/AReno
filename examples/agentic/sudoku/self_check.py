"""Self-contained CPU self-check for the Sudoku agentic example.

Runs with no model server, no network, no sandbox. It exercises two things:

1. Boundary / negative cases (issue acceptance):
   - illegal coordinates and bad digits raise SudokuError without leaking the
     solution;
   - undo at history start raises SudokuError;
   - action-budget exhaustion makes the episode terminal (truncated);
   - a fully-legal fill makes is_solved() True, detected from the visible board
     only (the solution is never read).

2. A deterministic heuristic policy rollout that reports per-difficulty
   ``solve_rate`` and ``invalid_action_rate`` — the observable metric the issue
   asks for — over a fixed dataset. The policy is a simple "naked single"
   heuristics: place a digit wherever exactly one candidate remains, else fall
   back to the first candidate. It is NOT a solver, so hard/extreme will
   truncate; that is expected and is the point: it makes the metric meaningful.

Usage (from the repo root, Python 3.10+):

    python3 examples/agentic/sudoku/self_check.py
    python3 examples/agentic/sudoku/self_check.py \
        --dataset /tmp/areno-sudoku-puzzles.jsonl --per-difficulty 8

Exit code is 0 iff every boundary check passes.

NOTE this confirms the environment contract ("environment correctness"). It does
NOT prove reinforcement learning is effective — that requires GPU training and
a before/after comparison, which is out of scope for this local check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import sudoku  # noqa: E402

DEFAULT_PER_DIFFICULTY = 4
DEFAULT_SEED = 2026
DEFAULT_DIFFICULTIES = "easy,medium,hard,extreme"


# --------------------------------------------------------------------------- #
# 1. Boundary / negative checks
# --------------------------------------------------------------------------- #


def run_boundary_checks(seed: int = DEFAULT_SEED) -> list[tuple[str, bool, str]]:
    """Return (name, passed, detail) for each boundary case."""

    results: list[tuple[str, bool, str]] = []
    env = sudoku.SudokuEnv.generate(difficulty="easy", seed=seed)

    # a) illegal coordinates raise (and never touch internals).
    for bad in [(9, 0), (0, 9), (-1, 0)]:
        passed = False
        detail = ""
        try:
            env.inspect_candidates(*bad)
        except sudoku.SudokuError as exc:
            passed = True
            detail = str(exc)
        results.append(("illegal_coord_inspect", passed, detail))

    # b) bad digit on a legal empty cell raises.
    passed = False
    detail = ""
    try:
        empty = _first_empty(env)
        env.place_digit(empty[0], empty[1], 10)
    except (sudoku.SudokuError, ValueError) as exc:
        passed = True
        detail = str(exc)
    results.append(("bad_digit_place", passed, detail))

    # c) undo at history start raises.
    passed = False
    detail = ""
    try:
        env.undo()
    except sudoku.SudokuError as exc:
        passed = True
        detail = str(exc)
    results.append(("undo_at_start", passed, detail))

    # d) place on an already-filled (given) cell is rejected + flagged invalid.
    given = next(iter(env.givens))
    res = env.place_digit(given[0], given[1], 1)
    results.append(("place_on_given_rejected", not res["placed"] and res["invalid_action"], res.get("reason", "")))

    # e) budget exhaustion => terminal (truncated), solved False.
    fresh = sudoku.SudokuEnv.generate(difficulty="easy", seed=seed, action_budget=3)
    for _ in range(3):
        if fresh.is_terminal():
            break
        empty = _first_empty(fresh)
        if empty is None:
            break
        fresh.place_digit(empty[0], empty[1], next(iter(fresh._candidates(empty[0], empty[1]))))  # noqa: SLF001
    results.append(
        (
            "budget_exhaustion_terminal",
            fresh.is_terminal() and not fresh.is_solved(),
            f"actions_used={fresh.actions_used}/{fresh.action_budget}",
        )
    )

    # f) a fully-legal fill of a generated easy puzzle solves it, detected from
    #    the visible board (solution never read; from_puzzle drops it anyway).
    solved_env = sudoku.SudokuEnv.generate(difficulty="easy", seed=seed + 1)
    solved_env.action_budget = 200  # generous so solving isn't truncated
    _force_solve_with_solver(solved_env)
    results.append(
        (
            "legal_fill_is_solved",
            solved_env.is_solved() and solved_env.is_terminal(),
            f"invalid_actions={solved_env.invalid_actions}",
        )
    )

    # g) the public state never carries the solution.
    state = solved_env.public_state()
    leaked = "solution" in state or "solution" in json.dumps(state, default=str).lower()
    results.append(("public_state_no_solution", not leaked, ""))

    return results


# --------------------------------------------------------------------------- #
# 2. Deterministic heuristic-policy rollout + per-difficulty metrics
# --------------------------------------------------------------------------- #


def run_rollout_metrics(
    *,
    dataset_path: str | None,
    per_difficulty: int,
    seed: int,
    difficulties: str,
    action_budget: int = sudoku.DEFAULT_ACTION_BUDGET,
    max_turns: int = 200,
    mode: str = "backtracking",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a heuristic policy over a fixed dataset and report per-band metrics."""

    records = _load_or_generate(dataset_path, per_difficulty, seed, difficulties, action_budget)
    bands = sorted({r["difficulty"] for r in records})
    by_band: dict[str, list[dict[str, Any]]] = {b: [] for b in bands}
    for record in records:
        outcome = _run_episode(record, max_turns=max_turns, mode=mode)
        by_band[record["difficulty"]].append(outcome)
        if verbose:
            print(f"[{record['difficulty']}] {record['id']}: solved={outcome['solved']} "
                  f"invalid={outcome['invalid_actions']} turns={outcome['turns']} "
                  f"status={outcome['status']}")

    summary: dict[str, Any] = {"per_difficulty": {}, "overall": {}}
    total_n = total_solved = total_invalid_actions = total_actions = 0
    for band in bands:
        outs = by_band[band]
        n = len(outs)
        solved = sum(1 for o in outs if o["solved"])
        invalid = sum(o["invalid_actions"] for o in outs)
        actions = sum(o["actions_used"] for o in outs)
        solve_rate = solved / n if n else 0.0
        invalid_rate = invalid / actions if actions else 0.0
        summary["per_difficulty"][band] = {
            "n": n,
            "solve_rate": round(solve_rate, 4),
            "invalid_action_rate": round(invalid_rate, 4),
            "solved": solved,
            "invalid_actions": invalid,
            "actions": actions,
            "avg_turns": round(statistics.fmean(o["turns"] for o in outs), 2) if outs else 0,
        }
        total_n += n
        total_solved += solved
        total_invalid_actions += invalid
        total_actions += actions
    summary["overall"] = {
        "n": total_n,
        "solve_rate": round(total_solved / total_n, 4) if total_n else 0.0,
        "invalid_action_rate": round(total_invalid_actions / total_actions, 4) if total_actions else 0.0,
    }
    return summary


def _run_episode(record: dict, *, max_turns: int, mode: str = "backtracking") -> dict[str, Any]:
    """One heuristic-policy episode. Returns an observable outcome dict.

    ``mode``:
    - ``backtracking``: MRV + naked-single + hidden-single reasoning with
      real branch backtracking. This is a *solving* policy: it follows the
      unique solution, so it solves every difficulty (success path) and never
      wastes actions on dead-end thrashing. ``invalid_actions`` stays ~0 here.
    - ``greedy``: naked-single only, no search; when stuck it guesses the MRV
      cell's first candidate and undoes on conflict. This is a deliberately
      weak baseline that *does* make mistakes, so ``invalid_action_rate`` has
      real spread by difficulty — the comparison target RL shouldbeat.
    """

    env = sudoku.SudokuEnv.from_puzzle(
        record["puzzle"],
        difficulty=record.get("difficulty", sudoku.DEFAULT_DIFFICULTY),
        seed=int(record.get("seed", 0)),
        action_budget=int(record.get("action_budget", sudoku.DEFAULT_ACTION_BUDGET)),
    )
    if mode == "backtracking":
        _run_backtracking_episode(env, max_turns=max_turns)
    else:
        _run_greedy_episode(env, max_turns=max_turns)

    if env.is_solved():
        status = "solved"
    elif env.is_terminal():
        status = "truncated"
    else:
        status = "gave_up"
    invalid = env.invalid_actions
    return {
        "solved": env.is_solved(),
        "status": status,
        "turns": env.actions_used,
        "actions_used": env.actions_used,
        "invalid_actions": invalid,
    }


def _run_greedy_episode(env: sudoku.SudokuEnv, *, max_turns: int) -> None:
    """Weak greedy policy: naked singles, guess + undo on conflict (baseline)."""

    guard = 0
    while not env.is_terminal() and guard < max_turns:
        guard += 1
        move = _forced_move(env) or _mrv_guess(env)
        if move is None:
            break
        r, c, d = move
        res = env.place_digit(r, c, d)
        if res.get("invalid_action") and env.history:
            env.undo()  # conflict: revert the faulty guess, try something else


def _run_backtracking_episode(env: sudoku.SudokuEnv, *, max_turns: int) -> None:
    """Solving policy: deep MRV backtracking with constraint propagation.

    Rather than re-implement search with undo-thrashing, we run the same fast
    bitmask MRV solver the generator uses on a *copy* of the visible board to
    find a completion of the current board, then replay that path legal-cell by
    legal-cell through the public ``place_digit`` interface. This:

    - gives real reasoning depth (MRV + early constraint pruning), so
      hard/extreme actually solve instead of truncating at ~53 dead-end turns;
    - keeps every placement legal, so ``invalid_actions`` stays ~0 (this is the
      success-path policy; use ``--mode greedy`` for a mistake-making baseline);
    - never reads ``env.solution`` — the solver operates on the visible board
      only, so the no-leak invariant is preserved.

    Bounded by ``max_turns`` place actions (≈ empty-cell count: 41 for easy up
    to ~61 for extreme; 200 by default is generous).
    """

    if env.is_terminal():
        return
    work = [row[:] for row in env.puzzle]
    row_used, col_used, box_used = sudoku._empty_masks()
    sudoku._apply_masks((row_used, col_used, box_used), work)
    if not _solve(work, row_used, col_used, box_used):
        return  # no completion on the visible board -> leave truncated
    for r in range(9):
        for c in range(9):
            if env.puzzle[r][c] != sudoku.EMPTY:
                continue
            if env.is_terminal() or (max_turns and env.actions_used >= max_turns):
                return
            env.place_digit(r, c, work[r][c])


def _forced_move(env: sudoku.SudokuEnv) -> tuple[int, int, int] | None:
    """A logically forced placement: naked single or hidden single.

    - Naked single: an empty cell with exactly one legal candidate.
    - Hidden single: a digit that has only one legal cell in some row, column,
      or 3x3 box. Finding these is the first real layer of "reasoning depth"
      beyond the bare MRV guess.
    """

    # Naked single.
    for r, c in sudoku.ALL_COORDS:
        if env.puzzle[r][c] != sudoku.EMPTY:
            continue
        cand = env._candidates(r, c)  # noqa: SLF001
        if len(cand) == 1:
            return r, c, next(iter(cand))

    # Hidden single over rows/cols/boxes.
    for group, axis in ((_rows_indices(), "row"), (_cols_indices(), "col"), (_boxes_indices(), "box")):
        for gi, cells in enumerate(group):
            placeable: dict[int, tuple[int, int]] = {}
            for r, c in cells:
                if env.puzzle[r][c] != sudoku.EMPTY:
                    continue
                for d in env._candidates(r, c):  # noqa: SLF001
                    placeable.setdefault(d, (r, c))
                    if placeable[d] != (r, c):
                        placeable[d] = None  # appears in >1 cell -> not forced
            for d, cell in placeable.items():
                if cell is not None:
                    return cell[0], cell[1], d
    return None


def _mrv_candidates(env: sudoku.SudokuEnv) -> tuple[tuple[int, int], list[int]] | tuple[None, None]:
    """The empty cell with the fewest candidates and its sorted candidates."""

    best: tuple[tuple[int, int], list[int]] | None = None
    for r, c in sudoku.ALL_COORDS:
        if env.puzzle[r][c] != sudoku.EMPTY:
            continue
        cand = sorted(env._candidates(r, c))  # noqa: SLF001
        if not cand:
            continue  # dead cell; skip (caller's forced/branch logic handles it)
        if best is None or len(cand) < len(best[1]):
            best = ((r, c), cand)
            if len(cand) == 1:
                break
    if best is None:
        return None, None
    return best


def _mrv_guess(env: sudoku.SudokuEnv) -> tuple[int, int, int] | None:
    cell, cand = _mrv_candidates(env)
    if cell is None or not cand:
        return None
    return cell[0], cell[1], cand[0]


def _rows_indices() -> list[list[tuple[int, int]]]:
    return [[(r, c) for c in range(9)] for r in range(9)]


def _cols_indices() -> list[list[tuple[int, int]]]:
    return [[(r, c) for r in range(9)] for c in range(9)]


def _boxes_indices() -> list[list[tuple[int, int]]]:
    return [
        [(br + i, bc + j) for i in range(3) for j in range(3)]
        for br in range(0, 9, 3)
        for bc in range(0, 9, 3)
    ]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _first_empty(env: sudoku.SudokuEnv) -> tuple[int, int] | None:
    for r, c in sudoku.ALL_COORDS:
        if env.puzzle[r][c] == sudoku.EMPTY:
            return r, c
    return None


def _force_solve_with_solver(env: sudoku.SudokuEnv) -> None:
    """Fill the env legally by MRV backtracking so we can prove is_solved works.

    Uses only the public place_digit path so history/budget stay consistent.
    """

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


def _load_or_generate(
    dataset_path: str | None,
    per_difficulty: int,
    seed: int,
    difficulties: str,
    action_budget: int,
) -> list[dict]:
    if dataset_path:
        path = Path(dataset_path).expanduser()
        if path.exists():
            records: list[dict] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        records.append(json.loads(stripped))
            return records
    bands = [d.strip().lower() for d in difficulties.split(",") if d.strip()]
    return dataset_generator.generate_records(
        per_difficulty * len(bands),
        seed=seed,
        difficulties=",".join(bands) if bands else difficulties,
        action_budget=action_budget,
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Sudoku agentic example CPU self-check.")
    parser.add_argument("--dataset", default=None, help="JSONL dataset path; omitted => generate in-memory.")
    parser.add_argument("--per-difficulty", type=int, default=DEFAULT_PER_DIFFICULTY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--difficulties", default=DEFAULT_DIFFICULTIES)
    parser.add_argument("--action-budget", type=int, default=sudoku.DEFAULT_ACTION_BUDGET)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=["backtracking", "greedy"],
        default="backtracking",
        help="backtracking: solving policy (naked+hidden single + DFS). "
        "greedy: weak baseline (naked single + guess/undo) that makes mistakes.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("1) Boundary / negative checks")
    print("=" * 72)
    boundary = run_boundary_checks(seed=args.seed)
    all_ok = True
    for name, passed, detail in boundary:
        flag = "PASS" if passed else "FAIL"
        all_ok &= passed
        print(f"  [{flag}] {name}  {('- ' + detail) if detail else ''}")

    print()
    print("=" * 72)
    print("2) Heuristic-policy rollout metrics by difficulty")
    print("=" * 72)
    summary = run_rollout_metrics(
        dataset_path=args.dataset,
        per_difficulty=args.per_difficulty,
        seed=args.seed,
        difficulties=args.difficulties,
        action_budget=args.action_budget,
        max_turns=args.max_turns,
        mode=args.mode,
        verbose=args.verbose,
    )
    print(f"  {'difficulty':<10} {'n':>4} {'solve_rate':>10} {'invalid_rate':>12} {'solved':>7} {'avg_turns':>10}")
    for band, m in summary["per_difficulty"].items():
        print(f"  {band:<10} {m['n']:>4} {m['solve_rate']:>10.4f} {m['invalid_action_rate']:>12.4f} "
              f"{m['solved']:>7} {m['avg_turns']:>10}")
    print(f"  {'overall':<10} {summary['overall']['n']:>4} {summary['overall']['solve_rate']:>10.4f} "
          f"{summary['overall']['invalid_action_rate']:>12.4f}")

    print()
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 72)

    print()
    all_ok = all_ok and (summary["overall"]["solve_rate"] > 0.0)
    print(f"RESULT: boundary={'OK' if all_ok else 'CHECK'}  overall solve_rate={summary['overall']['solve_rate']}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())