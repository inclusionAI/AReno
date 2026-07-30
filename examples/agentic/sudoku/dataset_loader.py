"""Dataset loader for the Sudoku agentic example.

Reads the JSONL produced by ``dataset_generator.py`` and converts each record
into an AReno prompt record. The prompt renders the visible board and the
rules; the solution is never part of the record. ``difficulty`` is carried
through so rewards/metrics can be grouped by band.

Curriculum: AReno's trainer consumes the dataset list sequentially (no
shuffling), so if the JSONL is ordered tutorial->easy->medium->hard->extreme
the first training pass is automatically an easy->hard curriculum. We
therefore keep the on-disk order and additionally re-sort defensively.
``max_turns`` is sized per puzzle as ``empty_cells + MAX_TURNS_HEADROOM`` so an
episode can afford the placements plus a few inspect/undo/recovery turns; it
scales naturally with difficulty (harder puzzles have more empties). Both
behaviors are gated on the ``SUDOKU_CURRICULUM`` env var (default "on"); set
it to "off" to restore the flat/legacy behavior.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import sudoku  # noqa: E402

DEFAULT_MAX_TURNS = 30

# Per-episode turn headroom on top of the cell count. One turn performs one
# tool call and fills at most one cell, so solving needs at least ``empty_cells``
# placement turns. The headroom leaves room for a weak policy to inspect
# candidates, make and recover from illegal placements, and undo/backtrack —
# without it a 0.6B model that inspects or errs even once can never fit the
# solved reward inside the budget, deadlocking the RL cold-start. For an
# 8-empty tutorial board this is ~16 turns (~1-2k re-rendered context), fine on
# a 16GB T4; raise only once the policy solves cleanly. The framework's
# overlong-trajectory filter still drops episodes that exceed the model context
# length, so larger bands stay safe.
MAX_TURNS_HEADROOM = 8


def _curriculum_enabled() -> bool:
    return os.environ.get("SUDOKU_CURRICULUM", "on").lower() not in ("off", "0", "false", "no")


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL puzzles and convert them to AReno prompt records."""

    records = _load_records(dataset_path)
    if _curriculum_enabled():
        order = {band: i for i, band in enumerate(dataset_generator.DIFFICULTY_ORDER)}
        records.sort(key=lambda rec: (order.get(str(rec.get("difficulty", "")).lower(), len(order)),
                                      int(rec.get("seed", 0))))
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "puzzles.jsonl"
    if not path.exists():
        # Fall back to in-memory generation so the example stays self-contained.
        return dataset_generator.generate_records()
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    puzzle = sudoku._normalize_puzzle(raw["puzzle"])  # noqa: SLF001
    difficulty = str(raw.get("difficulty", sudoku.DEFAULT_DIFFICULTY)).lower()
    empty_cells = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == sudoku.EMPTY)
    env = sudoku.SudokuEnv.from_puzzle(
        puzzle,
        difficulty=difficulty,
        seed=int(raw.get("seed", 0)),
        action_budget=int(raw.get("action_budget", sudoku.DEFAULT_ACTION_BUDGET)),
    )
    # max_turns is the per-episode LLM turn cap. Each turn performs exactly one
    # tool call and fills at most ONE cell, so a solvable episode needs at least
    # `empty_cells` place turns; MAX_TURNS_HEADROOM adds room for inspect
    # candidates, illegal-placement recovery, and undo/backtrack. For an 8-empty
    # tutorial board this is ~16 turns, well within a 16GB T4. A cap that is too
    # tight makes the solved reward unreachable for a model that inspects or errs
    # at all, deadlocking RL cold-start, so we keep real headroom rather than +1.
    # The framework's overlong-trajectory filter still drops episodes that exceed
    # the model context length, so larger bands stay safe; raise once solving is
    # clean.
    if _curriculum_enabled() and "max_turns" not in raw:
        max_turns = empty_cells + MAX_TURNS_HEADROOM
    else:
        max_turns = int(raw.get("max_turns", DEFAULT_MAX_TURNS))
    return {
        "id": raw.get("id", f"puzzle-{index:05d}"),
        "difficulty": difficulty,
        "seed": int(raw.get("seed", 0)),
        "action_budget": int(raw.get("action_budget", sudoku.DEFAULT_ACTION_BUDGET)),
        "max_turns": max_turns,
        "puzzle": [row[:] for row in puzzle],
        "empty_cells": empty_cells,
        "prompt": _make_prompt(env),
    }


def _make_prompt(env: sudoku.SudokuEnv) -> str:
    return (
        f"Solve this {env.difficulty} Sudoku puzzle within {env._actions_remaining()} actions. "  # noqa: SLF001
        "Rows and columns are 1-based (1..9). '.' marks an empty cell.\n\n"
        "Use inspect_candidates to see legal digits, place_digit to fill a cell, and undo to revert.\n\n"
        f"Board:\n{env.board_text()}\n"
    )