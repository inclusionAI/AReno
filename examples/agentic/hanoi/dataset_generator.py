"""Generate deterministic Towers of Hanoi fixtures for the agentic RL demo.

Unlike DuelGrid (which samples random states), Hanoi has a single canonical
start, so determinism here comes from **scripted scenarios** rather than a RNG.
For each n in 3..6 we emit four fixtures that together cover the issue's
acceptance surface:

- ``optimal``         — the oracle shortest solution; completes, 0 illegal, 0 excess.
- ``contains_illegal``— optimal with one no-op illegal move mid-run; completes, 1 illegal.
- ``boundary``        — triggers an empty-source rejection at the start, then optimal;
                        covers a boundary/invalid input while still completing.
- ``failure``         — shuffles disks without ever solving; does not complete.

Each record is a plain dict written as JSONL, mirroring DuelGrid's
``_state_to_record`` shape plus Hanoi-specific trace/expected fields so
``reward.py`` / ``run_agent.py`` / tests can consume it without external storage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game

SCENARIOS = ("optimal", "contains_illegal", "boundary", "failure")


def generate_records(count: int | None = None, *, seed: int = 2026) -> list[dict[str, Any]]:
    """Return deterministic Hanoi fixtures.

    ``count`` and ``seed`` are accepted for parity with DuelGrid's signature
    (and to keep CLI/test conventions aligned), but the fixtures are fully
    scripted: the same ``count``/``seed`` always yields byte-identical output.
    With the default ``count=None`` all 4 scenarios x 4 disk sizes (16 records)
    are emitted; a positive ``count`` truncates that deterministic list.
    """

    del seed  # scripted, not random — kept only for signature parity
    records: list[dict[str, Any]] = []
    for n in range(game.MIN_DISKS, game.MAX_DISKS + 1):
        for scenario in SCENARIOS:
            records.append(_record_for_scenario(n, scenario))
    if count is not None and count > 0:
        return records[:count]
    return records


def record_to_state(record: dict[str, Any]) -> game.HanoiState:
    """Rebuild a ``HanoiState`` from a fixture record."""

    n = int(record["n"])
    pegs_field = record.get("pegs")
    if pegs_field is None:
        return game.make_state(n)
    pegs = tuple(tuple(int(d) for d in stack) for stack in pegs_field)
    # Reconstruct the immutable state directly so non-canonical starts (if a
    # future fixture stores a mid-game state) round-trip correctly.
    return game.HanoiState(
        pegs=pegs,
        n=n,
        moves=int(record.get("moves", 0)),
        max_moves=int(record.get("max_moves", max(64, (2**n) * 4))),
    )


def record_to_trace(record: dict[str, Any]) -> list[tuple[int, int]]:
    """Extract a record's move trace as integer pairs."""

    return [tuple(int(x) for x in mv) for mv in record.get("trace", [])]


def expected_outcome(record: dict[str, Any]) -> dict[str, Any]:
    """The deterministic replay outcome a test should observe for this record."""

    return dict(record["expected"])


def write_jsonl(records: Iterable[dict[str, Any]], output: TextIO) -> None:
    """Write records as JSONL."""

    output.writelines(json.dumps(record, separators=(",", ":")) + "\n" for record in records)


# --- scenario builders ------------------------------------------------------


def _record_for_scenario(n: int, scenario: str) -> dict[str, Any]:
    builder = {
        "optimal": _optimal_trace,
        "contains_illegal": _contains_illegal_trace,
        "boundary": _boundary_trace,
        "failure": _failure_trace,
    }[scenario]
    trace = builder(n)
    # The expected outcome is NOT hand-written — we run the real rules engine
    # on the trace and store what it actually returns. This keeps "expected"
    # forever consistent with game.py, so tests only need to assert
    # expected == fresh-replay (no separate oracle to maintain).
    result = game.replay(trace, n)
    state = game.make_state(n)
    return {
        "id": f"hanoi-n{n}-{scenario}",
        "n": n,
        "scenario": scenario,
        "pegs": [list(stack) for stack in state.pegs],
        "moves": 0,
        "max_moves": state.max_moves,
        "legal_moves": [list(mv) for mv in game.legal_moves(state)],
        "oracle_steps": game.optimal_steps(n),
        "optimal_moves": [list(mv) for mv in game.optimal_solution(n)],
        "trace": [list(mv) for mv in trace],
        "expected": {
            "completed": result.completed,
            "legal_count": result.legal_count,
            "illegal_count": result.illegal_count,
            "excess_moves": result.excess_moves,
        },
    }


def _optimal_trace(n: int) -> list[tuple[int, int]]:
    return list(game.optimal_solution(n))


def _contains_illegal_trace(n: int) -> list[tuple[int, int]]:
    # Insert a no-op (source==target) illegal move after the first legal move.
    optimal = list(game.optimal_solution(n))
    return [optimal[0], (1, 1)] + optimal[1:]


def _boundary_trace(n: int) -> list[tuple[int, int]]:
    # Start by attempting to move from an empty peg (peg 1 is empty at start),
    # which exercises the empty-source boundary, then solve optimally.
    return [(1, 2)] + list(game.optimal_solution(n))


def _failure_trace(n: int) -> list[tuple[int, int]]:
    # Shuffle the smallest disk back and forth without ever completing. The
    # repeat count 2*(2**n-1) is chosen to stay well under make_state's
    # max_moves ceiling (max(64, 4*2**n)) while giving a clearly non-empty,
    # never-completing trace for the failure scenario.
    oscillate = [(0, 2), (2, 0)] * (2 * game.optimal_steps(n))
    return oscillate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic JSONL fixtures for the Areno Hanoi agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=None, help="Truncate to this many fixtures.")
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Kept for CLI parity (fixtures are scripted).",
    )
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
