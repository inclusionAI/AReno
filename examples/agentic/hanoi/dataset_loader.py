"""Dataset loader for the Hanoi agentic example.

Mirrors ``examples/agentic/duelgrid/dataset_loader.py``: read the JSONL
fixtures produced by ``dataset_generator.py`` and convert each raw record into
an Areno prompt record the trainer can consume. The split from the generator
is intentional — generation is offline/random-free production of fixtures,
loading is the training-time step that builds the ``prompt`` text and the
best-action hint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator
import game

DEFAULT_FILENAME = "hanoi_fixtures.jsonl"


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL fixtures and convert them to Areno prompt records.

    ``default_loader`` and any extra keyword args are accepted and ignored:
    AReno's CLI may inject a default loader / options when calling the loader
    function, and swallowing them keeps this loader drop-in compatible without
    asserting on a specific call signature.
    """

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        # If handed a directory (AReno may pass a run dir), look for the default
        # fixture filename inside it rather than failing opaquely.
        path = path / DEFAULT_FILENAME
    if not path.exists():
        # Fail fast with an actionable hint — the issue wants failures to
        # identify the affected input, not a generic stack trace.
        raise FileNotFoundError(
            f"Hanoi dataset not found: {path}. Generate it with "
            "`python examples/agentic/hanoi/dataset_generator.py --output <path>`."
        )
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                # JSONL: one JSON record per non-blank line.
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    """Build an Areno prompt record from a raw Hanoi fixture.

    Carries the full raw fixture under ``state`` plus the prompt and best-action
    fields Areno expects. ``best_action`` / ``best_actions`` use the true oracle
    solution (Hanoi's advantage over heuristic baselines).
    """

    state = dataset_generator.record_to_state(raw)
    optimal = [tuple(mv) for mv in raw.get("optimal_moves", [])]
    return {
        "id": raw.get("id", f"hanoi-{index:05d}"),
        # The prompt text is built here (not in the generator) so generation
        # stays a pure data step and the loader owns the model-facing surface.
        "prompt": game.format_prompt(state),
        # Keep the full raw fixture under "state" so reward_fn can rebuild the
        # board (source_record["state"]) without a second file format.
        "state": raw,
        "n": int(raw["n"]),
        "oracle_steps": int(raw["oracle_steps"]),
        # Hanoi has a true optimum, so best_action is the genuine first optimal
        # move (DuelGrid instead uses a heuristic baseline here).
        "best_action": list(optimal[0]) if optimal else None,
        "best_actions": [list(mv) for mv in optimal],
        "legal_actions": [list(mv) for mv in game.legal_moves(state)],
        "trace": raw.get("trace", []),
        "expected": raw.get("expected", {}),
    }
