"""Cross-module integration: dataset_generator -> loader -> reward, CPU-only."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "2048"

_SIBLING_MODULES = ("game", "dataset_generator", "dataset_loader")


def _load_module(name: str):
    saved_path = list(sys.path)
    saved_modules = {key: sys.modules.get(key) for key in _SIBLING_MODULES if key in sys.modules}
    for key in saved_modules:
        sys.modules.pop(key, None)
    try:
        path = EXAMPLE_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"agentic_2048_integration_{name}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        for key in _SIBLING_MODULES:
            if key in saved_modules:
                sys.modules[key] = saved_modules[key]
            else:
                sys.modules.pop(key, None)


def test_loader_to_reward_end_to_end(tmp_path, caplog):
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")
    reward = _load_module("reward")

    # 1. Generate a tiny dataset on disk (seeded -> reproducible).
    records = generator.generate_records(4, seed=2026, cap=16, trials=4)
    dataset_path = tmp_path / "boards.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        generator.write_jsonl(records, handle)

    # 2. Loader converts JSONL into prompt records (source_record shape).
    loaded = loader.load_training_dataset(str(dataset_path))
    assert len(loaded) == 4
    expected_keys = {"id", "prompt", "board", "seed", "random_baseline", "legal_moves"}
    for record in loaded:
        assert expected_keys <= set(record)
        assert game.normalize_board(record["board"])
        assert record["legal_moves"]

    # 3. Reward replays a policy move sequence per row; no model, no network.
    with caplog.at_level(logging.INFO):
        rewards = [
            reward.reward_fn(
                SimpleNamespace(
                    source_record=record,
                    completion="",
                    tool_calls=[
                        {
                            "name": "choose_moves",
                            "arguments": {"moves": ["left", "up", "right", "down", "left"]},
                        }
                    ],
                )
            )
            for record in loaded
        ]

    assert len(rewards) == 4
    assert all(isinstance(value, float) for value in rewards)
    # The reward is deterministic given the same source + moves.
    first_again = reward.reward_fn(
        SimpleNamespace(
            source_record=loaded[0],
            completion="",
            tool_calls=[{"name": "choose_moves", "arguments": {"moves": ["left", "up", "right", "down", "left"]}}],
        )
    )
    assert first_again == rewards[0]

    log_text = "\n".join(record.message for record in caplog.records)
    assert "improvement=" in log_text


def test_loader_raises_clear_error_when_dataset_missing(tmp_path):
    loader = _load_module("dataset_loader")

    missing = tmp_path / "does-not-exist.jsonl"
    try:
        loader.load_training_dataset(str(missing))
    except FileNotFoundError as exc:
        assert "dataset_generator.py" in str(exc)  # points at the fix
    else:  # pragma: no cover - defensive
        raise AssertionError("expected FileNotFoundError for a missing dataset")