from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "smoke_tutorial"


def _load_module(filename: str):
    path = EXAMPLE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"smoke_tutorial_{filename}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_loader_passthrough_prompt():
    """Test that rows with 'prompt' field pass through unchanged."""
    loader = _load_module("dataset_loader.py")
    raw = [{"prompt": "What is AI?", "reference": "Artificial intelligence"}]

    records = loader.load_training_dataset("unused", default_loader=lambda _: raw)

    assert records == [{"prompt": "What is AI?", "reference": "Artificial intelligence"}]


def test_dataset_loader_normalizes_question_answer():
    """Test that question/answer rows are normalized to prompt format."""
    loader = _load_module("dataset_loader.py")
    raw = [{"question": "What is 2+2?", "answer": "4"}]

    records = loader.load_training_dataset("unused", default_loader=lambda _: raw)

    assert len(records) == 1
    assert records[0]["prompt"] == "Question: What is 2+2?\nAnswer:"
    assert records[0]["reference"] == "4"


def test_dataset_loader_handles_empty_dataset():
    """Test that empty datasets are handled gracefully."""
    loader = _load_module("dataset_loader.py")

    records = loader.load_training_dataset("unused", default_loader=lambda _: [])

    assert records == []


def test_reward_fn_empty_completion():
    """Test that empty completions get 0 reward."""
    reward = _load_module("reward.py")

    class MockRecord:
        completion = ""

    assert reward.reward_fn(MockRecord()) == 0.0


def test_reward_fn_short_completion():
    """Test that very short completions get low reward."""
    reward = _load_module("reward.py")

    class MockRecord:
        completion = "Yes"

    assert reward.reward_fn(MockRecord()) == 0.0


def test_reward_fn_medium_completion():
    """Test that medium-length completions get medium reward."""
    reward = _load_module("reward.py")

    class MockRecord:
        completion = "This is a medium length response that has more than fifty characters in it."

    result = reward.reward_fn(MockRecord())
    assert 0.3 < result < 0.8


def test_reward_fn_long_with_reasoning():
    """Test that long completions with reasoning markers get high reward."""
    reward = _load_module("reward.py")

    class MockRecord:
        completion = "I believe this is correct because the evidence shows that the answer is 42. " * 3 + "Therefore, we can conclude that the solution is valid and complete."

    result = reward.reward_fn(MockRecord())
    assert result == 1.0


def test_reward_fn_long_without_reasoning():
    """Test that long completions without reasoning markers get slightly lower reward."""
    reward = _load_module("reward.py")

    class MockRecord:
        completion = "A" * 250  # Long but no reasoning markers

    result = reward.reward_fn(MockRecord())
    assert result == 0.8
