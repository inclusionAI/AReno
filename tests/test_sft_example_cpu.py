from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "alpaca"


def _load_dataset_loader():
    path = EXAMPLE_DIR / "dataset_loader.py"
    spec = importlib.util.spec_from_file_location("sft_alpaca_dataset_loader_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_alpaca_sft_loader_returns_prompt_response_rows():
    loader = _load_dataset_loader()
    raw = [
        {
            "instruction": "Convert text to uppercase.",
            "input": "hello",
            "output": "Use `hello.upper()`.",
        }
    ]

    records = loader.load_training_dataset("unused", default_loader=lambda _: raw)

    assert records == [
        {
            "prompt": "Instruction: Convert text to uppercase.\nInput: hello\nResponse:",
            "response": "Use `hello.upper()`.",
        }
    ]
