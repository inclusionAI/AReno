"""The tiny smoke benchmark rejects examples with no response targets."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "galore_smoke_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "bench_galore.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(range(len(text)))


@pytest.mark.parametrize("record", [{"text": "a"}, {"prompt": "p" * 64, "response": "answer"}])
def test_targetless_examples_are_rejected(record):
    with pytest.raises(ValueError, match="target"):
        benchmark.encode(Tokenizer(), [record])


def test_valid_smoke_encoding_remains_unchanged():
    rows = benchmark.encode(Tokenizer(), [{"text": "abc"}, {"prompt": "ab", "response": "cde"}])
    assert rows[0].tokens == [0, 1, 2] and rows[0].prompt_len == 1
    assert rows[1].tokens == [0, 1, 0, 1, 2] and rows[1].prompt_len == 2
