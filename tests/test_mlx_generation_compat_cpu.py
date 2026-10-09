"""CPU contracts for MLX-LM stop-sequence API compatibility."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from areno.api.backend.mlx.generation import _TextBatchGenerator
from areno.api.backend.mlx.numerics import float32_logits_processor
from areno.api.models import SamplingParams


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_text_insert_preserves_stop_sequences_across_mlx_lm_apis(monkeypatch, modern, ignore_eos):
    module = ModuleType("mlx_lm.generate")
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sampler = object()
    sample_utils.make_sampler = lambda **kwargs: sampler

    class StopState:
        def __init__(self, sequences):
            self.sequences = sequences

    option = "stop_sequences" if modern else "stop_matchers"
    setattr(module, "StopSequences" if modern else "StopSequenceMatcher", StopState)
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    calls = []

    class Generator:
        def insert(self, prompts, **kwargs):
            assert option in kwargs
            assert ("stop_matchers" if modern else "stop_sequences") not in kwargs
            calls.append((prompts, kwargs))
            return [10, 11]

    wrapper = object.__new__(_TextBatchGenerator)
    wrapper._tokenizer = SimpleNamespace(eos_token_ids=[2, 3], encode=lambda text, **kw: [7, 8])
    wrapper._generator = Generator()
    params = SamplingParams(max_new_tokens=5, stop=["END"], stop_token_ids=[3, 4], ignore_eos=ignore_eos)
    assert wrapper.insert([[1], [9]], [None, None], params) == [10, 11]
    prompts, kwargs = calls[0]
    expected = [[3], [4], [7, 8]] if ignore_eos else [[2], [3], [4], [7, 8]]
    assert prompts == [[1], [9]]
    assert [stop.sequences for stop in kwargs[option]] == [expected, expected]
    assert kwargs[option][0] is not kwargs[option][1]
    assert kwargs["samplers"] == [sampler, sampler]
    assert kwargs["max_tokens"] == [5, 5]
    assert kwargs["logits_processors"] == [[float32_logits_processor], [float32_logits_processor]]
