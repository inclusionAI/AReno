from types import SimpleNamespace

from areno.api.backend.cuda.generation import explicit_suppress_token_ids, rollout_options
from areno.api.config import CudaConfig
from areno.api.models import SamplingParams


class _StructuredOutputTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    unk_token_id = 2
    eos_token_id = 106
    # Gemma 4 tool-call delimiters and turn markers are special tokens. Qwen
    # multimodal tokenizers can also register generated protocol markers here.
    all_special_ids = [0, 1, 2, 48, 49, 105, 106]


def test_explicit_suppression_only_contains_never_generated_tokens():
    assert explicit_suppress_token_ids(_StructuredOutputTokenizer()) == (0, 1, 2)


def test_cuda_rollout_does_not_suppress_structured_output_special_tokens():
    ctx = SimpleNamespace(
        tokenizer=_StructuredOutputTokenizer(),
        eos_token_ids=(106,),
        custom_config=CudaConfig(),
    )

    options = rollout_options(ctx, SamplingParams())
    native = options["sampling_params"]

    assert native.suppress_token_ids == (0, 1, 2)
    assert native.suppress_special_tokens is False
    assert 48 not in native.suppress_token_ids
    assert 49 not in native.suppress_token_ids
    assert options["eos_token_id"] == (106,)


def test_cuda_rollout_keeps_stop_tokens_sampleable_until_stop_detection():
    ctx = SimpleNamespace(
        tokenizer=_StructuredOutputTokenizer(),
        eos_token_ids=(106,),
        custom_config=CudaConfig(),
    )

    options = rollout_options(ctx, SamplingParams(stop_token_ids=[49]))
    native = options["sampling_params"]

    assert native.stop_token_ids == (49,)
    assert 49 not in native.suppress_token_ids


def test_cuda_rollout_reserves_full_agentic_context_capacity():
    ctx = SimpleNamespace(
        tokenizer=_StructuredOutputTokenizer(),
        eos_token_ids=(106,),
        custom_config=CudaConfig(),
    )
    params = SamplingParams(max_new_tokens=256, max_prompt_len=2048, max_context_len=5000)

    options = rollout_options(ctx, params)

    assert options["max_prompt_len"] == 4744
