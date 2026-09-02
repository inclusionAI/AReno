"""CUDA rollout option translation."""

from __future__ import annotations

from areno.api.config import CudaConfig
from areno.api.context import Context
from areno.api.models import SamplingParams


def rollout_options(ctx: Context, sampling_params: SamplingParams):
    """Translate public sampling options to the CUDA engine representation."""

    eos_token_ids = () if sampling_params.ignore_eos else ctx.eos_token_ids
    stop_token_ids = tuple(sampling_params.stop_token_ids or ())
    # Protocol markers are often registered as special tokens.  Gemma 4 uses
    # <|tool_call> / <tool_call|> and Qwen-family templates may likewise rely
    # on special tokens in generated structured output, so suppressing every
    # tokenizer special id makes valid tool calls impossible.  EOS is handled
    # separately by the inference manager; only tokens that are never valid
    # generated content belong in this explicit mask.
    suppress_candidates = set(explicit_suppress_token_ids(ctx.tokenizer))
    suppress_token_ids = tuple(
        sorted(token_id for token_id in suppress_candidates if token_id not in {*eos_token_ids, *stop_token_ids})
    )
    cfg = ctx.custom_config
    if cfg is None:
        cfg = CudaConfig()
    if not isinstance(cfg, CudaConfig):
        raise TypeError(f"CudaBackend requires CudaConfig, got {type(cfg)!r}")

    from areno import SamplingParams as CudaSamplingParams

    max_prompt_len = sampling_params.max_prompt_len
    if sampling_params.max_context_len is not None:
        # Agentic turns repeatedly submit a growing transcript. Reserve the
        # full configured context on the first turn so paged KV storage and
        # CUDA graphs do not get rebuilt every time the transcript crosses a
        # new block boundary.
        context_prompt_capacity = max(int(sampling_params.max_context_len) - int(sampling_params.max_new_tokens), 1)
        max_prompt_len = max(int(max_prompt_len or 0), context_prompt_capacity)

    return {
        "max_prompt_len": max_prompt_len,
        "eos_token_id": eos_token_ids,
        "max_running_prompts": cfg.max_running_prompts,
        "decode_progress_interval_s": cfg.decode_progress_interval_s,
        "sampling_params": CudaSamplingParams(
            temperature=0.0 if sampling_params.greedy else sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=max(0, sampling_params.top_k),
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            suppress_special_tokens=False,
        ),
    }


def explicit_suppress_token_ids(tokenizer) -> tuple[int, ...]:
    """Return marker ids that should never be sampled as normal text."""

    values = []
    for attr in ("pad_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            values.append(value)
    return tuple(dict.fromkeys(values))


__all__ = ["explicit_suppress_token_ids", "rollout_options"]
