"""Pydantic models shared across the public API surface.

These types form the wire-format contract between user algorithm code and the
backends. Each schema is intentionally minimal: rollouts return tokens and
logprobs, training sequences carry the prompt+response packing plus the extra
fields (advantages, returns, ref-logprobs, ...) consumed by policy-gradient
loss functions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BackendType(Enum):
    """Available execution backends for an `Trainer`."""

    CUDA = "CUDA"
    MLX = "MLX"


class SamplingParams(BaseModel):
    """Generation controls used by rollout APIs.

    The fields intentionally mirror common inference-server sampling knobs so
    algorithm code can stay backend-agnostic. `greedy=True` overrides
    `temperature` on the backend side (temperature is forced to 0). `top_k=-1`
    disables top-k filtering; `max_prompt_len` is an upper bound the backend
    can enforce when truncating long prompts. `max_context_len` caps full
    agentic trajectory contexts, including the initial prompt and all generated
    turns concatenated for training.
    """

    greedy: bool = Field(default=False)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=-1)
    max_new_tokens: int = Field(default=16)
    max_context_len: int | None = Field(default=None)
    temperature: float = Field(default=1.0)
    stop: list[str] | None = Field(default=None)
    stop_token_ids: list[int] | None = Field(default=None)
    ignore_eos: bool = Field(default=False)
    seed: int | None = Field(default=None)
    skip_special_tokens: bool = Field(default=True)
    max_prompt_len: int | None = Field(default=None)


class RolloutSequence(BaseModel):
    """One sampled completion returned by rollout.

    `resp_logprobs[i]` is the log-probability of `resp_tokens[i]` under the
    rollout policy and becomes the "old logprobs" reference signal in
    importance-weighted losses (PPO/GSPO/GRPO).
    """

    resp_tokens: list[int] = Field(default_factory=list)
    resp_logprobs: list[float] = Field(default_factory=list)
    # Kept as ``Any`` so CUDA rollouts can retain a compact CPU tensor instead
    # of expanding every route id into millions of Python integers. Agentic
    # OpenAI responses serialize the same value to nested lists at the edge.
    routed_experts: Any | None = Field(default=None)


class RolloutResult(BaseModel):
    """All sampled completions for one prompt."""

    sequences: list[RolloutSequence] = Field(default_factory=list)
    adapter_version: int | None = Field(default=None)


class TrainSequence(BaseModel):
    """One rollout sequence converted into a policy-gradient training sample.

    `tokens` contains prompt and response tokens. `prompt_mask` marks prompt
    positions, so loss functions can train only on response tokens while still
    conditioning on the prompt. Optional fields (`returns`, `values`,
    `ref_logprobs`) are only populated for algorithms that need them (PPO with
    a critic and reference model).
    """

    prompt_mask: list[bool] = Field(default_factory=list)
    loss_mask: list[bool] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    advantages: list[float] = Field(default_factory=list)
    prompt_len: int | None = Field(default=None)
    scalar_advantage: float | None = Field(default=None)
    returns: list[float] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    ref_logprobs: list[float] = Field(default_factory=list)
    features: dict[str, Any] | list[dict[str, Any] | None] | None = Field(default=None)
    reward: float = Field(default=0.0)
    eos_token_id: int = Field(default=0)
    routed_experts: Any | None = Field(default=None)
