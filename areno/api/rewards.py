"""Reward function loading, batched execution, and group-relative advantage normalisation."""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class RewardEvent(BaseModel):
    type: Literal["request", "assistant_text", "assistant_tool_call", "tool_result", "finish", "error"]
    text: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | str | None = None
    content: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardRecord(BaseModel):
    prompt: str
    completion: str
    rendered_completion: str | None = None
    final_answer: str | None = None
    answer: Any | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[RewardEvent] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    loss_mask: list[bool] = Field(default_factory=list)
    source_record: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    return ((rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + eps)).tolist()


def load_reward_fn(path: str) -> Callable[[RewardRecord], float]:
    module_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reward function from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        reward_fn = module.reward_fn
    except AttributeError as exc:
        raise ValueError(f"{module_path} must define callable reward_fn(record)") from exc
    if not callable(reward_fn):
        raise ValueError(f"{module_path} must define callable reward_fn(record)")
    return reward_fn


def make_reward_record(
    *, prompt, completion, source_record, answer=None, tokens=None, logprobs=None, loss_mask=None, metadata=None
):
    return RewardRecord(
        prompt=prompt,
        completion=completion,
        rendered_completion=completion,
        final_answer=completion,
        answer=answer,
        tokens=list(tokens or []),
        logprobs=[float(v) for v in (logprobs or [])],
        loss_mask=list(loss_mask or []),
        source_record=dict(source_record),
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class RewardExecutionStats:
    path: Literal["batch", "scalar"]
    wall_time_s: float
    per_example_time_s: float
    count: int
    error: str | None = None


class RewardFnBundle(BaseModel):
    reward_fn: Callable[[RewardRecord], float] | None = None
    reward_batch: Callable[[list[RewardRecord]], list[float]] | None = None
    source_path: str
    model_config = {"arbitrary_types_allowed": True}


def load_reward(path: str) -> RewardFnBundle:
    module_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reward function from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward_fn = getattr(module, "reward_fn", None)
    reward_batch = getattr(module, "reward_batch", None)
    if reward_fn is None and reward_batch is None:
        raise ValueError(f"{module_path} must define callable reward_fn(record) or reward_batch(records)")
    if reward_fn is not None and not callable(reward_fn):
        raise ValueError(f"{module_path}.reward_fn must be callable")
    if reward_batch is not None and not callable(reward_batch):
        raise ValueError(f"{module_path}.reward_batch must be callable")
    return RewardFnBundle(reward_fn=reward_fn, reward_batch=reward_batch, source_path=str(module_path))


def _validate_cardinality(output: list[float], expected: int) -> None:
    got = len(output)
    if got != expected:
        first_bad = min(got, expected)
        raise ValueError(
            f"reward_batch returned {got} scores for {expected} records; first mismatched index is {first_bad}"
        )


def call_reward(bundle, records, *, prefer_batch=True):
    if not records:
        return [], RewardExecutionStats(path="scalar", wall_time_s=0.0, per_example_time_s=0.0, count=0, error=None)
    if prefer_batch and bundle.reward_batch is not None:
        start = time.perf_counter()
        output = list(bundle.reward_batch(list(records)))
        wall = time.perf_counter() - start
        _validate_cardinality(output, len(records))
        return [float(s) for s in output], RewardExecutionStats(
            path="batch", wall_time_s=wall, per_example_time_s=wall / len(records), count=len(records), error=None
        )
    scores, per_times = [], []
    start_all = time.perf_counter()
    for idx, record in enumerate(records):
        if bundle.reward_fn is None:
            s = time.perf_counter()
            out = list(bundle.reward_batch([record]))
            per_times.append(time.perf_counter() - s)
            _validate_cardinality(out, 1)
            scores.append(float(out[0]))
        else:
            s = time.perf_counter()
            try:
                score = bundle.reward_fn(record)
            except Exception as exc:
                raise type(exc)(f"reward_fn failed at index {idx}: {exc}") from exc
            per_times.append(time.perf_counter() - s)
            scores.append(float(score))
    wall = time.perf_counter() - start_all
    return scores, RewardExecutionStats(
        path="scalar",
        wall_time_s=wall,
        per_example_time_s=sum(per_times) / len(per_times),
        count=len(records),
        error=None,
    )
