"""Shared utilities for the engine/worker split.

`split_*` helpers shard user payloads across DP ranks. `dp_rank0_results`
picks one TP-rank-0 result per DP rank so the coordinator never duplicates
output. `merge_train_stats` averages losses from all DP ranks for the user.
`_check_token_ids` is a debug guardrail that catches degenerate sampler
outputs before they reach the next forward pass.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from areno.engine.data import TrainStats

_CHECK_TOKEN_IDS = os.getenv("ARENO_CHECK_TOKEN_IDS", "0").lower() in {"1", "true", "yes", "on"}


def ceil_div(a: int, b: int) -> int:
    """Integer ceil division used for block and graph sizing."""

    return (a + b - 1) // b


def pad_rows(rows: list[Any], *, dtype: torch.dtype, fill_value: int | float | bool = 0, width: int | None = None) -> torch.Tensor:
    """Pad variable-length 1D rows into a rectangular CPU tensor."""

    if width is None:
        width = max((len(row) for row in rows), default=0)
    out = torch.full((len(rows), width), fill_value, dtype=dtype)
    for row_idx, row in enumerate(rows):
        if len(row) == 0:
            continue
        out[row_idx, : len(row)] = torch.as_tensor(row, dtype=dtype)
    return out


def pad_rollout_rows(
    prompt_ids: list[list[int]],
    response_ids: list[list[int]],
    logprob_rows: list[Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded rollout tensors from prompt, response, and logprob rows."""

    input_rows = [prompt + response for prompt, response in zip(prompt_ids, response_ids, strict=True)]
    response_mask_rows = [
        ([0] * len(prompt)) + ([1] * len(response))
        for prompt, response in zip(prompt_ids, response_ids, strict=True)
    ]
    if logprob_rows is None:
        logprob_rows = [[] for _ in response_ids]
    max_response = max((len(row) for row in response_ids), default=0)
    input_ids = pad_rows(input_rows, dtype=torch.long)
    attention_mask = pad_rows([[1] * len(row) for row in input_rows], dtype=torch.long, width=input_ids.shape[1])
    response_mask = pad_rows(response_mask_rows, dtype=torch.long, width=input_ids.shape[1])
    logprobs = pad_rows(logprob_rows, dtype=torch.float32, width=max_response)
    return input_ids, attention_mask, response_mask, logprobs


def split_list_by_dp(items: list[Any], dp_size: int) -> list[list[Any]]:
    """Round-robin split a flat list so each DP rank gets a strided slice."""

    return [items[rank::dp_size] for rank in range(dp_size)]


def dp_rank0_results(results: list[Any], tp_size: int, dp_size: int) -> list[Any]:
    """Pick the result from TP rank 0 for each DP rank, dropping TP duplicates."""

    return [results[dp_rank * tp_size] for dp_rank in range(dp_size)]


def split_data_pack_by_dp(data_pack: dict[str, Any], dp_size: int) -> list[dict[str, Any]]:
    """Shard batch-major tensors and containers by data-parallel rank."""
    if dp_size == 1:
        return [data_pack]
    input_ids = data_pack["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("data_pack['input_ids'] must be a torch.Tensor")
    batch = int(input_ids.shape[0])
    # Tiny batches (smaller than dp_size) cannot be sliced cleanly, so just
    # replicate the whole pack to every rank; the loss will be averaged later.
    if batch < dp_size:
        return [data_pack for _ in range(dp_size)]
    return [slice_data_pack(data_pack, rank, dp_size, batch) for rank in range(dp_size)]


def slice_data_pack(obj: Any, rank: int, dp_size: int, batch: int) -> Any:
    """Recursively slice values whose leading dimension is the batch size."""
    if isinstance(obj, torch.Tensor):
        # Strided slice keeps DP shards balanced and aligns with the
        # round-robin layout used by `split_list_by_dp`.
        if obj.ndim > 0 and int(obj.shape[0]) == batch:
            return obj[rank::dp_size].contiguous()
        return obj
    if isinstance(obj, dict):
        return {key: slice_data_pack(value, rank, dp_size, batch) for key, value in obj.items()}
    if isinstance(obj, list):
        if len(obj) == batch:
            return obj[rank::dp_size]
        return [slice_data_pack(value, rank, dp_size, batch) for value in obj]
    if isinstance(obj, tuple):
        if len(obj) == batch:
            return tuple(obj[rank::dp_size])
        return tuple(slice_data_pack(value, rank, dp_size, batch) for value in obj)
    return obj


def merge_train_stats(results: list[dict[str, Any]]) -> TrainStats:
    """Average per-DP train losses into one user-visible `TrainStats`."""

    loss = sum(float(result["loss"]) for result in results) / len(results)
    stepped = all(bool(result["stepped"]) for result in results)
    metrics = merge_metric_dicts([result.get("metrics") for result in results])
    return TrainStats(loss=loss, stepped=stepped, metrics=metrics)


def merge_metric_dicts(metrics_list: list[dict[str, Any] | None]) -> dict[str, float] | None:
    """Average matching metric keys across DP results, dropping empty inputs."""

    merged: dict[str, list[float]] = {}
    for metrics in metrics_list:
        if not metrics:
            continue
        for key, value in metrics.items():
            merged.setdefault(key, []).append(float(value))
    if not merged:
        return None
    return {key: sum(values) / len(values) for key, values in merged.items()}


def _device_long(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to device and cast to int64; no-op if already there."""

    if tensor.device == device and tensor.dtype == torch.long:
        return tensor
    return tensor.to(device, non_blocking=True).long()


def _check_token_ids(tokens: torch.Tensor, vocab_size: int, name: str) -> None:
    """Raise a descriptive error if any sampled token id is out of range."""

    if not _CHECK_TOKEN_IDS:
        return
    if tokens.numel() == 0:
        return
    invalid = (tokens < 0) | (tokens >= vocab_size)
    if not bool(invalid.any().item()):
        return
    bad = tokens[invalid]
    raise RuntimeError(
        f"{name} out of vocab range [0, {vocab_size}): "
        f"min={int(tokens.min().item())} max={int(tokens.max().item())} "
        f"bad_min={int(bad.min().item())} bad_max={int(bad.max().item())} "
        f"shape={tuple(tokens.shape)}"
    )
