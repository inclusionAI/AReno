"""RolloutOutput merging helpers used by `ArenoEngine`.

The engine splits prompts both round-robin across DP ranks (see
`split_list_by_dp`) and into prefill-budget-friendly chunks. Workers emit one
`RolloutOutput` per chunk per rank; this module stitches them back together in
the user's original input order and produces padded tensors of consistent
shape, so downstream loss code can treat the output uniformly.
"""

from __future__ import annotations

import torch

from areno.engine.data import RolloutOutput
from areno.engine.runtime.common import pad_rollout_rows


def _merge_rollouts(outputs: list[RolloutOutput]) -> RolloutOutput:
    """Concatenate rollout chunks produced by prefill-budget chunking."""
    outputs = [output for output in outputs if len(output.prompt_ids) > 0]
    if not outputs:
        return _empty_rollout()
    prompt_ids = [row for output in outputs for row in output.prompt_ids]
    response_ids = [row for output in outputs for row in output.response_ids]
    finish_reason = [reason for output in outputs for reason in output.finish_reason]
    logprob_rows = []
    for output in outputs:
        for local_idx, (prompt, response) in enumerate(zip(output.prompt_ids, output.response_ids, strict=True)):
            response_len = len(response)
            logprob_rows.append(output.logprobs[local_idx, :response_len])
    input_ids, attention_mask, response_mask, logprobs = pad_rollout_rows(prompt_ids, response_ids, logprob_rows)
    return RolloutOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        logprobs=logprobs,
        finish_reason=finish_reason,
        metrics=_merge_rollout_metrics(outputs),
    )


def _merge_dp_rollouts_in_input_order(outputs: list[RolloutOutput | None], total_count: int) -> RolloutOutput:
    """Undo round-robin DP prompt splitting and rebuild original row order."""
    outputs = [output or _empty_rollout() for output in outputs]
    if total_count == 0:
        return _empty_rollout()
    dp_size = len(outputs)
    prompt_ids = []
    response_ids = []
    finish_reason = []
    logprob_rows = []
    # The split was `prompts[rank::dp_size]`, so inverse is to map
    # original_idx -> (dp_rank = original_idx % dp_size,
    #                  local_idx = original_idx // dp_size).
    for original_idx in range(total_count):
        dp_rank = original_idx % dp_size
        local_idx = original_idx // dp_size
        output = outputs[dp_rank]
        if local_idx >= len(output.prompt_ids):
            raise RuntimeError(
                f"missing rollout row for original_idx={original_idx} dp_rank={dp_rank} local_idx={local_idx}"
            )
        prompt_ids.append(output.prompt_ids[local_idx])
        response_ids.append(output.response_ids[local_idx])
        finish_reason.append(output.finish_reason[local_idx])
        response_len = len(output.response_ids[local_idx])
        logprob_rows.append(output.logprobs[local_idx, :response_len].detach().cpu())
    return _build_rollout_from_rows(
        prompt_ids,
        response_ids,
        finish_reason,
        logprob_rows,
        metrics=_merge_rollout_metrics(outputs),
    )


def _build_rollout_from_rows(
    prompt_ids: list[list[int]],
    response_ids: list[list[int]],
    finish_reason: list[str],
    logprob_rows: list[torch.Tensor],
    *,
    metrics: dict[str, float] | None,
) -> RolloutOutput:
    """Build padded rollout tensors from variable-length Python rows."""
    if not prompt_ids:
        return _empty_rollout()
    input_ids, attention_mask, response_mask, logprobs = pad_rollout_rows(prompt_ids, response_ids, logprob_rows)
    return RolloutOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        logprobs=logprobs,
        finish_reason=finish_reason,
        metrics=metrics,
    )


def _empty_rollout() -> RolloutOutput:
    """Return an empty rollout for the no-prompts path."""

    return RolloutOutput(
        prompt_ids=[],
        response_ids=[],
        input_ids=torch.empty(0, 0, dtype=torch.long),
        attention_mask=torch.empty(0, 0, dtype=torch.long),
        response_mask=torch.empty(0, 0, dtype=torch.long),
        logprobs=torch.empty(0, 0, dtype=torch.float32),
        finish_reason=[],
        metrics=None,
    )


def partial_tail_threshold(local_running: int, coalesce_timeout_s: float) -> int:
    """Return active-row cutoff for async rollout tail continuation."""

    if coalesce_timeout_s <= 0.0 or local_running <= 1:
        return 0
    return max(1, int(local_running) // 8)


def _merge_rollout_metrics(outputs: list[RolloutOutput]) -> dict[str, float] | None:
    """Sum same-named metrics across rollout chunks; drop empty inputs."""

    merged: dict[str, float] = {}
    for output in outputs:
        if not output.metrics:
            continue
        for key, value in output.metrics.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged or None
