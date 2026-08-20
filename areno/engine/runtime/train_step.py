"""Training-step helpers: pack rollouts, build metadata, sync gradients.

`_pack_train_data` converts the rectangular (B, T) rollout pack the user
hands in into a packed varlen layout used by the attention kernel. That
reduces the train forward FLOPs to just the valid tokens while keeping every
per-action signal (advantage, rollout logprob, ref logprob, value, return)
indexed in lock-step with the per-action axis used by `packed_next_token_logprobs`.

`_grad_norm` and `_clip_grad_norm` are TP-aware: they
combine local contributions across the TP group and ignore parameters whose
gradients are replicated across ranks (so each parameter contributes exactly
once to the global norm).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import TrainMeta


def _metrics_to_float(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    """Coerce metric values into Python floats (CPU) for the result wire."""

    if metrics is None:
        return None
    out = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().float().cpu())
        else:
            out[key] = float(value)
    return out


def _merge_metrics(*metrics_list: dict[str, Any] | None) -> dict[str, float] | None:
    """Merge several metric dicts; later values overwrite earlier ones."""

    out: dict[str, float] = {}
    for metrics in metrics_list:
        converted = _metrics_to_float(metrics)
        if converted:
            out.update(converted)
    return out or None


def _dense_train_meta(tokens: torch.Tensor, *, sequence_parallel_enabled: bool) -> TrainMeta:
    """Build attention metadata for a dense rectangular (B, T) train batch."""

    batch, seqlen = tokens.shape
    # FlashAttention varlen API expects cumulative seqlens. For dense batches
    # every row has the same length, so the prefix sum is regular: 0, T, 2T...
    cu_seqlens = torch.arange(
        0,
        (batch + 1) * seqlen,
        seqlen,
        device=tokens.device,
        dtype=torch.int32,
    )
    ctx = get_tp_context()
    return TrainMeta(
        cu_seqlens=cu_seqlens,
        max_seqlen=seqlen,
        sequence_parallel=sequence_parallel_enabled and ctx.world_size > 1 and seqlen % ctx.world_size == 0,
    )


def _train_meta(data_pack: dict[str, Any], tokens: torch.Tensor) -> TrainMeta:
    """Pick packed vs dense `TrainMeta` based on whether the pack is packed."""

    cu_seqlens = data_pack.get("train_cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor):
        max_seqlen = int(data_pack.get("train_max_seqlen", 0))
        return TrainMeta(
            cu_seqlens=cu_seqlens.to(device=tokens.device, dtype=torch.int32),
            max_seqlen=max_seqlen,
            packed=True,
            activation_checkpointing=bool(data_pack.get("_activation_checkpointing_enabled", False)),
        )
    meta = _dense_train_meta(tokens, sequence_parallel_enabled=bool(data_pack.get("_sequence_parallel_enabled", True)))
    meta.activation_checkpointing = bool(data_pack.get("_activation_checkpointing_enabled", False))
    return meta


def _pack_train_data(data_pack: dict[str, Any]) -> dict[str, Any]:
    """Convert padded rollout rows into a packed varlen training batch.

    Packed training removes pad tokens from the forward graph while preserving
    per-action masks, advantages, and rollout logprobs for the loss function.
    """
    lengths = data_pack.get("lengths")
    input_ids = data_pack.get("input_ids")
    if not isinstance(lengths, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
        return data_pack
    if input_ids.ndim != 2 or lengths.ndim != 1:
        return data_pack
    batch = int(input_ids.shape[0])
    if int(lengths.numel()) != batch:
        return data_pack

    # Token-axis bookkeeping. `cu_seqlens[i+1]` is the prefix sum of valid
    # tokens up to row i, matching the FlashAttention varlen contract.
    lengths = lengths.to(device=input_ids.device, dtype=torch.long).clamp(min=1, max=input_ids.shape[1])
    total_tokens = int(lengths.sum().item())
    packed_ids = torch.empty(total_tokens, device=input_ids.device, dtype=input_ids.dtype)
    position_ids = torch.empty(total_tokens, device=input_ids.device, dtype=torch.long)
    cu_seqlens = torch.empty(batch + 1, device=input_ids.device, dtype=torch.int32)
    cu_seqlens[0] = 0

    # Action-axis bookkeeping. Each row contributes `length-1` action sites
    # (the logits at position t predict the token at t+1). These flat tensors
    # let the loss treat the whole batch as one big sequence of actions.
    action_count = int((lengths - 1).clamp(min=0).sum().item())
    packed_response_mask = torch.empty(action_count, device=input_ids.device, dtype=torch.bool)
    packed_advantages = torch.empty(action_count, device=input_ids.device, dtype=torch.float32)
    packed_logprobs = torch.empty(action_count, device=input_ids.device, dtype=torch.float32)
    packed_ref_logprobs = torch.empty(action_count, device=input_ids.device, dtype=torch.float32)
    packed_values = torch.empty(action_count, device=input_ids.device, dtype=torch.float32)
    packed_returns = torch.empty(action_count, device=input_ids.device, dtype=torch.float32)
    packed_seq_ids = torch.empty(action_count, device=input_ids.device, dtype=torch.long)

    prompt_mask = data_pack.get("prompt_mask")
    loss_mask = data_pack.get("loss_mask")
    advantages = data_pack.get("advantages")
    rollout_logprobs = data_pack.get("logprobs")
    ref_logprobs = data_pack.get("ref_logprobs")
    values = data_pack.get("values")
    returns = data_pack.get("returns")
    features = data_pack.get("features")
    if not all(isinstance(x, torch.Tensor) for x in (prompt_mask, advantages, rollout_logprobs)):
        return data_pack
    has_ppo_fields = all(isinstance(x, torch.Tensor) for x in (ref_logprobs, values, returns))
    packed_features = _pack_multimodal_features(features, input_ids, lengths, packed_ids.device)

    token_offset = 0
    action_offset = 0
    max_seqlen = 0
    for row in range(batch):
        length = int(lengths[row].item())
        max_seqlen = max(max_seqlen, length)
        packed_ids[token_offset : token_offset + length] = input_ids[row, :length]
        # Position ids restart at 0 for each packed row, so rotary embeddings
        # treat the packed sequence as a concatenation of independent sequences.
        position_ids[token_offset : token_offset + length] = torch.arange(length, device=input_ids.device)
        cu_seqlens[row + 1] = token_offset + length
        if length > 1:
            # The per-action signals are sliced from index 1..length so that
            # `action_offset + k` corresponds to the prediction of token k+1.
            action_len = length - 1
            action_slice = slice(action_offset, action_offset + action_len)
            response_mask = ~prompt_mask[row, 1:length].to(dtype=torch.bool)
            if isinstance(loss_mask, torch.Tensor):
                response_mask = response_mask & loss_mask[row, 1:length].to(dtype=torch.bool)
            packed_response_mask[action_slice] = response_mask
            packed_advantages[action_slice] = advantages[row, 1:length].to(dtype=torch.float32)
            packed_logprobs[action_slice] = rollout_logprobs[row, 1:length].to(dtype=torch.float32)
            if has_ppo_fields:
                packed_ref_logprobs[action_slice] = ref_logprobs[row, 1:length].to(dtype=torch.float32)
                packed_values[action_slice] = values[row, 1:length].to(dtype=torch.float32)
                packed_returns[action_slice] = returns[row, 1:length].to(dtype=torch.float32)
            packed_seq_ids[action_slice] = row
            action_offset += action_len
        token_offset += length

    # The result keeps the original keys (so the loss can still read them) but
    # adds the new packed views. `input_ids` keeps shape (1, total_tokens) to
    # match the (B, T) contract used elsewhere with B=1.
    packed = dict(data_pack)
    packed.update(
        {
            "input_ids": packed_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "train_cu_seqlens": cu_seqlens,
            "train_max_seqlen": max_seqlen,
            "packed_response_mask": packed_response_mask,
            "packed_advantages": packed_advantages,
            "packed_logprobs": packed_logprobs,
            "packed_seq_ids": packed_seq_ids,
            "packed_num_sequences": batch,
        }
    )
    if packed_features is not None:
        packed["features"] = packed_features
    if has_ppo_fields:
        packed.update(
            {
                "packed_ref_logprobs": packed_ref_logprobs,
                "packed_values": packed_values,
                "packed_returns": packed_returns,
            }
        )
    return packed


def _pack_multimodal_features(
    features: object,
    input_ids: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> dict | None:
    if features is None:
        return None
    if not isinstance(features, list):
        return features if isinstance(features, dict) else None
    if len(features) != int(input_ids.shape[0]):
        return None
    packed_rows: list[dict] = []
    packed_masks: list[torch.Tensor] = []
    packed_positions: list[torch.Tensor] = []
    image_token_id = None
    has_any_feature = False
    for row_idx, row in enumerate(features):
        length = int(lengths[row_idx].item())
        row_tokens = input_ids[row_idx, :length]
        if not isinstance(row, dict):
            packed_masks.append(torch.zeros(length, device=device, dtype=torch.bool))
            packed_positions.append(torch.arange(length, device=device, dtype=torch.long).view(1, -1).expand(3, -1))
            continue
        has_any_feature = True
        packed_rows.append(row)
        row_image_token_id = row.get("image_token_id")
        if row_image_token_id is not None:
            image_token_id = int(row_image_token_id)
        packed_masks.append(_row_image_token_mask(row, row_tokens, row_idx, image_token_id))
        packed_positions.append(_row_mrope_positions(row, length, device))
    if not has_any_feature:
        return None
    packed: dict = {"image_feature_rows": packed_rows}
    if image_token_id is not None:
        packed["image_token_id"] = image_token_id
    if packed_masks:
        packed["image_token_mask"] = torch.cat(packed_masks, dim=0).to(device=device, dtype=torch.bool)
    if packed_positions:
        packed["mrope_position_ids"] = torch.cat(packed_positions, dim=1).to(device=device, dtype=torch.long)
    return packed


def _row_image_token_mask(
    row: dict, row_tokens: torch.Tensor, row_idx: int, image_token_id: int | None
) -> torch.Tensor:
    mask = row.get("image_token_mask")
    if mask is not None:
        mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        if mask.ndim == 2:
            mask = mask[row_idx]
        mask = mask.reshape(-1).to(device=row_tokens.device, dtype=torch.bool)
        if int(mask.numel()) >= int(row_tokens.numel()):
            return mask[: row_tokens.numel()]
    if image_token_id is not None:
        return row_tokens == int(image_token_id)
    return torch.zeros(int(row_tokens.numel()), device=row_tokens.device, dtype=torch.bool)


def _row_mrope_positions(row: dict, length: int, device: torch.device) -> torch.Tensor:
    positions = row.get("mrope_position_ids")
    if positions is None:
        return torch.arange(length, device=device, dtype=torch.long).view(1, -1).expand(3, -1)
    positions = positions if isinstance(positions, torch.Tensor) else torch.as_tensor(positions)
    positions = positions.to(device=device, dtype=torch.long)
    if positions.ndim == 3 and int(positions.shape[0]) == 3 and int(positions.shape[1]) == 1:
        positions = positions[:, 0, :]
    elif positions.ndim == 3 and int(positions.shape[0]) == 1 and int(positions.shape[1]) == 3:
        positions = positions[0]
    if positions.ndim != 2 or int(positions.shape[0]) != 3:
        return torch.arange(length, device=device, dtype=torch.long).view(1, -1).expand(3, -1)
    current = min(int(positions.shape[-1]), length)
    if current == length:
        return positions[:, :length]
    full = torch.empty((3, length), device=device, dtype=torch.long)
    if current > 0:
        full[:, :current] = positions[:, :current]
        next_pos = int(positions[:, :current].max().item()) + 1
    else:
        next_pos = 0
    tail_len = length - current
    full[:, current:] = torch.arange(tail_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + next_pos
    return full


def _grad_norm(parameters) -> float:
    """Compute global L2 grad norm across tensor-parallel ranks."""
    return _grad_norms(parameters)["global"]


def _grad_norms(parameters, groups=()) -> dict[str, float]:
    """Compute global and selected parameter-group L2 norms in one pass."""

    names = ("global", *groups)
    totals = None
    group_indexes = {name: index for index, name in enumerate(names[1:], start=1)}
    buckets = {}
    for param in _grads_for_norm(parameters):
        grad = _param_grad(param)
        if grad is None:
            continue
        key = (grad.device, grad.dtype)
        bucket = buckets.setdefault(key, ([], []))
        bucket[0].append(param)
        bucket[1].append(grad.detach())

    for params, grads in buckets.values():
        # foreach_norm batches kernel launches and only materializes one scalar
        # per parameter instead of a squared tensor as large as every gradient.
        if grads[0].dtype in {torch.float32, torch.float64}:
            norms = torch._foreach_norm(grads, 2)
        else:
            norms = [torch.linalg.vector_norm(grad, ord=2, dtype=torch.float32) for grad in grads]
        squared_norms = torch.stack(norms).float().square()
        if totals is None:
            totals = torch.zeros(len(names), device=squared_norms.device, dtype=torch.float32)
        totals[0].add_(squared_norms.sum())
        for group, group_index in group_indexes.items():
            indexes = [index for index, param in enumerate(params) if getattr(param, "_areno_lr_group", None) == group]
            if indexes:
                totals[group_index].add_(squared_norms[indexes].sum())

    if totals is None:
        return dict.fromkeys(names, 0.0)
    ctx = get_tp_context()
    # Summing only TP-sharded contributions (`_grads_for_norm`) and then
    # all-reducing across TP gives the same value every rank would compute on
    # the unsharded model, without double-counting replicated parameters.
    if ctx.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=ctx.group)
    values = totals.sqrt().cpu().tolist()
    return dict(zip(names, values, strict=True))


def _clip_grad_norm(parameters, grad_norm: float, max_norm: float) -> None:
    """Scale all gradients in place if their global norm exceeds `max_norm`."""

    if grad_norm <= 0.0:
        return
    clip_coef = float(max_norm) / (grad_norm + 1e-6)
    if clip_coef >= 1.0:
        return
    coef = None
    for param in parameters:
        grad = _param_grad(param)
        if grad is None:
            continue
        if coef is None:
            coef = torch.tensor(clip_coef, device=grad.device, dtype=grad.dtype)
        grad.mul_(coef)


def _grads_for_norm(parameters):
    """Yield params whose grads should contribute to the global TP grad norm.

    TP-sharded params contribute on every rank (they are different slices).
    Replicated params only contribute on rank 0 so the TP all-reduce does not
    double-count them.
    """

    ctx = get_tp_context()
    for param in parameters:
        if _param_grad(param) is None:
            continue
        is_tp_parallel = bool(getattr(param, "tensor_model_parallel", False))
        if is_tp_parallel or ctx.rank == 0:
            yield param


def _param_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    """Prefer the FP32 `main_grad` accumulator used by the optimizer split."""

    main_grad = getattr(param, "main_grad", None)
    if isinstance(main_grad, torch.Tensor):
        return main_grad
    return param.grad
