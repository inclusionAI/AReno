"""Per-batch mutable state used by the rollout scheduler.

`InferenceBatchState` is the bookkeeping object that walks a list of prompts
through the paged-KV decode loop. It admits prompts under both a sequence
budget (`max_running_seqs`) and a paged-block budget (`num_cache_blocks`),
packs admitted prompts into varlen prefill payloads, and converts the
finished Python rows back into the padded tensors that `RolloutOutput`
exposes to the user.
"""

from __future__ import annotations

import torch

from areno.engine.data import RolloutOutput
from areno.engine.runtime.common import ceil_div, pad_rollout_rows
from areno.engine.runtime.metadata import InferMeta


class InferenceBatchState:
    """Mutable scheduler state for one rollout batch.

    It tracks prompt admission, paged KV block ownership, generated tokens, and
    finished sequences. Prefill admits new prompts into free blocks; decode only
    advances currently active sequence ids.
    """

    def __init__(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        *,
        max_running_seqs: int | None = None,
        max_cache_len: int | None = None,
        max_prefill_tokens: int = 8192,
        kv_block_size: int = 256,
        num_cache_blocks: int | None = None,
        prompt_features: list[dict | None] | None = None,
    ):
        """Create rollout state and reserve bookkeeping for paged KV blocks."""

        self.prompts = prompts
        self.prompt_features = _normalize_prompt_features(prompt_features, len(prompts))
        self.generated = [[] for _ in prompts]
        self.logprobs = [[] for _ in prompts]
        self.max_new_tokens = max_new_tokens
        self.finished = [False for _ in prompts]
        self.finish_reason = ["" for _ in prompts]
        self.metrics: dict[str, float] = {}
        self.max_running_seqs = max_running_seqs or len(prompts)
        self._max_cache_len = max_cache_len or max(len(prompt) + max_new_tokens for prompt in prompts)
        self.max_prefill_tokens = max_prefill_tokens
        self.kv_block_size = kv_block_size
        self.max_blocks_per_seq = ceil_div(self.max_cache_len, kv_block_size)
        self.num_cache_blocks = num_cache_blocks or self.max_running_seqs * self.max_blocks_per_seq
        self._pending_seq_id = 0
        self._free_blocks = list(range(self.num_cache_blocks))
        self._seq_to_blocks: dict[int, list[int]] = {}
        self._free_recurrent_slots = list(range(self.max_running_seqs))
        self._seq_to_recurrent_slot: dict[int, int] = {}
        self._prefill_cursor_by_seq: dict[int, int] = {}
        self._last_active_ids: list[int] = []

    def append_prompts(self, prompts: list[list[int]], prompt_features: list[dict | None] | None = None) -> list[int]:
        """Append newly arrived prompts and return their row ids."""

        if not prompts:
            return []
        start = len(self.prompts)
        self.prompts.extend(prompts)
        self.prompt_features.extend(_normalize_prompt_features(prompt_features, len(prompts)))
        self.generated.extend([] for _ in prompts)
        self.logprobs.extend([] for _ in prompts)
        self.finished.extend(False for _ in prompts)
        self.finish_reason.extend("" for _ in prompts)
        return list(range(start, start + len(prompts)))

    @property
    def max_cache_len(self) -> int:
        """Maximum prompt-plus-response tokens allowed per sequence."""

        return self._max_cache_len

    @property
    def batch_size(self) -> int:
        """Maximum number of concurrently active sequences."""

        return self.max_running_seqs

    @property
    def has_pending_prompts(self) -> bool:
        """Whether there are prompts that have not fully entered decode."""

        return self._pending_seq_id < len(self.prompts)

    def build_prefill_payload(self) -> dict | None:
        """Admit as many pending prompts as the block and token budgets allow.

        The returned tensors are already packed for a varlen prefill call and
        include enough block metadata for the model to write each prompt token
        into its paged KV cache slot.
        """
        if not self._free_blocks or self._pending_seq_id >= len(self.prompts):
            return None
        # Packed prefill layout: `input_ids` and `position_ids` are flat 1-D
        # tensors of total tokens; `cu_seqlens` is the prefix sum boundary
        # of every admitted prompt (length B+1), and `sample_indices` points
        # at the last token of each prompt so the model only computes logits
        # for the next-token positions.
        input_ids: list[int] = []
        position_ids: list[int] = []
        mrope_position_parts: list[torch.Tensor] = []
        has_mrope_positions = False
        feature_mask: list[bool] = []
        audio_feature_mask: list[bool] = []
        image_features: list[dict] = []
        image_sequence_modes: list[bool] = []
        audio_sequence_modes: list[bool] = []
        cu_seqlens = [0]
        sample_indices: list[int] = []
        block_table: list[list[int]] = []
        cache_block_ids: list[int] = []
        cache_block_offsets: list[int] = []
        recurrent_slots: list[int] = []
        active_ids: list[int] = []

        while self._pending_seq_id < len(self.prompts):
            seq_id = self._pending_seq_id
            if seq_id not in self._seq_to_blocks and len(self._seq_to_blocks) >= self.max_running_seqs:
                break
            prompt = self.prompts[seq_id]
            if len(prompt) + self.max_new_tokens > self.max_cache_len:
                raise ValueError("request exceeds configured max_cache_len")
            cursor = self._prefill_cursor_by_seq.get(seq_id, 0)
            remaining_budget = self.max_prefill_tokens - len(input_ids)
            if remaining_budget <= 0:
                break
            chunk_len = min(len(prompt) - cursor, remaining_budget)
            if chunk_len <= 0:
                break
            blocks = self._seq_to_blocks.get(seq_id)
            if blocks is None:
                blocks = []
                self._seq_to_blocks[seq_id] = blocks
                if not self._free_recurrent_slots:
                    raise RuntimeError("recurrent state slots exhausted during prefill")
                self._seq_to_recurrent_slot[seq_id] = self._free_recurrent_slots.pop(0)
            required_blocks = ceil_div(cursor + chunk_len, self.kv_block_size)
            while len(blocks) < required_blocks:
                if not self._free_blocks:
                    if not input_ids:
                        raise RuntimeError("paged KV cache exhausted during prefill")
                    return (
                        None
                        if not input_ids
                        else self._prefill_payload(
                            input_ids,
                            position_ids,
                            mrope_position_parts if has_mrope_positions else None,
                            feature_mask,
                            audio_feature_mask,
                            image_features,
                            image_sequence_modes,
                            audio_sequence_modes,
                            cu_seqlens,
                            sample_indices,
                            block_table,
                            cache_block_ids,
                            cache_block_offsets,
                            recurrent_slots,
                            active_ids,
                        )
                    )
                blocks.append(self._free_blocks.pop(0))
            chunk = prompt[cursor : cursor + chunk_len]
            input_ids.extend(chunk)
            local_mask, local_features = _slice_prompt_image_features(
                self.prompt_features[seq_id],
                prompt,
                cursor,
                chunk_len,
            )
            feature_mask.extend(local_mask)
            audio_feature_mask.extend(
                _prompt_modality_mask(self.prompt_features[seq_id], prompt, "audio")[cursor : cursor + chunk_len]
            )
            image_sequence_modes.append(_prompt_has_image(self.prompt_features[seq_id], prompt))
            audio_sequence_modes.append(_prompt_has_audio(self.prompt_features[seq_id], prompt))
            if local_features is not None:
                image_features.append(local_features)
            local_mrope_positions = _slice_prompt_mrope_positions(
                self.prompt_features[seq_id],
                cursor,
                chunk_len,
            )
            if local_mrope_positions is not None:
                has_mrope_positions = True
                mrope_position_parts.append(local_mrope_positions)
            else:
                mrope_position_parts.append(
                    torch.arange(cursor, cursor + chunk_len, dtype=torch.long).view(1, -1).expand(3, -1)
                )
            position_ids.extend(range(cursor, cursor + chunk_len))
            # Per-token mapping from this prompt's token index to (block, offset)
            # inside the paged KV cache.
            for token_idx in range(cursor, cursor + chunk_len):
                cache_block_ids.append(blocks[token_idx // self.kv_block_size])
                cache_block_offsets.append(token_idx % self.kv_block_size)
            cu_seqlens.append(len(input_ids))
            # Pad the per-sequence block table to a uniform width so the model
            # can treat the entire batch as one rectangular tensor.
            block_table.append(_pad_blocks(blocks, self.max_blocks_per_seq))
            recurrent_slots.append(self._seq_to_recurrent_slot[seq_id])
            cursor += chunk_len
            if cursor >= len(prompt):
                sample_indices.append(len(input_ids) - 1)
                active_ids.append(seq_id)
                self._prefill_cursor_by_seq.pop(seq_id, None)
                self._pending_seq_id += 1
            else:
                self._prefill_cursor_by_seq[seq_id] = cursor
                break

        if not input_ids:
            return None

        return self._prefill_payload(
            input_ids,
            position_ids,
            mrope_position_parts if has_mrope_positions else None,
            feature_mask,
            audio_feature_mask,
            image_features,
            image_sequence_modes,
            audio_sequence_modes,
            cu_seqlens,
            sample_indices,
            block_table,
            cache_block_ids,
            cache_block_offsets,
            recurrent_slots,
            active_ids,
        )

    def _prefill_payload(
        self,
        input_ids: list[int],
        position_ids: list[int],
        mrope_position_parts: list[torch.Tensor] | None,
        feature_mask: list[bool],
        audio_feature_mask: list[bool],
        image_features: list[dict],
        image_sequence_modes: list[bool],
        audio_sequence_modes: list[bool],
        cu_seqlens: list[int],
        sample_indices: list[int],
        block_table: list[list[int]],
        cache_block_ids: list[int],
        cache_block_offsets: list[int],
        recurrent_slots: list[int],
        active_ids: list[int],
    ) -> dict:
        self._last_active_ids = active_ids
        payload = {
            "mode": "prefill",
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen": max((cu_seqlens[idx + 1] - cu_seqlens[idx] for idx in range(len(cu_seqlens) - 1)), default=0),
            "block_table": torch.tensor(block_table, dtype=torch.int32),
            "cache_block_ids": torch.tensor(cache_block_ids, dtype=torch.long),
            "cache_block_offsets": torch.tensor(cache_block_offsets, dtype=torch.long),
            "recurrent_slots": torch.tensor(recurrent_slots, dtype=torch.long),
        }
        if (
            any(feature_mask)
            or image_features
            or any(image_sequence_modes)
            or any(audio_sequence_modes)
            or mrope_position_parts is not None
        ):
            payload["features"] = _prefill_multimodal_features(
                feature_mask,
                image_features,
                mrope_position_parts,
                image_sequence_modes,
                audio_sequence_modes,
                audio_feature_mask,
            )
        return payload

    def ensure_decode_blocks(self, seq_ids: list[int], next_positions: list[int]) -> None:
        """Allocate one decode KV block for rows whose next token starts a block."""

        for seq_id, next_position in zip(seq_ids, next_positions, strict=True):
            if next_position < 0 or next_position % self.kv_block_size != 0:
                continue
            blocks = self._seq_to_blocks.get(int(seq_id))
            if blocks is None:
                continue
            required_blocks = next_position // self.kv_block_size + 1
            while len(blocks) < required_blocks:
                if not self._free_blocks:
                    raise RuntimeError("paged KV cache exhausted during decode")
                blocks.append(self._free_blocks.pop(0))

    def decode_position_deltas(self, seq_ids: list[int]) -> list[int]:
        """Return per-sequence MRoPE decode position deltas."""

        return [
            _prompt_mrope_position_delta(self.prompt_features[int(seq_id)], len(self.prompts[int(seq_id)]))
            for seq_id in seq_ids
        ]

    def to_rollout(self) -> RolloutOutput:
        """Materialize Python rollout state into padded tensors for the API."""
        input_ids, attention_mask, response_mask, logprobs = pad_rollout_rows(
            self.prompts, self.generated, self.logprobs
        )
        reasons = [reason or "unknown" for reason in self.finish_reason]
        return RolloutOutput(
            prompt_ids=self.prompts,
            response_ids=self.generated,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            logprobs=logprobs,
            finish_reason=reasons,
            metrics=self.metrics,
        )


def _normalize_prompt_features(features: list[dict | None] | None, count: int) -> list[dict | None]:
    if features is None:
        return [None for _ in range(count)]
    if len(features) != count:
        raise ValueError(f"prompt_features length mismatch: got {len(features)} for {count} prompts")
    return list(features)


def _slice_prompt_image_features(
    features: dict | None,
    prompt: list[int],
    cursor: int,
    chunk_len: int,
) -> tuple[list[bool], dict | None]:
    if features is None:
        return [False] * chunk_len, None
    image_embeds = _feature_tensor(features, "image_embeds")
    if image_embeds is None:
        image_embeds = _feature_tensor(features, "image_features")
    if image_embeds is None:
        image_embeds = _feature_tensor(features, "projected_image_embeds")
    if image_embeds is None:
        if not any(
            key in features
            for key in (
                "pixel_values",
                "input_image_embeds",
                "image_sizes",
                "image_attention_mask",
                "image_grid_thw",
                "target_sizes",
                "pixel_values_videos",
                "input_features",
                "input_audio_embeds",
                "audio_embeds",
                "audio_embed_sizes",
                "audio_attention_mask",
                "multimodal_feature_rows",
            )
        ):
            return [False] * chunk_len, None
    full_mask = _prompt_image_mask(features, prompt)
    local_mask = full_mask[cursor : cursor + chunk_len]
    local_count = sum(local_mask)
    if local_count == 0:
        return local_mask, None
    start = sum(full_mask[:cursor])
    end = start + local_count
    if image_embeds is None:
        payload_features = dict(features)
        modality_token_ids = dict(features.get("modality_token_ids") or {})
        if features.get("image_token_id") is not None:
            modality_token_ids.setdefault("image", int(features["image_token_id"]))
        modality_offsets = {}
        modality_counts = {}
        chunk = prompt[cursor : cursor + chunk_len]
        for modality, token_id in modality_token_ids.items():
            token_id = int(token_id)
            modality_offsets[modality] = sum(int(token) == token_id for token in prompt[:cursor])
            modality_counts[modality] = sum(int(token) == token_id for token in chunk)
        payload_features.update(
            {
                "multimodal_token_offset": start,
                "multimodal_token_count": local_count,
                "modality_token_offsets": modality_offsets,
                "modality_token_counts": modality_counts,
                "image_token_offset": start,
                "image_token_count": local_count,
            }
        )
        if modality_token_ids:
            payload_features["image_token_offset"] = modality_offsets.get("image", 0)
            payload_features["image_token_count"] = modality_counts.get("image", 0)
            payload_features["audio_token_offset"] = modality_offsets.get("audio", 0)
            payload_features["audio_token_count"] = modality_counts.get("audio", 0)
        for key in (
            "pixel_values",
            "input_image_embeds",
            "image_sizes",
            "image_attention_mask",
            "image_grid_thw",
            "target_sizes",
            "num_patches_per_image",
            "downsample_mode",
            "processor_expanded_image_tokens",
            "image_token_id",
            "input_audio_embeds",
            "audio_embed_sizes",
            "audio_attention_mask",
            "audio_token_id",
            "input_mode",
        ):
            if features.get(key) is not None:
                payload_features[key] = features[key]
        return local_mask, payload_features
    if image_embeds.ndim != 2:
        raise ValueError("image_embeds must have shape (num_image_tokens, hidden_size)")
    if int(image_embeds.shape[0]) < end:
        raise ValueError("image_embeds has fewer rows than image placeholder tokens")
    return local_mask, {"image_embeds": image_embeds[start:end]}


def _slice_prompt_mrope_positions(features: dict | None, cursor: int, chunk_len: int) -> torch.Tensor | None:
    if features is None or features.get("mrope_position_ids") is None:
        return None
    positions = _feature_tensor(features, "mrope_position_ids")
    if positions is None:
        return None
    if positions.ndim == 3 and int(positions.shape[0]) == 3 and int(positions.shape[1]) == 1:
        positions = positions[:, 0, :]
    elif positions.ndim == 3 and int(positions.shape[0]) == 1 and int(positions.shape[1]) == 3:
        positions = positions[0]
    if positions.ndim != 2 or int(positions.shape[0]) != 3:
        raise ValueError("mrope_position_ids must have shape (3, prompt_len)")
    end = cursor + chunk_len
    if int(positions.shape[-1]) < end:
        raise ValueError("mrope_position_ids length must cover the full prompt")
    return positions[:, cursor:end].to(dtype=torch.long).cpu()


def _prompt_mrope_position_delta(features: dict | None, prompt_len: int) -> int:
    positions = _slice_prompt_mrope_positions(features, 0, prompt_len)
    if positions is None or positions.numel() == 0:
        return 0
    return int(positions.max().item()) + 1 - int(prompt_len)


def _prefill_multimodal_features(
    feature_mask: list[bool],
    image_features: list[dict],
    mrope_position_parts: list[torch.Tensor] | None = None,
    image_sequence_modes: list[bool] | None = None,
    audio_sequence_modes: list[bool] | None = None,
    audio_feature_mask: list[bool] | None = None,
) -> dict:
    features = {}
    if image_sequence_modes is not None and any(image_sequence_modes):
        features["image_sequence_mask"] = torch.tensor(image_sequence_modes, dtype=torch.bool)
    if audio_sequence_modes is not None and any(audio_sequence_modes):
        features["audio_sequence_mask"] = torch.tensor(audio_sequence_modes, dtype=torch.bool)
    if mrope_position_parts is not None:
        features["mrope_position_ids"] = torch.cat(mrope_position_parts, dim=1).to(dtype=torch.long)
    if not image_features:
        if any(feature_mask):
            features.update(
                {"image_token_mask": torch.tensor(feature_mask, dtype=torch.bool), "image_embeds": torch.empty(0, 0)}
            )
        return features
    features.update(
        {
            "image_token_mask": torch.tensor(feature_mask, dtype=torch.bool),
            "image_feature_rows": image_features,
            "audio_feature_rows": image_features,
        }
    )
    if audio_feature_mask is not None:
        audio_mask = torch.tensor(audio_feature_mask, dtype=torch.bool)
        features["audio_token_mask"] = audio_mask
        features["image_token_mask"] &= ~audio_mask
    return features


def _feature_tensor(features: dict, key: str) -> torch.Tensor | None:
    value = features.get(key)
    if value is None:
        return None
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _prompt_image_mask(features: dict, prompt: list[int]) -> list[bool]:
    mask = features.get("image_token_mask")
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        mask_list = [bool(item) for item in mask.reshape(-1).tolist()]
        if len(mask_list) != len(prompt):
            raise ValueError("image_token_mask length must match prompt length")
        return mask_list
    token_ids = dict(features.get("modality_token_ids") or {})
    if features.get("image_token_id") is not None:
        token_ids.setdefault("image", int(features["image_token_id"]))
    if not token_ids:
        raise ValueError("multimodal features require a token mask or modality token ids")
    values = {int(value) for value in token_ids.values()}
    return [int(token) in values for token in prompt]


def _prompt_has_image(features: dict | None, prompt: list[int]) -> bool:
    if features is None:
        return False
    mask = features.get("image_token_mask")
    if mask is not None:
        return bool(torch.as_tensor(mask, dtype=torch.bool).any())
    image_token_id = features.get("image_token_id")
    if image_token_id is None:
        image_token_id = (features.get("modality_token_ids") or {}).get("image")
    return image_token_id is not None and any(int(token) == int(image_token_id) for token in prompt)


def _prompt_has_audio(features: dict | None, prompt: list[int]) -> bool:
    if features is None:
        return False
    mask = features.get("audio_token_mask")
    if mask is not None:
        return bool(torch.as_tensor(mask, dtype=torch.bool).any())
    audio_token_id = features.get("audio_token_id")
    if audio_token_id is None:
        audio_token_id = (features.get("modality_token_ids") or {}).get("audio")
    return audio_token_id is not None and any(int(token) == int(audio_token_id) for token in prompt)


def _prompt_modality_mask(features: dict | None, prompt: list[int], modality: str) -> list[bool]:
    if features is None:
        return [False] * len(prompt)
    mask = features.get(f"{modality}_token_mask")
    if mask is not None:
        values = [bool(item) for item in torch.as_tensor(mask).reshape(-1).tolist()]
        if len(values) != len(prompt):
            raise ValueError(f"{modality}_token_mask length must match prompt length")
        return values
    token_id = features.get(f"{modality}_token_id")
    if token_id is None:
        token_id = (features.get("modality_token_ids") or {}).get(modality)
    if token_id is None:
        return [False] * len(prompt)
    return [int(token) == int(token_id) for token in prompt]


def payload_to_infer_meta(payload: dict, device: torch.device) -> InferMeta:
    """Move a scheduler payload to device and expose it as model metadata."""

    if payload["mode"] == "prefill":
        # Prefill consumes a packed-varlen layout: `cu_seqlens` and `max_seqlen`
        # drive the attention kernel, while `cache_block_ids/offsets` tell the
        # KV writer where each prompt token's KV should be stored.
        return InferMeta(
            mode="prefill",
            sample_indices=payload["sample_indices"].to(device, non_blocking=True),
            cu_seqlens=payload["cu_seqlens"].to(device, non_blocking=True),
            max_seqlen=int(payload["max_seqlen"]),
            block_table=payload["block_table"].to(device, non_blocking=True),
            cache_block_ids=payload["cache_block_ids"].to(device, non_blocking=True),
            cache_block_offsets=payload["cache_block_offsets"].to(device, non_blocking=True),
            recurrent_slots=payload["recurrent_slots"].to(device, non_blocking=True),
        )
    # Decode runs one token per active sequence and reads previously written
    # KV through the same `block_table`, with `cache_seqlens` giving how many
    # tokens have already been written into each block table row.
    return InferMeta(
        mode="decode",
        sample_indices=payload["sample_indices"].to(device, non_blocking=True),
        cache_seqlens=payload["cache_seqlens"].to(device, non_blocking=True),
        block_table=payload["block_table"].to(device, non_blocking=True),
        recurrent_slots=(
            payload["recurrent_slots"].to(device, non_blocking=True)
            if payload.get("recurrent_slots") is not None
            else None
        ),
    )


def load_tokenizer(model_path: str | None):
    """Load tokenizer when a checkpoint path is available."""

    if model_path is None:
        return None
    from areno.engine.data.tokenizer import load_tokenizer as _load_tokenizer

    return _load_tokenizer(model_path)


def _pad_blocks(blocks: list[int], width: int) -> list[int]:
    if not blocks:
        raise ValueError("block table row cannot be empty")
    return blocks + [blocks[-1]] * (width - len(blocks))
