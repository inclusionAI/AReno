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

from areno.engine.runtime.metadata import InferMeta
from areno.engine.data import RolloutOutput
from areno.engine.runtime.common import ceil_div, pad_rollout_rows


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
    ):
        """Create rollout state and reserve bookkeeping for paged KV blocks."""

        self.prompts = prompts
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
        self._prefill_cursor_by_seq: dict[int, int] = {}
        self._last_active_ids: list[int] = []

    def append_prompts(self, prompts: list[list[int]]) -> list[int]:
        """Append newly arrived prompts and return their row ids."""

        if not prompts:
            return []
        start = len(self.prompts)
        self.prompts.extend(prompts)
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
        cu_seqlens = [0]
        sample_indices: list[int] = []
        block_table: list[list[int]] = []
        cache_block_ids: list[int] = []
        cache_block_offsets: list[int] = []
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
            required_blocks = ceil_div(cursor + chunk_len, self.kv_block_size)
            while len(blocks) < required_blocks:
                if not self._free_blocks:
                    if not input_ids:
                        raise RuntimeError("paged KV cache exhausted during prefill")
                    return None if not input_ids else self._prefill_payload(
                        input_ids,
                        position_ids,
                        cu_seqlens,
                        sample_indices,
                        block_table,
                        cache_block_ids,
                        cache_block_offsets,
                        active_ids,
                    )
                blocks.append(self._free_blocks.pop(0))
            chunk = prompt[cursor : cursor + chunk_len]
            input_ids.extend(chunk)
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
            cu_seqlens,
            sample_indices,
            block_table,
            cache_block_ids,
            cache_block_offsets,
            active_ids,
        )

    def _prefill_payload(
        self,
        input_ids: list[int],
        position_ids: list[int],
        cu_seqlens: list[int],
        sample_indices: list[int],
        block_table: list[list[int]],
        cache_block_ids: list[int],
        cache_block_offsets: list[int],
        active_ids: list[int],
    ) -> dict:
        self._last_active_ids = active_ids
        return {
            "mode": "prefill",
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen": max((cu_seqlens[idx + 1] - cu_seqlens[idx] for idx in range(len(cu_seqlens) - 1)), default=0),
            "block_table": torch.tensor(block_table, dtype=torch.int32),
            "cache_block_ids": torch.tensor(cache_block_ids, dtype=torch.long),
            "cache_block_offsets": torch.tensor(cache_block_offsets, dtype=torch.long),
        }

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

    def to_rollout(self) -> RolloutOutput:
        """Materialize Python rollout state into padded tensors for the API."""
        input_ids, attention_mask, response_mask, logprobs = pad_rollout_rows(self.prompts, self.generated, self.logprobs)
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
        )
    # Decode runs one token per active sequence and reads previously written
    # KV through the same `block_table`, with `cache_seqlens` giving how many
    # tokens have already been written into each block table row.
    return InferMeta(
        mode="decode",
        sample_indices=payload["sample_indices"].to(device, non_blocking=True),
        cache_seqlens=payload["cache_seqlens"].to(device, non_blocking=True),
        block_table=payload["block_table"].to(device, non_blocking=True),
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
