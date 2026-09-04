"""CUDA graph capture/replay for the decode step.

Decode runs `tokens_per_seq` tokens per active sequence (one for plain
decode, `k + 1` for speculative verify), so the graph's batch-size degrees of
freedom are `bucket` and `tokens_per_seq`. `DecodeGraph` owns the static input
buffers that the captured graph reads from; replay copies the current step's
tensors into those buffers (and pads the tail with the scratch block) so the
captured shape and pointer set never change. With `draft_hidden_size` set the
graph captures the model's MTP draft forward instead of the trunk.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from areno.engine.runtime.common import ceil_div as ceil_div  # noqa: F401
from areno.engine.runtime.metadata import InferMeta
from areno.engine.runtime.routing_replay import captured_routing, routing_replay_context


def bucket_for(batch_size: int, buckets: list[int]) -> int:
    """Return the smallest configured bucket that covers `batch_size`."""

    for bucket in buckets:
        if batch_size <= bucket:
            return bucket
    return batch_size


def sync_before_graph_capture(device: torch.device, group) -> None:
    """Place all ranks at a clean synchronization point before graph capture."""
    # The sequence CUDA-sync → NCCL barrier → CUDA-sync guarantees: all
    # outstanding kernels on this device finished, every TP rank reached the
    # barrier, then any cross-stream queueing introduced by NCCL is drained
    # before capture begins. Without this, in-flight work could leak into the
    # captured graph and cause replay corruption.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_available() and dist.is_initialized():
        if device.type == "cuda":
            dist.barrier(
                group=group, device_ids=[device.index if device.index is not None else torch.cuda.current_device()]
            )
        else:
            dist.barrier(group=group)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def has_graph_capture_memory(device: torch.device, group, warmup_bytes: int) -> bool:
    """Return true only if every rank has enough free memory for capture."""
    if device.type != "cuda":
        return True
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Capture itself adds bookkeeping over the warmup peak, so demand a 20%
    # headroom margin before letting any rank start to capture.
    required = int(max(warmup_bytes, 1) * 1.2)
    ok = torch.tensor([1 if free_bytes > required else 0], device=device, dtype=torch.int32)
    # MIN reduce so the result is true only when EVERY rank is happy; one
    # tight rank causes all ranks to skip capture in lockstep.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    return bool(ok.item())


class DecodeGraph:
    """Reusable CUDA graph for one decode batch bucket.

    The graph owns static input buffers sized to `bucket` rows of
    `tokens_per_seq` tokens each. Replay copies the current token/cache
    metadata into those buffers and replays the captured model call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bucket: int,
        max_blocks_per_seq: int,
        scratch_block: int,
        scratch_recurrent_slot: int,
        device: torch.device,
        *,
        capture_routing: bool = False,
        tokens_per_seq: int = 1,
        draft_hidden_size: int | None = None,
        hidden_dtype: torch.dtype = torch.bfloat16,
    ):
        """Allocate static input buffers and the `InferMeta` baked into capture."""

        self.model = model
        self.bucket = bucket
        self.tokens_per_seq = tokens_per_seq
        self.scratch_block = scratch_block
        self.scratch_recurrent_slot = scratch_recurrent_slot
        self.device = device
        tokens = bucket * tokens_per_seq
        # Stable input pointers. The captured CUDA graph remembers these as
        # source/destination addresses, so replay must write through the same
        # tensors rather than swapping in fresh allocations.
        self.input_ids = torch.zeros((1, tokens), device=device, dtype=torch.long)
        self.position_ids = torch.zeros((1, tokens), device=device, dtype=torch.long)
        # Draft graphs also read the hidden states the MTP layer fuses with.
        self.hidden_states = (
            torch.zeros((1, tokens, draft_hidden_size), device=device, dtype=hidden_dtype)
            if draft_hidden_size is not None
            else None
        )
        self.cache_seqlens = torch.zeros(bucket, device=device, dtype=torch.int32)
        # Graph warmup/capture executes the model before any request is
        # admitted. Point every dummy row at the dedicated scratch recurrent
        # slot so it cannot seed live request state with synthetic tokens.
        self.recurrent_slots = torch.full((bucket,), scratch_recurrent_slot, device=device, dtype=torch.long)
        # Padding columns point to `scratch_block`, a dedicated block that the
        # scheduler never assigns to a real sequence. This keeps the attention
        # kernel safe when actual batch size < bucket.
        self.block_table = torch.full((bucket, max_blocks_per_seq), scratch_block, device=device, dtype=torch.int32)
        self.meta = InferMeta(
            mode="decode",
            sample_indices=torch.arange(tokens, device=device, dtype=torch.long),
            cache_seqlens=self.cache_seqlens,
            block_table=self.block_table,
            recurrent_slots=self.recurrent_slots,
            capture_routing=capture_routing,
            tokens_per_seq=tokens_per_seq,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.logits_shard: torch.Tensor | None = None
        self.output_hidden: torch.Tensor | None = None
        self.routing_capture: torch.Tensor | None = None

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the captured call on the static buffers; returns (logits_shard, hidden_states)."""
        if self.hidden_states is not None:
            return self.model.mtp_draft_forward(
                input_ids=self.input_ids,
                hidden_states=self.hidden_states,
                position_ids=self.position_ids,
                infer_meta=self.meta,
            )
        out = self.model(input_ids=self.input_ids, position_ids=self.position_ids, infer_meta=self.meta)
        return out.logits_shard, out.hidden_states

    @torch.inference_mode()
    def warmup(self, iterations: int = 3) -> int:
        """Run eager decode a few times and return the extra peak bytes observed."""
        before = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        # Warmup on a side stream so any one-time allocator behavior happens
        # before capture; the result is the additional bytes we need to keep
        # available when the graph is captured.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            for _ in range(iterations):
                with routing_replay_context(self.meta):
                    outputs = self._forward()
                del outputs
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        return max(0, torch.cuda.max_memory_allocated(self.device) - before)

    @torch.inference_mode()
    def capture(self) -> None:
        """Capture the model decode call using the graph-owned static buffers."""
        # The torch.cuda.graph context records every kernel launched inside it.
        # All inputs referenced here must already live on the graph's stream
        # and must remain alive at the same addresses for the lifetime of the
        # graph, which is exactly what `self.input_ids/...` provide.
        with torch.cuda.device(self.device), torch.cuda.graph(self.graph):
            with routing_replay_context(self.meta):
                self.logits_shard, self.output_hidden = self._forward()
            # Stack per-layer routes while capture is active. Replay then
            # updates this fixed contiguous tensor without launching a new
            # stack kernel or allocating an output tensor from Python.
            self.routing_capture = captured_routing(self.meta)

    @torch.inference_mode()
    def replay_tensors(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        recurrent_slots: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Copy one dynamic decode step into static buffers and replay the graph.

        Returns the logits shard; `output_hidden` holds the matching hidden
        states until the next replay.
        """
        actual = int(cache_seqlens.numel())
        if actual > self.bucket:
            raise ValueError(f"decode payload has {actual} rows, graph bucket is {self.bucket}")
        live_tokens = actual * self.tokens_per_seq
        if int(input_ids.numel()) != live_tokens:
            raise ValueError(f"decode payload has {input_ids.numel()} tokens, expected {live_tokens}")

        # Copy the live values into the captured-stable buffers. The graph
        # was recorded against these buffer addresses so `copy_` here is what
        # makes the replay reflect the current step.
        self.input_ids[0, :live_tokens].copy_(input_ids.view(-1))
        self.position_ids[0, :live_tokens].copy_(position_ids.view(-1))
        if self.hidden_states is not None:
            if hidden_states is None:
                raise ValueError("draft graph replay requires hidden_states")
            self.hidden_states[0, :live_tokens].copy_(hidden_states.reshape(live_tokens, -1))
        self.cache_seqlens[:actual].copy_(cache_seqlens)
        self.recurrent_slots[:actual].copy_(recurrent_slots)
        block_cols = int(block_table.shape[1])
        if block_cols > self.block_table.shape[1]:
            raise ValueError(
                f"decode block table has {block_cols} columns, graph buffer has {self.block_table.shape[1]}"
            )
        self.block_table[:actual, :block_cols].copy_(block_table)
        if block_cols < self.block_table.shape[1]:
            # Pad the unused columns to scratch so attention reads stay valid.
            self.block_table[:actual, block_cols:].fill_(self.scratch_block)

        # Fill the unused tail rows with no-op values so the captured kernel
        # runs over the full bucket without touching live KV cache slots.
        if actual < self.bucket:
            self.input_ids[0, live_tokens:].fill_(0)
            self.position_ids[0, live_tokens:].fill_(0)
            if self.hidden_states is not None:
                self.hidden_states[0, live_tokens:].zero_()
            self.block_table[actual : self.bucket].fill_(self.scratch_block)
            self.cache_seqlens[actual : self.bucket].fill_(0)
            self.recurrent_slots[actual : self.bucket].fill_(self.scratch_recurrent_slot)

        self.graph.replay()
        assert self.logits_shard is not None
        return self.logits_shard
