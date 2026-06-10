"""CUDA graph capture/replay for the decode step.

Decode runs one token per active sequence, so the graph's only batch-size
degree of freedom is `bucket`. `DecodeGraph` owns the static input buffers
that the captured graph reads from; replay copies the current step's tensors
into those buffers (and pads the tail with the scratch block) so the captured
shape and pointer set never change.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from areno.engine.runtime.common import ceil_div
from areno.engine.runtime.metadata import InferMeta


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
            dist.barrier(group=group, device_ids=[device.index if device.index is not None else torch.cuda.current_device()])
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

    Decode always runs one token per active sequence. The graph owns static
    input buffers sized to a bucket. Replay copies the current token/cache
    metadata into those buffers and replays the captured model call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bucket: int,
        max_blocks_per_seq: int,
        scratch_block: int,
        device: torch.device,
    ):
        """Allocate static input buffers and the `InferMeta` baked into capture."""

        self.model = model
        self.bucket = bucket
        self.scratch_block = scratch_block
        self.device = device
        # Stable input pointers. The captured CUDA graph remembers these as
        # source/destination addresses, so replay must write through the same
        # tensors rather than swapping in fresh allocations.
        self.input_ids = torch.zeros((1, bucket), device=device, dtype=torch.long)
        self.position_ids = torch.zeros((1, bucket), device=device, dtype=torch.long)
        self.cache_seqlens = torch.zeros(bucket, device=device, dtype=torch.int32)
        # Padding columns point to `scratch_block`, a dedicated block that the
        # scheduler never assigns to a real sequence. This keeps the attention
        # kernel safe when actual batch size < bucket.
        self.block_table = torch.full((bucket, max_blocks_per_seq), scratch_block, device=device, dtype=torch.int32)
        self.meta = InferMeta(
            mode="decode",
            sample_indices=torch.arange(bucket, device=device, dtype=torch.long),
            cache_seqlens=self.cache_seqlens,
            block_table=self.block_table,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.logits_shard: torch.Tensor | None = None

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
                logits_shard = self.model(
                    input_ids=self.input_ids,
                    position_ids=self.position_ids,
                    infer_meta=self.meta,
                ).logits_shard
                del logits_shard
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
            self.logits_shard = self.model(input_ids=self.input_ids, position_ids=self.position_ids, infer_meta=self.meta).logits_shard

    @torch.inference_mode()
    def replay_tensors(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Copy one dynamic decode step into static buffers and replay the graph."""
        actual = int(input_ids.numel())
        if actual > self.bucket:
            raise ValueError(f"decode payload has {actual} tokens, graph bucket is {self.bucket}")

        # Copy the live values into the captured-stable buffers. The graph
        # was recorded against these buffer addresses so `copy_` here is what
        # makes the replay reflect the current step.
        self.input_ids[0, :actual].copy_(input_ids)
        self.position_ids[0, :actual].copy_(position_ids)
        self.cache_seqlens[:actual].copy_(cache_seqlens)
        block_cols = int(block_table.shape[1])
        if block_cols > self.block_table.shape[1]:
            raise ValueError(f"decode block table has {block_cols} columns, graph buffer has {self.block_table.shape[1]}")
        self.block_table[:actual, :block_cols].copy_(block_table)
        if block_cols < self.block_table.shape[1]:
            # Pad the unused columns to scratch so attention reads stay valid.
            self.block_table[:actual, block_cols:].fill_(self.scratch_block)

        # Fill the unused tail rows with no-op values so the captured kernel
        # runs over the full bucket without touching live KV cache slots.
        if actual < self.bucket:
            self.input_ids[0, actual:self.bucket].fill_(0)
            self.position_ids[0, actual:self.bucket].fill_(0)
            self.block_table[actual:self.bucket].fill_(self.scratch_block)
            self.cache_seqlens[actual:self.bucket].fill_(0)

        self.graph.replay()
        assert self.logits_shard is not None
        return self.logits_shard
