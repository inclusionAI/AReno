"""Scheduler-owned metadata for ordinary one-token decode.

Device tables retain their storage across iterations. Host lengths advance
from the scheduler's known admissions, rather than being read back from CUDA.
Completion flags use one bounded pinned transfer with an explicit event; the
CPU advances its logical metadata before waiting for that transfer.
"""

from __future__ import annotations

import torch

from areno.engine.data.rollout_state import InferenceBatchState


class DecodeMetadata:
    def __init__(self, state: InferenceBatchState, device: torch.device):
        self.device = device
        capacity = state.max_running_seqs
        self.tables = torch.empty((capacity, state.max_blocks_per_seq), device=device, dtype=torch.int32)
        self.slots = torch.empty(capacity, device=device, dtype=torch.long)
        self.completion = torch.empty(capacity, dtype=torch.bool, pin_memory=device.type == "cuda")
        self.completion_event = torch.cuda.Event() if device.type == "cuda" else None
        self.active_rows: torch.Tensor | None = None
        self.rows_cpu = torch.empty(0, dtype=torch.long)
        self.rows: list[int] = []
        self.lengths: list[int] = []
        self.until_block = 0
        self.until_full = 0
        self.dirty = False
        self.block_table = self.tables[:0]
        self.recurrent_slots = self.slots[:0]

    def sync_rows(self, state: InferenceBatchState, active_rows: torch.Tensor) -> None:
        # Admission and compaction replace this tensor; a steady decode keeps
        # its identity. Only membership changes need a row-ID transfer.
        if active_rows is self.active_rows:
            return
        previous = dict(zip(self.rows, self.lengths, strict=True))
        self.rows_cpu = active_rows.detach().to(device="cpu")
        self.rows = self.rows_cpu.tolist()
        if len(self.rows) > self.tables.shape[0]:
            raise RuntimeError("active decode rows exceed scheduler capacity")
        self.lengths = [previous.get(row, len(state.prompts[row])) for row in self.rows]
        self.active_rows = active_rows
        self.until_block = min((-length) % state.kv_block_size for length in self.lengths) if self.rows else 0
        self.until_full = min(
            (
                state.max_new_tokens - (length - len(state.prompts[row]) + 1)
                for row, length in zip(self.rows, self.lengths, strict=True)
            ),
            default=0,
        )
        self.block_table = self.tables[: len(self.rows)]
        self.recurrent_slots = self.slots[: len(self.rows)]
        self.dirty = True

    def prepare(self, state: InferenceBatchState) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rows and (self.dirty or self.until_block == 0):
            state.ensure_decode_blocks(self.rows, self.lengths)
            table = []
            for row in self.rows:
                blocks = state._seq_to_blocks[row]
                table.append(blocks + [blocks[-1]] * (state.max_blocks_per_seq - len(blocks)))
            # These infrequent H2D copies are blocking: the temporary host
            # tensors remain alive until the copies complete. No pinned
            # staging buffer can be overwritten while a copy is in flight.
            self.block_table.copy_(torch.tensor(table, dtype=torch.int32))
            if self.dirty:
                self.recurrent_slots.copy_(torch.tensor([state._seq_to_recurrent_slot[row] for row in self.rows]))
            self.until_block = min(state.kv_block_size - length % state.kv_block_size for length in self.lengths)
            self.dirty = False
        return self.block_table, self.recurrent_slots

    def finish_step(self, finished: torch.Tensor | None) -> torch.Tensor | None:
        host_finished = None
        if finished is not None:
            host_finished = self.completion[: len(self.rows)]
            host_finished.copy_(finished, non_blocking=self.completion_event is not None)
            if self.completion_event is not None:
                self.completion_event.record()
        # This work does not depend on the just-sampled tokens. Do it while
        # the model/sampler and completion transfer can still be executing.
        self.lengths = [length + 1 for length in self.lengths]
        self.until_block -= 1
        self.until_full -= 1
        if finished is not None and self.completion_event is not None:
            # The event also guards reuse of the pinned destination next step.
            self.completion_event.synchronize()
        return host_finished

    def full_length_mask(self, state: InferenceBatchState) -> torch.Tensor | None:
        if self.until_full > 0:
            return None
        return torch.tensor(
            [
                length - len(state.prompts[row]) + 1 >= state.max_new_tokens
                for row, length in zip(self.rows, self.lengths, strict=True)
            ],
            dtype=torch.bool,
        )
