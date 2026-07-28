"""Lightweight dataclasses that flow through the rollout/training pipeline.

`PromptItem` is the unit produced by `Trainer.load_prompt_batches` after
tokenising a dataset row. `PromptBatch` groups a fixed-size set of items
together and carries diagnostic counters so the trainer can surface how many
records were skipped for exceeding the prompt-length budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

DATASET_MIX_METADATA_KEY = "__areno_meta__"
DatasetExhaustionPolicy = Literal["stop", "cycle", "renormalize"]
DATASET_MIX_SAMPLER_VERSION = 1
DATASET_MIX_WEIGHT_UNIT = "sample"


@dataclass(slots=True)
class PromptItem:
    """A dataset record after prompt tokenization and length filtering.

    `prompt` keeps the raw text used for downstream decoding/rewards,
    `input_tokens` holds the tokenized prefix that will be prepended to every
    rollout response, and `record` preserves the original row so reward
    functions can read task-specific fields (gold answers, test cases, ...).
    """

    prompt: str
    solutions: list[str] | None
    input_tokens: list[int]
    record: dict[str, Any]


@dataclass(slots=True)
class PromptBatch:
    """A batch of prompts plus counters for skipped over-length examples.

    `scanned` is how many raw dataset rows were inspected to build this batch
    (including skips), `skipped_long` is how many were dropped this round, and
    `total_skipped_long` accumulates the drop count across the epoch so the
    metric logger can report it as a cumulative counter.
    """

    items: list[PromptItem]
    scanned: int
    skipped_long: int
    total_skipped_long: int

    @property
    def prompts(self) -> list[str]:
        """Return raw prompt strings in batch order for rollout."""

        return [item.prompt for item in self.items]


@dataclass(frozen=True, slots=True)
class DatasetMixSource:
    """One named, weighted map-style dataset used by ``WeightedMixedDataset``."""

    name: str
    dataset: Sequence
    weight: float


@dataclass(frozen=True, slots=True)
class _DatasetMixEntry:
    source_index: int
    row_index: int
    cycle: int


class WeightedMixedDataset:
    """Deterministically interleave weighted map-style datasets.

    ``stop`` ends when a selected source is exhausted. ``cycle`` restarts
    exhausted sources until ``samples_per_epoch`` is reached, or until every
    source has exhausted once when no budget is supplied. ``renormalize``
    removes exhausted sources and continues with the remaining weights,
    emitting every source row exactly once.
    """

    def __init__(
        self,
        sources: Sequence[DatasetMixSource],
        *,
        seed: int,
        exhaustion: DatasetExhaustionPolicy,
        shuffle_within_sources: bool = True,
        samples_per_epoch: int | None = None,
    ) -> None:
        self.sources = tuple(sources)
        self.seed = seed
        self.exhaustion = exhaustion
        self.shuffle_within_sources = shuffle_within_sources
        self.samples_per_epoch = samples_per_epoch
        self.epoch = 0
        self._normalized_weights: tuple[float, ...] = ()
        self._validate()
        self._entries: list[_DatasetMixEntry] = []
        self._termination_reason = ""
        self._summary_cache: dict[str, Any] = {}
        self.set_epoch(0)

    def _validate(self) -> None:
        if len(self.sources) < 2:
            raise ValueError("dataset mix requires at least two sources")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("dataset mix seed must be an integer in [0, 2^63)")
        if self.exhaustion not in {"stop", "cycle", "renormalize"}:
            raise ValueError("dataset mix exhaustion must be one of: stop, cycle, renormalize")
        if self.samples_per_epoch is not None and (
            isinstance(self.samples_per_epoch, bool)
            or not isinstance(self.samples_per_epoch, int)
            or self.samples_per_epoch <= 0
        ):
            raise ValueError("dataset mix samples_per_epoch must be a positive integer")
        if self.samples_per_epoch is not None and self.exhaustion != "cycle":
            raise ValueError("dataset mix samples_per_epoch is only supported with exhaustion='cycle'")
        if not isinstance(self.shuffle_within_sources, bool):
            raise ValueError("dataset mix shuffle_within_sources must be a boolean")

        names: set[str] = set()
        numeric_weights: list[float] = []
        for source in self.sources:
            if not isinstance(source.name, str) or not source.name.strip():
                raise ValueError("dataset mix source name must be a non-empty string")
            if source.name != source.name.strip():
                raise ValueError("dataset mix source name must not have surrounding whitespace")
            if not source.name.isprintable():
                raise ValueError(f"dataset mix source name contains non-printable characters: {source.name!r}")
            if source.name in names:
                raise ValueError(f"duplicate dataset mix source name: {source.name}")
            names.add(source.name)
            try:
                numeric_weight = float(source.weight)
            except (OverflowError, TypeError, ValueError):
                numeric_weight = math.nan
            if isinstance(source.weight, bool) or not math.isfinite(numeric_weight) or numeric_weight <= 0:
                raise ValueError(f"dataset mix source '{source.name}' weight must be finite and positive")
            numeric_weights.append(numeric_weight)
            if not hasattr(source.dataset, "__len__") or not hasattr(source.dataset, "__getitem__"):
                raise ValueError(f"dataset mix source '{source.name}' must support len() and indexed access")
            if len(source.dataset) == 0:
                raise ValueError(f"dataset mix source '{source.name}' is empty")
            first = source.dataset[0]
            if not isinstance(first, Mapping):
                raise ValueError(f"dataset mix source '{source.name}' rows must be mappings")
            if DATASET_MIX_METADATA_KEY in first:
                raise ValueError(
                    f"dataset mix source '{source.name}' contains reserved field '{DATASET_MIX_METADATA_KEY}'"
                )
        max_weight = max(numeric_weights)
        scaled_weights = [weight / max_weight for weight in numeric_weights]
        if any(weight == 0.0 for weight in scaled_weights):
            raise ValueError("dataset mix weights have an unsupported numeric range")
        scaled_total = sum(scaled_weights)
        normalized_weights = tuple(weight / scaled_total for weight in scaled_weights)
        if any(weight == 0.0 for weight in normalized_weights):
            raise ValueError("dataset mix weights have an unsupported numeric range")
        self._normalized_weights = normalized_weights

    def set_epoch(self, epoch: int) -> None:
        """Build the deterministic schedule for one epoch."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("dataset mix epoch must be a non-negative integer")
        if epoch == self.epoch and self._entries:
            return
        self.epoch = epoch
        self._entries, self._termination_reason = self._build_schedule()
        self._summary_cache = self._build_summary()

    def _build_schedule(self) -> tuple[list[_DatasetMixEntry], str]:
        rng = random.Random(_stable_seed(self.seed, self.epoch, "source-selection"))
        orders = [self._source_order(index, cycle=0) for index in range(len(self.sources))]
        positions = [0] * len(self.sources)
        cycles = [0] * len(self.sources)
        exhausted_once: set[int] = set()
        active = list(range(len(self.sources)))
        entries: list[_DatasetMixEntry] = []

        while active:
            if self.samples_per_epoch is not None and len(entries) >= self.samples_per_epoch:
                return entries, "samples_per_epoch"
            source_index = _weighted_choice(rng, active, [self._normalized_weights[index] for index in active])

            row_index = orders[source_index][positions[source_index]]
            positions[source_index] += 1
            entries.append(_DatasetMixEntry(source_index=source_index, row_index=row_index, cycle=cycles[source_index]))
            if positions[source_index] < len(orders[source_index]):
                continue

            exhausted_once.add(source_index)
            if self.exhaustion == "stop":
                return entries, f"source_exhausted:{self.sources[source_index].name}"
            if self.exhaustion == "renormalize":
                active.remove(source_index)
                continue
            if self.samples_per_epoch is None and len(exhausted_once) == len(self.sources):
                return entries, "all_sources_exhausted_once"
            cycles[source_index] += 1
            orders[source_index] = self._source_order(source_index, cycle=cycles[source_index])
            positions[source_index] = 0

        return entries, "all_sources_exhausted"

    def _source_order(self, source_index: int, *, cycle: int) -> Sequence[int]:
        if not self.shuffle_within_sources:
            return range(len(self.sources[source_index].dataset))
        order = list(range(len(self.sources[source_index].dataset)))
        source = self.sources[source_index]
        random.Random(_stable_seed(self.seed, self.epoch, source.name, cycle)).shuffle(order)
        return order

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._entries[index]
        source = self.sources[entry.source_index]
        record = dict(source.dataset[entry.row_index])
        if DATASET_MIX_METADATA_KEY in record:
            raise ValueError(
                f"dataset mix source '{source.name}' row {entry.row_index} contains reserved field "
                f"'{DATASET_MIX_METADATA_KEY}'"
            )
        record[DATASET_MIX_METADATA_KEY] = {
            "source": source.name,
            "source_index": entry.row_index,
            "cycle": entry.cycle,
        }
        return record

    def summary(self) -> dict[str, Any]:
        """Return sample-free metadata suitable for logs and JSON artifacts."""

        return {
            **self._summary_cache,
            "sources": [dict(source) for source in self._summary_cache["sources"]],
        }

    def _build_summary(self) -> dict[str, Any]:
        selected = Counter(entry.source_index for entry in self._entries)
        duplicates = Counter(entry.source_index for entry in self._entries if entry.cycle > 0)
        total = len(self._entries)
        source_summaries = []
        warnings = []
        for index, source in enumerate(self.sources):
            count = selected[index]
            expected_rows = (
                self.samples_per_epoch * self._normalized_weights[index] if self.samples_per_epoch is not None else None
            )
            if expected_rows is not None and expected_rows < 1:
                warnings.append(
                    f"source '{source.name}' has expected_rows={expected_rows:.6g}; "
                    "it may receive zero samples in an epoch"
                )
            source_summaries.append(
                {
                    "name": source.name,
                    "weight_requested": self._normalized_weights[index],
                    "expected_rows": expected_rows,
                    "rows_available": len(source.dataset),
                    "rows_selected": count,
                    "duplicates": duplicates[index],
                    "observed_proportion": count / total if total else 0.0,
                }
            )
        schedule_hash = hashlib.sha256()
        for entry in self._entries:
            schedule_hash.update(f"{self.sources[entry.source_index].name}:{entry.row_index}:{entry.cycle}\n".encode())
        mix_spec = {
            "sampler_version": DATASET_MIX_SAMPLER_VERSION,
            "weight_unit": DATASET_MIX_WEIGHT_UNIT,
            "seed": self.seed,
            "policy": self.exhaustion,
            "shuffle_within_sources": self.shuffle_within_sources,
            "samples_per_epoch": self.samples_per_epoch,
            "sources": [
                {
                    "name": source.name,
                    "rows_available": len(source.dataset),
                    "weight": self._normalized_weights[index],
                }
                for index, source in enumerate(self.sources)
            ],
        }
        serialized_mix_spec = json.dumps(mix_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        mix_spec_hash = hashlib.sha256(serialized_mix_spec.encode()).hexdigest()
        return {
            "version": 1,
            "sampler_version": DATASET_MIX_SAMPLER_VERSION,
            "weight_unit": DATASET_MIX_WEIGHT_UNIT,
            "seed": self.seed,
            "epoch": self.epoch,
            "policy": self.exhaustion,
            "shuffle_within_sources": self.shuffle_within_sources,
            "samples_per_epoch": self.samples_per_epoch,
            "planned_rows": total,
            "termination_reason": self._termination_reason,
            "mix_spec_hash": f"sha256:{mix_spec_hash}",
            "schedule_hash": f"sha256:{schedule_hash.hexdigest()}",
            "warnings": warnings,
            "sources": source_summaries,
        }


def _weighted_choice(rng: random.Random, candidates: Sequence[int], weights: Sequence[float]) -> int:
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return candidate
    return candidates[-1]


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
