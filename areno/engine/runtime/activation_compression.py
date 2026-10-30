"""Opt-in compression of saved activations (GACT, Liu et al., ICML 2022).

While an actor training forward runs inside :meth:`ActivationCompressor.capture`,
``torch.autograd.graph.saved_tensors_hooks`` intercept every tensor autograd
saves for backward. Each eligible tensor is a *slot*, numbered in forward order,
and is either kept or quantized to 8, 4 or 2 bits; backward reconstructs the
original shape and dtype. With activation checkpointing the slots are the
checkpoint boundaries (decoder-layer inputs) and the tensors saved outside
checkpointed layers; without it, every saved activation of the model forward.
Weights and buffers (and views of them), integer tensors, 1-D tensors such as
per-row statistics, tensors with fewer than ``MIN_NUMEL`` elements,
non-contiguous tensors, everything saved inside :func:`saved_uncompressed`
(the attention kernels' Q, K, V, output and softmax statistics) and any tensor
whose storage one of those uncompressed saves keeps alive are never
compressed.

A quantized slot stores, per group of ``GROUP_SIZE`` contiguous elements, an
FP32 minimum and step plus stochastically rounded codes (unbiased, as in
GACT). Transformer hidden states carry a few channels that are orders of
magnitude larger than the rest; the ``1 / CHANNELS_PER_OUTLIER`` channels with
the largest magnitude are therefore kept exactly, in the original dtype, and
excluded from the group ranges.

:meth:`ActivationCompressor.calibrate` picks each slot's precision from its
measured gradient sensitivity: the squared relative L2 change of all parameter
gradients when only that slot is compressed. A greedy search minimizes stored
bytes for a target error; the plan is then measured on the calibration batch
and a held-out batch, and the target is rescaled from the measured error until
the plan is within budget on both (ultimately falling back to no compression).
:meth:`ActivationCompressor.check` re-measures the plan periodically during
training and recalibrates when it no longer meets the budget.
"""

from __future__ import annotations

import json
import logging
import math
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

GROUP_SIZE = 64
CHANNELS_PER_OUTLIER = 64
MIN_NUMEL = 1 << 16
UNCOMPRESSED = 16
SUPPORTED_BITS = (8, 4, 2)
# Aim plans at this fraction of the budget so held-out batches stay inside it.
PLAN_MARGIN = 0.8
PLAN_ROUNDS = 6
STATE_VERSION = 1
PLAN_FILE = "activation_compression.json"
_FLOATING = (torch.bfloat16, torch.float16, torch.float32)
_M32 = 0xFFFFFFFF
# Depth of saved_uncompressed() contexts; a module counter, not a ContextVar,
# because activation recompute replays forwards on autograd's device thread.
_UNCOMPRESSED_DEPTH = 0


@contextmanager
def saved_uncompressed() -> Iterator[None]:
    """Store tensors saved inside the context uncompressed (and not as slots)."""

    global _UNCOMPRESSED_DEPTH
    _UNCOMPRESSED_DEPTH += 1
    try:
        yield
    finally:
        _UNCOMPRESSED_DEPTH -= 1


@dataclass(slots=True)
class PackedActivation:
    """Reconstruction metadata and packed codes for one saved tensor."""

    codes: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    outlier_index: torch.Tensor
    outliers: torch.Tensor
    shape: torch.Size
    dtype: torch.dtype
    bits: int
    group_size: int = GROUP_SIZE

    @property
    def packed_bytes(self) -> int:
        return self.codes.numel()

    @property
    def scale_bytes(self) -> int:
        return self.minimum.numel() * 4 + self.scale.numel() * 4

    @property
    def outlier_bytes(self) -> int:
        return (
            self.outliers.numel() * self.outliers.element_size()
            + self.outlier_index.numel() * self.outlier_index.element_size()
        )


def outlier_count(channels: int) -> int:
    """Number of exactly kept channels for a last dimension of ``channels``."""

    return -(-channels // CHANNELS_PER_OUTLIER)


def _mul32(h: torch.Tensor, c: int) -> torch.Tensor:
    # (h * c) mod 2**32 for h, c < 2**32 without int64 overflow.
    return ((h & 0xFFFF) * c + ((((h >> 16) * c) & 0xFFFF) << 16)) & _M32


def _lowbias32(h: torch.Tensor) -> torch.Tensor:
    # C. Wellons' lowbias32 integer hash of h < 2**32.
    h = h ^ (h >> 16)
    h = _mul32(h, 0x7FEB352D)
    h = h ^ (h >> 15)
    h = _mul32(h, 0x846CA68B)
    return h ^ (h >> 16)


def uniform_noise(index: torch.Tensor, seed: int) -> torch.Tensor:
    """Counter-based uniform draws in [0, 1), identical to the CUDA kernel's.

    The index is hashed before the seed keys it; XOR-ing the seed into the raw
    index would only permute indices and reuse the same draws across seeds.
    """

    h = _lowbias32(_lowbias32(index & _M32) ^ seed)
    return (h >> 8).to(torch.float32) * (1.0 / (1 << 24))


def _use_kernel(x: torch.Tensor) -> bool:
    if x.device.type != "cuda":
        return False
    try:
        import areno.accel.kernels.activation_quant  # noqa: F401
    except ImportError:
        return False
    return True


def quantize(
    x: torch.Tensor,
    bits: int,
    *,
    stochastic: bool = True,
    seed: int = 0,
    group_size: int = GROUP_SIZE,
    use_kernel: bool | None = None,
) -> PackedActivation:
    """Quantize contiguous ``x`` (at least 2-D) per group of ``group_size`` elements."""

    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    if not x.is_contiguous() or x.dtype not in _FLOATING or x.dim() < 2 or x.numel() == 0:
        raise ValueError("quantize expects a non-empty contiguous floating tensor with at least two dimensions")
    seed = int(seed) & 0x7FFFFFFF
    channels = x.shape[-1]
    rows = x.reshape(-1, channels)
    low, high = torch.aminmax(rows, dim=0)
    peak = torch.maximum(high, low.neg()).float()
    outlier_index = torch.topk(peak, outlier_count(channels)).indices
    outliers = rows.index_select(1, outlier_index)
    keep = torch.zeros(channels, dtype=torch.uint8, device=x.device)
    keep[outlier_index] = 1
    if _use_kernel(x) if use_kernel is None else use_kernel:
        from areno.accel.kernels.activation_quant import quantize_groups

        codes, minimum, scale = quantize_groups(x, keep, bits, group_size, stochastic=stochastic, seed=seed)
        return PackedActivation(codes, minimum, scale, outlier_index, outliers, x.shape, x.dtype, bits, group_size)
    numel = x.numel()
    groups = -(-numel // group_size)
    index = torch.arange(groups * group_size, device=x.device, dtype=torch.int64).view(groups, group_size)
    valid = index < numel
    used = valid & (keep[index % channels] == 0)
    values = torch.zeros(groups * group_size, dtype=torch.float32, device=x.device)
    values[:numel] = x.reshape(-1)
    values = values.view(groups, group_size)
    inf = torch.tensor(float("inf"), device=x.device)
    minimum = torch.where(used, values, inf).amin(dim=1)
    maximum = torch.where(used, values, -inf).amax(dim=1)
    # Groups without quantized elements store a zero range.
    minimum = torch.where(minimum == inf, 0.0, minimum)
    maximum = torch.where(maximum == -inf, 0.0, maximum)
    levels = (1 << bits) - 1
    # A tensor divisor keeps IEEE division; CUDA divides by a Python scalar
    # through its reciprocal, which would differ from the kernel's step.
    scale = (maximum - minimum) / torch.full_like(maximum, levels)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (values - minimum[:, None]) / scale[:, None]
    q = q + (uniform_noise(index, seed) if stochastic else 0.5)
    codes = torch.where(used, torch.floor(q).clamp(0, levels), 0.0).to(torch.uint8)
    per_byte = 8 // bits
    codes = codes.view(-1, per_byte)
    packed = codes[:, 0].clone()
    for position in range(1, per_byte):
        packed |= codes[:, position] << (bits * position)
    return PackedActivation(packed, minimum, scale, outlier_index, outliers, x.shape, x.dtype, bits, group_size)


def dequantize(packed: PackedActivation, *, use_kernel: bool | None = None) -> torch.Tensor:
    """Reconstruct a contiguous tensor with the original shape and dtype."""

    out = torch.empty(packed.shape, dtype=packed.dtype, device=packed.codes.device)
    if _use_kernel(out) if use_kernel is None else use_kernel:
        from areno.accel.kernels.activation_quant import dequantize_groups

        dequantize_groups(packed.codes, packed.minimum, packed.scale, out, packed.bits, packed.group_size)
    else:
        bits = packed.bits
        shifts = torch.arange(8 // bits, device=out.device, dtype=torch.uint8) * bits
        codes = (packed.codes[:, None] >> shifts) & ((1 << bits) - 1)
        values = codes.view(-1, packed.group_size).float() * packed.scale[:, None] + packed.minimum[:, None]
        out.view(-1).copy_(values.view(-1)[: out.numel()])
    out.view(-1, packed.shape[-1]).index_copy_(1, packed.outlier_index, packed.outliers)
    return out


def stored_bytes(numel: int, element_size: int, channels: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    """Bytes one slot occupies at ``bits`` precision, all metadata included."""

    if bits == UNCOMPRESSED:
        return numel * element_size
    groups = -(-numel // group_size)
    outliers = outlier_count(channels)
    return groups * (group_size * bits // 8 + 8) + (numel // channels) * outliers * element_size + outliers * 8


@dataclass(slots=True)
class SavedActivationStats:
    """Byte accounting of the slots saved by one captured forward."""

    slots: int = 0
    compressed_slots: int = 0
    saved_bytes: int = 0
    packed_bytes: int = 0
    scale_bytes: int = 0
    outlier_bytes: int = 0
    kept_bytes: int = 0
    workspace_bytes: int = 0
    plan_misses: int = 0

    @property
    def stored_bytes(self) -> int:
        return self.packed_bytes + self.scale_bytes + self.outlier_bytes + self.kept_bytes

    def metrics(self) -> dict[str, float]:
        """Per-forward metrics; ``workspace`` is the largest single reconstruction."""

        saved = max(self.saved_bytes, 1)
        return {
            "actc_slots": float(self.slots),
            "actc_compressed_slots": float(self.compressed_slots),
            "actc_saved_bytes": float(self.saved_bytes),
            "actc_stored_bytes": float(self.stored_bytes),
            "actc_packed_bytes": float(self.packed_bytes),
            "actc_scale_bytes": float(self.scale_bytes),
            "actc_outlier_bytes": float(self.outlier_bytes),
            "actc_kept_bytes": float(self.kept_bytes),
            "actc_workspace_bytes": float(self.workspace_bytes),
            "actc_stored_fraction": self.stored_bytes / saved,
            "actc_plan_misses": float(self.plan_misses),
        }


def _plan_lookup(signatures: Sequence[tuple], plan: Sequence[int]) -> dict[tuple, int]:
    """Key a plan by (signature, occurrence of that signature) instead of position.

    Compiled regions may stop saving tensors they can recompute cheaply after a
    recompilation (for example RoPE tables); keys of other signatures then stay
    aligned with the slots they were calibrated on.
    """

    seen: dict[tuple, int] = {}
    lookup = {}
    for signature, bits in zip(signatures, plan, strict=True):
        occurrence = seen.get(signature, 0)
        seen[signature] = occurrence + 1
        lookup[(signature, occurrence)] = bits
    return lookup


@dataclass(slots=True)
class _Capture:
    plan: dict[tuple, int] | None
    excluded: set[int]
    seed: int
    stochastic: bool
    stats: SavedActivationStats = field(default_factory=SavedActivationStats)
    signatures: list[tuple] = field(default_factory=list)
    numels: list[int] = field(default_factory=list)
    element_sizes: list[int] = field(default_factory=list)
    occurrences: dict[tuple, int] = field(default_factory=dict)
    cache: dict[tuple, tuple[weakref.ref, Any]] = field(default_factory=dict)
    # Storages that saved-uncompressed tensors keep alive, and the slots that
    # share one (compressing those would add a copy without freeing anything).
    held: set[int] = field(default_factory=set)
    slot_storage: dict[int, int] = field(default_factory=dict)
    aliased: set[int] = field(default_factory=set)

    def pack(self, tensor: torch.Tensor) -> Any:
        storage = tensor.untyped_storage().data_ptr()
        if (
            _UNCOMPRESSED_DEPTH
            or tensor.dtype not in _FLOATING
            or tensor.dim() < 2
            or tensor.numel() < MIN_NUMEL
            or not tensor.is_contiguous()
        ):
            self.held.add(storage)
            if storage in self.slot_storage:
                self.aliased.add(self.slot_storage[storage])
            return tensor
        if storage in self.excluded or storage in self.held:
            return tensor
        key = (
            storage,
            tensor.storage_offset(),
            tuple(tensor.shape),
            tensor.dtype,
            tensor._version,
        )
        cached = self.cache.get(key)
        # A live first owner guarantees the address was not reused.
        if cached is not None and cached[0]() is not None:
            return cached[1]
        slot = len(self.signatures)
        self.slot_storage.setdefault(storage, slot)
        signature = (str(tensor.dtype), tensor.dim(), int(tensor.shape[-1]))
        self.signatures.append(signature)
        self.numels.append(tensor.numel())
        self.element_sizes.append(tensor.element_size())
        occurrence = self.occurrences.get(signature, 0)
        self.occurrences[signature] = occurrence + 1
        bits = UNCOMPRESSED
        if self.plan is not None:
            bits = self.plan.get((signature, occurrence))
            if bits is None:
                self.stats.plan_misses += 1
                bits = UNCOMPRESSED
        stats = self.stats
        stats.slots += 1
        stats.saved_bytes += tensor.numel() * tensor.element_size()
        if bits == UNCOMPRESSED:
            handle: Any = tensor
            stats.kept_bytes += tensor.numel() * tensor.element_size()
        else:
            with torch.no_grad():
                handle = quantize(
                    tensor.detach(), bits, stochastic=self.stochastic, seed=self.seed * 1_000_003 + slot * 7919
                )
            stats.compressed_slots += 1
            stats.packed_bytes += handle.packed_bytes
            stats.scale_bytes += handle.scale_bytes
            stats.outlier_bytes += handle.outlier_bytes
            stats.workspace_bytes = max(stats.workspace_bytes, tensor.numel() * tensor.element_size())
        self.cache[key] = (weakref.ref(tensor), handle)
        return handle


def _unpack(handle: Any) -> torch.Tensor:
    if isinstance(handle, PackedActivation):
        return dequantize(handle)
    return handle


def relative_error(grads: Sequence[torch.Tensor | None], reference: Sequence[torch.Tensor | None]) -> float:
    """``||g - ref|| / ||ref||`` over all parameters, accumulated in FP32."""

    difference = None
    norm = None
    for grad, ref in zip(grads, reference, strict=True):
        if ref is None and grad is None:
            continue
        ref32 = torch.zeros_like(grad, dtype=torch.float32) if ref is None else ref.float()
        grad32 = torch.zeros_like(ref32) if grad is None else grad.float()
        term = (grad32 - ref32).square().sum()
        size = ref32.square().sum()
        difference = term if difference is None else difference + term
        norm = size if norm is None else norm + size
    if norm is None:
        return 0.0
    return math.sqrt(float(difference)) / max(math.sqrt(float(norm)), 1.0e-30)


def plan_bits(
    sensitivity: Sequence[dict[int, float]],
    slots: Sequence[tuple[int, int, int]],
    target: float,
    bits: Sequence[int] = SUPPORTED_BITS,
) -> list[int]:
    """Greedy multiple-choice knapsack: fewest bytes with summed sensitivity <= target.

    ``sensitivity[slot][b]`` is the squared relative gradient error caused by
    compressing only ``slot`` at ``b`` bits (absent: not allowed), and
    ``slots[slot]`` is its ``(numel, element_size, channels)``. Each round
    lowers the precision of the slot whose next level adds the least error per
    byte saved.
    """

    levels = (UNCOMPRESSED, *sorted(bits, reverse=True))
    plan = [UNCOMPRESSED] * len(sensitivity)
    used = 0.0
    while True:
        best = None
        for slot, current in enumerate(plan):
            step = levels.index(current) + 1
            if step == len(levels) or levels[step] not in sensitivity[slot]:
                continue
            lower = levels[step]
            added = sensitivity[slot][lower] - sensitivity[slot].get(current, 0.0)
            if used + added > target:
                continue
            saved = stored_bytes(*slots[slot], current) - stored_bytes(*slots[slot], lower)
            if saved <= 0:
                continue
            score = (max(added, 0.0) / saved, -saved)
            if best is None or score < best[0]:
                best = (score, slot, lower, added)
        if best is None:
            return plan
        _, slot, lower, added = best
        plan[slot] = lower
        used += added


class ActivationCompressor:
    """Per-worker state: the calibrated plan and the last forward's accounting."""

    def __init__(
        self,
        model: nn.Module,
        *,
        error_budget: float = 0.01,
        bits: Sequence[int] = SUPPORTED_BITS,
        stochastic: bool = True,
        seed: int = 0,
    ):
        if error_budget <= 0:
            raise ValueError("error_budget must be positive")
        if not bits or any(value not in SUPPORTED_BITS for value in bits):
            raise ValueError(f"bits must be a non-empty subset of {SUPPORTED_BITS}")
        self.model = model
        self.error_budget = float(error_budget)
        self.bits = tuple(sorted(set(bits), reverse=True))
        self.stochastic = stochastic
        self.seed = seed
        self.plan: list[int] | None = None
        self.signatures: list[tuple] | None = None
        self.calibration: dict[str, Any] = {}
        self.last_capture: _Capture | None = None
        self._captures = 0
        logger.info(
            "activation compression: adaptive, error budget %.4g, bits %s, rounding %s, group %d",
            self.error_budget,
            self.bits,
            "stochastic" if stochastic else "nearest",
            GROUP_SIZE,
        )

    @contextmanager
    def capture(self, plan: Sequence[int] | None | str = "current") -> Iterator[_Capture]:
        """Compress the eligible tensors saved inside the context with ``plan``.

        The default uses the calibrated plan; ``plan=None`` keeps every slot and
        only records slots and bytes.
        """

        if isinstance(plan, str):
            plan = self.plan
        lookup = None
        if plan is not None:
            if self.signatures is None or len(plan) != len(self.signatures):
                raise ValueError("a compression plan needs one entry per recorded slot signature")
            lookup = _plan_lookup(self.signatures, plan)
        excluded = {t.untyped_storage().data_ptr() for t in chain(self.model.parameters(), self.model.buffers())}
        self._captures += 1
        state = _Capture(plan=lookup, excluded=excluded, seed=self.seed + self._captures, stochastic=self.stochastic)
        try:
            with torch.autograd.graph.saved_tensors_hooks(state.pack, _unpack):
                yield state
        finally:
            state.cache.clear()
            self.last_capture = state

    def calibrate(
        self,
        run: Callable[[Sequence[int] | None], Sequence[torch.Tensor | None]],
        held_out: Callable[[Sequence[int] | None], Sequence[torch.Tensor | None]] | None = None,
    ) -> dict[str, float]:
        """Choose per-slot bits and return calibration metrics.

        ``run(plan)`` and ``held_out(plan)`` recompute, from scratch, the
        parameter gradients of one fixed calibration microbatch and one
        held-out microbatch under :meth:`capture` with ``plan`` (``None``:
        exact). Callers should make the backward deterministic, otherwise its
        run-to-run noise is counted as compression error.
        """

        started = time.perf_counter()
        passes = 0

        def error(fn, plan, reference) -> float:
            nonlocal passes
            passes += 1
            return relative_error(fn(plan), reference)

        def reference_of(fn) -> list[torch.Tensor | None]:
            nonlocal passes
            passes += 1
            return _clone(fn(None))

        self.plan = None
        self.signatures = None
        if held_out is not None:
            # Run the held-out batch once first, so state its first pass may
            # change (compiled graphs, kernel choices for new shapes) is
            # settled before any reference gradient is taken.
            passes += 1
            held_out(None)
        reference = reference_of(run)
        record = self.last_capture
        count = len(record.signatures)
        self.signatures = list(record.signatures)
        slots = [
            (numel, size, signature[2])
            for numel, size, signature in zip(record.numels, record.element_sizes, record.signatures, strict=True)
        ]
        # Slots whose storage an uncompressed save also keeps alive stay as
        # they are: compressing them would add memory, not free it.
        pinned = sorted(record.aliased)
        floor = error(run, None, reference)
        budget = self.error_budget
        sensitivity = []
        for slot in range(count):
            row = {}
            for bits in () if slot in record.aliased else self.bits:
                plan = [UNCOMPRESSED] * count
                plan[slot] = bits
                value = error(run, plan, reference)
                row[bits] = max(value * value - floor * floor, 0.0)
                if row[bits] > budget * budget:
                    # Fewer bits only add error; no plan can afford this slot lower.
                    break
            sensitivity.append(row)
        held_reference = reference_of(held_out) if held_out is not None else None
        # A reference that no longer matches the exact backward would be
        # charged to compression on every plan; re-measure and rebuild it.
        drift = error(run, None, reference)
        if drift > max(2.0 * floor, 0.1 * budget):
            logger.warning("activation compression: calibration reference drifted by %.4g; rebuilding it", drift)
            reference = reference_of(run)
        aim = PLAN_MARGIN * budget
        target = aim * aim
        best = ([UNCOMPRESSED] * count, floor, float("nan"))
        rounds = []
        measured: dict[tuple[int, ...], float] = {}
        # Largest target whose plan met the budget, smallest whose plan did not.
        fits = misses = None
        for _ in range(PLAN_ROUNDS):
            plan = plan_bits(sensitivity, slots, target, self.bits)
            key = tuple(plan)
            uncompressed = all(bits == UNCOMPRESSED for bits in plan)
            if (uncompressed or key in measured) and misses is None:
                break
            if uncompressed:
                worst = 0.0
            elif key in measured:
                worst = measured[key]
            else:
                calibration_error = error(run, plan, reference)
                held_out_error = error(held_out, plan, held_reference) if held_out is not None else float("nan")
                worst = calibration_error if held_out is None else max(calibration_error, held_out_error)
                measured[key] = worst
                rounds.append((plan, calibration_error, held_out_error))
                if worst <= budget and _bytes(plan, slots) < _bytes(best[0], slots):
                    best = (plan, calibration_error, held_out_error)
                if aim * 0.9 <= worst <= budget:
                    break
            if worst <= budget:
                fits = target if fits is None else max(fits, target)
            else:
                misses = target if misses is None else min(misses, target)
            if fits is not None and misses is not None:
                # The joint error can jump between neighbouring targets (8-bit
                # slots add little, 4-bit ones add fully); bisect the bracket.
                target = math.sqrt(fits * misses)
            else:
                # Summed single-slot sensitivities misestimate the joint error;
                # rescale the target by the measured ratio and plan again.
                target *= (aim / max(worst, 1.0e-9)) ** 2
        plan, calibration_error, held_out_error = best
        self.plan = list(plan)
        stored = _bytes(plan, slots)
        saved = _bytes([UNCOMPRESSED] * count, slots)
        predicted = floor * floor + sum(sensitivity[s][b] for s, b in enumerate(plan) if b != UNCOMPRESSED)
        self.calibration = {
            "slots": count,
            "plan": list(plan),
            "signatures": [list(signature) for signature in self.signatures],
            "numels": [numel for numel, _, _ in slots],
            "pinned_slots": pinned,
            "sensitivity": [{str(bits): value for bits, value in row.items()} for row in sensitivity],
            "noise_floor": floor,
            "reference_drift": drift,
            "predicted_error": math.sqrt(predicted),
            "calibration_error": calibration_error,
            "held_out_error": held_out_error,
            "rounds": [
                {"stored_fraction": _bytes(p, slots) / max(saved, 1), "calibration_error": c, "held_out_error": h}
                for p, c, h in rounds
            ],
            "uncompressed_fallback": not any(bits != UNCOMPRESSED for bits in plan),
            "stored_fraction": stored / max(saved, 1),
            "passes": passes,
            "seconds": time.perf_counter() - started,
        }
        logger.info("activation compression calibration: %s", json.dumps(self.calibration))
        return self.calibration_metrics()

    def check(
        self,
        run: Callable[[Sequence[int] | None | str], Sequence[torch.Tensor | None]],
        held_out: Callable[[Sequence[int] | None], Sequence[torch.Tensor | None]] | None = None,
    ) -> dict[str, float]:
        """Measure the current plan on ``run``'s microbatch; recalibrate when over budget.

        Gradients shrink and lose coherence as training converges, so the same
        plan's relative error grows; GACT likewise re-estimates periodically.
        """

        reference = _clone(run(None))
        error = relative_error(run("current"), reference)
        del reference
        metrics = {"actc_check_error": error}
        if error > self.error_budget:
            logger.warning(
                "activation compression: plan error %.4g exceeds the budget %.4g; recalibrating",
                error,
                self.error_budget,
            )
            metrics.update(self.calibrate(run, held_out))
            metrics["actc_recalibrated"] = 1.0
        return metrics

    def calibration_metrics(self) -> dict[str, float]:
        """Flatten the last calibration into ``actc_*`` training metrics."""

        c = self.calibration
        metrics = {
            "actc_calibration_error": c["calibration_error"],
            "actc_noise_floor": c["noise_floor"],
            "actc_reference_drift": c["reference_drift"],
            "actc_predicted_error": c["predicted_error"],
            "actc_plan_rounds": float(len(c["rounds"])),
            "actc_pinned_slots": float(len(c.get("pinned_slots", ()))),
            "actc_uncompressed_fallback": float(c["uncompressed_fallback"]),
            "actc_plan_stored_fraction": c["stored_fraction"],
            "actc_calibration_passes": float(c["passes"]),
            "actc_calibration_s": c["seconds"],
        }
        if not math.isnan(c["held_out_error"]):
            metrics["actc_held_out_error"] = c["held_out_error"]
        for bits in (*self.bits, UNCOMPRESSED):
            metrics[f"actc_plan_slots_{bits}bit"] = float(sum(1 for value in c["plan"] if value == bits))
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """JSON-serializable plan, saved next to checkpoints for resume."""

        return {
            "version": STATE_VERSION,
            "group_size": GROUP_SIZE,
            "error_budget": self.error_budget,
            "bits": list(self.bits),
            "plan": self.plan,
            "signatures": None if self.signatures is None else [list(s) for s in self.signatures],
            "calibration": self.calibration,
        }

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        """Adopt a saved plan when it was calibrated with the same settings."""

        if (
            state.get("version") != STATE_VERSION
            or state.get("group_size") != GROUP_SIZE
            or state.get("error_budget") != self.error_budget
            or tuple(state.get("bits") or ()) != self.bits
            or state.get("plan") is None
        ):
            return False
        self.plan = [int(bits) for bits in state["plan"]]
        self.signatures = [tuple(signature) for signature in state["signatures"]]
        self.calibration = dict(state.get("calibration") or {})
        return True

    def save(self, directory: str | Path) -> None:
        """Write the calibrated plan to ``directory``; nothing before calibration."""

        if self.plan is not None:
            (Path(directory) / PLAN_FILE).write_text(json.dumps(self.state_dict()) + "\n", encoding="utf-8")

    def load(self, directory: str | Path | None) -> bool:
        """Reuse a plan saved by :meth:`save` with matching settings."""

        path = None if directory is None else Path(directory) / PLAN_FILE
        if path is None or not path.is_file():
            return False
        loaded = self.load_state_dict(json.loads(path.read_text(encoding="utf-8")))
        logger.info("activation compression: %s the plan in %s", "reusing" if loaded else "ignoring", path)
        return loaded


def _clone(grads: Sequence[torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [None if grad is None else grad.detach().clone() for grad in grads]


def _bytes(plan: Sequence[int], slots: Sequence[tuple[int, int, int]]) -> int:
    return sum(stored_bytes(*slot, bits) for slot, bits in zip(slots, plan, strict=True))
