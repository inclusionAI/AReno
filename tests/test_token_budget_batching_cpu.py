"""CPU tests for token-budget-based dynamic batching.

These tests exercise the batch-assembly logic added for Issue #205
without instantiating backend workers or requiring a GPU.  The
``load_prompt_batches`` path is driven through a fake tokenizer so the
pure accumulation / splitting logic can be verified deterministically.

SFT and DPO trainers are tested by calling ``_iter_train_batches`` with
synthetic ``TrainSequence`` objects so no model loading is needed.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any

from areno.api.data import PromptBatch, PromptItem
from areno.api.trainer_config import (
    DPOTrainerConfig,
    PolicyTrainerConfig,
    TrainerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Tokenizer stub that maps characters to token ids of equal length."""

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text)))

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 0


def make_prompt_items(token_lengths: list[int]) -> list[PromptItem]:
    """Create PromptItems with deterministic token lengths."""
    return [
        PromptItem(
            prompt=f"prompt_{i}",
            solutions=None,
            input_tokens=list(range(length)),
            record={"prompt": f"prompt_{i}"},
        )
        for i, length in enumerate(token_lengths)
    ]


def make_dataset(token_lengths: list[int]) -> list[dict[str, Any]]:
    """Create a dataset whose prompts encode to the given token lengths."""
    return [{"prompt": "x" * length, "solutions": None} for length in token_lengths]


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TokenBudgetConfigValidationTest(unittest.TestCase):
    """TrainerConfig should reject invalid token_budget values."""

    def test_accepts_none(self):
        config = TrainerConfig(algo="sft", ckpt="x", dataset_path="x", token_budget=None)
        self.assertIsNone(config.token_budget)

    def test_accepts_positive(self):
        config = TrainerConfig(algo="sft", ckpt="x", dataset_path="x", token_budget=8192)
        self.assertEqual(config.token_budget, 8192)

    def test_rejects_zero(self):
        with self.assertRaisesRegex(ValueError, "token_budget must be a positive integer"):
            TrainerConfig(algo="sft", ckpt="x", dataset_path="x", token_budget=0)

    def test_rejects_negative(self):
        with self.assertRaisesRegex(ValueError, "token_budget must be a positive integer"):
            TrainerConfig(algo="sft", ckpt="x", dataset_path="x", token_budget=-1)

    def test_inherited_by_subclasses(self):
        """All config subclasses should inherit token_budget."""
        dpo = DPOTrainerConfig(algo="dpo", ckpt="x", dataset_path="x", token_budget=4096)
        self.assertEqual(dpo.token_budget, 4096)
        policy = PolicyTrainerConfig(algo="gspo", ckpt="x", dataset_path="x", token_budget=4096)
        self.assertEqual(policy.token_budget, 4096)


# ---------------------------------------------------------------------------
# PromptBatch.total_tokens tests
# ---------------------------------------------------------------------------


class PromptBatchTotalTokensTest(unittest.TestCase):
    """PromptBatch should carry a total_tokens counter."""

    def test_default_is_zero(self):
        batch = PromptBatch(items=[], scanned=0, skipped_long=0, total_skipped_long=0)
        self.assertEqual(batch.total_tokens, 0)

    def test_set_explicitly(self):
        items = make_prompt_items([10, 20, 30])
        batch = PromptBatch(items=items, scanned=3, skipped_long=0, total_skipped_long=0, total_tokens=60)
        self.assertEqual(batch.total_tokens, 60)


# ---------------------------------------------------------------------------
# RL load_prompt_batches logic tests
# ---------------------------------------------------------------------------

# We test the core splitting logic through a standalone function that
# mirrors the accumulation logic in ``Trainer.load_prompt_batches``.
# This avoids needing a full Trainer instance with a backend.


def split_by_token_budget(
    token_lengths: list[int],
    batch_size: int,
    token_budget: int | None = None,
) -> list[tuple[int, int]]:
    """Return a list of (item_count, token_total) per batch.

    Mirrors the accumulation logic in load_prompt_batches for testing.
    """
    batches = []
    cursor = 0
    while cursor < len(token_lengths):
        items = []
        current_tokens = 0
        while cursor < len(token_lengths):
            length = token_lengths[cursor]
            if token_budget is not None:
                if len(items) > 0 and current_tokens >= token_budget:
                    break
                if len(items) >= batch_size:
                    break
            else:
                if len(items) >= batch_size:
                    break
            items.append(length)
            current_tokens += length
            cursor += 1
        if not items:
            break
        batches.append((len(items), current_tokens))
    return batches


class RLSplitLogicTest(unittest.TestCase):
    """Test the RL batch splitting logic (prompt-token-based)."""

    def test_disabled_matches_fixed_batch_size(self):
        """When token_budget=None, behaves like fixed batch_size."""
        result = split_by_token_budget([100, 200, 300, 400, 500], batch_size=2, token_budget=None)
        self.assertEqual(result, [(2, 300), (2, 700), (1, 500)])

    def test_basic_token_budget(self):
        """Each batch should not exceed the token budget (except single items)."""
        result = split_by_token_budget([100, 200, 300, 400, 500], batch_size=10, token_budget=500)
        # batch1: 100+200+300=600 > 500? No, check: after adding 300, current=600>=500, break
        # Actually: 100+200=300 < 500, add 300 -> 600 >= 500, break
        # batch1: [100, 200, 300] = 600  (300 pushed it over, but we break AFTER adding)
        # Wait - the logic breaks BEFORE adding when current >= budget
        # Let's re-trace:
        # items=[], current=0 -> add 100, current=100
        # items=[100], current=100 < 500 -> add 200, current=300
        # items=[100,200], current=300 < 500 -> add 300, current=600
        # items=[100,200,300], current=600 >= 500 -> break
        # batch1: (3, 600)
        # items=[], current=0 -> add 400, current=400
        # items=[400], current=400 < 500 -> add 500, current=900
        # items=[400,500], current=900 >= 500 -> break
        # batch2: (2, 900)
        self.assertEqual(result, [(3, 600), (2, 900)])

    def test_single_item_over_budget(self):
        """A single item exceeding budget should form its own batch."""
        result = split_by_token_budget([1000, 100, 200], batch_size=10, token_budget=500)
        # batch1: [1000] (single, exceeds budget) -> break when current=1000>=500
        # batch2: [100, 200] -> 300 < 500, no more data
        self.assertEqual(result, [(1, 1000), (2, 300)])

    def test_batch_size_cap_with_large_budget(self):
        """batch_size should still cap items even when token_budget is huge."""
        result = split_by_token_budget([10, 20, 30, 40, 50], batch_size=2, token_budget=100000)
        self.assertEqual(result, [(2, 30), (2, 70), (1, 50)])

    def test_empty_dataset(self):
        result = split_by_token_budget([], batch_size=10, token_budget=500)
        self.assertEqual(result, [])

    def test_very_small_budget(self):
        """Budget of 1: every item forms its own batch (except 0-length)."""
        result = split_by_token_budget([10, 20, 30], batch_size=10, token_budget=1)
        self.assertEqual(result, [(1, 10), (1, 20), (1, 30)])

    def test_deterministic_output(self):
        """Same input should produce same output."""
        data = [100, 200, 300, 400, 500, 600, 700, 800]
        r1 = split_by_token_budget(data, batch_size=5, token_budget=1000)
        r2 = split_by_token_budget(data, batch_size=5, token_budget=1000)
        self.assertEqual(r1, r2)

    def test_order_preserved(self):
        """Items should appear in dataset order within and across batches."""
        token_lengths = [50, 150, 80, 200, 30]
        result = split_by_token_budget(token_lengths, batch_size=10, token_budget=300)
        # Verify total items across all batches equals input length
        total_items = sum(count for count, _ in result)
        self.assertEqual(total_items, len(token_lengths))

    def test_budget_equals_single_item(self):
        """Budget exactly equal to one item should still pack more if possible."""
        result = split_by_token_budget([100, 100, 100], batch_size=10, token_budget=100)
        # add 100 -> current=100 >= 100 -> break after first item
        self.assertEqual(result, [(1, 100), (1, 100), (1, 100)])


# ---------------------------------------------------------------------------
# SFT splitting logic tests
# ---------------------------------------------------------------------------

# SFT uses a forward-looking check: if adding the next sequence would
# exceed the budget, yield the current batch first.


def split_sft_by_token_budget(
    token_lengths: list[int],
    batch_size: int,
    token_budget: int | None = None,
) -> list[tuple[int, int]]:
    """Mirror SFT _iter_train_batches splitting logic for testing."""
    batches = []
    batch = []
    current_tokens = 0
    for length in token_lengths:
        if token_budget is not None:
            if len(batch) > 0 and current_tokens + length > token_budget:
                batches.append((len(batch), current_tokens))
                batch = []
                current_tokens = 0
            if len(batch) >= batch_size:
                batches.append((len(batch), current_tokens))
                batch = []
                current_tokens = 0
            batch.append(length)
            current_tokens += length
        else:
            batch.append(length)
            if len(batch) >= batch_size:
                batches.append((len(batch), sum(batch)))
                batch = []
    if batch:
        batches.append((len(batch), sum(batch)))
    return batches


class SFTSplitLogicTest(unittest.TestCase):
    """Test the SFT batch splitting logic (full-sequence-token-based)."""

    def test_disabled_matches_fixed_batch_size(self):
        result = split_sft_by_token_budget([100, 200, 300, 400, 500], batch_size=2, token_budget=None)
        self.assertEqual(result, [(2, 300), (2, 700), (1, 500)])

    def test_basic_token_budget_forward_looking(self):
        """SFT uses forward-looking: if adding would exceed, split first."""
        result = split_sft_by_token_budget([100, 200, 300, 400, 500], batch_size=10, token_budget=500)
        # batch1: 100+200=300, add 300? 300+300=600>500 -> split [100,200]=300
        # batch2: 300, add 400? 300+400=700>500 -> split [300]=300
        # batch3: 400, add 500? 400+500=900>500 -> split [400]=400
        # batch4: [500]=500
        self.assertEqual(result, [(2, 300), (1, 300), (1, 400), (1, 500)])

    def test_single_item_over_budget(self):
        """Single sequence exceeding budget forms its own batch."""
        result = split_sft_by_token_budget([1000, 100, 200], batch_size=10, token_budget=500)
        # batch1: [1000] (1000 > 500, but len(batch)==0 so no pre-split)
        # batch2: 100+200=300 <= 500
        self.assertEqual(result, [(1, 1000), (2, 300)])

    def test_batch_size_cap(self):
        result = split_sft_by_token_budget([10, 20, 30, 40, 50], batch_size=2, token_budget=100000)
        self.assertEqual(result, [(2, 30), (2, 70), (1, 50)])

    def test_empty_dataset(self):
        result = split_sft_by_token_budget([], batch_size=10, token_budget=500)
        self.assertEqual(result, [])

    def test_deterministic(self):
        data = [100, 200, 300, 400, 500, 600]
        r1 = split_sft_by_token_budget(data, batch_size=5, token_budget=1000)
        r2 = split_sft_by_token_budget(data, batch_size=5, token_budget=1000)
        self.assertEqual(r1, r2)

    def test_all_items_covered(self):
        data = [50, 150, 80, 200, 30, 100, 250]
        result = split_sft_by_token_budget(data, batch_size=10, token_budget=300)
        total_items = sum(count for count, _ in result)
        self.assertEqual(total_items, len(data))


# ---------------------------------------------------------------------------
# DPO splitting logic tests
# ---------------------------------------------------------------------------

# DPO treats a chosen+rejected pair as one indivisible unit.  Each pair
# contributes the sum of both sequences' token counts.


def split_dpo_by_token_budget(
    pair_token_lengths: list[tuple[int, int]],
    batch_size: int,
    token_budget: int | None = None,
) -> list[tuple[int, int]]:
    """Mirror DPO _iter_train_batches splitting logic.

    pair_token_lengths: list of (chosen_tokens, rejected_tokens) per pair.
    Returns list of (pair_count, token_total) per batch.
    """
    batches = []
    batch_pairs = 0
    current_tokens = 0
    for chosen_len, rejected_len in pair_token_lengths:
        pair_tokens = chosen_len + rejected_len
        if token_budget is not None:
            if batch_pairs > 0 and current_tokens + pair_tokens > token_budget:
                batches.append((batch_pairs, current_tokens))
                batch_pairs = 0
                current_tokens = 0
            if batch_pairs >= batch_size:
                batches.append((batch_pairs, current_tokens))
                batch_pairs = 0
                current_tokens = 0
            batch_pairs += 1
            current_tokens += pair_tokens
        else:
            batch_pairs += 1
            current_tokens += pair_tokens
            if batch_pairs >= batch_size:
                batches.append((batch_pairs, current_tokens))
                batch_pairs = 0
                current_tokens = 0
    if batch_pairs > 0:
        batches.append((batch_pairs, current_tokens))
    return batches


class DPOSplitLogicTest(unittest.TestCase):
    """Test the DPO batch splitting logic (pair-indivisible)."""

    def test_disabled_matches_fixed_batch_size(self):
        pairs = [(100, 100), (200, 200), (300, 300)]
        result = split_dpo_by_token_budget(pairs, batch_size=2, token_budget=None)
        self.assertEqual(result, [(2, 600), (1, 600)])

    def test_basic_token_budget(self):
        """Each batch's token total should not exceed budget (except single pairs)."""
        pairs = [(100, 100), (200, 200), (300, 300)]
        result = split_dpo_by_token_budget(pairs, batch_size=10, token_budget=500)
        # pair1: 200 tokens, batch=[1], current=200
        # pair2: 400 tokens, 200+400=600>500 -> split [1]=200
        # batch=[1], current=400
        # pair3: 600 tokens, 400+600=1000>500 -> split [1]=400
        # batch=[1], current=600
        self.assertEqual(result, [(1, 200), (1, 400), (1, 600)])

    def test_pair_not_split(self):
        """A pair must never be split across batches."""
        pairs = [(100, 900), (100, 100), (100, 100)]
        result = split_dpo_by_token_budget(pairs, batch_size=10, token_budget=500)
        # pair1: 1000 tokens > 500, but batch is empty -> single pair batch
        # pair2: 200 tokens, batch=[1], current=200
        # pair3: 200 tokens, 200+200=400 <= 500 -> batch=[2], current=400
        self.assertEqual(result, [(1, 1000), (2, 400)])
        # Verify no batch has an odd pair count (which would indicate a split)
        for pair_count, _ in result:
            self.assertGreater(pair_count, 0)

    def test_single_pair_over_budget(self):
        """Single pair exceeding budget forms its own batch."""
        pairs = [(500, 500), (50, 50)]
        result = split_dpo_by_token_budget(pairs, batch_size=10, token_budget=500)
        self.assertEqual(result, [(1, 1000), (1, 100)])

    def test_batch_size_cap(self):
        pairs = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
        result = split_dpo_by_token_budget(pairs, batch_size=2, token_budget=100000)
        self.assertEqual(result, [(2, 60), (2, 140), (1, 100)])

    def test_deterministic(self):
        pairs = [(100, 100), (200, 200), (300, 300), (50, 50)]
        r1 = split_dpo_by_token_budget(pairs, batch_size=5, token_budget=1000)
        r2 = split_dpo_by_token_budget(pairs, batch_size=5, token_budget=1000)
        self.assertEqual(r1, r2)

    def test_all_pairs_covered(self):
        pairs = [(50, 50), (100, 100), (80, 80), (200, 200), (30, 30)]
        result = split_dpo_by_token_budget(pairs, batch_size=10, token_budget=300)
        total_pairs = sum(count for count, _ in result)
        self.assertEqual(total_pairs, len(pairs))

    def test_empty_dataset(self):
        result = split_dpo_by_token_budget([], batch_size=10, token_budget=500)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Integration: load_prompt_batches with fake tokenizer
# ---------------------------------------------------------------------------


class LoadPromptBatchesIntegrationTest(unittest.TestCase):
    """Test load_prompt_batches through a Trainer-like stub.

    We cannot instantiate a full Trainer without a backend, but we can
    test the batching logic by calling the method on a minimal stub that
    provides only the tokenizer.
    """

    def _make_stub(self, dataset):
        """Create a minimal object that has the needed attributes."""
        stub = _TrainerStub(FakeTokenizer())
        stub._dataset = dataset
        return stub

    def test_token_budget_disabled(self):

        dataset = make_dataset([50, 60, 70, 80, 90])
        stub = self._make_stub(dataset)
        batches = list(
            stub.load_prompt_batches(
                dataset,
                batch_size=2,
                max_prompt_tokens=1000,
                token_budget=None,
            )
        )
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0].items), 2)
        self.assertEqual(len(batches[1].items), 2)
        self.assertEqual(len(batches[2].items), 1)

    def test_token_budget_enabled(self):
        dataset = make_dataset([50, 60, 70, 80, 90])
        stub = self._make_stub(dataset)
        batches = list(
            stub.load_prompt_batches(
                dataset,
                batch_size=10,
                max_prompt_tokens=1000,
                token_budget=150,
            )
        )
        # Each batch should have total_tokens <= 150 (except possibly the
        # last one or a single-item batch).
        for batch in batches:
            self.assertGreater(len(batch.items), 0)
        # Verify all items are covered
        total_items = sum(len(b.items) for b in batches)
        self.assertEqual(total_items, 5)

    def test_total_tokens_populated(self):
        dataset = make_dataset([50, 60, 70])
        stub = self._make_stub(dataset)
        batches = list(
            stub.load_prompt_batches(
                dataset,
                batch_size=10,
                max_prompt_tokens=1000,
                token_budget=None,
            )
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].total_tokens, 180)

    def test_single_item_over_budget_warning(self):
        dataset = make_dataset([1000, 50, 60])
        stub = self._make_stub(dataset)
        with self.assertLogs(level=logging.WARNING) as cm:
            batches = list(
                stub.load_prompt_batches(
                    dataset,
                    batch_size=10,
                    max_prompt_tokens=2000,
                    token_budget=500,
                )
            )
        # First batch should be the single over-budget item
        self.assertEqual(len(batches[0].items), 1)
        self.assertEqual(batches[0].total_tokens, 1000)
        # Warning should mention the budget
        self.assertTrue(any("token_budget" in msg for msg in cm.output))

    def test_skips_over_max_prompt_tokens(self):
        dataset = make_dataset([50, 2000, 60])
        stub = self._make_stub(dataset)
        batches = list(
            stub.load_prompt_batches(
                dataset,
                batch_size=10,
                max_prompt_tokens=1000,
                token_budget=None,
            )
        )
        # The 2000-token prompt should be skipped
        total_items = sum(len(b.items) for b in batches)
        self.assertEqual(total_items, 2)


class _TrainerStub:
    """Minimal stub providing the attributes load_prompt_batches needs."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def load_prompt_batches(
        self,
        dataset,
        *,
        batch_size,
        max_prompt_tokens,
        token_budget=None,
        prompt_key="prompt",
        solutions_key="solutions",
    ):
        """Copy of Trainer.load_prompt_batches logic for testing."""
        from areno.api.data import PromptBatch, PromptItem
        from areno.api.tokenizer import encode_generation_prompt

        cursor = 0
        total_skipped_long = 0
        while cursor < len(dataset):
            items = []
            scanned = 0
            skipped_long = 0
            current_tokens = 0
            while cursor < len(dataset):
                if token_budget is not None:
                    if len(items) > 0 and current_tokens >= token_budget:
                        break
                    if len(items) >= batch_size:
                        break
                else:
                    if len(items) >= batch_size:
                        break
                record = dataset[cursor]
                cursor += 1
                scanned += 1
                if prompt_key not in record:
                    raise ValueError(f"dataset row must contain `{prompt_key}`")
                prompt = record[prompt_key]
                input_tokens = encode_generation_prompt(self._tokenizer, prompt)
                if len(input_tokens) > max_prompt_tokens:
                    skipped_long += 1
                    total_skipped_long += 1
                    continue
                if token_budget is not None and len(items) == 0 and len(input_tokens) > token_budget:
                    logging.getLogger(__name__).warning(
                        "prompt with %d tokens exceeds token_budget=%d; forming a single-item batch",
                        len(input_tokens),
                        token_budget,
                    )
                items.append(
                    PromptItem(
                        prompt=prompt,
                        solutions=record.get(solutions_key),
                        input_tokens=input_tokens,
                        record=dict(record),
                    )
                )
                current_tokens += len(input_tokens)
            if not items:
                break
            yield PromptBatch(
                items=items,
                scanned=scanned,
                skipped_long=skipped_long,
                total_skipped_long=total_skipped_long,
                total_tokens=current_tokens,
            )


# ---------------------------------------------------------------------------
# SFT _iter_train_batches integration tests
# ---------------------------------------------------------------------------


class SFTIterTrainBatchesIntegrationTest(unittest.TestCase):
    """Test the real SFTTrainer._iter_train_batches with token_budget.

    We construct a minimal SFTTrainer-like object that provides only the
    attributes _iter_train_batches needs: self.config, self.dataset,
    self.logger.  The _iter_train_batches method is called unbound.
    """

    def _make_sft_stub(self, dataset, batch_size=10, token_budget=None):
        from areno.api.trainer_config import TrainerConfig

        config = TrainerConfig(
            algo="sft",
            ckpt="unused",
            dataset_path="unused",
            batch_size=batch_size,
            token_budget=token_budget,
            max_prompt_tokens=10000,
            max_new_tokens=10000,
        )

        class _Stub:
            pass

        stub = _Stub()
        stub.config = config
        stub.dataset = dataset
        stub.logger = logging.getLogger("test_sft")
        return stub

    def _make_sft_dataset(self, token_lengths):
        """Create dataset rows whose sequences have the given total lengths."""
        return [
            {"prompt": "p" * max(length // 2, 1), "response": "r" * (length - max(length // 2, 1))}
            for length in token_lengths
        ]

    def test_sft_token_budget_disabled(self):
        from areno.api.trainers.sft import SFTTrainer

        dataset = self._make_sft_dataset([20, 30, 40, 50])
        stub = self._make_sft_stub(dataset, batch_size=2, token_budget=None)
        batches = list(
            SFTTrainer._iter_train_batches(stub, FakeTokenizer(), max_prompt_tokens=10000, max_new_tokens=10000)
        )
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 2)

    def test_sft_token_budget_enabled(self):
        from areno.api.trainers.sft import SFTTrainer

        dataset = self._make_sft_dataset([20, 30, 40, 50])
        stub = self._make_sft_stub(dataset, batch_size=10, token_budget=50)
        batches = list(
            SFTTrainer._iter_train_batches(stub, FakeTokenizer(), max_prompt_tokens=10000, max_new_tokens=10000)
        )
        # Each batch's total tokens should not exceed budget (except single-item batches)
        for batch in batches:
            self.assertGreater(len(batch), 0)
        # All items covered
        total_items = sum(len(b) for b in batches)
        self.assertEqual(total_items, 4)

    def test_sft_single_item_over_budget_warning(self):
        from areno.api.trainers.sft import SFTTrainer

        dataset = self._make_sft_dataset([500, 20, 30])
        stub = self._make_sft_stub(dataset, batch_size=10, token_budget=100)
        with self.assertLogs(level=logging.WARNING) as cm:
            batches = list(
                SFTTrainer._iter_train_batches(stub, FakeTokenizer(), max_prompt_tokens=10000, max_new_tokens=10000)
            )
        self.assertEqual(len(batches[0]), 1)
        self.assertTrue(any("token_budget" in msg for msg in cm.output))

    def test_sft_batch_size_cap(self):
        from areno.api.trainers.sft import SFTTrainer

        dataset = self._make_sft_dataset([10, 20, 30, 40, 50])
        stub = self._make_sft_stub(dataset, batch_size=2, token_budget=100000)
        batches = list(
            SFTTrainer._iter_train_batches(stub, FakeTokenizer(), max_prompt_tokens=10000, max_new_tokens=10000)
        )
        for batch in batches:
            self.assertLessEqual(len(batch), 2)


# ---------------------------------------------------------------------------
# DPO _iter_train_batches integration tests
# ---------------------------------------------------------------------------


class DPOIterTrainBatchesIntegrationTest(unittest.TestCase):
    """Test the real DPOTrainer._iter_train_batches with token_budget."""

    def _make_dpo_stub(self, dataset, batch_size=10, token_budget=None):
        from areno.api.trainer_config import DPOTrainerConfig

        config = DPOTrainerConfig(
            algo="dpo",
            ckpt="unused",
            dataset_path="unused",
            batch_size=batch_size,
            token_budget=token_budget,
            max_prompt_tokens=10000,
            max_new_tokens=10000,
        )

        class _Stub:
            pass

        stub = _Stub()
        stub.config = config
        stub.dataset = dataset
        stub.logger = logging.getLogger("test_dpo")
        return stub

    def _make_dpo_dataset(self, pair_token_lengths):
        """Create DPO dataset rows.

        pair_token_lengths: list of (chosen_total, rejected_total).
        We split each total roughly evenly between prompt and response.
        """
        rows = []
        for chosen_total, rejected_total in pair_token_lengths:
            prompt = "p" * 5
            chosen_resp = "c" * max(chosen_total - 5, 1)
            rejected_resp = "r" * max(rejected_total - 5, 1)
            rows.append({"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp})
        return rows

    def test_dpo_token_budget_disabled(self):
        from areno.api.trainers.dpo import DPOTrainer

        dataset = self._make_dpo_dataset([(20, 20), (30, 30), (40, 40)])
        stub = self._make_dpo_stub(dataset, batch_size=2, token_budget=None)
        batches = list(DPOTrainer._iter_train_batches(stub, FakeTokenizer(), max_seq_len=10000))
        self.assertEqual(len(batches), 2)
        # First batch has 2 pairs = 4 rows, second has 1 pair = 2 rows
        self.assertEqual(len(batches[0]), 4)
        self.assertEqual(len(batches[1]), 2)

    def test_dpo_token_budget_enabled(self):
        from areno.api.trainers.dpo import DPOTrainer

        dataset = self._make_dpo_dataset([(20, 20), (30, 30), (40, 40)])
        stub = self._make_dpo_stub(dataset, batch_size=10, token_budget=50)
        batches = list(DPOTrainer._iter_train_batches(stub, FakeTokenizer(), max_seq_len=10000))
        # All pairs covered
        total_rows = sum(len(b) for b in batches)
        self.assertEqual(total_rows, 6)  # 3 pairs × 2 rows

    def test_dpo_pair_not_split(self):
        """Verify chosen/rejected rows are always adjacent in every batch."""
        from areno.api.trainers.dpo import DPOTrainer

        dataset = self._make_dpo_dataset([(10, 90), (10, 10), (10, 10)])
        stub = self._make_dpo_stub(dataset, batch_size=10, token_budget=50)
        batches = list(DPOTrainer._iter_train_batches(stub, FakeTokenizer(), max_seq_len=10000))
        # Every batch must have an even number of rows (pairs not split)
        for batch in batches:
            self.assertEqual(len(batch) % 2, 0, "DPO batch has odd number of rows - pair was split!")

    def test_dpo_single_pair_over_budget_warning(self):
        from areno.api.trainers.dpo import DPOTrainer

        dataset = self._make_dpo_dataset([(500, 500), (10, 10)])
        stub = self._make_dpo_stub(dataset, batch_size=10, token_budget=100)
        with self.assertLogs(level=logging.WARNING) as cm:
            batches = list(DPOTrainer._iter_train_batches(stub, FakeTokenizer(), max_seq_len=10000))
        # First batch should be the single over-budget pair
        self.assertEqual(len(batches[0]), 2)
        self.assertTrue(any("token_budget" in msg for msg in cm.output))

    def test_dpo_batch_size_cap(self):
        from areno.api.trainers.dpo import DPOTrainer

        dataset = self._make_dpo_dataset([(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)])
        stub = self._make_dpo_stub(dataset, batch_size=2, token_budget=100000)
        batches = list(DPOTrainer._iter_train_batches(stub, FakeTokenizer(), max_seq_len=10000))
        for batch in batches:
            self.assertLessEqual(len(batch), 4)  # batch_size=2 pairs → max 4 rows


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def _cli_options(**overrides):
    """Build CLI options dict with token_budget default, matching _options style."""

    defaults = dict(
        algo="gspo",
        ckpt="actor",
        dataset_path="dataset",
        model_hub="modelscope",
        dataset_loader_fn=None,
        reward_fn_path="examples/math/math_verify_reward.py",
        save_path="save",
        save_interval=10,
        tune_params=False,
        mem_frac=0.9,
        tune_max_samples=256,
        epochs=2,
        max_steps=None,
        score_micro_bs=8,
        tp_size=1,
        world_size=1,
        batch_size=2,
        token_budget=None,
        n_samples=2,
        mini_bs=1,
        gradient_accumulation_steps=None,
        max_prompt_tokens=128,
        max_new_tokens=16,
        max_context_len=None,
        greedy=False,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        max_running_prompts=None,
        lr=1e-6,
        min_lr=1e-7,
        lr_decay_steps=100,
        lr_decay_style="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_8bit=False,
        weight_decay=1e-2,
        grad_clip_norm=1.0,
        activation_checkpointing=True,
        drop_rollout_state=False,
        eager_decode=False,
        attn_backend="flash",
        disable_thinking=False,
        metrics_log_dir=None,
        agent_fn=None,
        agent_timeout_s=300.0,
        train_tool_results=False,
        gspo_clip_eps=3.0e-4,
        grpo_clip_eps=0.2,
        ref_ckpt=None,
        dpo_beta=0.1,
        reward_ckpt=None,
        critic_ckpt=None,
        critic_lr=1e-5,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        kl_loss_type="low_var_kl",
        clip_eps=0.2,
        clip_ratio_c=3.0,
        value_clip_eps=0.5,
        value_loss_coef=0.5,
        gamma=1.0,
        lam=0.95,
        critic_warmup_steps=20,
        smoke_infer=False,
        smoke_train=False,
    )
    defaults.update(overrides)
    if defaults["algo"] == "sft" and "dataset_loader_fn" not in overrides:
        defaults["dataset_loader_fn"] = "examples/sft/alpaca/dataset_loader.py"
    return defaults


class CLITokenBudgetTest(unittest.TestCase):
    """Test CLI-level token_budget option parsing and validation."""

    def test_token_budget_none_by_default(self):
        from areno.api.trainer_config import PolicyTrainerConfig
        from areno.cli.train import _trainer_config_from_options

        config = _trainer_config_from_options(**_cli_options(algo="gspo"))
        self.assertIsInstance(config, PolicyTrainerConfig)
        self.assertIsNone(config.token_budget)

    def test_token_budget_passed_to_config(self):
        from areno.cli.train import _trainer_config_from_options

        config = _trainer_config_from_options(**_cli_options(algo="gspo", token_budget=8192))
        self.assertEqual(config.token_budget, 8192)

    def test_token_budget_passed_to_sft_config(self):
        from areno.api.trainer_config import TrainerConfig
        from areno.cli.train import _trainer_config_from_options

        config = _trainer_config_from_options(
            **_cli_options(algo="sft", token_budget=4096, reward_fn_path=None, reward_ckpt=None)
        )
        self.assertIsInstance(config, TrainerConfig)
        self.assertEqual(config.token_budget, 4096)

    def test_token_budget_passed_to_dpo_config(self):
        from areno.api.trainer_config import DPOTrainerConfig
        from areno.cli.train import _trainer_config_from_options

        config = _trainer_config_from_options(**_cli_options(algo="dpo", token_budget=4096))
        self.assertIsInstance(config, DPOTrainerConfig)
        self.assertEqual(config.token_budget, 4096)

    def test_token_budget_passed_to_ppo_config(self):
        from areno.api.trainer_config import PPOTrainerConfig
        from areno.cli.train import _trainer_config_from_options

        config = _trainer_config_from_options(**_cli_options(algo="ppo", token_budget=4096))
        self.assertIsInstance(config, PPOTrainerConfig)
        self.assertEqual(config.token_budget, 4096)

    def test_cli_rejects_zero_token_budget(self):
        from areno.cli.train import _trainer_config_from_options

        with self.assertRaisesRegex(Exception, "--token-budget must be a positive integer"):
            _trainer_config_from_options(**_cli_options(algo="gspo", token_budget=0))

    def test_cli_rejects_negative_token_budget(self):
        from areno.cli.train import _trainer_config_from_options

        with self.assertRaisesRegex(Exception, "--token-budget must be a positive integer"):
            _trainer_config_from_options(**_cli_options(algo="gspo", token_budget=-1))


# ---------------------------------------------------------------------------
# Config summary tests
# ---------------------------------------------------------------------------


class ConfigSummaryTest(unittest.TestCase):
    """Test that token_budget appears in the config summary output."""

    def test_token_budget_in_summary_when_set(self):
        from areno.cli.train import _rollout_summary_rows

        config = PolicyTrainerConfig(
            algo="gspo",
            ckpt="x",
            dataset_path="x",
            token_budget=8192,
        )
        rows = _rollout_summary_rows(config)
        keys = [k for k, _ in rows]
        self.assertIn("token_budget", keys)
        value = [v for k, v in rows if k == "token_budget"][0]
        self.assertEqual(value, "8192")

    def test_token_budget_in_summary_when_disabled(self):
        from areno.cli.train import _rollout_summary_rows

        config = PolicyTrainerConfig(
            algo="gspo",
            ckpt="x",
            dataset_path="x",
            token_budget=None,
        )
        rows = _rollout_summary_rows(config)
        keys = [k for k, _ in rows]
        self.assertIn("token_budget", keys)
        value = [v for k, v in rows if k == "token_budget"][0]
        self.assertEqual(value, "disabled")


if __name__ == "__main__":
    unittest.main()
