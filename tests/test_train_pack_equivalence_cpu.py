"""CPU equivalence tests for the train-pack vectorization refactor.

The policy-only trainer now emits `TrainSequence` rows with the structured
`prompt_len`/`scalar_advantage` layout instead of full per-token
`prompt_mask`/`advantages` lists, and `pad_rows` builds the rectangular pack
tensor with one vectorized numpy pass. These tests pin the behavior contract:
the scalar-convention pack is tensor-identical to the list convention, the
vectorized `pad_rows` matches the per-row reference, and the response-only
metrics are unchanged.
"""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from areno.api.backend.cuda.training import make_train_pack
from areno.api.metrics import collect_train_batch_stats
from areno.api.models import TrainSequence
from areno.engine.runtime.common import pad_rows


def _reference_pad_rows(rows, *, dtype, fill_value=0, width=None):
    """Per-row reference implementation of pad_rows (pre-vectorization)."""

    if width is None:
        width = max((len(row) for row in rows), default=0)
    out = torch.full((len(rows), width), fill_value, dtype=dtype)
    for row_idx, row in enumerate(rows):
        if len(row) == 0:
            continue
        out[row_idx, : len(row)] = torch.as_tensor(row, dtype=dtype)
    return out


def _build_list_rows(rng, n=8, prefix=40, resp=400):
    """GRPO-shaped rows with the legacy full-list layout.

    Advantages are constant per sample and broadcast over the response tokens,
    matching what the policy-only trainer produces; the scalar layout
    expresses exactly this invariant.
    """

    rows = []
    for _ in range(n):
        resp_len = resp + rng.randint(-20, 20)
        advantage = rng.random()
        rows.append(
            TrainSequence(
                prompt_mask=[True] * prefix + [False] * resp_len,
                tokens=list(range(prefix + resp_len)),
                logprobs=[0.0] * prefix + [rng.random() - 0.5 for _ in range(resp_len)],
                advantages=[0.0] * prefix + [advantage] * resp_len,
                reward=float(rng.random()),
                eos_token_id=0,
            )
        )
    return rows


def _build_scalar_rows_from(list_rows):
    """Derive scalar-layout rows from list-layout rows (identical content)."""

    rows = []
    for row in list_rows:
        prefix_len = sum(1 for is_prompt in row.prompt_mask if is_prompt)
        rows.append(
            TrainSequence(
                prompt_len=prefix_len,
                tokens=list(row.tokens),
                logprobs=list(row.logprobs),
                scalar_advantage=float(row.advantages[prefix_len]),
                reward=row.reward,
                eos_token_id=row.eos_token_id,
            )
        )
    return rows


class PadRowsVectorizationTest(unittest.TestCase):
    """Vectorized pad_rows must match the per-row reference exactly."""

    def test_int_rows(self):
        rng = random.Random(0)
        rows = [[rng.randint(0, 100) for _ in range(rng.randint(0, 30))] for _ in range(12)]
        for width in (None, 40):
            a = pad_rows(rows, dtype=torch.long, fill_value=7, width=width)
            b = _reference_pad_rows(rows, dtype=torch.long, fill_value=7, width=width)
            self.assertTrue(torch.equal(a, b), "int64 pad_rows mismatch")

    def test_float_rows(self):
        rng = random.Random(1)
        rows = [[rng.random() for _ in range(rng.randint(0, 30))] for _ in range(10)]
        a = pad_rows(rows, dtype=torch.float32, width=32)
        b = _reference_pad_rows(rows, dtype=torch.float32, width=32)
        self.assertTrue(torch.equal(a, b), "float32 pad_rows mismatch")

    def test_bool_rows(self):
        rng = random.Random(2)
        rows = [[bool(rng.getrandbits(1)) for _ in range(rng.randint(0, 30))] for _ in range(9)]
        a = pad_rows(rows, dtype=torch.bool, fill_value=True, width=32)
        b = _reference_pad_rows(rows, dtype=torch.bool, fill_value=True, width=32)
        self.assertTrue(torch.equal(a, b), "bool pad_rows mismatch")

    def test_all_empty_rows(self):
        rows = [[], [], []]
        a = pad_rows(rows, dtype=torch.long, width=5)
        b = _reference_pad_rows(rows, dtype=torch.long, width=5)
        self.assertTrue(torch.equal(a, b))


class TrainPackEquivalenceTest(unittest.TestCase):
    """The scalar layout must produce tensor-identical packs to the list layout."""

    def _assert_packs_equal(self, a, b):
        self.assertEqual(set(a), set(b), "pack keys differ")
        for key in a:
            if isinstance(a[key], torch.Tensor):
                self.assertTrue(torch.equal(a[key], b[key]), f"pack[{key}] differs")
            else:
                self.assertEqual(a[key], b[key], f"pack[{key}] differs")

    def test_grpo_shaped_packs_identical(self):
        rng = random.Random(3)
        list_rows = _build_list_rows(rng)
        scalar_rows = _build_scalar_rows_from(list_rows)
        self._assert_packs_equal(make_train_pack(list_rows), make_train_pack(scalar_rows))

    def test_scalar_pack_semantics(self):
        """Spot-check the scalar pack values for one small batch."""

        seq = TrainSequence(
            prompt_len=3,
            tokens=[1, 2, 3, 4, 5, 6],
            logprobs=[0.0, 0.0, 0.0, 0.5, 0.6, 0.7],
            scalar_advantage=0.25,
            reward=1.0,
            eos_token_id=0,
        )
        pack = make_train_pack([seq])
        self.assertEqual(pack["prompt_mask"].tolist(), [[True, True, True, False, False, False]])
        self.assertEqual(pack["advantages"].tolist(), [[0.0, 0.0, 0.0, 0.25, 0.25, 0.25]])
        self.assertIsNone(pack.get("loss_mask"))


class TrainBatchStatsEquivalenceTest(unittest.TestCase):
    """Response-only metrics must be identical across both layouts."""

    def test_stats_identical(self):
        rng = random.Random(4)
        list_rows = _build_list_rows(rng, n=6, prefix=10, resp=50)
        scalar_rows = _build_scalar_rows_from(list_rows)
        list_stats = collect_train_batch_stats(list_rows)
        scalar_stats = collect_train_batch_stats(scalar_rows)
        self.assertEqual(list_stats.keys(), scalar_stats.keys())
        for key in list_stats:
            np.testing.assert_allclose(
                np.asarray(list_stats[key], dtype=np.float64),
                np.asarray(scalar_stats[key], dtype=np.float64),
                rtol=0,
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
