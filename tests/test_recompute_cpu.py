from __future__ import annotations

import unittest

import torch

from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer, should_checkpoint_layer


class RecomputeTest(unittest.TestCase):
    """Activation checkpoint tests use tiny CPU tensors and no model weights."""

    def test_should_checkpoint_layer_requires_training_forward(self):
        """Recompute is enabled only for grad-enabled training forwards."""
        train_meta = TrainMeta(activation_checkpointing=True)
        infer_meta = InferMeta(mode="decode")

        self.assertTrue(should_checkpoint_layer(train_meta, None))
        self.assertFalse(should_checkpoint_layer(TrainMeta(activation_checkpointing=False), None))
        self.assertFalse(should_checkpoint_layer(train_meta, infer_meta))
        with torch.no_grad():
            self.assertFalse(should_checkpoint_layer(train_meta, None))

    def test_checkpoint_layer_matches_direct_forward_and_backpropagates(self):
        """Checkpointed execution should preserve forward value and gradients."""
        calls = {"count": 0}

        def layer_fn(x, bias):
            calls["count"] += 1
            return (x * x + bias).sum()

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        bias = torch.tensor([0.5, 0.5, 0.5])

        out = checkpoint_layer(layer_fn, x, bias, train_meta=TrainMeta(activation_checkpointing=True), infer_meta=None)
        out.backward()

        self.assertEqual(float(out.detach()), 15.5)
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 4.0, 6.0])))
        self.assertGreaterEqual(calls["count"], 1)

    def test_checkpoint_layer_bypasses_checkpoint_for_infer_meta(self):
        """Inference metadata must bypass checkpointing even when train_meta opts in."""
        calls = {"count": 0}

        def layer_fn(x):
            calls["count"] += 1
            return x + 1

        x = torch.tensor([1.0], requires_grad=True)
        out = checkpoint_layer(
            layer_fn,
            x,
            train_meta=TrainMeta(activation_checkpointing=True),
            infer_meta=InferMeta(mode="prefill"),
        )

        self.assertEqual(out.tolist(), [2.0])
        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
