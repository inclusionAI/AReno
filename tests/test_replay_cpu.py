"""CPU tests for rollout record save/load and replay batch reconstruction.

These tests exercise the replay serialization layer and the ``Trainer``
save/load wrappers without requiring a GPU or backend initialization.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from areno.api.models import TrainSequence
from areno.engine.data.replay import (
    REPLAY_FORMAT_VERSION,
    RolloutRecord,
    load_rollout_records,
    save_rollout_records,
)


def _make_record(**overrides) -> RolloutRecord:
    """Return a minimal valid ``RolloutRecord`` with optional overrides."""

    defaults = dict(
        format_version=REPLAY_FORMAT_VERSION,
        epoch=0,
        step=0,
        prompt_index=0,
        sample_index=0,
        tokens=[1, 2, 3],
        prompt_mask=[True, False, False],
        loss_mask=[False, True, False],
        logprobs=[0.0, -1.2, 0.0],
        advantages=[0.0, 0.5, 0.0],
        reward=1.0,
        eos_token_id=99,
        metadata={},
    )
    defaults.update(overrides)
    return RolloutRecord(**defaults)


def _make_train_sequence(**overrides) -> TrainSequence:
    """Return a minimal valid ``TrainSequence`` with optional overrides."""

    defaults = dict(
        prompt_mask=[True, False, False],
        loss_mask=[False, True, False],
        tokens=[1, 2, 3],
        logprobs=[0.0, -1.2, 0.0],
        advantages=[0.0, 0.5, 0.0],
        reward=1.0,
        eos_token_id=99,
    )
    defaults.update(overrides)
    return TrainSequence(**defaults)


def _write_jsonl(path: Path, *records: dict) -> None:
    """Write raw JSON objects to a ``.jsonl`` file for negative tests."""

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class RolloutRecordIOTest(unittest.TestCase):
    """Save/load round-trip and validation tests."""

    def test_save_then_load_preserves_all_fields(self):
        """Round-trip save→load should preserve every field exactly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.jsonl"
            original = [
                _make_record(epoch=1, step=5, prompt_index=2, sample_index=3),
                _make_record(tokens=[10, 20], prompt_mask=[True, False],
                             loss_mask=[False, True], logprobs=[0.0, -0.5],
                             advantages=[0.0, 1.0], reward=0.8),
            ]
            save_rollout_records(path, original)
            loaded = load_rollout_records(path)

            self.assertEqual(len(loaded), 2)
            for orig, got in zip(original, loaded, strict=True):
                self.assertEqual(got.format_version, orig.format_version)
                self.assertEqual(got.epoch, orig.epoch)
                self.assertEqual(got.step, orig.step)
                self.assertEqual(got.prompt_index, orig.prompt_index)
                self.assertEqual(got.sample_index, orig.sample_index)
                self.assertEqual(got.tokens, orig.tokens)
                self.assertEqual(got.prompt_mask, orig.prompt_mask)
                self.assertEqual(got.loss_mask, orig.loss_mask)
                self.assertEqual(got.logprobs, orig.logprobs)
                self.assertEqual(got.advantages, orig.advantages)
                self.assertEqual(got.reward, orig.reward)
                self.assertEqual(got.eos_token_id, orig.eos_token_id)
                self.assertEqual(got.metadata, orig.metadata)

    def test_load_rejects_version_mismatch(self):
        """Incompatible format_version must raise ValueError, not coerce."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            _write_jsonl(path, {
                "format_version": 999,
                "tokens": [1], "prompt_mask": [True], "loss_mask": [False],
                "logprobs": [0.0], "advantages": [0.0],
                "reward": 1.0, "eos_token_id": 0,
            })
            with self.assertRaisesRegex(ValueError, "format_version 999 is incompatible"):
                load_rollout_records(path)

    def test_load_rejects_missing_format_version(self):
        """A missing format_version must produce a clear error."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            _write_jsonl(path, {
                "tokens": [1], "prompt_mask": [True], "loss_mask": [False],
                "logprobs": [0.0], "advantages": [0.0],
                "reward": 1.0, "eos_token_id": 0,
            })
            with self.assertRaisesRegex(ValueError, "missing format_version"):
                load_rollout_records(path)

    def test_load_rejects_missing_required_field(self):
        """A missing required field must produce a clear error naming the field."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            _write_jsonl(path, {
                "format_version": REPLAY_FORMAT_VERSION,
                "tokens": [1], "prompt_mask": [True], "loss_mask": [False],
                "logprobs": [0.0], "advantages": [0.0],
                "reward": 1.0,
                # eos_token_id missing
            })
            with self.assertRaisesRegex(ValueError, "missing required field 'eos_token_id'"):
                load_rollout_records(path)

    def test_load_rejects_misaligned_lengths(self):
        """Tokens/mask length mismatch must be caught."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            _write_jsonl(path, {
                "format_version": REPLAY_FORMAT_VERSION,
                "tokens": [1, 2, 3], "prompt_mask": [True, False],
                "loss_mask": [False, True, False],
                "logprobs": [0.0, -1.0, 0.0], "advantages": [0.0, 0.5, 0.0],
                "reward": 1.0, "eos_token_id": 0,
            })
            with self.assertRaisesRegex(ValueError, "'prompt_mask' length 2"):
                load_rollout_records(path)

    def test_load_empty_file_raises(self):
        """An empty replay file should fail explicitly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replay file is empty"):
                load_rollout_records(path)

    def test_load_nonexistent_file_raises(self):
        """A missing path should fail with a clear message."""

        with self.assertRaisesRegex(ValueError, "replay file not found"):
            load_rollout_records("/nonexistent/path/to/replay.jsonl")

    def test_load_skips_blank_lines(self):
        """Blank lines in the JSONL file should be silently skipped."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.jsonl"
            record = _make_record()
            save_rollout_records(path, [record])
            # Append a blank line
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n\n")
            loaded = load_rollout_records(path)
            self.assertEqual(len(loaded), 1)


class ReplayBatchReconstructionTest(unittest.TestCase):
    """Trainer save/load wrapper tests using fake backends."""

    def test_loaded_records_rebuild_identical_train_sequence(self):
        """``load_rollout_batch`` should produce field-identical ``TrainSequence`` objects."""

        from areno import Trainer

        trainer = Trainer(world_size=1, model_path="unused")
        original = [
            _make_train_sequence(tokens=[10, 20, 30], prompt_mask=[True, False, False],
                                 loss_mask=[False, True, False], logprobs=[0.0, -2.0, 0.0],
                                 advantages=[0.0, 1.0, 0.0], reward=0.5, eos_token_id=7),
            _make_train_sequence(tokens=[40, 50], prompt_mask=[True, False],
                                 loss_mask=[False, True], logprobs=[0.0, -0.3],
                                 advantages=[0.0, -0.5], reward=0.9, eos_token_id=7),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "step_000000.jsonl")
            trainer.save_rollout_batch(path, epoch=0, step=0, train_batch=original)
            replayed = trainer.load_rollout_batch(path)

        self.assertEqual(len(replayed), 2)
        for orig, got in zip(original, replayed, strict=True):
            self.assertEqual(got.tokens, orig.tokens)
            self.assertEqual(got.prompt_mask, orig.prompt_mask)
            self.assertEqual(got.loss_mask, orig.loss_mask)
            self.assertEqual(got.logprobs, orig.logprobs)
            self.assertEqual(got.advantages, orig.advantages)
            self.assertEqual(got.reward, orig.reward)
            self.assertEqual(got.eos_token_id, orig.eos_token_id)

    def test_save_creates_valid_jsonl_with_provenance(self):
        """Saved records should contain epoch/step/index provenance fields."""

        from areno import Trainer

        trainer = Trainer(world_size=1, model_path="unused")
        batch = [_make_train_sequence(), _make_train_sequence(reward=0.3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "step_000005.jsonl")
            trainer.save_rollout_batch(path, epoch=2, step=5, train_batch=batch)
            records = load_rollout_records(path)

        self.assertEqual(records[0].epoch, 2)
        self.assertEqual(records[0].step, 5)
        self.assertEqual(records[0].prompt_index, 0)
        self.assertEqual(records[1].prompt_index, 1)
        self.assertEqual(records[1].reward, 0.3)

    def test_load_rollout_batch_raises_on_bad_file(self):
        """``load_rollout_batch`` should propagate validation errors."""

        from areno import Trainer

        trainer = Trainer(world_size=1, model_path="unused")
        with self.assertRaisesRegex(ValueError, "replay file not found"):
            trainer.load_rollout_batch("/nonexistent/file.jsonl")


class ReplayConfigTest(unittest.TestCase):
    """Config defaults and replay_path propagation."""

    def test_replay_path_defaults_to_none(self):
        """Default config should have replay_path=None for backward compatibility."""

        from areno.api.trainer_config import PolicyTrainerConfig, PPOTrainerConfig

        gspo_config = PolicyTrainerConfig(algo="gspo", ckpt="unused", dataset_path="unused")
        self.assertIsNone(gspo_config.replay_path)

        ppo_config = PPOTrainerConfig(algo="ppo", ckpt="unused", dataset_path="unused")
        self.assertIsNone(ppo_config.replay_path)

    def test_replay_path_can_be_set(self):
        """Setting replay_path should store the value."""

        from areno.api.trainer_config import PolicyTrainerConfig

        config = PolicyTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            replay_path="/tmp/replay",
        )
        self.assertEqual(config.replay_path, "/tmp/replay")

    def test_save_replay_path_defaults_to_none(self):
        """Default config should have save_replay_path=None."""

        from areno.api.trainer_config import PolicyTrainerConfig

        config = PolicyTrainerConfig(algo="gspo", ckpt="unused", dataset_path="unused")
        self.assertIsNone(config.save_replay_path)

    def test_save_replay_path_can_be_set(self):
        """Setting save_replay_path should store the value."""

        from areno.api.trainer_config import PolicyTrainerConfig

        config = PolicyTrainerConfig(
            algo="gspo", ckpt="unused", dataset_path="unused",
            save_replay_path="/tmp/save-replay",
        )
        self.assertEqual(config.save_replay_path, "/tmp/save-replay")


if __name__ == "__main__":
    unittest.main()