from __future__ import annotations

import unittest

from areno.engine.runtime.load_progress import (
    ModelLoadTracker,
    STAGE_CONFIG_TOKENIZER,
    STAGE_DEVICE_PLACEMENT,
    STAGE_REFERENCE_RESOLUTION,
    STAGE_WEIGHT_SHARD_READING,
    STAGE_WORKER_DISTRIBUTION,
)


def _fake_resolve(model_ref: str) -> str:
    """Stand-in for resolve_model_path; returns a fake local path."""

    return f"/cache/{model_ref.split('/')[-1].lower()}"


def _fake_config(path: str) -> dict:
    """Stand-in for config_from_hf; returns a minimal config dict."""

    return {"model_type": "fake", "path": path, "hidden_size": 8}


def _fake_build(config: dict) -> dict:
    """Stand-in for build_model_on_device; returns a fake model object."""

    return {"config": config, "params": []}


def _fake_load_weights(model: dict, path: str) -> None:
    """Stand-in for load_model_weights; mutates the fake model in place."""

    model["loaded_from"] = path


class TrackedLoadFlowTest(unittest.TestCase):
    """Integration-style test: drive a fake 5-stage load flow with the tracker.

    Mirrors the call sequence in ArenoEngine.from_pretrained + ArenoWorker.__init__
    without importing torch, so it runs in any CPU environment. Asserts the
    tracker records each stage in order and surfaces the failing stage on error.
    """

    def _run_load(self, tracker: ModelLoadTracker, model_ref: str) -> dict:
        """Drive the five stages with the tracker, returning the fake model."""

        with tracker.stage(STAGE_REFERENCE_RESOLUTION, detail=model_ref):
            path = _fake_resolve(model_ref)
        with tracker.stage(STAGE_CONFIG_TOKENIZER, detail=path):
            config = _fake_config(path)
        with tracker.stage(STAGE_DEVICE_PLACEMENT, detail="cpu"):
            model = _fake_build(config)
        with tracker.stage(STAGE_WEIGHT_SHARD_READING, detail=path):
            _fake_load_weights(model, path)
        with tracker.stage(STAGE_WORKER_DISTRIBUTION):
            model["distributed"] = True
        return model

    def test_full_flow_completes_all_stages_in_order(self):
        tracker = ModelLoadTracker(rank0=False)
        model = self._run_load(tracker, "Qwen/Qwen3.5-0.8B")

        self.assertEqual(model["loaded_from"], "/cache/qwen3.5-0.8b")
        self.assertTrue(model["distributed"])
        self.assertEqual(tracker.last_completed_stage, STAGE_WORKER_DISTRIBUTION)

    def test_failure_in_weight_stage_keeps_prior_completed_stage(self):
        tracker = ModelLoadTracker(rank0=False)

        with self.assertRaises(KeyError):
            with tracker.stage(STAGE_REFERENCE_RESOLUTION):
                path = _fake_resolve("Qwen/Qwen3.5-0.8B")
            with tracker.stage(STAGE_CONFIG_TOKENIZER):
                config = _fake_config(path)
            with tracker.stage(STAGE_DEVICE_PLACEMENT):
                model = _fake_build(config)
            with tracker.stage(STAGE_WEIGHT_SHARD_READING):
                raise KeyError("missing weight shard")
            # Later stages never run.

        # device_placement completed; weight_shard_reading did not.
        self.assertEqual(tracker.last_completed_stage, STAGE_DEVICE_PLACEMENT)

    def test_failure_surfaces_failing_stage_not_completed(self):
        tracker = ModelLoadTracker(rank0=False)

        with self.assertRaises(ValueError):
            with tracker.stage(STAGE_REFERENCE_RESOLUTION):
                _fake_resolve("Qwen/Qwen3.5-0.8B")
            with tracker.stage(STAGE_CONFIG_TOKENIZER):
                raise ValueError("malformed config.json")

        # reference_resolution completed; config_tokenizer_load did not, so the
        # tracker points at the last *completed* stage, not the failing one.
        self.assertEqual(tracker.last_completed_stage, STAGE_REFERENCE_RESOLUTION)


if __name__ == "__main__":
    unittest.main()