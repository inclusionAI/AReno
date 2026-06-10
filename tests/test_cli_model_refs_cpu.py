from __future__ import annotations

import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from areno.cli import model_refs


class CliModelReferenceTest(unittest.TestCase):
    """CLI model-reference tests avoid network access by replacing Hugging Face Hub."""

    def test_resolve_model_ref_keeps_existing_local_path(self):
        """Local checkpoints should pass through without importing hub download code."""
        with tempfile.TemporaryDirectory() as path:
            self.assertEqual(model_refs.resolve_model_ref(path), path)

    def test_resolve_model_ref_downloads_repo_id_once_with_cache(self):
        """Repeated role references should share a single snapshot_download result."""
        calls: list[str] = []
        fake_hub = types.SimpleNamespace(snapshot_download=lambda repo: calls.append(repo) or f"/cache/{repo}")

        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            cache: dict[str, str] = {}
            first = model_refs.resolve_model_ref("Qwen/Qwen3-4B", cache)
            second = model_refs.resolve_model_ref("Qwen/Qwen3-4B", cache)

        self.assertEqual(first, "/cache/Qwen/Qwen3-4B")
        self.assertEqual(second, first)
        self.assertEqual(calls, ["Qwen/Qwen3-4B"])

    def test_resolve_model_refs_for_ppo_config_reuses_duplicate_roles(self):
        """Train config resolution should not download the same repo once per role."""
        calls: list[str] = []
        fake_hub = types.SimpleNamespace(snapshot_download=lambda repo: calls.append(repo) or f"/cache/{repo}")
        config = SimpleNamespace(
            algo="ppo",
            ckpt="org/actor",
            dataset_path="/data",
            ref_ckpt="org/actor",
            reward_ckpt="org/reward",
            critic_ckpt="org/actor",
        )

        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            resolved = model_refs.resolve_model_refs_for_config(config)

        self.assertEqual(resolved.ckpt, "/cache/org/actor")
        self.assertEqual(resolved.ref_ckpt, "/cache/org/actor")
        self.assertEqual(resolved.critic_ckpt, "/cache/org/actor")
        self.assertEqual(resolved.reward_ckpt, "/cache/org/reward")
        self.assertEqual(calls, ["org/actor", "org/reward"])


if __name__ == "__main__":
    unittest.main()
