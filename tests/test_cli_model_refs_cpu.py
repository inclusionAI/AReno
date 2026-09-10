from __future__ import annotations

import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from areno.cli import model_refs


class CliModelReferenceTest(unittest.TestCase):
    """CLI model-reference tests avoid network access by replacing hub downloaders."""

    def test_resolve_model_ref_keeps_existing_local_path(self):
        """Local checkpoints should pass through without importing hub download code."""
        with tempfile.TemporaryDirectory() as path:
            self.assertEqual(model_refs.resolve_model_ref(path), path)

    def test_resolve_model_ref_downloads_repo_id_once_with_cache(self):
        """Repeated role references should share a single snapshot_download result."""
        calls: list[str] = []
        fake_modelscope = types.SimpleNamespace(snapshot_download=lambda repo: calls.append(repo) or f"/ms/{repo}")

        with patch.dict(sys.modules, {"modelscope": fake_modelscope}):
            cache: dict[str, str] = {}
            first = model_refs.resolve_model_ref("Qwen/Qwen3-4B", cache)
            second = model_refs.resolve_model_ref("Qwen/Qwen3-4B", cache)

        self.assertEqual(first, "/ms/Qwen/Qwen3-4B")
        self.assertEqual(second, first)
        self.assertEqual(calls, ["Qwen/Qwen3-4B"])

    def test_resolve_model_ref_uses_hugging_face_when_selected(self):
        """Hugging Face model refs should use the HF downloader explicitly."""
        calls: list[str] = []
        fake_hub = types.SimpleNamespace(snapshot_download=lambda repo: calls.append(repo) or f"/cache/{repo}")

        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            resolved = model_refs.resolve_model_ref("Qwen/Qwen3-4B", model_hub="hf")

        self.assertEqual(resolved, "/cache/Qwen/Qwen3-4B")
        self.assertEqual(calls, ["Qwen/Qwen3-4B"])

    def test_resolve_model_refs_for_ppo_config_reuses_duplicate_roles(self):
        """Train config resolution should not download the same repo once per role."""
        calls: list[str] = []
        fake_modelscope = types.SimpleNamespace(snapshot_download=lambda repo: calls.append(repo) or f"/ms/{repo}")
        config = SimpleNamespace(
            algo="ppo",
            ckpt="org/actor",
            dataset_path="/data",
            ref_ckpt="org/actor",
            reward_ckpt="org/reward",
            critic_ckpt="org/actor",
        )

        with patch.dict(sys.modules, {"modelscope": fake_modelscope}):
            resolved = model_refs.resolve_model_refs_for_config(config)

        self.assertEqual(resolved.ckpt, "/ms/org/actor")
        self.assertEqual(resolved.ref_ckpt, "/ms/org/actor")
        self.assertEqual(resolved.critic_ckpt, "/ms/org/actor")
        self.assertEqual(resolved.reward_ckpt, "/ms/org/reward")
        self.assertEqual(calls, ["org/actor", "org/reward"])

    def test_resolve_model_refs_for_config_honors_modelscope_for_all_roles(self):
        calls: list[str] = []
        fake_modelscope = types.SimpleNamespace(snapshot_download=lambda repo: calls.append(repo) or f"/ms/{repo}")
        config = SimpleNamespace(
            algo="ppo",
            model_hub="modelscope",
            ckpt="org/actor",
            dataset_path="/data",
            ref_ckpt="org/actor",
            reward_ckpt="org/reward",
            critic_ckpt="org/actor",
        )

        with patch.dict(sys.modules, {"modelscope": fake_modelscope}):
            resolved = model_refs.resolve_model_refs_for_config(config)

        self.assertEqual(resolved.ckpt, "/ms/org/actor")
        self.assertEqual(resolved.ref_ckpt, "/ms/org/actor")
        self.assertEqual(resolved.critic_ckpt, "/ms/org/actor")
        self.assertEqual(resolved.reward_ckpt, "/ms/org/reward")
        self.assertEqual(calls, ["org/actor", "org/reward"])


if __name__ == "__main__":
    unittest.main()
