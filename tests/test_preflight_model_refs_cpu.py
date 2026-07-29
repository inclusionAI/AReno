"""CPU tests for the model-reference preflight capability (#229).

All tests use local temp fixtures or mocked hub clients — no network access,
no GPU, no weight materialisation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli import preflight as preflight_mod
from areno.cli import diagnostics
from areno.cli.preflight import (
    PreflightResult,
    format_preflight_text,
    preflight_model_ref,
    preflight_model_refs_for_config,
    preflight_results_to_json,
)


def _make_complete_model(tmpdir: Path) -> Path:
    """Create a minimal complete checkpoint directory."""
    (tmpdir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 8}), encoding="utf-8"
    )
    (tmpdir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmpdir / "model.safetensors").write_bytes(b"\x00" * 16)
    return tmpdir


class PreflightLocalTest(unittest.TestCase):
    """Tests for local path preflight."""

    def test_local_complete_fixture_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_complete_model(Path(tmp))
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.stage, "local")
            self.assertEqual(result.missing_artifacts, [])

    def test_local_missing_config_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"\x00")
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "format")
            self.assertEqual(result.stage, "config")
            self.assertIn("config.json", result.missing_artifacts)

    def test_local_config_json_not_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text("not json {{{", encoding="utf-8")
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"\x00")
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "format")
            self.assertEqual(result.stage, "config")

    def test_local_config_missing_model_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"\x00")
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "format")
            self.assertEqual(result.stage, "config")
            self.assertIn("model_type", result.detail)

    def test_local_missing_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            (path / "model.safetensors").write_bytes(b"\x00")
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "format")
            self.assertEqual(result.stage, "tokenizer")
            self.assertTrue(len(result.missing_artifacts) > 0)
            self.assertIn("tokenizer.json", result.missing_artifacts)

    def test_local_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "format")
            self.assertEqual(result.stage, "weights")
            self.assertTrue(len(result.missing_artifacts) > 0)

    def test_local_safetensors_index_with_missing_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            # Index references two shards but only one exists on disk.
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                }}),
                encoding="utf-8",
            )
            (path / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 16)
            # shard 2 is deliberately missing.
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "format")
            self.assertEqual(result.stage, "weights")
            self.assertIn("model-00002-of-00002.safetensors", result.missing_artifacts)

    def test_local_safetensors_index_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.0.weight": "model-00001.safetensors"}}),
                encoding="utf-8",
            )
            (path / "model-00001.safetensors").write_bytes(b"\x00" * 16)
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "ok")

    def test_local_nonexistent_path(self):
        """A nonexistent path that doesn't look like a repo ID should be not_found."""
        # Use a path with no slash so it can't be mistaken for a remote repo ID.
        result = preflight_model_ref("nonexistent_local_path_no_slash")
        self.assertEqual(result.status, "format")
        self.assertEqual(result.stage, "remote")

    def test_local_path_is_file_not_dir(self):
        with tempfile.NamedTemporaryFile() as f:
            result = preflight_model_ref(f.name)
            self.assertEqual(result.status, "format")
            self.assertEqual(result.stage, "local")
            self.assertIn("not a directory", result.detail)

    def test_local_empty_ref(self):
        result = preflight_model_ref("")
        self.assertEqual(result.status, "format")
        self.assertEqual(result.stage, "local")

    def test_local_vocab_and_merges_tokenizer_ok(self):
        """Alternative tokenizer set: vocab.json + merges.txt."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "llama"}), encoding="utf-8"
            )
            (path / "vocab.json").write_text("{}", encoding="utf-8")
            (path / "merges.txt").write_text("", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"\x00" * 16)
            result = preflight_model_ref(str(path))
            self.assertEqual(result.status, "ok")


class PreflightRemoteTest(unittest.TestCase):
    """Tests for remote ID preflight (no network access)."""

    def test_remote_invalid_format_no_slash(self):
        result = preflight_model_ref("invalid-no-slash", model_hub="modelscope")
        self.assertEqual(result.status, "format")
        self.assertEqual(result.stage, "remote")

    def test_remote_modelscope_client_not_installed(self):
        with patch.dict(sys.modules, {"modelscope": None}):
            result = preflight_model_ref("Qwen/Qwen3-0.6B", model_hub="modelscope")
        self.assertEqual(result.status, "network")
        self.assertEqual(result.stage, "remote")
        self.assertIn("modelscope", result.detail)

    def test_remote_hf_client_not_installed(self):
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            result = preflight_model_ref("Qwen/Qwen3-0.6B", model_hub="hf")
        self.assertEqual(result.status, "network")
        self.assertEqual(result.stage, "remote")
        self.assertIn("huggingface_hub", result.detail)

    def test_remote_valid_format_client_installed(self):
        fake_modelscope = types.SimpleNamespace()
        with patch.dict(sys.modules, {"modelscope": fake_modelscope}):
            result = preflight_model_ref("Qwen/Qwen3-0.6B", model_hub="modelscope")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stage, "remote")

    def test_remote_hf_valid_format_client_installed(self):
        fake_hub = types.SimpleNamespace()
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            result = preflight_model_ref("Qwen/Qwen3-0.6B", model_hub="hf")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stage, "remote")

    def test_remote_cached_model_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_dir = cache_dir / "Qwen" / "Qwen3-0.6B"
            model_dir.mkdir(parents=True)
            fake_modelscope = types.SimpleNamespace()
            with (
                patch.dict(sys.modules, {"modelscope": fake_modelscope}),
                patch.dict(os.environ, {"MODELSCOPE_CACHE": str(cache_dir)}),
            ):
                result = preflight_model_ref("Qwen/Qwen3-0.6B", model_hub="modelscope")
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.stage, "remote")
            self.assertIsNotNone(result.resolved_path)

    def test_remote_unknown_hub(self):
        result = preflight_model_ref("Qwen/Qwen3-0.6B", model_hub="unknown")
        self.assertEqual(result.status, "format")
        self.assertEqual(result.stage, "remote")


class PreflightConfigTest(unittest.TestCase):
    """Tests for preflight_model_refs_for_config."""

    def test_ppo_config_preflights_all_roles(self):
        fake_modelscope = types.SimpleNamespace()
        config = SimpleNamespace(
            algo="ppo",
            model_hub="modelscope",
            ckpt="org/actor",
            ref_ckpt="org/ref",
            reward_ckpt="org/reward",
            critic_ckpt="org/critic",
        )
        with patch.dict(sys.modules, {"modelscope": fake_modelscope}):
            results = preflight_model_refs_for_config(config)
        self.assertEqual(len(results), 4)
        refs = [r.model_ref for r in results]
        self.assertIn("org/actor", refs)
        self.assertIn("org/ref", refs)
        self.assertIn("org/reward", refs)
        self.assertIn("org/critic", refs)
        for r in results:
            self.assertEqual(r.status, "ok")

    def test_sft_config_preflights_only_ckpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_complete_model(Path(tmp))
            config = SimpleNamespace(
                algo="sft",
                model_hub="modelscope",
                ckpt=str(path),
                ref_ckpt=None,
                reward_ckpt=None,
                critic_ckpt=None,
            )
            results = preflight_model_refs_for_config(config)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "ok")

    def test_config_with_failing_ref_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            # Missing tokenizer and weights.
            config = SimpleNamespace(
                algo="gspo",
                model_hub="modelscope",
                ckpt=str(path),
                ref_ckpt=None,
                reward_ckpt=None,
                critic_ckpt=None,
            )
            results = preflight_model_refs_for_config(config)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "format")
            self.assertEqual(results[0].stage, "tokenizer")


class PreflightFormattingTest(unittest.TestCase):
    """Tests for output formatting helpers."""

    def test_format_preflight_text_ok(self):
        result = PreflightResult(
            model_ref="/path/to/model",
            resolved_path="/path/to/model",
            status="ok",
            stage="local",
            detail="all good",
        )
        text = format_preflight_text(result)
        self.assertIn("OK", text)
        self.assertIn("/path/to/model", text)
        self.assertNotIn("missing", text)

    def test_format_preflight_text_failure(self):
        result = PreflightResult(
            model_ref="/path/to/model",
            resolved_path="/path/to/model",
            status="format",
            stage="tokenizer",
            detail="no tokenizer files",
            missing_artifacts=["tokenizer.json", "vocab.json"],
            next_step="Download tokenizer files",
        )
        text = format_preflight_text(result)
        self.assertIn("FORMAT", text)
        self.assertIn("tokenizer.json", text)
        self.assertIn("Download tokenizer files", text)

    def test_preflight_results_to_json(self):
        results = [
            PreflightResult(
                model_ref="org/model",
                resolved_path=None,
                status="ok",
                stage="remote",
                detail="hub client available",
            )
        ]
        json_str = preflight_results_to_json(results)
        parsed = json.loads(json_str)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["status"], "ok")
        self.assertEqual(parsed[0]["model_ref"], "org/model")


class CheckCommandIntegrationTest(unittest.TestCase):
    """Integration tests for `areno check --model-ref`."""

    def test_check_with_model_ref_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_complete_model(Path(tmp))
            runner = CliRunner()
            with patch.object(diagnostics, "collect_env", return_value=_fake_ready_report()):
                result = runner.invoke(
                    diagnostics.check_command,
                    ["--model-ref", str(path)],
                )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("model preflight", result.output)
            self.assertIn("OK", result.output)

    def test_check_with_model_ref_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            runner = CliRunner()
            with patch.object(diagnostics, "collect_env", return_value=_fake_ready_report()):
                result = runner.invoke(
                    diagnostics.check_command,
                    ["--model-ref", str(path)],
                )
            self.assertEqual(result.exit_code, 1)
            self.assertIn("FAIL", result.output)
            self.assertIn("model preflight", result.output)

    def test_check_with_model_ref_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_complete_model(Path(tmp))
            runner = CliRunner()
            with patch.object(diagnostics, "collect_env", return_value=_fake_ready_report()):
                result = runner.invoke(
                    diagnostics.check_command,
                    ["--model-ref", str(path), "--json"],
                )
            self.assertEqual(result.exit_code, 0)
            parsed = json.loads(result.output)
            self.assertIn("preflight", parsed)
            self.assertEqual(parsed["preflight"][0]["status"], "ok")

    def test_check_without_model_ref_unchanged(self):
        """Existing `areno check` behavior is preserved when --model-ref is absent."""
        runner = CliRunner()
        with patch.object(diagnostics, "collect_env", return_value=_fake_ready_report()):
            result = runner.invoke(diagnostics.check_command, [])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("AReno check: ready", result.output)
        self.assertNotIn("model preflight", result.output)


class TrainPreflightIntegrationTest(unittest.TestCase):
    """Tests for preflight integration with train config.

    These tests validate the preflight logic that `areno train --preflight`
    would invoke, without importing the full train CLI (which requires torch).
    """

    def test_train_preflight_logic_fails_on_missing_artifacts(self):
        """The preflight logic used by --preflight correctly detects missing files."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            config = SimpleNamespace(
                algo="gspo",
                model_hub="modelscope",
                ckpt=str(path),
                ref_ckpt=None,
                reward_ckpt=None,
                critic_ckpt=None,
            )
            results = preflight_model_refs_for_config(config)
            failed = [r for r in results if r.status != "ok"]
            self.assertTrue(len(failed) > 0)
            self.assertEqual(failed[0].status, "format")
            self.assertEqual(failed[0].stage, "tokenizer")
            # Verify the output includes the missing file names.
            text = format_preflight_text(failed[0])
            self.assertIn("tokenizer", text.lower())

    def test_train_preflight_logic_passes_on_complete_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_complete_model(Path(tmp))
            config = SimpleNamespace(
                algo="gspo",
                model_hub="modelscope",
                ckpt=str(path),
                ref_ckpt=None,
                reward_ckpt=None,
                critic_ckpt=None,
            )
            results = preflight_model_refs_for_config(config)
            failed = [r for r in results if r.status != "ok"]
            self.assertEqual(len(failed), 0)


class ServePreflightIntegrationTest(unittest.TestCase):
    """Tests for preflight logic used by `areno serve --preflight`."""

    def test_serve_preflight_logic_detects_nonexistent_path(self):
        """The preflight logic used by --preflight catches nonexistent paths."""
        # Use a path with no slash so it fails remote format check.
        result = preflight_model_ref("nonexistent_serve_path_no_slash")
        self.assertNotEqual(result.status, "ok")


def _fake_ready_report() -> dict:
    """A minimal env report that passes all checks."""
    return {
        "areno": {"version": "0.1.0"},
        "python": {"version": "3.11.0", "executable": "/python"},
        "platform": {"system": "Linux", "release": "6.0", "machine": "x86_64", "platform": "Linux"},
        "torch": {
            "imported": True, "error": None, "version": "2.6.0",
            "cuda_build": "12.4", "cuda_runtime": "12.4.0",
            "cuda_runtime_error": None, "cuda_available": True,
            "device_count": 1,
            "gpus": [{"index": 0, "name": "NVIDIA H100", "capability": "9.0"}],
        },
        "cuda": {
            "cuda_home": "/usr/local/cuda", "inferred_cuda_home": "/usr/local/cuda",
            "nvcc": {"path": "/usr/local/cuda/bin/nvcc", "version": "release 12.4"},
            "driver": {"path": "/usr/bin/nvidia-smi", "driver_version": "550.0",
                       "cuda_version": "12.4", "error": None},
        },
        "gpus": [{"index": 0, "name": "NVIDIA H100", "capability": "9.0"}],
        "dependencies": {
            "flash_attn": {"distribution": "flash-attn", "module": "flash_attn",
                           "version": "2.7.0", "imported": True, "error": None},
            "flash_linear_attention": {"distribution": "flash-linear-attention", "module": "fla",
                                       "version": "0.2.0", "imported": True, "error": None},
            "areno_accel": {"distribution": None, "module": "areno.accel._areno_accel",
                            "version": None, "imported": True, "error": None},
        },
        "install": {"build_ext_disabled": False},
        "env": {"CUDA_HOME": "/usr/local/cuda", "MAX_JOBS": "8"},
        "paths": {"metrics_log_dir": "/tmp/areno", "hf_cache": "/tmp/cache"},
    }


if __name__ == "__main__":
    unittest.main()
