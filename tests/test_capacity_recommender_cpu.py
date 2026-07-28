"""CPU tests for the capacity-tuning recommendation tool skill.

Exercises the recommender script at ``.agents/skills/areno-tune-capacity/scripts/recommend_capacity.py``
covering success paths, invalid input, boundary values, algorithm correctness,
output format, determinism, file export, and the safety constraint that no
training run is ever submitted.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_DEPS_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _DEPS_ROOT / ".agents" / "skills" / "areno-tune-capacity" / "scripts" / "recommend_capacity.py"


def _load_module():
    """Import the recommender script as a module for function-level tests."""

    spec = importlib.util.spec_from_file_location("recommend_capacity", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["recommend_capacity"] = module
    spec.loader.exec_module(module)
    return module


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run the recommender as a subprocess with the given arguments."""

    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=str(_DEPS_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


class TestRecommenderWithProfile(unittest.TestCase):
    """Recommender produces three modes when measured profile data is given."""

    def test_generates_three_modes(self):
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--peak-mem-frac", "0.82", "--throughput-tps", "1200.0",
            "--json",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertIn("recommendations", result)
        self.assertEqual(set(result["recommendations"].keys()), {"conservative", "balanced", "throughput"})
        self.assertEqual(result["profile"]["source"], "measured")
        self.assertAlmostEqual(result["profile"]["peak_mem_frac"], 0.82)

    def test_each_recommendation_has_overrides_and_explanation(self):
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--peak-mem-frac", "0.82", "--json",
        ])
        result = json.loads(proc.stdout)
        for mode in ("conservative", "balanced", "throughput"):
            rec = result["recommendations"][mode]
            self.assertIn("overrides", rec)
            self.assertIn("explanation", rec)
            self.assertIn("estimated_mem_frac", rec)
            self.assertIn("validation", rec)
            self.assertIsInstance(rec["overrides"], dict)
            self.assertTrue(len(rec["overrides"]) > 0)


class TestRecommenderWithoutProfile(unittest.TestCase):
    """Recommender uses fallback estimation when no profile data is given."""

    def test_gpu_memory_fallback(self):
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--gpu-memory-gb", "80", "--model-params-billions", "7.0",
            "--json",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["profile"]["source"], "estimated")
        self.assertGreater(result["profile"]["peak_mem_frac"], 0)

    def test_defaults_when_no_inputs(self):
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--json",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["profile"]["source"], "default")
        self.assertGreater(result["profile"]["peak_mem_frac"], 0)


# ---------------------------------------------------------------------------
# Invalid input and boundary
# ---------------------------------------------------------------------------


class TestRecommenderInvalidInput(unittest.TestCase):
    """Recommender rejects invalid input with clear error messages."""

    def test_rejects_invalid_tp_world_ratio(self):
        proc = _run_cli([
            "--tp-size", "3", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--json",
        ])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("divisible", proc.stderr)

    def test_rejects_negative_values(self):
        proc = _run_cli([
            "--tp-size", "-1", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--json",
        ])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("must be positive", proc.stderr)

    def test_rejects_invalid_mem_frac(self):
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--mem-frac", "0", "--json",
        ])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("mem-frac", proc.stderr)

        proc2 = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--mem-frac", "0.95", "--json",
        ])
        self.assertEqual(proc2.returncode, 1)


# ---------------------------------------------------------------------------
# Algorithm correctness
# ---------------------------------------------------------------------------


class TestRecommenderAlgorithm(unittest.TestCase):
    """Verify that each recommendation mode applies the correct adjustments."""

    def test_conservative_halves_concurrency(self):
        mod = _load_module()
        inp = mod.RecommenderInput(
            tp_size=4, world_size=8, batch_size=32, n_samples=8, mini_bs=16,
            max_running_prompts=256, adam_8bit=False, activation_checkpointing=True,
            keep_rollout_state=True, max_new_tokens=3071, max_context_len=None,
            max_prompt_tokens=1024,
        )
        profile = mod.ProfileData(peak_mem_frac=0.82, throughput_tps=1200.0, source="measured")
        recs = mod.generate_recommendations(inp, profile)
        cons = recs["conservative"]
        self.assertEqual(cons.overrides["max_running_prompts"], 128)
        self.assertEqual(cons.overrides["mini_bs"], 8)
        self.assertTrue(cons.overrides["activation_checkpointing"])
        self.assertFalse(cons.overrides["keep_rollout_state"])
        self.assertTrue(cons.overrides["adam_8bit"])

    def test_throughput_does_not_reduce_values(self):
        mod = _load_module()
        inp = mod.RecommenderInput(
            tp_size=4, world_size=8, batch_size=32, n_samples=8, mini_bs=16,
            max_running_prompts=64, adam_8bit=False, activation_checkpointing=True,
            keep_rollout_state=True, max_new_tokens=3071, max_context_len=None,
            max_prompt_tokens=1024,
        )
        profile = mod.ProfileData(peak_mem_frac=0.50, throughput_tps=None, source="measured")
        recs = mod.generate_recommendations(inp, profile)
        thr = recs["throughput"]
        self.assertGreaterEqual(thr.overrides["max_running_prompts"], 64)
        self.assertTrue(thr.overrides["keep_rollout_state"])

    def test_balanced_keeps_adam_choice(self):
        mod = _load_module()
        inp = mod.RecommenderInput(
            tp_size=4, world_size=8, batch_size=32, n_samples=8, mini_bs=16,
            max_running_prompts=256, adam_8bit=True, activation_checkpointing=True,
            keep_rollout_state=True, max_new_tokens=3071, max_context_len=None,
            max_prompt_tokens=1024,
        )
        profile = mod.ProfileData(peak_mem_frac=0.82, throughput_tps=None, source="measured")
        recs = mod.generate_recommendations(inp, profile)
        bal = recs["balanced"]
        self.assertEqual(bal.overrides["adam_8bit"], True)


# ---------------------------------------------------------------------------
# Safety constraints
# ---------------------------------------------------------------------------


class TestRecommenderSafety(unittest.TestCase):
    """Semantic parameters must never be modified by any recommendation."""

    def test_preserves_semantic_token_limits(self):
        mod = _load_module()
        inp = mod.RecommenderInput(
            tp_size=4, world_size=8, batch_size=32, n_samples=8, mini_bs=16,
            max_running_prompts=256, adam_8bit=False, activation_checkpointing=True,
            keep_rollout_state=True, max_new_tokens=4096, max_context_len=8192,
            max_prompt_tokens=2048,
        )
        profile = mod.ProfileData(peak_mem_frac=0.82, throughput_tps=None, source="measured")
        recs = mod.generate_recommendations(inp, profile)
        for mode, rec in recs.items():
            self.assertNotIn("max_new_tokens", rec.overrides, f"{mode} should not override max_new_tokens")
            self.assertNotIn("max_context_len", rec.overrides, f"{mode} should not override max_context_len")
            self.assertNotIn("max_prompt_tokens", rec.overrides, f"{mode} should not override max_prompt_tokens")


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestRecommenderOutputFormat(unittest.TestCase):
    """Verify JSON and human-readable output formats."""

    def test_json_output_structure(self):
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--peak-mem-frac", "0.82", "--json",
        ])
        result = json.loads(proc.stdout)
        self.assertIn("ok", result)
        self.assertIn("input", result)
        self.assertIn("profile", result)
        self.assertIn("recommendations", result)

    def test_human_readable_contains_all_modes(self):
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--peak-mem-frac", "0.82",
        ])
        output = proc.stdout
        self.assertIn("Conservative", output)
        self.assertIn("Balanced", output)
        self.assertIn("Throughput", output)

    def test_deterministic_output(self):
        args = [
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--peak-mem-frac", "0.82", "--throughput-tps", "1200.0",
            "--json",
        ]
        proc1 = _run_cli(args)
        proc2 = _run_cli(args)
        self.assertEqual(proc1.stdout, proc2.stdout)


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------


class TestRecommenderFileExport(unittest.TestCase):
    """Override files are written when --output-dir is provided."""

    def test_writes_three_override_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = _run_cli([
                "--tp-size", "4", "--world-size", "8",
                "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
                "--peak-mem-frac", "0.82",
                "--output-dir", tmpdir,
            ])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out_dir = Path(tmpdir)
            files = sorted(out_dir.glob("capacity_override_*.json"))
            self.assertEqual(len(files), 3)
            for f in files:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.assertIn("max_running_prompts", data)
                self.assertIn("mini_bs", data)

    def test_does_not_submit_training(self):
        """The script exits 0 (or 1 for validation) but never calls a training API."""
        proc = _run_cli([
            "--tp-size", "4", "--world-size", "8",
            "--batch-size", "32", "--n-samples", "8", "--mini-bs", "16",
            "--peak-mem-frac", "0.82", "--json",
        ])
        self.assertIn(proc.returncode, (0, 1))
        self.assertNotIn("areno train", proc.stdout)
        self.assertNotIn("launching", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()