"""CPU tests for the training-scale calculator.

Run with:  pytest tests/test_training_scale_cpu.py -v
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest  # noqa: F401  -- pytest is a test-time dependency

# ---------------------------------------------------------------------------
# Import the module under test.
#
# calc_training_scale.py is a standalone script in skills/, not a proper
# Python package.  We locate it by path and import it via importlib so the
# import works regardless of whether the script is on sys.path.
# ---------------------------------------------------------------------------

_SKILL_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "areno-training-scale"
    / "scripts"
    / "calc_training_scale.py"
)

if not _SKILL_SCRIPT.exists():
    raise FileNotFoundError(
        f"Cannot find calc_training_scale.py at {_SKILL_SCRIPT}. "
        "Make sure the skill directory exists under skills/."
    )

_spec = importlib.util.spec_from_file_location("calc_training_scale", _SKILL_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load module spec from {_SKILL_SCRIPT}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["calc_training_scale"] = _mod  # needed for dataclass resolution on 3.9
_spec.loader.exec_module(_mod)

# Re-export the symbols we test, so the rest of the file uses normal names.
ScaleInput = _mod.ScaleInput
ScaleResult = _mod.ScaleResult
_ALL_ALGOS = _mod._ALL_ALGOS
_is_online_rl = _mod._is_online_rl
calculate = _mod.calculate
suggest_combinations = _mod.suggest_combinations
_validate_input = _mod._validate_input


# ===========================================================================
# Validation tests
# ===========================================================================


class TestValidation:
    """Malformed or boundary inputs must raise ValueError."""

    def test_unknown_algo(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            calculate(ScaleInput(algo="unknown", dataset_size=100))

    def test_negative_dataset_size(self):
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=-1))

    def test_zero_dataset_size(self):
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=0))

    def test_zero_mini_bs(self):
        with pytest.raises(ValueError, match="mini_bs must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, mini_bs=0))

    def test_negative_mini_bs(self):
        with pytest.raises(ValueError, match="mini_bs must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, mini_bs=-4))

    def test_zero_grad_accum(self):
        with pytest.raises(ValueError, match="gradient_accumulation_steps must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, gradient_accumulation_steps=0))

    def test_negative_grad_accum(self):
        with pytest.raises(ValueError, match="gradient_accumulation_steps must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, gradient_accumulation_steps=-1))

    def test_world_size_not_divisible_by_tp_size(self):
        with pytest.raises(ValueError, match="divisible by tp_size"):
            calculate(ScaleInput(algo="sft", dataset_size=100, world_size=7, tp_size=4))

    def test_zero_epochs(self):
        with pytest.raises(ValueError, match="epochs must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, epochs=0))

    def test_zero_avg_seq_len(self):
        with pytest.raises(ValueError, match="avg_seq_len must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, avg_seq_len=0))

    def test_rl_negative_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            calculate(ScaleInput(algo="gspo", dataset_size=100, batch_size=-1))

    def test_rl_negative_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be positive"):
            calculate(ScaleInput(algo="gspo", dataset_size=100, n_samples=-1))

    def test_negative_target_global_batch(self):
        with pytest.raises(ValueError, match="target_global_batch must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, target_global_batch=-1))

    def test_zero_world_size(self):
        with pytest.raises(ValueError, match="world_size must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, world_size=0))

    def test_zero_tp_size(self):
        with pytest.raises(ValueError, match="tp_size must be positive"):
            calculate(ScaleInput(algo="sft", dataset_size=100, tp_size=0))


# ===========================================================================
# SFT calculation tests
# ===========================================================================


class TestSFT:
    """SFT: 1 row = 1 sample."""

    def test_basic(self):
        """Simple SFT: 10k rows, mini_bs=16, grad_accum=2, 2 GPUs (dp=2)."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=10000,
            mini_bs=16,
            gradient_accumulation_steps=2,
            world_size=8,
            tp_size=4,
            epochs=1,
        ))
        assert r.dp_size == 2
        assert r.global_batch == 64  # 16 * 2 * 2
        assert r.gradient_accumulation_steps == 2
        assert r.samples_per_step == 64
        assert r.updates_per_epoch == math.ceil(10000 / 64)
        assert r.total_updates == r.updates_per_epoch * 1

    def test_default_grad_accum(self):
        """When gradient_accumulation_steps is None, default to 1."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=500,
            mini_bs=8,
            world_size=1,
            tp_size=1,
        ))
        assert r.gradient_accumulation_steps == 1
        assert r.global_batch == 8  # 8 * 1 * 1

    def test_multi_epoch(self):
        """3 epochs."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=1000,
            mini_bs=16,
            gradient_accumulation_steps=1,
            world_size=8,
            tp_size=4,
            epochs=3,
        ))
        assert r.total_updates == r.updates_per_epoch * 3

    def test_uneven_dataset_warning(self):
        """Dataset not divisible by global_batch produces a warning."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=1000,
            mini_bs=16,
            gradient_accumulation_steps=2,
            world_size=8,
            tp_size=4,
        ))
        assert r.global_batch == 64
        assert 1000 % 64 != 0
        assert any("not divisible" in w for w in r.warnings)

    def test_exact_division_no_warning(self):
        """Dataset exactly divisible by global_batch: no warning."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=64,
            mini_bs=16,
            gradient_accumulation_steps=2,
            world_size=8,
            tp_size=4,
        ))
        assert r.global_batch == 64
        assert not any("not divisible" in w for w in r.warnings)

    def test_token_estimation(self):
        """approx_tokens = total_updates * samples_per_step * avg_seq_len."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=64,
            mini_bs=16,
            gradient_accumulation_steps=1,
            world_size=1,
            tp_size=1,
            avg_seq_len=1024,
        ))
        assert r.global_batch == 16
        assert r.updates_per_epoch == 4  # 64 / 16
        assert r.approx_tokens == 4 * 16 * 1024  # 65536


# ===========================================================================
# DPO calculation tests
# ===========================================================================


class TestDPO:
    """DPO: 1 row = 1 chosen/rejected pair. Counting is same as SFT."""

    def test_basic(self):
        """DPO uses same counting as SFT for batch/steps."""
        r = calculate(ScaleInput(
            algo="dpo",
            dataset_size=5000,
            mini_bs=16,
            gradient_accumulation_steps=1,
            world_size=8,
            tp_size=4,
        ))
        assert r.dp_size == 2
        assert r.global_batch == 32  # 16 * 1 * 2
        assert r.updates_per_epoch == math.ceil(5000 / 32)

    def test_dpo_is_offline(self):
        """DPO is not online RL."""
        assert not _is_online_rl("dpo")


# ===========================================================================
# Online RL (GSPO/GRPO/PPO) calculation tests
# ===========================================================================


class TestOnlineRL:
    """Online RL: 1 row = 1 prompt with n_samples rollouts."""

    def test_gspo_basic(self):
        """GSPO: 500 prompts, batch_size=32, n_samples=8."""
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=500,
            batch_size=32,
            n_samples=8,
            mini_bs=16,
            world_size=8,
            tp_size=4,
        ))
        assert r.dp_size == 2
        assert r.global_batch == 32  # prompt-level batch
        assert r.samples_per_step == 256  # 32 * 8
        assert r.updates_per_epoch == math.ceil(500 / 32)

    def test_grpo_basic(self):
        """GRPO: same counting as GSPO."""
        r = calculate(ScaleInput(
            algo="grpo",
            dataset_size=1000,
            batch_size=16,
            n_samples=4,
            mini_bs=8,
            world_size=2,
            tp_size=1,
        ))
        assert r.global_batch == 16
        assert r.samples_per_step == 64  # 16 * 4

    def test_ppo_basic(self):
        """PPO: same counting as GSPO/GRPO."""
        r = calculate(ScaleInput(
            algo="ppo",
            dataset_size=2000,
            batch_size=64,
            n_samples=1,
            mini_bs=16,
            world_size=8,
            tp_size=4,
        ))
        assert r.global_batch == 64
        assert r.samples_per_step == 64  # 64 * 1

    def test_rl_default_batch_size(self):
        """When batch_size is None, default to 32."""
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=1000,
            n_samples=8,
            mini_bs=16,
            world_size=8,
            tp_size=4,
        ))
        assert r.global_batch == 32

    def test_rl_default_n_samples(self):
        """When n_samples is None, default to 8."""
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=1000,
            batch_size=32,
            mini_bs=16,
            world_size=8,
            tp_size=4,
        ))
        assert r.samples_per_step == 256  # 32 * 8

    def test_rl_token_estimation(self):
        """approx_tokens = total_updates * samples_per_step * avg_seq_len."""
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=32,
            batch_size=32,
            n_samples=8,
            mini_bs=16,
            world_size=8,
            tp_size=4,
            avg_seq_len=2048,
        ))
        assert r.updates_per_epoch == 1
        assert r.samples_per_step == 256
        assert r.approx_tokens == 1 * 256 * 2048  # 524288

    def test_rl_uneven_dataset_warning(self):
        """Dataset not divisible by batch_size produces a warning."""
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=100,
            batch_size=32,
            n_samples=8,
            mini_bs=16,
            world_size=8,
            tp_size=4,
        ))
        assert any("not divisible" in w for w in r.warnings)

    def test_rl_grad_accum_auto_covers_rollout(self):
        """Default grad_accum should cover the full rollout in one optimizer step."""
        # batch_size=32, n_samples=8 => 256 sequences
        # mini_bs=16, dp_size=2 => 32 per grad_accum step
        # Need 256/32 = 8 grad_accum steps
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=1000,
            batch_size=32,
            n_samples=8,
            mini_bs=16,
            world_size=8,
            tp_size=4,
        ))
        assert r.gradient_accumulation_steps == 8

    def test_rl_grad_accum_mismatch_warning(self):
        """If grad_accum doesn't cover the full rollout, emit a warning."""
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=1000,
            batch_size=32,
            n_samples=8,
            mini_bs=16,
            gradient_accumulation_steps=1,  # too small: 16*1*2=32 != 256
            world_size=8,
            tp_size=4,
        ))
        assert any("does not equal" in w for w in r.warnings)

    def test_is_online_rl(self):
        """Verify algorithm classification."""
        assert _is_online_rl("gspo")
        assert _is_online_rl("grpo")
        assert _is_online_rl("ppo")
        assert not _is_online_rl("sft")
        assert not _is_online_rl("dpo")


# ===========================================================================
# Reverse solve (target_global_batch) tests
# ===========================================================================


class TestReverseSolve:
    """Solve gradient_accumulation for a target global batch."""

    def test_sft_exact_division(self):
        """target=128, mini_bs=16, dp=2 => grad_accum=4."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=1000,
            mini_bs=16,
            world_size=8,
            tp_size=4,
            target_global_batch=128,
        ))
        assert r.gradient_accumulation_steps == 4  # 128 / (16 * 2)
        assert r.global_batch == 128

    def test_sft_uneven_division_warning(self):
        """target=100, mini_bs=16, dp=2 => 100/32 not integer => warning."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=1000,
            mini_bs=16,
            world_size=8,
            tp_size=4,
            target_global_batch=100,
        ))
        assert r.gradient_accumulation_steps == 4  # ceil(100/32) = 4
        assert r.global_batch == 128  # 16 * 4 * 2
        assert any("not divisible" in w for w in r.warnings)

    def test_rl_reverse_solve(self):
        """RL: target_global_batch refers to prompt batch size."""
        # target=64 prompts, n_samples=8 => 512 sequences
        # mini_bs=16, dp=2 => 32 per grad_accum step
        # Need 512/32 = 16 grad_accum steps
        r = calculate(ScaleInput(
            algo="gspo",
            dataset_size=1000,
            mini_bs=16,
            world_size=8,
            tp_size=4,
            n_samples=8,
            target_global_batch=64,
        ))
        assert r.global_batch == 64
        assert r.gradient_accumulation_steps == 16
        assert r.samples_per_step == 512  # 64 * 8

    def test_target_overrides_grad_accum(self):
        """target_global_batch takes precedence over gradient_accumulation_steps."""
        r = calculate(ScaleInput(
            algo="sft",
            dataset_size=1000,
            mini_bs=16,
            gradient_accumulation_steps=99,  # should be overridden
            world_size=8,
            tp_size=4,
            target_global_batch=64,
        ))
        assert r.gradient_accumulation_steps == 2  # 64 / (16 * 2)
        assert r.global_batch == 64


# ===========================================================================
# Suggest combinations tests
# ===========================================================================


class TestSuggestCombinations:
    """suggest_combinations returns valid mini_bs/dp_size/grad_accum combos."""

    def test_target_64(self):
        combos = suggest_combinations(target_global_batch=64)
        # 64 = 8*1*8, 8*2*4, 8*4*2, 8*8*1, 16*1*4, 16*2*2, 16*4*1,
        #      32*1*2, 32*2*1, 64*1*1
        assert len(combos) > 0
        for c in combos:
            assert c["global_batch"] == 64
            assert c["mini_bs"] * c["dp_size"] * c["gradient_accumulation_steps"] == 64

    def test_target_prime(self):
        """A prime target has no combos with standard mini_bs/dp_size values."""
        combos = suggest_combinations(target_global_batch=17)
        assert len(combos) == 0

    def test_sorted_by_grad_accum(self):
        combos = suggest_combinations(target_global_batch=64)
        ga_values = [c["gradient_accumulation_steps"] for c in combos]
        assert ga_values == sorted(ga_values)


# ===========================================================================
# Deterministic output tests
# ===========================================================================


class TestDeterministic:
    """Same input always produces same output."""

    def test_sft_deterministic(self):
        inp = ScaleInput(algo="sft", dataset_size=999, mini_bs=16, world_size=2, tp_size=1)
        r1 = calculate(inp)
        r2 = calculate(inp)
        assert r1 == r2

    def test_gspo_deterministic(self):
        inp = ScaleInput(
            algo="gspo", dataset_size=500, batch_size=32, n_samples=8,
            mini_bs=16, world_size=8, tp_size=4,
        )
        r1 = calculate(inp)
        r2 = calculate(inp)
        assert r1 == r2


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestCLI:
    """Run the script as a subprocess and check output."""

    @pytest.fixture
    def script_path(self):
        """Find the script at its canonical location in the skills directory."""
        p = (
            Path(__file__).resolve().parent.parent
            / "skills"
            / "areno-training-scale"
            / "scripts"
            / "calc_training_scale.py"
        )
        if not p.exists():
            pytest.skip("calc_training_scale.py not found in skills/")
        return p

    def test_sft_json_output(self, script_path):
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--algo", "sft", "--dataset-size", "1000",
             "--mini-bs", "16", "--world-size", "1", "--tp-size", "1",
             "--json"],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        assert data["algo"] == "sft"
        assert data["dp_size"] == 1
        assert data["global_batch"] == 16

    def test_gspo_json_output(self, script_path):
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--algo", "gspo", "--dataset-size", "500",
             "--batch-size", "32", "--n-samples", "8",
             "--mini-bs", "16", "--world-size", "8", "--tp-size", "4",
             "--json"],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        assert data["algo"] == "gspo"
        assert data["global_batch"] == 32
        assert data["samples_per_step"] == 256

    def test_target_global_batch_json(self, script_path):
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--algo", "sft", "--dataset-size", "1000",
             "--mini-bs", "16", "--world-size", "8", "--tp-size", "4",
             "--target-global-batch", "128", "--json"],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        assert data["gradient_accumulation_steps"] == 4
        assert data["global_batch"] == 128

    def test_suggest_mode(self, script_path):
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--suggest", "--target-global-batch", "64", "--json"],
            capture_output=True, text=True, check=True,
        )
        combos = json.loads(result.stdout)
        assert len(combos) > 0
        for c in combos:
            assert c["global_batch"] == 64

    def test_missing_dataset_size_error(self, script_path):
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--algo", "sft"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_human_readable_output(self, script_path):
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--algo", "sft", "--dataset-size", "1000",
             "--mini-bs", "16", "--world-size", "2", "--tp-size", "1"],
            capture_output=True, text=True, check=True,
        )
        assert "AReno Training Scale" in result.stdout
        assert "global_batch" in result.stdout