from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_repository_agent_skills_are_valid():
    """Project skills should retain valid metadata, links, and script entrypoints."""

    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(root / ".agents/scripts/validate_skills.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        raise AssertionError(
            f"validate_skills.py failed with exit code {process.returncode}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    result = json.loads(process.stdout)
    assert result["skill_count"] == 10
    assert result["script_count"] >= 15


def test_transcript_validator_accepts_normalized_argument_objects(tmp_path):
    root = Path(__file__).resolve().parents[1]
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "guess_code", "arguments": {"code": "0123"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": '{"solved":true}'},
                ]
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [
            sys.executable,
            str(root / ".agents/skills/areno-build-agentic-workflow/scripts/validate_transcript.py"),
            str(transcript),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout)["ok"] is True


# ---------------------------------------------------------------------------
# generate_recipe.py tests
# ---------------------------------------------------------------------------

_RECIPE_SCRIPT = ".agents/skills/areno-run-training/scripts/generate_recipe.py"
_VALID_MODES = ("sft", "dpo", "gspo", "grpo", "ppo")
_ROLLOUT_MODES = {"gspo", "grpo", "ppo"}


def _run_recipe(root: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(root / _RECIPE_SCRIPT), *extra_args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_generate_recipe_valid_modes():
    """The recipe generator should produce valid output for all five modes."""

    root = Path(__file__).resolve().parents[1]
    for mode in _VALID_MODES:
        proc = _run_recipe(
            root,
            "--mode",
            mode,
            "--gpu-count",
            "2",
            "--context-length",
            "4096",
            "--target-batch",
            "8",
        )
        assert proc.returncode == 0, f"mode={mode} failed: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert data["ok"] is True, f"mode={mode} ok is False"
        assert data["mode"] == mode
        assert data["command"].startswith("areno train")
        assert f"--algo {mode}" in data["command"]
        # Rollout modes should include n_samples; non-rollout should not.
        if mode in _ROLLOUT_MODES:
            assert "n_samples" in data["recipe"]
            assert "--reward-fn-path" in data["command"]
        else:
            assert "n_samples" not in data["recipe"]
        # PPO should include clip_eps and critic fields.
        if mode == "ppo":
            assert "clip_eps" in data["recipe"]
            assert "critic_lr" in data["recipe"]
            assert "gamma" in data["recipe"]
            assert "lam" in data["recipe"]
            assert "critic_warmup_steps" in data["recipe"]
        # DPO should include dpo_beta.
        if mode == "dpo":
            assert "dpo_beta" in data["recipe"]
        # GSPO should include gspo_clip_eps.
        if mode == "gspo":
            assert "gspo_clip_eps" in data["recipe"]
        # GRPO should include grpo_clip_eps.
        if mode == "grpo":
            assert "grpo_clip_eps" in data["recipe"]


def test_generate_recipe_invalid_mode():
    """An invalid mode should be rejected by argparse (non-zero exit)."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "invalid",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
    )
    assert proc.returncode != 0


def test_generate_recipe_zero_gpu():
    """Zero GPU count should produce a validation error."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "0",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
    )
    assert proc.returncode == 1
    data = json.loads(proc.stdout)
    assert data["ok"] is False
    assert any("gpu_count" in err for err in data["errors"])


def test_generate_recipe_small_gpu_defaults():
    """Small GPU count (<=2) should reduce tp_size and mini_bs."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "1",
        "--context-length",
        "2048",
        "--target-batch",
        "4",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["recipe"]["tp_size"] == 1
    assert data["recipe"]["mini_bs"] <= 4
    assert data["recipe"]["score_micro_bs"] <= 4
    prov = data["provenance"]
    assert any("small" in v.lower() or "<= 2" in v for v in prov.values())


def test_generate_recipe_ppo_has_all_fields():
    """PPO recipe should include all PPO-specific fields and a reward-fn-path flag."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "ppo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    recipe = data["recipe"]
    for field in ("clip_eps", "critic_lr", "gamma", "lam", "critic_warmup_steps", "use_kl_loss", "kl_loss_coef"):
        assert field in recipe, f"PPO recipe missing {field}"
    assert "--reward-fn-path" in data["command"]
    assert "--ref-ckpt" in data["command"]


def test_generate_recipe_sft_has_loader():
    """SFT recipe command should include --dataset-loader-fn."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "sft",
        "--gpu-count",
        "2",
        "--context-length",
        "2048",
        "--target-batch",
        "4",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "--dataset-loader-fn" in data["command"]


def test_generate_recipe_provenance_covers_all_fields():
    """Every recipe field should have a corresponding provenance entry."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    recipe_keys = set(data["recipe"].keys())
    provenance_keys = set(data["provenance"].keys())
    # Provenance may contain supplementary keys like _dp_size, but every recipe
    # key must be covered.
    missing = recipe_keys - provenance_keys
    assert not missing, f"Recipe fields without provenance: {missing}"


def test_generate_recipe_override_works():
    """The --override flag should set the value and update provenance."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
        "--override",
        "optimizer_lr=2e-5",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["recipe"]["optimizer_lr"] == 2e-5
    assert data["provenance"]["optimizer_lr"] == "user override"


def test_generate_recipe_command_is_concise():
    """The generated command should only contain required fields, not all defaults."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    cmd = data["command"]
    # Required fields should be present.
    assert "--algo" in cmd
    assert "--tp-size" in cmd
    assert "--world-size" in cmd
    assert "--batch-size" in cmd
    assert "--mini-bs" in cmd
    assert "--n-samples" in cmd
    # Non-required defaults should NOT be in the command.
    assert "--epochs" not in cmd
    assert "--weight-decay" not in cmd
    assert "--grad-clip-norm" not in cmd
    assert "--lr-decay-steps" not in cmd


def test_generate_recipe_command_includes_user_overrides():
    """User-overridden non-required fields should appear in the command."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
        "--override",
        "epochs=5",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["recipe"]["epochs"] == 5
    assert "--epochs 5" in data["command"]


def test_generate_recipe_sft_context_split():
    """SFT should allocate all context to prompt, zero to generation."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "sft",
        "--gpu-count",
        "2",
        "--context-length",
        "2048",
        "--target-batch",
        "4",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["recipe"]["max_prompt_tokens"] == 2048
    assert data["recipe"]["max_new_tokens"] == 0


def test_generate_recipe_rl_context_split():
    """RL modes should split context 25% prompt / 75% generation."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    # 25% of 4096 = 1024, capped at 1024.
    assert data["recipe"]["max_prompt_tokens"] == 1024
    # 75% of 4096 = 3072, capped at 3071.
    assert data["recipe"]["max_new_tokens"] == 3071


def test_generate_recipe_memory_estimation():
    """Memory estimation should work when --ckpt is provided."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
        "--ckpt",
        "Qwen/Qwen3-0.6B",
        "--gpu-type",
        "T4",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "memory" in data
    mem = data["memory"]
    assert mem["weights_bytes"] > 0
    assert mem["optimizer_bytes"] > 0
    assert mem["kv_cache_bytes"] > 0
    assert mem["total_estimated_bytes"] > 0
    assert mem["headroom_ok"] is False  # 0.6B on T4 with batch=8 should OOM
    assert any("OOM" in w or "exceeds" in w for w in data["warnings"])


def test_generate_recipe_memory_no_ckpt():
    """Memory estimation should be absent when --ckpt is not provided."""

    root = Path(__file__).resolve().parents[1]
    proc = _run_recipe(
        root,
        "--mode",
        "gspo",
        "--gpu-count",
        "2",
        "--context-length",
        "4096",
        "--target-batch",
        "8",
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert "memory" not in data
