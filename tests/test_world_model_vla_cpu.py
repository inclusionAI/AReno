from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from areno.experimental.world_model_vla.assets import create_snapshot_manifest, verify_snapshot
from areno.experimental.world_model_vla.backend import RlinfWanBackend
from areno.experimental.world_model_vla.cli import main
from areno.experimental.world_model_vla.config import SlurmResources, WorldModelVLAConfig
from areno.experimental.world_model_vla.metrics import parse_metric_log
from areno.experimental.world_model_vla.slurm import SlurmLauncher
from areno.experimental.world_model_vla.state import WorkflowStateStore
from areno.experimental.world_model_vla.workflow import WorldModelVLAWorkflow


def _write_snapshot(path: Path, content: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "weights.bin").write_bytes(content)
    (path / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "weights.bin",
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _config(tmp_path: Path, *, stage_ends: tuple[int, ...] = (2, 10)) -> WorldModelVLAConfig:
    rlinf = tmp_path / "rlinf"
    python = rlinf / ".venv" / "bin" / "python"
    ray = python.with_name("ray")
    (rlinf / ".venv" / "wan").mkdir(parents=True, exist_ok=True)
    runtime_python = rlinf / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.touch()
    runtime_python.chmod(0o755)
    python.parent.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        python.symlink_to(runtime_python)
    train_config = rlinf / "examples" / "embodiment" / "config"
    compatibility_files = {
        rlinf / "rlinf" / "hybrid_engines" / "fsdp" / "strategy" / "fsdp.py": 'get("sync_module_states", True)',
        rlinf
        / "rlinf"
        / "models"
        / "embodiment"
        / "openvla_oft"
        / "rlinf"
        / "openvla_oft_action_model.py": "_supports_sdpa = True",
        rlinf
        / "rlinf"
        / "models"
        / "embodiment"
        / "openvla_oft"
        / "rlinf"
        / "__init__.py": 'attn_implementation="sdpa"',
        rlinf / "rlinf" / "runners" / "embodied_runner.py": "self.rollout.init_worker().wait()",
    }
    for path, content in compatibility_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for path in (
        ray,
        rlinf / "examples" / "embodiment" / "train_embodied_agent.py",
        rlinf / "evaluations" / "eval_embodied_agent.py",
        train_config / "wan_libero_spatial_grpo_openvlaoft.yaml",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    ray.chmod(0o755)

    world_model = tmp_path / "world-model"
    policy = tmp_path / "policy"
    _write_snapshot(world_model, b"world")
    _write_snapshot(policy, b"policy")
    return WorldModelVLAConfig(
        rlinf_root=rlinf,
        rlinf_python=python,
        world_model_path=world_model,
        policy_path=policy,
        output_dir=tmp_path / "output",
        cache_root=tmp_path / "cache",
        stage_ends=stage_ends,
        eval_trajectories=4,
        require_rlinf_compatibility_patchset=True,
        slurm=SlurmResources(
            partition="gpu",
            train_qos="train",
            eval_qos="short",
            constraint="test-gpu",
            gpus=2,
            cpus=8,
            memory="32G",
        ),
    )


def test_config_validates_stages_and_resolves_json_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "workflow.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "rlinf_root": "../rlinf",
                "rlinf_python": "../rlinf/python",
                "world_model_path": "../world",
                "policy_path": "../policy",
                "output_dir": "../output",
                "cache_root": "../cache",
                "stage_ends": [2, 5],
                "eval_trajectories": 4,
                "slurm": {"partition": "gpu", "gpus": 2},
            }
        ),
        encoding="utf-8",
    )

    config = WorldModelVLAConfig.from_json(config_path)

    assert config.rlinf_root == (tmp_path / "rlinf").resolve()
    assert config.stage_ends == (2, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        _config(tmp_path / "invalid", stage_ends=(2, 2))
    with pytest.raises(ValueError, match="divisible"):
        WorldModelVLAConfig(
            **{
                **config.to_dict(),
                "eval_trajectories": 3,
                "slurm": SlurmResources(partition="gpu", gpus=2),
            }
        )
    with pytest.raises(ValueError, match="slurm.gpus"):
        WorldModelVLAConfig(
            **{
                **config.to_dict(),
                "slurm": SlurmResources(partition="gpu", gpus=0),
            }
        )
    with pytest.raises(ValueError, match="require_rlinf_compatibility_patchset"):
        replace(config, require_rlinf_compatibility_patchset="false")

    config_path.write_text(json.dumps({"output_dir": "output"}), encoding="utf-8")
    with pytest.raises(ValueError, match="slurm"):
        WorldModelVLAConfig.from_json(config_path)


def test_snapshot_manifest_checks_size_hash_and_unsafe_paths(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_snapshot(snapshot, b"verified")

    assert verify_snapshot(snapshot, check_hashes=True).ok
    (snapshot / "weights.bin").write_bytes(b"corrupt!")
    result = verify_snapshot(snapshot, check_hashes=True)
    assert not result.ok
    assert "sha256 mismatch" in result.errors[0]

    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps({"files": [{"path": "../escape", "size": 1}]}), encoding="utf-8"
    )
    result = verify_snapshot(snapshot)
    assert not result.ok
    assert "unsafe manifest path" in result.errors[0]


def test_snapshot_manifest_creation_hashes_files_and_protects_existing(tmp_path: Path) -> None:
    snapshot = tmp_path / "downloaded-model"
    (snapshot / "nested").mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "nested" / "weights.bin").write_bytes(b"weights")

    created = create_snapshot_manifest(snapshot)

    assert created.files == 2
    assert created.total_bytes == 9
    assert created.includes_sha256
    assert verify_snapshot(snapshot, check_hashes=True).ok
    with pytest.raises(FileExistsError, match="already exists"):
        create_snapshot_manifest(snapshot)

    replaced = create_snapshot_manifest(snapshot, include_hashes=False, overwrite=True)
    assert not replaced.includes_sha256

    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="contains no files"):
        create_snapshot_manifest(tmp_path / "empty")


def test_backend_builds_staged_train_and_real_eval_commands(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = RlinfWanBackend(config)
    first, second = backend.stages()

    assert first.start_step == 0
    assert first.resume_dir is None
    train = backend.build_train_command(second)
    assert "runner.max_steps=10" in train
    assert "runner.save_interval=10" in train
    assert f"runner.resume_dir={backend.checkpoint_dir(2)}" in train
    assert "actor.micro_batch_size=1" in train
    assert "actor.fsdp_config.limit_all_gathers=true" in train
    assert "++actor.fsdp_config.sync_module_states=false" in train
    assert "++weight_syncer.patch.transport_device=cpu" in train

    stock_backend = RlinfWanBackend(replace(config, require_rlinf_compatibility_patchset=False))
    assert "++actor.fsdp_config.sync_module_states=false" not in stock_backend.build_train_command(first)

    checkpoint_2 = backend.full_weights_path(2)
    checkpoint_10 = backend.full_weights_path(10)
    checkpoint_2.parent.mkdir(parents=True)
    checkpoint_2.touch()
    checkpoint_10.parent.mkdir(parents=True)
    checkpoint_10.touch()

    assert backend.latest_full_weights() == checkpoint_10
    evaluation = backend.build_eval_command()
    assert f"runner.ckpt_path={checkpoint_10}" in evaluation
    assert "env.eval.total_num_envs=4" in evaluation
    assert "env.eval.use_fixed_reset_state_ids=true" in evaluation


def test_runtime_compatibility_and_cache_environment(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = RlinfWanBackend(config)

    assert all(check.ok for check in backend.runtime_checks())
    assert config.rlinf_python == config.rlinf_root / ".venv" / "bin" / "python"
    assert backend.ray_executable == config.rlinf_root / ".venv" / "bin" / "ray"
    environment = backend.environment()
    assert environment["HF_HOME"].startswith(str(config.cache_root))
    assert environment["MODELSCOPE_CACHE"].startswith(str(config.cache_root))
    assert environment["TORCH_HOME"].startswith(str(config.cache_root))
    assert environment["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(config.rlinf_python.parent.parent / "wan"),
        str(config.rlinf_root),
    ]

    patched_file = config.rlinf_root / "rlinf" / "runners" / "embodied_runner.py"
    patched_file.unlink()
    assert not all(check.ok for check in backend.runtime_checks())
    stock_checks = RlinfWanBackend(replace(config, require_rlinf_compatibility_patchset=False)).runtime_checks()
    assert all(check.ok for check in stock_checks)


def test_metric_parser_uses_latest_rich_table_values(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.log"
    metrics.write_text(
        "\x1b[32m│ Global Step: 1/10 │\x1b[0m\n│success_once=0.125 │\n│ Global Step: 2/10 │\n│success_once=7.75e-1 │\n",
        encoding="utf-8",
    )

    parsed = parse_metric_log(metrics)

    assert parsed.global_step == 2
    assert parsed.total_steps == 10
    assert parsed.success_once == pytest.approx(0.775)


def test_slurm_commands_include_resources_dependency_and_worker(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "workflow.json"
    launcher = SlurmLauncher(config, config_path)

    stage = launcher.build_stage_command(2, dependency="afterok:123")
    evaluation = launcher.build_eval_command(dependency="afterok:456")

    assert stage[0] == "sbatch"
    assert "--gres=gpu:2" in stage
    assert "--dependency=afterok:123" in stage
    assert str(launcher.worker_script) in stage
    assert sys.executable in stage
    assert stage[-5:] == ["--config", str(config_path.resolve()), "--end-step", "2", "--auto-continue"]
    assert "--qos=short" in evaluation
    assert "--dependency=afterok:456" in evaluation
    assert evaluation[-3:] == ["run-eval", "--config", str(config_path.resolve())]

    portable = replace(
        config,
        slurm=SlurmResources(partition="gpu", gpus=2, cpus=8, memory="32G"),
    )
    portable_command = SlurmLauncher(portable, config_path).build_stage_command(2)
    assert not any(value.startswith("--qos=") for value in portable_command)
    assert not any(value.startswith("--constraint=") for value in portable_command)
    assert not any(value.startswith("--account=") for value in portable_command)


def test_state_store_is_atomic_and_rejects_stage_plan_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state = WorkflowStateStore(config.output_dir)

    initialized = state.ensure(config)
    updated = state.update(lambda value: {**value, "marker": 1})

    assert initialized["stage_ends"] == [2, 10]
    assert updated["marker"] == 1
    assert json.loads(state.path.read_text(encoding="utf-8"))["marker"] == 1
    changed = _config(tmp_path, stage_ends=(2, 20))
    with pytest.raises(RuntimeError, match="stage_ends"):
        state.ensure(changed)


def test_workflow_verifies_assets_and_prevents_duplicate_submit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workflow = WorldModelVLAWorkflow(config)

    assert workflow.verify(check_hashes=True)["ok"]
    with patch.object(SlurmLauncher, "submit", return_value="123"):
        job_id, command = workflow.submit_initial()
    assert job_id == "123"
    assert command[-1] == "--auto-continue"

    with (
        patch.object(SlurmLauncher, "submit", return_value="456"),
        pytest.raises(RuntimeError, match="already submitted"),
    ):
        workflow.submit_initial()


def test_completed_stage_is_reused_without_starting_ray(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workflow = WorldModelVLAWorkflow(config)
    checkpoint = workflow.backend.full_weights_path(2)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    with patch.object(workflow, "_run_with_ray") as run:
        assert workflow.run_stage(2) == checkpoint

    run.assert_not_called()
    assert workflow.state.read()["stages"]["2"]["reused"] is True


def test_resume_rejects_noncontiguous_or_incomplete_checkpoints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workflow = WorldModelVLAWorkflow(config)

    later_weights = workflow.backend.full_weights_path(10)
    later_weights.parent.mkdir(parents=True)
    later_weights.touch()
    with pytest.raises(RuntimeError, match="non-contiguous"):
        workflow.submit_resume(dry_run=True)

    later_weights.unlink()
    workflow.backend.checkpoint_dir(2).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="complete resume checkpoint"):
        workflow.run_stage(10)


def test_internal_cli_plan_and_auto_continue_guard(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "workflow.json"
    config.write_json(config_path)

    assert main(["plan", "--config", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["roles"] == [
        "world_model_env",
        "policy_rollout",
        "fsdp_actor",
    ]

    with patch.dict("os.environ", {}, clear=True):
        assert (
            main(
                [
                    "run-stage",
                    "--config",
                    str(config_path),
                    "--end-step",
                    "2",
                    "--auto-continue",
                ]
            )
            == 1
        )
    assert "requires SLURM_JOB_ID" in capsys.readouterr().err


def test_internal_cli_creates_snapshot_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.bin").write_bytes(b"weights")

    assert main(["manifest", "--snapshot", str(snapshot)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["files"] == 1
    assert report["includes_sha256"] is True
    assert Path(report["path"]).is_file()
