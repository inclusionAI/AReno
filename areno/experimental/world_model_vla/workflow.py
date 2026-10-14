"""Restartable lifecycle for world-model-driven VLA post-training."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from areno.experimental.world_model_vla.assets import AssetVerification, verify_snapshot
from areno.experimental.world_model_vla.backend import RlinfWanBackend, TrainingStage
from areno.experimental.world_model_vla.config import WorldModelVLAConfig
from areno.experimental.world_model_vla.metrics import parse_metric_log
from areno.experimental.world_model_vla.slurm import SlurmLauncher
from areno.experimental.world_model_vla.state import WorkflowStateStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorldModelVLAWorkflow:
    """Coordinate validation, execution, recovery, and real evaluation."""

    def __init__(self, config: WorldModelVLAConfig) -> None:
        self.config = config
        self.backend = RlinfWanBackend(config)
        self.state = WorkflowStateStore(config.output_dir)

    @property
    def persisted_config_path(self) -> Path:
        return self.config.output_dir / "workflow_config.json"

    def plan(self) -> dict[str, Any]:
        return {
            "backend": "rlinf_wan",
            "roles": ["world_model_env", "policy_rollout", "fsdp_actor"],
            "stages": [
                {
                    "start_step": stage.start_step,
                    "end_step": stage.end_step,
                    "resume_dir": None if stage.resume_dir is None else str(stage.resume_dir),
                    "checkpoint_dir": str(stage.checkpoint_dir),
                }
                for stage in self.backend.stages()
            ],
            "evaluation": {
                "trajectories": self.config.eval_trajectories,
                "metric": "eval/success_once",
                "target": self.config.target_success_rate,
            },
        }

    def verify(self, *, check_hashes: bool | None = None) -> dict[str, Any]:
        hash_check = self.config.verify_sha256 if check_hashes is None else check_hashes
        runtime = self.backend.runtime_checks()
        assets = (
            verify_snapshot(self.config.world_model_path, check_hashes=hash_check),
            verify_snapshot(self.config.policy_path, check_hashes=hash_check),
        )
        errors = [f"{check.name}: {check.detail}" for check in runtime if not check.ok]
        for result in assets:
            errors.extend(f"{result.name}: {error}" for error in result.errors)
        report = {
            "runtime": [{"name": check.name, "ok": check.ok, "detail": check.detail} for check in runtime],
            "assets": [self._asset_report(result) for result in assets],
            "sha256": hash_check,
            "ok": not errors,
            "errors": errors,
        }
        return report

    def require_valid(self, *, check_hashes: bool | None = None) -> dict[str, Any]:
        report = self.verify(check_hashes=check_hashes)
        if not report["ok"]:
            formatted = "\n".join(f"- {error}" for error in report["errors"])
            raise RuntimeError(f"world-model VLA preflight failed:\n{formatted}")
        return report

    def persist_config(self) -> Path:
        self.backend.prepare_directories()
        if self.persisted_config_path.is_file():
            persisted = json.loads(self.persisted_config_path.read_text(encoding="utf-8"))
            if persisted != self.config.to_dict():
                raise RuntimeError(f"configuration differs from the existing workflow: {self.persisted_config_path}")
            return self.persisted_config_path
        self.config.write_json(self.persisted_config_path)
        return self.persisted_config_path

    def submit_initial(self, *, dry_run: bool = False, force: bool = False) -> tuple[str | None, list[str]]:
        self.require_valid()
        config_path = self.persist_config()
        self.state.ensure(self.config)
        launcher = SlurmLauncher(self.config, config_path)
        command = launcher.build_stage_command(self.config.stage_ends[0])
        if dry_run:
            return None, command
        key = f"train_{self.config.stage_ends[0]}"
        self._require_submission_available(key, force=force)
        job_id = launcher.submit(command)
        self._record_job(key, job_id)
        return job_id, command

    def submit_resume(self, *, dry_run: bool = False, force: bool = False) -> tuple[str | None, list[str]]:
        self.require_valid()
        config_path = self.persist_config()
        self.state.ensure(self.config)
        completed = self._completed_stage_prefix()
        if completed and completed[-1] == self.config.stage_ends[-1]:
            launcher = SlurmLauncher(self.config, config_path)
            command = launcher.build_eval_command()
            key = "eval"
        else:
            start_index = self.config.stage_ends.index(completed[-1]) + 1 if completed else 0
            end_step = self.config.stage_ends[start_index]
            launcher = SlurmLauncher(self.config, config_path)
            command = launcher.build_stage_command(end_step)
            key = f"train_{end_step}"
        if dry_run:
            return None, command
        self._require_submission_available(key, force=force)
        job_id = launcher.submit(command)
        self._record_job(key, job_id)
        return job_id, command

    def run_stage(self, end_step: int) -> Path:
        self.require_valid()
        self.backend.prepare_directories()
        self.state.ensure(self.config)
        stage = self.backend.stage(end_step)
        weights = self.backend.full_weights_path(end_step)
        if weights.is_file():
            self._record_stage(stage, status="completed", checkpoint=str(weights), reused=True)
            return weights
        if stage.resume_dir is not None:
            resume_weights = self.backend.full_weights_path(stage.start_step)
            if not stage.resume_dir.is_dir() or not resume_weights.is_file():
                raise FileNotFoundError(
                    f"complete resume checkpoint not found: {stage.resume_dir} "
                    f"(expected actor weights at {resume_weights})"
                )
        self._record_stage(stage, status="running")
        try:
            self._run_with_ray(self.backend.build_train_command(stage), label=f"train-{end_step}")
            if not weights.is_file():
                raise RuntimeError(f"stage {end_step} exited without full actor weights: {weights}")
            metrics = parse_metric_log(self.config.output_dir / "metrics.log")
            self._record_stage(stage, status="completed", metrics=metrics.to_dict(), checkpoint=str(weights))
            return weights
        except Exception as error:
            self._record_stage(stage, status="failed", error=f"{type(error).__name__}: {error}")
            raise

    def run_evaluation(self, checkpoint: Path | None = None) -> dict[str, Any]:
        self.require_valid()
        self.backend.prepare_directories()
        self.state.ensure(self.config)
        checkpoint = checkpoint or self.backend.latest_full_weights()
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"evaluation checkpoint not found: {checkpoint}")
        self._record_evaluation(status="running", checkpoint=str(checkpoint))
        try:
            self._run_with_ray(self.backend.build_eval_command(checkpoint), label="eval")
            metrics = parse_metric_log(self.backend.evaluation_dir / "metrics.log")
            result = metrics.to_dict()
            result["target_success_rate"] = self.config.target_success_rate
            result["target_reached"] = (
                metrics.success_once is not None and metrics.success_once >= self.config.target_success_rate
            )
            self._record_evaluation(status="completed", **result)
            return result
        except Exception as error:
            self._record_evaluation(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    def submit_successor(self, end_step: int, *, current_job_id: str) -> tuple[str, list[str]]:
        stages = self.backend.stages()
        index = next((index for index, stage in enumerate(stages) if stage.end_step == end_step), None)
        if index is None:
            raise ValueError(f"step {end_step} is not a configured stage end")
        launcher = SlurmLauncher(self.config, self.persisted_config_path)
        dependency = f"afterok:{current_job_id}"
        if index + 1 < len(stages):
            next_step = stages[index + 1].end_step
            command = launcher.build_stage_command(next_step, dependency=dependency)
            key = f"train_{next_step}"
        else:
            command = launcher.build_eval_command(dependency=dependency)
            key = "eval"
        self._require_submission_available(key, force=False)
        job_id = launcher.submit(command)
        self._record_job(key, job_id)
        return job_id, command

    def status(self) -> dict[str, Any]:
        state = self.state.read()
        training = parse_metric_log(self.config.output_dir / "metrics.log")
        evaluation = parse_metric_log(self.backend.evaluation_dir / "metrics.log")
        latest_checkpoint = self.backend.latest_full_weights()
        return {
            "config": str(self.persisted_config_path),
            "state": state,
            "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint),
            "training_metrics": training.to_dict(),
            "evaluation_metrics": evaluation.to_dict(),
        }

    @contextmanager
    def _ray_session(self, label: str) -> Iterator[dict[str, str]]:
        env = self.backend.environment()
        job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        local_base = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"areno-wm-vla-{job_id}"
        ray_dir = local_base / "ray"
        plasma_dir = local_base / "plasma"
        ray_dir.mkdir(parents=True, exist_ok=True)
        plasma_dir.mkdir(parents=True, exist_ok=True)
        env["RAY_TMPDIR"] = str(ray_dir)
        env["TMPDIR"] = str(local_base)

        ray = str(self.backend.ray_executable)
        subprocess.run([ray, "stop", "--force"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        monitor = self._start_gpu_monitor(label)
        try:
            subprocess.run(
                [
                    ray,
                    "start",
                    "--head",
                    f"--temp-dir={ray_dir}",
                    f"--plasma-directory={plasma_dir}",
                    f"--object-store-memory={self.config.ray_object_store_bytes}",
                    f"--num-cpus={self.config.slurm.cpus}",
                    f"--num-gpus={self.config.slurm.gpus}",
                    "--disable-usage-stats",
                ],
                check=True,
                env=env,
            )
            yield env
        finally:
            if monitor is not None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
            subprocess.run([ray, "stop", "--force"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _run_with_ray(self, command: list[str], *, label: str) -> None:
        with self._ray_session(label) as env:
            subprocess.run(command, cwd=self.config.rlinf_root, env=env, check=True)

    def _start_gpu_monitor(self, label: str) -> subprocess.Popen | None:
        if not self.config.gpu_monitor or shutil.which("nvidia-smi") is None:
            return None
        path = self.config.output_dir / "logs" / f"gpu-memory-{label}-{os.environ.get('SLURM_JOB_ID', os.getpid())}.csv"
        stream = path.open("w", encoding="utf-8")
        try:
            return subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    f"--loop={self.config.gpu_monitor_interval}",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        finally:
            stream.close()

    @staticmethod
    def _asset_report(result: AssetVerification) -> dict[str, Any]:
        return {
            "name": result.name,
            "ok": result.ok,
            "verified_bytes": result.verified_bytes,
            "expected_bytes": result.expected_bytes,
            "errors": list(result.errors),
        }

    def _record_job(self, key: str, job_id: str) -> None:
        def transform(state: dict[str, Any]) -> dict[str, Any]:
            state = state or self.state.initial_state(self.config)
            state.setdefault("jobs", {})[key] = {
                "job_id": job_id,
                "status": "submitted",
                "submitted_at": _now(),
            }
            return state

        self.state.update(transform)

    def _completed_stage_prefix(self) -> list[int]:
        completed: list[int] = []
        missing_seen = False
        noncontiguous: list[int] = []
        for step in self.config.stage_ends:
            if self.backend.full_weights_path(step).is_file():
                if missing_seen:
                    noncontiguous.append(step)
                else:
                    completed.append(step)
            else:
                missing_seen = True
        if noncontiguous:
            raise RuntimeError(
                "non-contiguous stage checkpoints found; refusing to resume past a missing stage: "
                + ", ".join(str(step) for step in noncontiguous)
            )
        return completed

    def _require_submission_available(self, key: str, *, force: bool) -> None:
        existing = self.state.read().get("jobs", {}).get(key)
        if existing is not None and not force:
            job_id = existing.get("job_id", "unknown")
            raise RuntimeError(
                f"workflow job {key!r} was already submitted as {job_id}; use --force to submit it again"
            )

    def _record_stage(self, stage: TrainingStage, *, status: str, **values: Any) -> None:
        def transform(state: dict[str, Any]) -> dict[str, Any]:
            state = state or self.state.initial_state(self.config)
            record = state.setdefault("stages", {}).setdefault(str(stage.end_step), {})
            if status in {"running", "completed"}:
                record.pop("error", None)
            record.update(
                {
                    "start_step": stage.start_step,
                    "end_step": stage.end_step,
                    "status": status,
                    "updated_at": _now(),
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    **values,
                }
            )
            return state

        self.state.update(transform)

    def _record_evaluation(self, *, status: str, **values: Any) -> None:
        def transform(state: dict[str, Any]) -> dict[str, Any]:
            state = state or self.state.initial_state(self.config)
            state["evaluation"] = {
                **state.get("evaluation", {}),
                "status": status,
                "updated_at": _now(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                **values,
            }
            if status in {"running", "completed"}:
                state["evaluation"].pop("error", None)
            return state

        self.state.update(transform)


__all__ = ["WorldModelVLAWorkflow"]
