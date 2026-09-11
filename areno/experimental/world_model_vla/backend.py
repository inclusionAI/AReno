"""RLinf frozen-Wan backend for AReno's world-model VLA workflow."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from areno.experimental.world_model_vla.config import WorldModelVLAConfig


@dataclass(frozen=True, slots=True)
class TrainingStage:
    """One restartable interval in a long world-model training run."""

    start_step: int
    end_step: int
    resume_dir: Path | None
    checkpoint_dir: Path


@dataclass(frozen=True, slots=True)
class RuntimeCheck:
    name: str
    ok: bool
    detail: str


class RlinfWanBackend:
    """Build and validate commands for RLinf's Wan LIBERO-Spatial example."""

    training_config_name = "wan_libero_spatial_grpo_openvlaoft"
    experiment_name = "wan_libero_spatial_train"

    def __init__(self, config: WorldModelVLAConfig) -> None:
        self.config = config

    @property
    def training_config_dir(self) -> Path:
        return self.config.rlinf_root / "examples" / "embodiment" / "config"

    @property
    def experiment_dir(self) -> Path:
        return self.config.output_dir / self.experiment_name

    @property
    def evaluation_dir(self) -> Path:
        return self.config.output_dir / "evaluation"

    @property
    def ray_executable(self) -> Path:
        return self.config.rlinf_python.parent / "ray"

    def stages(self) -> tuple[TrainingStage, ...]:
        result: list[TrainingStage] = []
        previous = 0
        for end_step in self.config.stage_ends:
            resume_dir = self.checkpoint_dir(previous) if previous else None
            result.append(
                TrainingStage(
                    start_step=previous,
                    end_step=end_step,
                    resume_dir=resume_dir,
                    checkpoint_dir=self.checkpoint_dir(end_step),
                )
            )
            previous = end_step
        return tuple(result)

    def stage(self, end_step: int) -> TrainingStage:
        for stage in self.stages():
            if stage.end_step == end_step:
                return stage
        raise ValueError(f"step {end_step} is not a configured stage end")

    def checkpoint_dir(self, step: int) -> Path:
        return self.experiment_dir / "checkpoints" / f"global_step_{step}"

    def full_weights_path(self, step: int) -> Path:
        return self.checkpoint_dir(step) / "actor" / "model_state_dict" / "full_weights.pt"

    def latest_full_weights(self) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        checkpoint_root = self.experiment_dir / "checkpoints"
        for path in checkpoint_root.glob("global_step_*/actor/model_state_dict/full_weights.pt"):
            match = re.fullmatch(r"global_step_(\d+)", path.parents[2].name)
            if match:
                candidates.append((int(match.group(1)), path))
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def environment(self) -> dict[str, str]:
        cache_root = self.config.cache_root
        env = dict(os.environ)
        python_paths = [str(self.config.rlinf_root)]
        wan_source = self.config.rlinf_python.parent.parent / "wan"
        if wan_source.is_dir():
            python_paths.insert(0, str(wan_source))
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env.update(
            {
                "XDG_CACHE_HOME": str(cache_root),
                "HF_HOME": str(cache_root / "huggingface"),
                "MODELSCOPE_CACHE": str(cache_root / "modelscope"),
                "TORCH_HOME": str(cache_root / "torch"),
                "MPLCONFIGDIR": str(cache_root / "matplotlib"),
                "PYTHONPATH": os.pathsep.join(python_paths),
                "EMBODIED_PATH": str(self.training_config_dir),
                "REPO_PATH": str(self.config.rlinf_root),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "ROBOT_PLATFORM": "LIBERO",
                "LIBERO_TYPE": "standard",
                "TOKENIZERS_PARALLELISM": "false",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.update(self.config.extra_env)
        return env

    def prepare_directories(self) -> None:
        for path in (
            self.config.output_dir,
            self.evaluation_dir,
            self.config.cache_root,
            self.config.cache_root / "huggingface",
            self.config.cache_root / "modelscope",
            self.config.cache_root / "torch",
            self.config.cache_root / "matplotlib",
            self.config.output_dir / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def build_train_command(self, stage: TrainingStage) -> list[str]:
        command = [
            str(self.config.rlinf_python),
            str(self.config.rlinf_root / "examples" / "embodiment" / "train_embodied_agent.py"),
            "--config-path",
            str(self.training_config_dir),
            "--config-name",
            self.training_config_name,
            f"runner.logger.log_path={self.config.output_dir}",
            f"runner.logger.experiment_name={self.experiment_name}",
            f"runner.max_steps={stage.end_step}",
            "runner.val_check_interval=-1",
            f"runner.save_interval={stage.end_step}",
            f"env.train.wan_wm_hf_ckpt_path={self.config.world_model_path}",
            "env.train.video_cfg.save_video=false",
            f"rollout.model.model_path={self.config.policy_path}",
            f"actor.model.model_path={self.config.policy_path}",
            f"actor.micro_batch_size={self.config.actor_micro_batch_size}",
            "actor.fsdp_config.limit_all_gathers=true",
            "++weight_syncer.patch.transport_device=cpu",
        ]
        if self.config.require_rlinf_compatibility_patchset:
            command.append("++actor.fsdp_config.sync_module_states=false")
        if stage.resume_dir is not None:
            command.append(f"runner.resume_dir={stage.resume_dir}")
        command.extend(self.config.train_overrides)
        return command

    def build_eval_command(self, checkpoint: Path | None = None) -> list[str]:
        checkpoint = checkpoint or self.latest_full_weights()
        if checkpoint is None:
            raise FileNotFoundError(f"no full_weights.pt found under {self.experiment_dir / 'checkpoints'}")
        return [
            str(self.config.rlinf_python),
            str(self.config.rlinf_root / "evaluations" / "eval_embodied_agent.py"),
            "--config-path",
            str(self.training_config_dir),
            "--config-name",
            self.training_config_name,
            f"runner.logger.log_path={self.evaluation_dir}",
            "runner.logger.experiment_name=wan_libero_spatial_eval",
            f"runner.ckpt_path={checkpoint}",
            f"rollout.model.model_path={self.config.policy_path}",
            f"env.eval.total_num_envs={self.config.eval_trajectories}",
            "env.eval.use_fixed_reset_state_ids=true",
            "env.eval.video_cfg.save_video=false",
            *self.config.eval_overrides,
        ]

    def runtime_checks(self) -> tuple[RuntimeCheck, ...]:
        files = {
            "RLinf Python": self.config.rlinf_python,
            "RLinf Ray CLI": self.ray_executable,
            "training entrypoint": self.config.rlinf_root / "examples" / "embodiment" / "train_embodied_agent.py",
            "evaluation entrypoint": self.config.rlinf_root / "evaluations" / "eval_embodied_agent.py",
            "Wan training config": self.training_config_dir / f"{self.training_config_name}.yaml",
        }
        directories = {
            "world model snapshot": self.config.world_model_path,
            "policy snapshot": self.config.policy_path,
        }
        checks = [RuntimeCheck(name, path.is_file(), str(path)) for name, path in files.items()]
        for name in ("RLinf Python", "RLinf Ray CLI"):
            path = files[name]
            checks.append(RuntimeCheck(f"{name} executable", os.access(path, os.X_OK), str(path)))
        checks.extend(RuntimeCheck(name, path.is_dir(), str(path)) for name, path in directories.items())
        if self.config.setup_script is not None:
            checks.append(
                RuntimeCheck("setup script", self.config.setup_script.is_file(), str(self.config.setup_script))
            )
        if self.config.require_rlinf_compatibility_patchset:
            checks.extend(self._compatibility_checks())
        return tuple(checks)

    def _compatibility_checks(self) -> list[RuntimeCheck]:
        requirements = (
            (
                "configurable FSDP module synchronization",
                self.config.rlinf_root / "rlinf" / "hybrid_engines" / "fsdp" / "strategy" / "fsdp.py",
                '"sync_module_states", True',
            ),
            (
                "OpenVLA SDPA declaration",
                self.config.rlinf_root
                / "rlinf"
                / "models"
                / "embodiment"
                / "openvla_oft"
                / "rlinf"
                / "openvla_oft_action_model.py",
                "_supports_sdpa = True",
            ),
            (
                "OpenVLA SDPA selection",
                self.config.rlinf_root / "rlinf" / "models" / "embodiment" / "openvla_oft" / "rlinf" / "__init__.py",
                'attn_implementation="sdpa"',
            ),
            (
                "sequential embodied worker initialization",
                self.config.rlinf_root / "rlinf" / "runners" / "embodied_runner.py",
                "self.rollout.init_worker().wait()",
            ),
        )
        checks: list[RuntimeCheck] = []
        for name, path, marker in requirements:
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            checks.append(RuntimeCheck(name, marker in text, f"{path}: requires {marker!r}"))
        return checks


__all__ = ["RlinfWanBackend", "RuntimeCheck", "TrainingStage"]
