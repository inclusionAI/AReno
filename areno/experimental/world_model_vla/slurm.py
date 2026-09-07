"""Slurm command construction for staged world-model VLA runs."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from areno.experimental.world_model_vla.config import WorldModelVLAConfig


class SlurmLauncher:
    def __init__(self, config: WorldModelVLAConfig, config_path: str | Path) -> None:
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.worker_script = Path(__file__).with_name("worker.sbatch")

    def build_stage_command(self, end_step: int, *, dependency: str | None = None) -> list[str]:
        command = self._base_command(mode="train", dependency=dependency)
        command.extend(self._worker_prefix())
        command.extend(
            [
                "run-stage",
                "--config",
                str(self.config_path),
                "--end-step",
                str(end_step),
                "--auto-continue",
            ]
        )
        return command

    def build_eval_command(self, *, dependency: str | None = None) -> list[str]:
        command = self._base_command(mode="eval", dependency=dependency)
        command.extend(self._worker_prefix())
        command.extend(["run-eval", "--config", str(self.config_path)])
        return command

    def submit(self, command: list[str]) -> str:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        output = "\n".join(value for value in (result.stdout.strip(), result.stderr.strip()) if value)
        match = re.search(r"(?:Submitted batch job\s+)?(\d+)\s*$", result.stdout.strip())
        if match is None:
            raise RuntimeError(f"could not parse Slurm job ID from: {output}")
        return match.group(1)

    def _base_command(self, *, mode: str, dependency: str | None) -> list[str]:
        resources = self.config.slurm
        qos = resources.train_qos if mode == "train" else resources.eval_qos
        time_limit = resources.train_time_limit if mode == "train" else resources.eval_time_limit
        logs = self.config.output_dir / "logs"
        command = [
            "sbatch",
            f"--job-name=areno-wm-vla-{mode}",
            f"--partition={resources.partition}",
            f"--time={time_limit}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory}",
            f"--gres=gpu:{resources.gpus}",
            f"--output={logs}/%x-%j.out",
            f"--error={logs}/%x-%j.err",
        ]
        if qos:
            command.append(f"--qos={qos}")
        if resources.constraint:
            command.append(f"--constraint={resources.constraint}")
        if resources.account:
            command.append(f"--account={resources.account}")
        if resources.disable_gpu_binding:
            command.append("--gres-flags=disable-binding")
        if resources.node:
            command.append(f"--nodelist={resources.node}")
        if dependency:
            command.append(f"--dependency={dependency}")
        return command

    def _worker_prefix(self) -> list[str]:
        setup_script = str(self.config.setup_script) if self.config.setup_script is not None else "-"
        return [str(self.worker_script), sys.executable, setup_script]


__all__ = ["SlurmLauncher"]
