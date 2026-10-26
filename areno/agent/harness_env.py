"""Harbor environments for shared local tools and prebuilt Docker images.

The local environment provides separate working directories, not OS isolation.
It uses preinstalled tools and always reaps the process group on exit.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from areno.api.harness import HarnessAttempt, HarnessEnvironment


class LocalEnvironment(BaseEnvironment):
    def __init__(
        self, *args: Any, task_env: dict[str, str] | None = None, inherit_env: list[str] | None = None, **kwargs: Any
    ) -> None:
        self.task_env = dict(task_env or {})
        self.inherit_env = inherit_env
        self.workspace = kwargs.pop("workspace", None)
        self._processes: set[asyncio.subprocess.Process] = set()
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "areno-local"

    def _validate_definition(self) -> None:
        pass

    async def start(self, force_build: bool = False) -> None:
        self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
        if self.workspace:
            Path(self.workspace).mkdir(parents=True, exist_ok=True)

    async def stop(self, delete: bool = True) -> None:
        for process in tuple(self._processes):
            self._kill(process)
            await process.wait()

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.upload_file(source_path, str(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self.upload_dir(source_dir, str(target_dir))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self._resolve_user(user) is not None:
            raise ValueError("local environment runs as the current user; user switching is unsupported")
        process_env = (
            dict(os.environ)
            if self.inherit_env is None
            else {key: os.environ[key] for key in self.inherit_env if key in os.environ}
        )
        process_env.update(self.task_env)
        process_env.update(self._merge_env(env) or {})
        # A tool can write arbitrarily large output; retain a bounded tail.
        with tempfile.TemporaryFile() as log:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd or self.workspace,
                env=process_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._processes.add(process)
            try:
                code = await asyncio.wait_for(process.wait(), timeout_sec)
            finally:
                self._kill(process)
                await process.wait()
                self._processes.discard(process)
            log.seek(0, 2)
            log.seek(max(0, log.tell() - 8000))
            return ExecResult(return_code=code, stdout=log.read().decode(errors="replace"))


def local_environment(
    attempt: HarnessAttempt, *, python: str = sys.executable, task_env: dict[str, str] | None = None
) -> HarnessEnvironment:
    executable = shutil.which(python)
    if executable is None:
        raise ValueError(f"task Python executable not found: {python}")
    root = attempt.root
    definition = root / "environment"
    definition.mkdir()
    env = dict(
        PATH=str(Path(executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    env.update(task_env or {})
    environment = LocalEnvironment(
        environment_dir=definition,
        environment_name="areno-local",
        session_id=root.name,
        trial_paths=TrialPaths(trial_dir=root / "trial"),
        task_env_config=EnvironmentConfig(),
        task_env=env,
    )
    return HarnessEnvironment(environment, str(root / "workspace"), str(root / "control"), executable)


def local_spec(
    attempt: HarnessAttempt,
    *,
    python: str = sys.executable,
    task_env: dict[str, str] | None = None,
    inherit_env: list[str] | None = None,
):
    """Optional custom local provider; Harbor has no built-in local provider.

    Production installed agents with fixed absolute config paths need isolated
    environments. This provider is useful for custom agents and CPU checks.
    """
    from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig

    from areno.agent.harness_trial import EnvironmentSpec

    workspace = str(attempt.root / "workspace")
    return EnvironmentSpec(
        config=TrialEnvironmentConfig(
            import_path="areno.agent.harness_env:LocalEnvironment",
            kwargs={
                "task_env": {"CUDA_VISIBLE_DEVICES": "", **dict(task_env or {})},
                "workspace": workspace,
                "inherit_env": inherit_env,
            },
        ),
        workspace=workspace,
        control_dir=str(attempt.root / "control"),
        python=python,
        proxy_bind_host="127.0.0.1",
        proxy_connect_host="127.0.0.1",
        isolated=False,
        log_root=attempt.root / "local-root",
    )
