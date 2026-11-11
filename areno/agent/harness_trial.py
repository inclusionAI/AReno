"""AReno training bridge around Harbor's native Trial lifecycle and factories."""

from __future__ import annotations

import asyncio
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.task.config import TaskConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TrialConfig, VerifierConfig
from harbor.models.trial.paths import EnvironmentPaths
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial

from areno.agent.harness_proxy import ModelProxy
from areno.api.harness import HarnessEnvironment


@dataclass(slots=True)
class EnvironmentSpec:
    """Native Harbor configuration plus dataset workspace and endpoint routing.

    ``task_options`` is passed to Harbor's task EnvironmentConfig, while
    ``config`` is its trial EnvironmentConfig. Custom factories may use any
    Harbor provider or an environment import path without changing the runner.
    """

    config: EnvironmentConfig
    workspace: str = "/workspace"
    control_dir: str = "/areno-controller"
    python: str = "python"
    proxy_bind_host: str = "0.0.0.0"
    proxy_connect_host: str = "host.docker.internal"
    task_options: dict[str, Any] = field(default_factory=dict)
    isolated: bool = True
    log_root: str | Path | None = None


def native_agent_config(name: str, **options) -> AgentConfig:
    """Delegate built-in names and user import paths to Harbor's agent factory."""
    return AgentConfig(name=name, **options)


def native_environment_spec(
    name: str,
    *,
    image: str | None = None,
    python: str = "python",
    task_env: dict | None = None,
    task_options: dict | None = None,
    workspace: str = "/workspace",
    control_dir: str = "/areno-controller",
    proxy_bind_host: str = "0.0.0.0",
    proxy_connect_host: str = "host.docker.internal",
    **options,
) -> EnvironmentSpec:
    task = dict(task_options or {})
    if image is not None:
        task["docker_image"] = image
    configured = dict(options)
    environment = {**configured.pop("env", {}), **dict(task_env or {})}
    if ":" in name:
        configured.setdefault("import_path", name)
    elif not configured.get("import_path"):
        configured.setdefault("type", name)
    config = EnvironmentConfig(env=environment, **configured)
    return EnvironmentSpec(config, workspace, control_dir, python, proxy_bind_host, proxy_connect_host, task)


async def run_trial(workflow, attempt, adapter, ctx, *, max_turns, agent_factory, environment_factory):
    spec = environment_factory(attempt, **workflow.environment_options)
    if not isinstance(spec, EnvironmentSpec):
        raise TypeError("environment factory must return an EnvironmentSpec with native Harbor configuration")
    result = attempt.item.record["harness_result"]
    proxy = ModelProxy(
        ctx,
        max_turns=max_turns,
        timeout=attempt.timeout,
        bind_host=spec.proxy_bind_host,
        connect_host=spec.proxy_connect_host,
    )
    attempt.proxy = proxy
    trial = None
    async with proxy:
        native = agent_factory(attempt, **workflow.harness_options)
        if not isinstance(native, AgentConfig):
            raise TypeError("harness factory must return a native Harbor AgentConfig")
        native = native.model_copy(deep=True)
        native.model_name = native.model_name or "openai/policy"
        native.env.update(OPENAI_API_KEY=proxy.api_key, OPENAI_BASE_URL=proxy.base_url)
        native.override_timeout_sec = attempt.timeout
        agent_class = AgentFactory.get_agent_class_from_config(native)
        if not spec.isolated and issubclass(agent_class, BaseInstalledAgent):
            raise ValueError("Harbor installed agents require an isolated environment; select an isolated provider")
        options = agent_class.options_model
        if options is not None and "max_turns" in options.model_fields:
            native.kwargs.setdefault("max_turns", max_turns)
        if workflow.preinstalled:
            agent = native.model_copy(
                deep=True,
                update={
                    "name": None,
                    "import_path": "areno.agent.harbor_agent:PreinstalledAgent",
                    "kwargs": {
                        "native_config": native.model_dump(mode="json", context={"redact_sensitive_env": False})
                    },
                },
            )
        else:
            agent = native
        definition = attempt.root / "task"
        (definition / "environment").mkdir(parents=True)
        (definition / "instruction.md").write_text(attempt.item.prompt)
        task = TaskConfig.model_validate({"environment": spec.task_options})
        (definition / "task.toml").write_text(task.model_dump_toml())
        image = task.environment.docker_image
        if image:
            (definition / "environment" / "Dockerfile").write_text(f"FROM {image}\nWORKDIR {spec.workspace}\n")
        config = TrialConfig(
            task={"path": definition},
            agent=agent,
            environment=spec.config,
            trials_dir=attempt.root / "trials",
            trial_name=attempt.root.name,
            verifier=VerifierConfig(disable=True),
        )
        try:
            trial = await Trial.create(config)
            attempt.environment = HarnessEnvironment(
                trial.agent_environment,
                spec.workspace,
                spec.control_dir,
                spec.python,
                spec.proxy_bind_host,
                spec.proxy_connect_host,
            )
            # The optional local provider shares the controller filesystem;
            # use sample-local log paths rather than Harbor's container paths.
            if spec.log_root is not None:
                trial.agent_env_paths = EnvironmentPaths._with_root(Path(spec.log_root))
                trial.agent.environment_logs_dir = trial.agent_env_paths.agent_dir
                for directory in (
                    trial.agent_env_paths.agent_dir,
                    trial.agent_env_paths.artifacts_dir,
                    trial.agent_env_paths.verifier_dir,
                ):
                    Path(directory).mkdir(parents=True, exist_ok=True)

            async def prepare(event):
                await asyncio.wait_for(adapter.prepare(attempt), attempt.timeout)

            async def grade(event):
                if asyncio.current_task().cancelling():
                    return
                try:
                    grading = await asyncio.wait_for(adapter.verify(attempt), attempt.verify_timeout)
                except TimeoutError:
                    grading = {"reward": 0.0, "verify_returncode": None, "verify_timed_out": True}
                reward = float(grading.pop("reward"))
                if not math.isfinite(reward):
                    raise ValueError("dataset reward must be finite")
                result.update(grading, reward=reward)

            trial.add_hook(TrialEvent.AGENT_START, prepare)
            trial.add_hook(TrialEvent.AGENT_END, grade)
            completed = await trial.run()
            failure = completed.exception_info
            agent_failure = failure is not None and failure.exception_type in {
                "AgentTimeoutError",
                "NonZeroAgentExitCodeError",
                "AgentSafetyRefusalError",
            }
            result.update(
                valid=bool(proxy.trace)
                and not proxy.errors
                and (failure is None or agent_failure)
                and "reward" in result,
                returncode=0 if failure is None else None,
                timed_out=bool(failure and failure.exception_type == "AgentTimeoutError"),
                proxy_errors=list(proxy.errors),
                turn_limit_reached=proxy.limit_reached,
                trial_id=str(completed.id),
                agent_context=completed.agent_result.model_dump(mode="json") if completed.agent_result else None,
            )
            if failure:
                result["error"] = f"{failure.exception_type}: {failure.exception_message}"
            if workflow.require_agent_success and (failure or proxy.limit_reached):
                result["reward"] = 0.0
            if not result["valid"]:
                result.pop("reward", None)
        finally:
            if workflow.result_dir is not None:
                artifact = Path(workflow.result_dir).expanduser()
                artifact.mkdir(parents=True, exist_ok=True)
                if trial is not None:
                    destination = artifact / attempt.root.name
                    shutil.copytree(trial.paths.trial_dir, destination, dirs_exist_ok=True)
                    grading_artifacts = attempt.root / "artifacts"
                    if grading_artifacts.exists():
                        shutil.copytree(grading_artifacts, destination / "artifacts", dirs_exist_ok=True)
    return proxy
