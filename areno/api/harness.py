"""Registry-based Harbor harness bridge for AReno agentic rollouts.

Factories receive a sample-local :class:`HarnessAttempt`. Harness factories
return native Harbor ``AgentConfig`` objects; environment factories return
``EnvironmentSpec`` containing native Harbor environment configuration. Dataset adapters
own normalization, workspace preparation and grading, independently of agents.
Harbor is a required dependency, imported lazily to keep SDK startup lightweight.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import inspect
import json
import logging
import math
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.trial.config import AgentConfig

    from areno.agent.harness_proxy import ModelProxy
    from areno.agent.harness_trial import EnvironmentSpec
    from areno.api.agentic import AgentBatch, AgentItem, AgentTrajectory, RolloutSession

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HarnessEnvironment:
    """Harbor environment plus paths and endpoint routing for one attempt."""

    environment: BaseEnvironment
    workspace: str
    control_dir: str
    python: str = "python"
    proxy_bind_host: str = "127.0.0.1"
    proxy_connect_host: str = "127.0.0.1"


@dataclass(slots=True)
class HarnessAttempt:
    item: AgentItem
    root: Path
    timeout: float
    verify_timeout: float
    environment: HarnessEnvironment | None = None
    proxy: ModelProxy | None = None


class HarnessDataset(Protocol):
    def normalize(self, record: dict[str, Any]) -> dict[str, Any]: ...

    async def prepare(self, attempt: HarnessAttempt) -> None: ...

    async def verify(self, attempt: HarnessAttempt) -> dict[str, Any]:
        """Return grading details including a numeric ``reward``."""
        ...


HarnessFactory = Callable[..., "AgentConfig"]
EnvironmentFactory = Callable[..., "EnvironmentSpec"]
DatasetFactory = Callable[..., HarnessDataset]
RecordLoader = Callable[..., Iterable[Mapping[str, Any]]]
_HARNESSES: dict[str, HarnessFactory] = {}
_ENVIRONMENTS: dict[str, EnvironmentFactory] = {}
_DATASETS: dict[str, DatasetFactory] = {}


def _register(registry: dict, name: str, factory: Callable, replace: bool) -> None:
    key = name.strip().lower()
    if not key or not callable(factory):
        raise ValueError("registration requires a nonempty name and callable factory")
    if key in registry and not replace:
        raise ValueError(f"{key!r} is already registered")
    registry[key] = factory


def register_harness(name: str, factory: HarnessFactory, *, replace: bool = False) -> None:
    """Register ``factory(attempt, **options) -> harbor.models.trial.config.AgentConfig``."""
    _register(_HARNESSES, name, factory, replace)


def register_environment(name: str, factory: EnvironmentFactory, *, replace: bool = False) -> None:
    """Register a native Harbor EnvironmentSpec configuration factory."""
    _register(_ENVIRONMENTS, name, factory, replace)


def register_dataset(name: str, factory: DatasetFactory, *, replace: bool = False) -> None:
    """Register an adapter factory for normalization, preparation and grading."""
    _register(_DATASETS, name, factory, replace)


def list_harnesses() -> list[str]:
    from harbor.agents.factory import AgentFactory

    return sorted(set(_HARNESSES) | set(AgentFactory.registered_names()))


def list_environments() -> list[str]:
    from harbor.models.environment_type import EnvironmentType

    return sorted(set(_ENVIRONMENTS) | {value.value for value in EnvironmentType})


def list_datasets() -> list[str]:
    return sorted(_DATASETS)


def _resolve(registry: dict, name: str) -> Callable:
    try:
        return registry[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown adapter {name!r}; registered: {', '.join(sorted(registry))}") from None


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _record_loader(loader: RecordLoader | str | Path) -> RecordLoader:
    if callable(loader):
        return loader
    path_text, separator, function = str(loader).rpartition(":")
    path = Path(path_text if separator else str(loader)).expanduser().resolve()
    function = function if separator else "load_training_dataset"
    if not function:
        raise ValueError("custom loader requires a function name after ':'")
    name = f"areno_harness_loader_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load dataset loader from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    result = getattr(module, function, None)
    if not callable(result):
        raise ValueError(f"{path} must define callable {function}(dataset_path, ...)")
    return result


@dataclass(slots=True)
class HarnessRunner:
    """Compose registered adapters; callable as ``workflow(ctx, batch)`` in the SDK.

    ``dataset_loader`` and ``reward_fn`` implement the AReno hook contracts.
    CLI files expose them through top-level function wrappers, as in the Pi
    example, so selecting adapters needs no trainer or CLI changes. Each
    sample owns its record, environment, credentials and exact model trace.

    ``loader_fn`` accepts a callable or ``file.py[:function]`` (default:
    ``load_training_dataset``). It reads user-defined source formats and may
    yield records lazily. ``loader_kwargs`` contains explicit user options;
    invocation options override them. Framework helpers such as ``default_loader``
    and ``load_dataset`` are supplied only when the callable accepts them.
    The independently registered dataset adapter owns the task record schema.
    The core provides no dataset implementations or default dataset selection.
    Set ``require_agent_success=False`` when a trusted grader should score the
    submitted state even if the agent reaches its turn/time budget or exits.
    ``harness_options`` contains native Harbor AgentConfig fields, including
    ``kwargs``, ``env``, ``skills`` and ``mcp_servers``. Registered factories
    return AgentConfig/EnvironmentSpec rather than duplicating Harbor agents.
    Installed tools are reused by default; ``preinstalled=False`` explicitly
    enables the native agent's installation/setup. For Docker with an external
    daemon, place ``attempt_dir`` on a path shared with that daemon's host.
    """

    harness: str
    dataset: str
    environment: str = "docker"
    harness_options: dict[str, Any] = field(default_factory=dict)
    environment_options: dict[str, Any] = field(default_factory=dict)
    dataset_options: dict[str, Any] = field(default_factory=dict)
    result_dir: str | Path | None = None
    loader_fn: RecordLoader | str | Path | None = None
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    require_agent_success: bool = True
    preinstalled: bool = True
    attempt_dir: str | Path | None = None

    def dataset_loader(
        self, dataset_path: str, *, default_loader: Callable | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        adapter = _resolve(_DATASETS, self.dataset)(**self.dataset_options)
        context = {"default_loader": default_loader} if default_loader is not None else {}
        context.update({key: value for key, value in kwargs.items() if key in {"load_dataset", "load_from_disk"}})
        options = {**self.loader_kwargs, **{key: value for key, value in kwargs.items() if key not in context}}
        if self.loader_fn is None:
            if default_loader is None:
                raise ValueError("dataset loading requires loader_fn or default_loader")
            rows = default_loader(dataset_path, **options)
        else:
            loader = _record_loader(self.loader_fn)
            parameters = inspect.signature(loader).parameters
            accepts_kwargs = any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values())
            options.update(
                {
                    key: value
                    for key, value in context.items()
                    if accepts_kwargs
                    or (
                        key in parameters
                        and parameters[key].kind
                        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    )
                }
            )
            rows = loader(dataset_path, **options)
        return [adapter.normalize(dict(row)) for row in rows]

    @staticmethod
    def reward_fn(record: Any) -> float:
        return float(record.source_record["harness_result"]["reward"])

    async def __call__(self, ctx: RolloutSession, batch: AgentBatch) -> AgentTrajectory:
        from areno.agent.harness_trial import native_agent_config, native_environment_spec, run_trial
        from areno.api.agentic import AgentTrajectory

        agent_factory = _HARNESSES.get(self.harness.strip().lower())
        if agent_factory is None:

            def agent_factory(attempt, **options):
                return native_agent_config(self.harness, **options)

        environment_factory = _ENVIRONMENTS.get(self.environment.strip().lower())
        if environment_factory is None:

            def environment_factory(attempt, **options):
                return native_environment_spec(self.environment, **options)

        dataset_factory = _resolve(_DATASETS, self.dataset)
        semaphore = asyncio.Semaphore(max(1, int(ctx.max_running_prompts)))
        attempt_parent = Path(self.attempt_dir).expanduser() if self.attempt_dir is not None else None
        if attempt_parent is not None:
            attempt_parent.mkdir(parents=True, exist_ok=True)

        async def run_sample(original: AgentItem) -> AgentTrajectory:
            async with semaphore:
                item = copy.deepcopy(original)
                adapter = dataset_factory(**self.dataset_options)
                item.record = adapter.normalize(item.record)
                timeout = _positive(item.record.get("timeout", 1800), "timeout")
                verify_timeout = _positive(item.record.get("verify_timeout", 120), "verify_timeout")
                max_turns = item.record.get("max_turns", 32)
                if isinstance(max_turns, bool) or int(max_turns) != max_turns or max_turns < 1:
                    raise ValueError("max_turns must be a positive integer")
                with tempfile.TemporaryDirectory(prefix="areno-harness-", dir=attempt_parent) as directory:
                    attempt = HarnessAttempt(item, Path(directory), timeout, verify_timeout)
                    result = {"harness": self.harness, "environment": self.environment, "dataset": self.dataset}
                    item.record["harness_result"] = result
                    try:
                        proxy = await run_trial(
                            self,
                            attempt,
                            adapter,
                            ctx,
                            max_turns=int(max_turns),
                            agent_factory=agent_factory,
                            environment_factory=environment_factory,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        result.update(valid=False, error=f"{type(exc).__name__}: {exc}")
                        result.pop("reward", None)
                        logger.warning("invalid Harbor trial: %s", result)
                        return AgentTrajectory(invalid_items=[item])
                    finally:
                        if self.result_dir is not None:
                            artifact = Path(self.result_dir).expanduser()
                            artifact.mkdir(parents=True, exist_ok=True)
                            (artifact / f"{attempt.root.name}.json").write_text(
                                json.dumps(
                                    {
                                        "prompt_index": item.prompt_index,
                                        "sample_index": item.sample_index,
                                        "result": result,
                                        "trace": attempt.proxy.trace if attempt.proxy is not None else [],
                                    }
                                )
                            )
                    logger.info(
                        "harness=%s prompt=%d sample=%d valid=%s reward=%s",
                        self.harness,
                        item.prompt_index,
                        item.sample_index,
                        result.get("valid"),
                        result.get("reward"),
                    )
                    if not result.get("valid"):
                        logger.warning("invalid Harbor trial: %s", result)
                        return AgentTrajectory(invalid_items=[item])
                    return AgentTrajectory(turns=proxy.turns(item))

        tasks = [asyncio.create_task(run_sample(item)) for item in batch.iter_samples()]
        try:
            trajectories = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return AgentTrajectory(
            turns=[turn for result in trajectories for turn in result.turns],
            invalid_items=[item for result in trajectories for item in result.invalid_items],
        )


def _local_environment(attempt: HarnessAttempt, **options: Any) -> EnvironmentSpec:
    from areno.agent.harness_env import local_spec

    return local_spec(attempt, **options)


def _docker_environment(attempt: HarnessAttempt, **options: Any) -> EnvironmentSpec:
    from areno.agent.harness_trial import native_environment_spec

    return native_environment_spec("docker", **options)


register_environment("local", _local_environment)
register_environment("docker", _docker_environment)
