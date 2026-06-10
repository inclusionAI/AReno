"""Algorithm registry for trainer and loss-function dispatch."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import pkgutil
from typing import Callable, Protocol

from areno.api.trainer_config import TrainerConfig
from areno.api.loss_fns import dpo_loss_fn, grpo_loss_fn, gspo_loss_fn, ppo_loss_fn, sft_loss_fn


class TrainerFactory(Protocol):
    """Callable returning a concrete trainer class without importing it early."""

    def __call__(self) -> type:
        """Return the trainer implementation class."""


LossFnFactory = Callable[[TrainerConfig, Callable], Callable]


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Declarative metadata for an algorithm implementation."""

    name: str
    trainer_cls: type | TrainerFactory
    default_loss_fn: Callable
    requires_rollout: bool
    loss_fn_factory: LossFnFactory | None = None
    experimental: bool = False

    def resolve_trainer_cls(self) -> type:
        """Resolve a lazy trainer loader to the concrete trainer class."""

        if isinstance(self.trainer_cls, type):
            return self.trainer_cls
        return self.trainer_cls()

    def make_loss_fn(self, config: TrainerConfig) -> Callable:
        """Build the callable loss used by the selected trainer."""

        if self.loss_fn_factory is not None:
            return self.loss_fn_factory(config, self.default_loss_fn)
        return self.default_loss_fn


_ALGORITHMS: dict[str, AlgorithmSpec] = {}
_EXPERIMENTAL_LOADED = False


def register_algorithm(spec: AlgorithmSpec, *, replace: bool = False) -> None:
    """Register an algorithm spec by name.

    Duplicate registration is rejected by default so plugin import-order bugs
    fail loudly instead of silently changing the trainer used by a run.
    """

    name = spec.name.strip().lower()
    if not name:
        raise ValueError("algorithm name must be non-empty")
    if name in _ALGORITHMS and not replace:
        raise ValueError(f"algorithm {name!r} is already registered")
    if name != spec.name:
        spec = AlgorithmSpec(
            name=name,
            trainer_cls=spec.trainer_cls,
            default_loss_fn=spec.default_loss_fn,
            requires_rollout=spec.requires_rollout,
            loss_fn_factory=spec.loss_fn_factory,
            experimental=spec.experimental,
        )
    _ALGORITHMS[name] = spec


def get_algorithm(name: str) -> AlgorithmSpec:
    """Return a registered algorithm, loading experimental plugins on demand."""

    key = name.strip().lower()
    if key not in _ALGORITHMS:
        load_experimental_algorithms()
    if key not in _ALGORITHMS:
        known = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(f"unknown algorithm {name!r}; registered: {known}")
    return _ALGORITHMS[key]


def list_algorithms(*, include_experimental: bool = True) -> dict[str, AlgorithmSpec]:
    """Return the registered algorithm specs keyed by algorithm name."""

    if include_experimental:
        load_experimental_algorithms()
    return dict(_ALGORITHMS)


def load_experimental_algorithms() -> None:
    """Import packages under ``areno.experimental`` once."""

    global _EXPERIMENTAL_LOADED
    if _EXPERIMENTAL_LOADED:
        return
    _EXPERIMENTAL_LOADED = True
    try:
        package = importlib.import_module("areno.experimental")
    except ModuleNotFoundError:
        return
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    for module in pkgutil.iter_modules(package_path):
        if module.ispkg:
            importlib.import_module(f"{package.__name__}.{module.name}")


def _load_policy_trainer() -> type:
    from areno.api.trainers.policy_only import PolicyOnlyTrainer

    return PolicyOnlyTrainer


def _load_sft_trainer() -> type:
    from areno.api.trainers.sft import SFTTrainer

    return SFTTrainer


def _load_dpo_trainer() -> type:
    from areno.api.trainers.dpo import DPOTrainer

    return DPOTrainer


def _load_ppo_trainer() -> type:
    from areno.api.trainers.ppo import PPOTrainer

    return PPOTrainer


def _bind_gspo_loss(config: TrainerConfig, loss_fn: Callable) -> Callable:
    from functools import partial

    return partial(loss_fn, clip_eps=getattr(config, "gspo_clip_eps"))


def _bind_grpo_loss(config: TrainerConfig, loss_fn: Callable) -> Callable:
    from functools import partial

    return partial(loss_fn, clip_eps=getattr(config, "grpo_clip_eps"))


def _register_builtin_algorithms() -> None:
    register_algorithm(
        AlgorithmSpec(
            name="sft",
            trainer_cls=_load_sft_trainer,
            default_loss_fn=sft_loss_fn,
            requires_rollout=False,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            name="dpo",
            trainer_cls=_load_dpo_trainer,
            default_loss_fn=dpo_loss_fn,
            requires_rollout=False,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            name="gspo",
            trainer_cls=_load_policy_trainer,
            default_loss_fn=gspo_loss_fn,
            requires_rollout=True,
            loss_fn_factory=_bind_gspo_loss,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            name="grpo",
            trainer_cls=_load_policy_trainer,
            default_loss_fn=grpo_loss_fn,
            requires_rollout=True,
            loss_fn_factory=_bind_grpo_loss,
        )
    )
    register_algorithm(
        AlgorithmSpec(
            name="ppo",
            trainer_cls=_load_ppo_trainer,
            default_loss_fn=ppo_loss_fn,
            requires_rollout=True,
        )
    )


_register_builtin_algorithms()


__all__ = [
    "AlgorithmSpec",
    "get_algorithm",
    "list_algorithms",
    "load_experimental_algorithms",
    "register_algorithm",
]
