"""Backend contract + lazy registry shared by every execution engine.

The `Backend` ABC enumerates everything `Trainer` needs from a concrete
backend: weight sync, rollout, training, optional checkpointing, and the
auxiliary `score_*`/`train_values` operations PPO drives through model roles.
Implementations register themselves with `@register_backend(BackendType.X)`
and are imported lazily from `BACKEND_MODULES`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from areno.api.context import Context
from areno.api.models import SamplingParams, RolloutResult, TrainSequence
from areno.api.roles import ModelRole

# Map BackendType -> implementation class, populated by `register_backend`.
BACKEND_CLS = {}
# Map BackendType.value -> Python module path; `get_backend_cls` imports the
# module the first time a backend is requested, which both registers the class
# and avoids paying the import cost for backends that are never used.
BACKEND_MODULES = {
    "Areno": "areno.api.backend.areno",
}

def register_backend(backend_type):
    """Register a backend implementation class for `Trainer` construction."""

    def decorator(cls):
        BACKEND_CLS[backend_type] = cls
        return cls
    return decorator


def get_backend_cls(backend_type):
    """Return a backend class, importing only the selected backend package."""

    cls = BACKEND_CLS.get(backend_type)
    if cls is not None:
        return cls
    # Lazy import: triggering `__import__` runs the decorator and populates
    # `BACKEND_CLS`, so the lookup below succeeds on the second pass.
    module_name = BACKEND_MODULES.get(backend_type.value)
    if module_name is None:
        return None
    __import__(module_name)
    return BACKEND_CLS.get(backend_type)


class Backend(ABC):
    """Backend contract implemented by concrete execution engines.

    Required ops (initialize/rollout/rollout_batch/train) keep the core RL
    loop functional. Optional ops (save_checkpoint/ensure_roles/score_* /
    train_values) raise `NotImplementedError` by default so feature-poorer
    backends only opt into what they can support.
    """

    @abstractmethod
    def initialize(self, ctx: Context):
        """Start backend resources and load model state."""

        pass

    @abstractmethod
    def rollout_batch(
        self,
        ctx: Context,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
    ) -> list[RolloutResult]:
        """Generate samples for a prompt batch, preserving input order."""

        pass

    @abstractmethod
    def train(
        self,
        ctx: Context,
        batch_data: list[TrainSequence],
        loss_fn: Callable,
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        """Consume rollout sequences and apply one optimization step."""

        pass

    def save_checkpoint(self, ctx: Context, path: str) -> str:
        """Persist model weights, or raise when the backend cannot save."""

        raise NotImplementedError(f"{type(self).__name__} does not support checkpoint saving")

    def ensure_roles(self, ctx: Context, roles: dict[str, ModelRole]) -> None:
        """Prepare backend-owned auxiliary model roles, or raise if unsupported."""

        raise NotImplementedError(f"{type(self).__name__} does not support model roles")

    def begin_rollout_session(self, ctx: Context) -> None:
        """Prepare backend state for one or more rollout calls."""

        del ctx

    async def begin_rollout_session_async(self, ctx: Context) -> None:
        """Async rollout-session begin hook."""

        self.begin_rollout_session(ctx)

    async def sync_rollout_session_async(self, ctx: Context) -> None:
        """Optional synchronization hook before request-driven rollout."""

        del ctx

    def dp_size(self, ctx: Context) -> int:
        """Return the backend's effective data-parallel size."""

        del ctx
        return 1

    def model_context_len(self, ctx: Context) -> int | None:
        """Return the model's configured maximum context length when known."""

        del ctx
        return None

    def end_rollout_session(self, ctx: Context) -> None:
        """Finalize rollout state before scoring or training."""

        del ctx

    async def end_rollout_session_async(self, ctx: Context) -> None:
        """Async rollout-session end hook."""

        self.end_rollout_session(ctx)

    def score_logprobs(self, ctx: Context, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        """Score fixed token rows with a backend-owned role."""

        raise NotImplementedError(f"{type(self).__name__} does not support role logprob scoring")

    def score_values(self, ctx: Context, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        """Score fixed token rows with a backend-owned critic role."""

        raise NotImplementedError(f"{type(self).__name__} does not support role value scoring")

    def score_rewards(self, ctx: Context, role: str, token_rows: list[list[int]]) -> list[float]:
        """Score fixed token rows with a backend-owned reward role."""

        raise NotImplementedError(f"{type(self).__name__} does not support role reward scoring")

    def train_values(
        self,
        ctx: Context,
        role: str,
        batch_data: list[TrainSequence],
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
        *,
        cliprange_value: float = 0.5,
        value_loss_coef: float = 0.5,
    ) -> dict[str, float]:
        """Train a backend-owned critic role."""

        raise NotImplementedError(f"{type(self).__name__} does not support role value training")
