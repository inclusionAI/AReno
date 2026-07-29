"""High-level entrypoint that algorithm scripts interact with.

`Trainer` ties together tokenizer loading, backend creation, the rollout/train
cycle, and (optionally) TensorBoard recording. A typical RL script constructs
one `Trainer`, calls ``init()`` once, and then loops:
``rollout_batch() -> train()``. PPO additionally calls `ensure_roles` so that
ref/reward/critic models become available behind the backend boundary.
"""

import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from areno.api.agentic import LossMaskPolicy, RolloutSession
from areno.api.backend.base import Backend, get_backend_cls
from areno.api.config import BackendConfig, coerce_backend_config, resolve_backend_type
from areno.api.context import Context
from areno.api.data import PromptBatch, PromptItem
from areno.api.health_check import (
    HealthCheckConfig,
    HealthCheckError,
    HealthReport,
    WindowSignals,
    run_health_check,
)
from areno.api.metrics import MetricsRecorder, collect_train_batch_stats
from areno.api.models import BackendType, RolloutResult, SamplingParams, TrainSequence
from areno.api.roles import ModelRole
from areno.api.tokenizer import encode_generation_prompt, eos_token_ids, load_tokenizer, normalize_token_ids


class Trainer:
    """High-level API used by algorithm code.

    `Trainer` owns tokenizer loading, backend construction, rollout, training,
    checkpointing, and optional metric recording. A typical RL loop calls
    `init()`, repeatedly runs `rollout_batch() -> train()`, and finally
    `close()`.
    """

    def __init__(
        self,
        world_size: int,
        model_path: str,
        backend_type: BackendType | None = None,
        custom_config: BackendConfig | None = None,
        metrics_log_dir: str | None = None,
    ) -> None:
        """Create a trainer without starting backend workers.

        Call `init()` before rollout or training. `world_size` is the total
        number of devices/workers visible to the selected backend.
        """

        self._tokenizer = None
        self._backend: Backend | None = None
        # Resolve backend type from the explicit value or default to Areno.
        self._backend_type = resolve_backend_type(backend_type, custom_config)
        self._model_path = model_path
        self._ctx: Context | None = None
        self._world_size = world_size
        self._initialized = False
        self._custom_config = coerce_backend_config(self._backend_type, custom_config)
        self._metrics = MetricsRecorder(metrics_log_dir) if metrics_log_dir else None
        # Per-step wall-time bag accumulated by the rollout/train helpers
        # so `record_train_step` can flush a complete timing snapshot.
        self._metric_timings: dict[str, float] = {}
        self._step_active = False
        self._step_wall_start: float | None = None
        self._rollout_session_depth = 0
        self._rollout_wall_start: float | None = None
        # Optional startup-window health checker (Issue #249). Stays `None`
        # unless `configure_health_check` is called with an enabled config, so
        # the default run path never pays for it.
        self._health_checker: TrainingHealthChecker | None = None

    def init(self) -> None:
        """Load tokenizer, create backend context, and initialize workers."""

        real_path = self._model_path
        self._tokenizer = load_tokenizer(real_path)
        self._ctx = Context(
            self._world_size, real_path, self._tokenizer, self._custom_config, eos_token_ids(real_path, self._tokenizer)
        )
        backend_cls = get_backend_cls(self._backend_type)
        if backend_cls is None:
            raise ValueError(f"unsupported backend type: {self._backend_type}")
        self._backend = backend_cls()
        self._backend.initialize(self._ctx)
        self._initialized = True

    def get_tokenizer(self) -> Any:
        """Return the initialized tokenizer for prompt and completion handling."""

        return self._tokenizer

    def _begin_step(self) -> None:
        """Open a trainer-owned step if rollout/train has not already done so."""

        if self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._step_active:
            return
        self._ctx.step()
        self._metric_timings = {}
        self._step_active = True
        self._step_wall_start = time.perf_counter()

    def finish_step(self) -> None:
        """Close the current trainer-owned step without running actor train."""

        self._step_active = False
        self._step_wall_start = None

    def begin_rollout_session(self) -> None:
        """Prepare backend rollout state for one or more rollout calls."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth == 0:
            self._begin_step()
            self._rollout_wall_start = time.perf_counter()
            self._backend.begin_rollout_session(self._ctx)
        self._rollout_session_depth += 1

    async def begin_rollout_session_async(self) -> None:
        """Async variant of :meth:`begin_rollout_session`."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth == 0:
            self._begin_step()
            self._rollout_wall_start = time.perf_counter()
            await self._backend.begin_rollout_session_async(self._ctx)
        self._rollout_session_depth += 1

    async def sync_rollout_session_async(self) -> None:
        """Synchronize backend rollout workers before request-driven rollout."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth <= 0:
            raise RuntimeError(
                "sync_rollout_session_async must be called inside `async with trainer.rollout_session(...)`"
            )
        await self._backend.sync_rollout_session_async(self._ctx)

    def dp_size(self) -> int:
        """Return the initialized backend's effective data-parallel size."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        dp_size_fn = getattr(self._backend, "dp_size", None)
        if callable(dp_size_fn):
            return int(dp_size_fn(self._ctx))
        config = self._ctx.custom_config
        if config is None:
            return int(self._ctx.world_size)
        return max(int(self._ctx.world_size) // int(config.tp_size), 1)

    def model_context_len(self) -> int | None:
        """Return the loaded model's context length when the backend exposes it."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        return self._backend.model_context_len(self._ctx)

    def probe_rollout_cache(self, *, max_new_tokens: int, max_running_prompts: int, max_prompt_len: int) -> float:
        """Allocate rollout KV cache/decode graphs without running rollout decode."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        return self._backend.probe_rollout_cache(
            self._ctx,
            max_new_tokens=max_new_tokens,
            max_running_prompts=max_running_prompts,
            max_prompt_len=max_prompt_len,
        )

    def end_rollout_session(self) -> None:
        """Finalize backend rollout state when a rollout group completes."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth <= 0:
            logger.warning("end_rollout_session called without matching begin")
            return
        self._rollout_session_depth -= 1
        if self._rollout_session_depth == 0:
            try:
                self._backend.end_rollout_session(self._ctx)
            finally:
                self._finish_rollout_timing()

    async def end_rollout_session_async(self) -> None:
        """Async variant of :meth:`end_rollout_session`."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth <= 0:
            return
        self._rollout_session_depth -= 1
        if self._rollout_session_depth == 0:
            try:
                await self._backend.end_rollout_session_async(self._ctx)
            finally:
                self._finish_rollout_timing()

    def _finish_rollout_timing(self) -> None:
        """Record one rollout-session wall time for the current policy step."""

        if self._rollout_wall_start is None:
            return
        self._metric_timings["rollout"] = (
            self._metric_timings.get("rollout", 0.0) + time.perf_counter() - self._rollout_wall_start
        )
        self._rollout_wall_start = None

    def load_prompt_batches(
        self,
        dataset,
        *,
        batch_size: int,
        max_prompt_tokens: int,
        prompt_key: str = "prompt",
        solutions_key: str = "solutions",
    ) -> Iterable[PromptBatch]:
        """Yield tokenized prompt batches from a dataset-like object.

        Records whose prompt exceeds `max_prompt_tokens` are skipped. The full
        original record is preserved on each `PromptItem` so reward functions
        can read task-specific fields. The cursor advances even when records
        are skipped, so the iterator eventually walks the entire dataset.
        """

        cursor = 0
        total_skipped_long = 0
        while cursor < len(dataset):
            items = []
            scanned = 0
            skipped_long = 0
            # Keep scanning until we accumulate `batch_size` accepted rows or
            # exhaust the dataset; over-long prompts increment the skip counter
            # but do not fill the batch.
            while len(items) < batch_size and cursor < len(dataset):
                record = dataset[cursor]
                cursor += 1
                scanned += 1
                if prompt_key not in record:
                    raise ValueError(
                        f"dataset row must contain `{prompt_key}`; use --dataset-loader-fn to normalize raw rows"
                    )
                prompt = record[prompt_key]
                input_tokens = encode_generation_prompt(self._tokenizer, prompt)
                if len(input_tokens) > max_prompt_tokens:
                    skipped_long += 1
                    total_skipped_long += 1
                    continue
                items.append(
                    PromptItem(
                        prompt=prompt,
                        solutions=record[solutions_key] if solutions_key in record else None,
                        input_tokens=input_tokens,
                        record=dict(record),
                    )
                )
            if not items:
                break
            yield PromptBatch(
                items=items,
                scanned=scanned,
                skipped_long=skipped_long,
                total_skipped_long=total_skipped_long,
            )

    def rollout_batch(self, prompts: list[str], n_samples: int, sampling_params: SamplingParams) -> list[RolloutResult]:
        """Generate `n_samples` completions for each prompt in order."""

        prompt_tokens = [encode_generation_prompt(self._tokenizer, prompt) for prompt in prompts]
        return self.rollout_token_batch(prompt_tokens, n_samples, sampling_params)

    def rollout_token_batch(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
    ) -> list[RolloutResult]:
        """Generate completions for prompts that were already tokenized."""

        # Rollout is the natural boundary of a new policy step. Consecutive
        # rollouts before train stay on the same step instead of bumping twice.
        if self._rollout_session_depth <= 0:
            raise RuntimeError("rollout_token_batch must be called inside `async with trainer.rollout_session(...)`")
        self._begin_step()
        return self._backend.rollout_batch(
            self._ctx, _normalize_prompt_token_batch(prompt_tokens), n_samples, sampling_params
        )

    async def rollout_token_batch_async(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
    ) -> list[RolloutResult]:
        """Async rollout variant for request-concurrent callers."""

        if self._rollout_session_depth <= 0:
            raise RuntimeError(
                "rollout_token_batch_async must be called inside `async with trainer.rollout_session(...)`"
            )
        self._begin_step()
        rollout_async = getattr(self._backend, "rollout_batch_async")
        return await rollout_async(self._ctx, _normalize_prompt_token_batch(prompt_tokens), n_samples, sampling_params)

    def rollout_session(
        self,
        *,
        sampling_params: SamplingParams,
        loss_mask_policy: LossMaskPolicy | None = None,
        max_running_prompts: int | None = None,
        timeout_s: float = 300.0,
        proxy: bool = True,
    ) -> RolloutSession:
        """Create an async rollout session, optionally with an OpenAI-compatible proxy."""

        return RolloutSession(
            self,
            sampling_params=sampling_params,
            loss_mask_policy=loss_mask_policy,
            max_running_prompts=max_running_prompts,
            timeout_s=timeout_s,
            proxy=proxy,
        )

    def train(
        self,
        batch_data: list[TrainSequence],
        loss_fn: Callable,
        mini_bs: int = 8,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        """Run one backend training step with a caller-provided loss function.

        Returns whatever scalar metric dict the backend produces; when a
        `MetricsRecorder` is attached the dict and the accumulated step timings
        are also dispatched to TensorBoard.
        """

        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        self._begin_step()
        start = time.perf_counter()
        result = self._backend.train(self._ctx, batch_data, loss_fn, mini_bs, gradient_accumulation_steps)
        self._metric_timings["train"] = time.perf_counter() - start
        if isinstance(result, dict):
            if "rollout" in self._metric_timings:
                result["step_rollout_time_s"] = self._metric_timings["rollout"]
            result["step_train_time_s"] = self._metric_timings["train"]
            if self._step_wall_start is not None:
                result["step_e2e_time_s"] = time.perf_counter() - self._step_wall_start
        if self._metrics is not None:
            self._metrics.record_train_step(
                step=self._ctx.global_step,
                train_result=result,
                train_batch=batch_data,
                timings=self._metric_timings,
            )
        if self._health_checker is not None:
            try:
                self._health_checker.observe(step=self._ctx.global_step, train_result=result, train_batch=batch_data)
            except Exception:
                logger.warning("health checker observe failed", exc_info=True)
        self.finish_step()
        return result

    def record_rollout_sample(self, sample: dict[str, Any]) -> None:
        """Persist a representative rollout sample when metrics recording is enabled."""

        if self._metrics is not None:
            self._metrics.record_rollout_sample(sample)

    def record_dashboard_state(
        self,
        *,
        stage: str,
        step: int | None = None,
        epoch: int | None = None,
        role: str | None = None,
        status: str = "running",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist dashboard state independently from TensorBoard scalar events."""

        if self._metrics is not None:
            self._metrics.record_dashboard_state(
                stage=stage, step=step, epoch=epoch, role=role, status=status, extra=extra
            )

    def ensure_roles(self, roles: dict[str, ModelRole]) -> None:
        """Prepare backend-owned auxiliary model roles for algorithms like PPO."""

        self._backend.ensure_roles(self._ctx, roles)

    def score_logprobs(self, role: str, token_rows: list[list[int]], *, microbatch_size: int = 8) -> list[list[float]]:
        """Score fixed token sequences with a backend-owned model role."""

        return self._backend.score_logprobs(self._ctx, role, token_rows, microbatch_size=microbatch_size)

    def score_values(self, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        """Score per-token critic values with a backend-owned model role."""

        return self._backend.score_values(self._ctx, role, token_rows)

    def score_rewards(self, role: str, token_rows: list[list[int]]) -> list[float]:
        """Score sequence rewards with a backend-owned reward model role."""

        return self._backend.score_rewards(self._ctx, role, token_rows)

    def train_values(
        self,
        role: str,
        batch_data: list[TrainSequence],
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
        *,
        cliprange_value: float = 0.5,
        value_loss_coef: float = 0.5,
    ) -> dict[str, float]:
        """Train a backend-owned critic/value role.

        `cliprange_value` is the value-function clipping range from the PPO
        paper; `value_loss_coef` scales the MSE loss before it is added to the
        critic's objective.
        """

        return self._backend.train_values(
            self._ctx,
            role,
            batch_data,
            mini_bs,
            gradient_accumulation_steps,
            cliprange_value=cliprange_value,
            value_loss_coef=value_loss_coef,
        )

    def save_checkpoint(self, path: str) -> str:
        """Save a HuggingFace-compatible checkpoint when supported by backend."""

        return self._backend.save_checkpoint(self._ctx, path)

    def configure_health_check(
        self,
        config: HealthCheckConfig | None,
        *,
        artifact_dir: str | None = None,
    ) -> None:
        """Attach (or detach) the startup-window health checker.

        ``config`` or ``config.enabled=False`` detaches the checker so
        ``train()`` short-circuits and produces no artifact / metric / log.
        ``artifact_dir`` defaults to ``<metrics_log_dir>/health_check`` and is
        only used when a checker is actually attached.

        Called by trainer implementations at the top of ``fit()``; not part of
        the rollout/train hot path.
        """

        if config is None or not config.enabled:
            self._health_checker = None
            return
        if self._ctx is None:
            raise RuntimeError("Trainer is not initialized; call init() before configure_health_check()")
        sink = artifact_dir
        if sink is None:
            base = self._metrics.log_dir if self._metrics is not None else None
            sink = str(Path(base) / "health_check") if base is not None else None
        self._health_checker = TrainingHealthChecker(config, sink=sink, metrics=self._metrics)

    def record_rollout_skipped(self, skipped_long: int) -> None:
        """Feed rollout-side overlong-prompt skip counts into the health window.

        Optional: trainers that hold a `PromptBatch` with a ``skipped_long``
        field call this so the skipped-batches check sees the real count. When
        never called, the window assumes zero rollout skips for that step,
        which is correct for batches that contained no overlong prompts.

        Must be called **before** the corresponding ``train()`` call so the
        skips are attributed to the current step's window, not the next one.
        """

        if self._health_checker is not None:
            self._health_checker.record_skipped(skipped_long)

    def close(self) -> None:
        """Release backend workers and local resources such as metric writers."""

        # Each cleanup step is independent so a failure in one does not skip
        # the others (e.g. a metrics-writer error must not prevent the health
        # checker from writing its artifact).
        # Step 1: backend
        try:
            if self._backend is not None:
                self._backend.close()
        except Exception:
            logger.warning("backend close failed", exc_info=True)
        finally:
            self._backend = None
            self._initialized = False
        # Step 2: metrics
        try:
            if self._metrics is not None:
                self._metrics.close()
        except Exception:
            logger.warning("metrics close failed", exc_info=True)
        # Step 3: health checker — finalize the window early if the run ends
        # before the configured startup window completes.
        try:
            if self._health_checker is not None:
                self._health_checker.finalize_early()
        except Exception:
            logger.warning("health check finalize failed", exc_info=True)
        finally:
            self._health_checker = None


def _normalize_prompt_token_batch(prompt_tokens: list[list[int]]) -> list[list[int]]:
    return [normalize_token_ids(row) for row in prompt_tokens]


def _extract_grad_zero_ratio(train_result: Any) -> float | None:
    """Read the gradient-zero ratio from a backend train result.

    `ArenoBackend.train` flattens per-step metrics to top-level scalars
    (`{"loss": ..., "grad_zero_ratio": ...}`). Some callers/tests group them
    under a nested ``"metrics"`` dict; support both shapes.
    """

    if not isinstance(train_result, dict):
        return None
    value = train_result.get("grad_zero_ratio")
    if value is None:
        nested = train_result.get("metrics")
        if isinstance(nested, dict):
            value = nested.get("grad_zero_ratio")
    return value


class TrainingHealthChecker:
    """Coordinator-side startup-window health checker (Issue #249).

    Accumulates per-step signals from `Trainer.train()` (reusing
    `collect_train_batch_stats` for the sample-side signals and the backend
    train result for `loss` / `grad_zero_ratio`), evaluates the window once it
    fills, and writes a structured artifact + human-readable log + `health/*`
    TensorBoard scalars. `on_fail='warn'` (default) only logs; `on_fail='fail'`
    raises `HealthCheckError` carrying stage + input (never sample text).
    """

    _STATUS_VALUE = {"pass": 0, "warn": 1, "fail": 2}

    def __init__(
        self,
        config: HealthCheckConfig,
        *,
        sink: str | None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._config = config
        self._sink = Path(sink) if sink is not None else None
        if self._sink is not None:
            self._sink.mkdir(parents=True, exist_ok=True)
        self._metrics = metrics
        self._logger = logging.getLogger("areno.health_check")
        self._signals = WindowSignals()
        self._steps_seen = 0
        self._completed_at_step = 0
        self._finalized = False
        self._pending_skipped = 0

    def record_skipped(self, skipped_long: int) -> None:
        """Accumulate rollout-side overlong-prompt skips for the current step."""

        self._pending_skipped += int(skipped_long)

    def observe(self, *, step: int, train_result: Any, train_batch: list[TrainSequence]) -> None:
        """Feed one training step's signals; evaluate when the window fills."""

        if self._finalized:
            return
        # Reuse the existing sample-side summarizer so reward / response-length
        # signals are computed exactly as `MetricsRecorder` computes them.
        stats = collect_train_batch_stats(train_batch)
        self._signals.rewards.extend(float(r) for r in stats.get("rewards", []))
        response_lens = stats.get("response_len", [])
        # Per-batch effective token count = sum of response lengths this step.
        self._signals.effective_tokens_per_batch.append(int(sum(response_lens)))
        self._signals.total_batches += len(train_batch)
        self._signals.skipped_long += self._pending_skipped
        self._pending_skipped = 0
        # Backend-reported signals. `ArenoBackend.train` returns a flat dict
        # (`{"loss": ..., "grad_zero_ratio": ...}`); fall back to a nested
        # `train_result["metrics"]["grad_zero_ratio"]` for callers that group
        # diagnostics under a "metrics" key.
        loss = train_result.get("loss") if isinstance(train_result, dict) else None
        if loss is not None:
            self._signals.losses.append(float(loss))
        gz = _extract_grad_zero_ratio(train_result)
        if gz is not None:
            self._signals.grad_zero_ratios.append(float(gz))
        self._steps_seen += 1
        self._completed_at_step = step
        if self._steps_seen >= self._config.startup_window_updates:
            self._evaluate()

    def finalize_early(self) -> None:
        """Evaluate the partial window if the run ends before it fills.

        If the window is incomplete (``steps_seen < startup_window_updates``),
        a warning is logged so the user knows the result may be less reliable
        than a full-window evaluation.
        """

        if self._finalized or self._steps_seen == 0:
            self._finalized = True
            return
        if self._steps_seen < self._config.startup_window_updates:
            self._logger.warning(
                "health check window incomplete: steps_seen=%d < startup_window_updates=%d; "
                "results may be less reliable",
                self._steps_seen,
                self._config.startup_window_updates,
            )
        self._evaluate()

    def _evaluate(self) -> None:
        """Run the pure checks, emit outputs, and optionally raise."""

        if self._finalized:
            return
        self._finalized = True
        report = run_health_check(
            self._config,
            self._signals,
            completed_at_step=self._completed_at_step,
        )
        if report is None:
            return
        self._write_artifact(report)
        self._emit_metrics(report)
        self._log_report(report)
        if report.summary == "fail" and self._config.on_fail == "fail":
            stages = sorted({c.stage for c in report.checks if c.status == "fail"})
            inputs = [c.input for c in report.checks if c.status == "fail" and c.input != "-"]
            detail = "; ".join(c.message for c in report.checks if c.status == "fail")
            raise HealthCheckError(
                f"health-check FAIL stage={','.join(stages)} input={','.join(inputs)} detail={detail}"
            )

    def _write_artifact(self, report: HealthReport) -> None:
        if self._sink is None:
            return
        path = self._sink / f"{report.run_id}.json"
        path.write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _emit_metrics(self, report: HealthReport) -> None:
        # Emit through the shared TensorBoard writer when one is attached, so
        # the `health/*` namespace lives alongside `rollout/*` and `train/*`.
        if self._metrics is None:
            return
        step = report.completed_at_step
        self._metrics.add_scalar("health/summary", float(self._STATUS_VALUE[report.summary]), step)
        for check in report.checks:
            self._metrics.add_scalar(f"health/{check.name}", float(self._STATUS_VALUE[check.status]), step)

    def _log_report(self, report: HealthReport) -> None:
        window = self._config.startup_window_updates
        self._logger.info(
            "stage=health_check startup_window=%d updates completed_at_step=%d",
            window,
            report.completed_at_step,
        )
        for check in report.checks:
            self._logger.info(
                "stage=health_check name=%s status=%s stage=%s msg=%s metric_ref=%s",
                check.name,
                check.status,
                check.stage,
                check.message,
                check.metric_ref,
            )
        artifact = str(self._sink / f"{report.run_id}.json") if self._sink is not None else "n/a"
        self._logger.info(
            "stage=health_check SUMMARY=%s artifact=%s",
            report.summary,
            artifact,
        )
