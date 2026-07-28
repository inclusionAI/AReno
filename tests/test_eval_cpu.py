"""CPU-side unit tests for periodic evaluation during SFT/DPO training.

Task 1: Evaluation config and CLI entrypoint.
Task 2: Trainer.evaluate() core method.
Task 3: Evaluation metric recording.
Task 4: SFT training evaluation integration.
Task 5: DPO training evaluation integration.
"""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim

from areno import Trainer
from areno.api.context import Context
from areno.api.models import TrainSequence


def _default_sft_options(**overrides):
    """Return kwargs for _trainer_config_from_options with SFT-safe defaults.

    Every attribute that _trainer_config_from_options reads directly (i.e. not
    via ``getattr``) must be present; missing values cause an
    ``AttributeError`` deep inside the validation chain before our eval check
    ever fires.
    """
    base = {
        "algo": "sft",
        "ckpt": "/fake/ckpt",
        "dataset_path": "/fake/train.jsonl",
        "dataset_loader_fn": "loader.py:load",
        "model_hub": "modelscope",
        "save_interval": 100,
        "epochs": 10,
        "tp_size": 2,
        "world_size": 2,
        "batch_size": 4,
        "mini_bs": 2,
        "max_prompt_tokens": 512,
        "max_new_tokens": 512,
        "max_context_len": None,
        "agent_timeout_s": 300.0,
        "gradient_accumulation_steps": None,
        "lr": 1.0e-6,
        "min_lr": 1.0e-7,
        "lr_decay_steps": 100,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 1.0e-2,
        "grad_clip_norm": 1.0,
        "critic_warmup_steps": 0,
        "eval_dataset_path": None,
        "eval_interval": 0,
        "eval_batches": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Helpers shared across Task 2 / Task 4 / Task 5
# ---------------------------------------------------------------------------


class _FakeEvalModel(nn.Module):
    """Minimal module with a trainable parameter for eval safety tests."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.weight


class _EvalBackendStub:
    """Backend stub: tracks model mode and has a simple evaluate() impl."""

    def __init__(self, model: _FakeEvalModel, optimizer: optim.Optimizer | None = None):
        self._model = model
        self._optimizer = optimizer
        self.eval_calls = 0

    def evaluate(
        self,
        ctx: Context,
        batch_data: list,
        loss_fn,
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        self.eval_calls += 1
        self._model.eval()
        try:
            # Simulate forward pass + loss computation.
            _ = self._model(torch.zeros(1, 4, dtype=torch.long))
            return {"loss": 0.5, "sft_loss": 0.5, "sft_target_tokens": 4.0}
        finally:
            self._model.train()


# ---------------------------------------------------------------------------
# Task 1: Evaluation config and CLI entrypoint
# ---------------------------------------------------------------------------


class EvalCliConfigTest(unittest.TestCase):
    """Tests for eval CLI options and TrainerConfig fields."""

    # -- _trainer_config_from_options ----------------------------------------

    def test_eval_invalid_dataset_path(self):
        """Non-existent --eval-dataset-path raises click.UsageError with path info."""
        from areno.cli.train import _trainer_config_from_options

        with tempfile.TemporaryDirectory() as td:
            non_existent = Path(td) / "does_not_exist.jsonl"
            options = _default_sft_options(eval_dataset_path=str(non_existent))
            with self.assertRaises(click.UsageError) as ctx:
                _trainer_config_from_options(**options)
            self.assertIn("--eval-dataset-path", str(ctx.exception))
            self.assertIn(str(non_existent), str(ctx.exception))


# ---------------------------------------------------------------------------
# Task 2: Trainer.evaluate() core method
# ---------------------------------------------------------------------------


class EvalTrainerCoreTest(unittest.TestCase):
    """Tests for Trainer.evaluate() — param safety, mode restore, optimizer."""

    def _make_trainer(self) -> tuple[Trainer, _FakeEvalModel, _EvalBackendStub]:
        model = _FakeEvalModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        backend = _EvalBackendStub(model, optimizer)
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._backend = backend
        trainer._ctx = Context(1, "unused", object())
        trainer._initialized = True
        return trainer, model, backend

    @staticmethod
    def _dummy_loss_fn(_data_pack, _logprobs):
        return torch.tensor(0.0), {"loss": torch.tensor(0.0)}

    @staticmethod
    def _make_simple_batch() -> list[TrainSequence]:
        seq = TrainSequence(
            prompt_mask=[True, True, True, False],
            tokens=[1, 2, 3, 4],
            logprobs=[-0.1, -0.2, -0.3, -0.4],
            advantages=[0.0, 0.0, 0.0, 0.0],
            eos_token_id=0,
        )
        return [seq]

    # -- param safety ---------------------------------------------------------

    def test_eval_params_unchanged(self):
        """All model parameter values must be identical before and after eval."""
        trainer, model, _ = self._make_trainer()
        batch = self._make_simple_batch()

        params_before = {name: param.clone() for name, param in model.named_parameters()}
        trainer.evaluate(batch, self._dummy_loss_fn, mini_bs=4)
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.equal(params_before[name], param),
                f"parameter '{name}' changed after evaluate()",
            )

    # -- model mode -----------------------------------------------------------

    def test_eval_model_mode_restored(self):
        """After evaluate(), model.training must be True."""
        trainer, model, _ = self._make_trainer()
        batch = self._make_simple_batch()

        self.assertTrue(model.training, "model should be in train mode before eval")
        trainer.evaluate(batch, self._dummy_loss_fn, mini_bs=4)
        self.assertTrue(model.training, "model should be in train mode after eval")

    # -- optimizer safety -----------------------------------------------------

    def test_eval_optimizer_unchanged(self):
        """Optimizer step count and momentum state must be unchanged by eval."""
        trainer, model, backend = self._make_trainer()

        # Do one real optimizer step so there is meaningful state to check.
        opt: optim.SGD = backend._optimizer
        opt.zero_grad()
        loss = model(torch.zeros(1, 4, dtype=torch.long)).sum()
        loss.backward()
        opt.step()
        step_after_train = copy.deepcopy(opt.state_dict())["state"]

        batch = self._make_simple_batch()
        trainer.evaluate(batch, self._dummy_loss_fn, mini_bs=4)
        step_after_eval = copy.deepcopy(opt.state_dict())["state"]

        # Compare per-parameter momentum buffers — must be identical.
        for param_id, state_after_train in step_after_train.items():
            self.assertIn(param_id, step_after_eval, f"param {param_id} missing after eval")
            state_after_eval = step_after_eval[param_id]
            for key in state_after_train:
                if isinstance(state_after_train[key], torch.Tensor):
                    self.assertTrue(
                        torch.equal(state_after_train[key], state_after_eval[key]),
                        f"opt state '{key}' for param {param_id} changed",
                    )
                else:
                    self.assertEqual(state_after_train[key], state_after_eval[key])


# ---------------------------------------------------------------------------
# Task 3: Evaluation metric recording
# ---------------------------------------------------------------------------


class EvalMetricsRecorderTest(unittest.TestCase):
    """Tests for MetricsRecorder.record_eval_step() — eval/ namespace."""

    def test_eval_metrics_namespace(self):
        """Verify eval/ and train/ namespaces are independent in TensorBoard events."""
        import tensorboard.backend.event_processing.event_accumulator as ea

        with tempfile.TemporaryDirectory() as log_dir:
            from areno.api.metrics import MetricsRecorder

            recorder = MetricsRecorder(log_dir)

            # Record a training step (writes train/*, rollout/* tags).
            seq = TrainSequence(
                prompt_mask=[True, False],
                tokens=[1, 2],
                logprobs=[-0.1, -0.2],
                advantages=[0.0, 0.1],
                reward=1.0,
            )
            train_result = {"loss": 0.5, "sft_loss": 0.5}
            recorder.record_train_step(step=10, train_result=train_result, train_batch=[seq])

            # Record an eval step (writes eval/* tags).
            eval_result = {"sft_loss": 2.3, "sample_count": 50.0, "duration_s": 1.2}
            recorder.record_eval_step(step=10, eval_result=eval_result)

            recorder.close()

            # Read back TensorBoard events and verify namespace separation.
            acc = ea.EventAccumulator(log_dir)
            acc.Reload()
            scalar_tags = acc.Tags()["scalars"]

            # eval/ tags must exist with the expected keys.
            eval_tags = [t for t in scalar_tags if t.startswith("eval/")]
            self.assertEqual(len(eval_tags), 3, f"Expected 3 eval tags, got {eval_tags}")
            self.assertIn("eval/sft_loss", eval_tags)
            self.assertIn("eval/sample_count", eval_tags)
            self.assertIn("eval/duration_s", eval_tags)

            # train/ tags exist and were not overwritten.
            train_tags = [t for t in scalar_tags if t.startswith("train/")]
            self.assertGreater(len(train_tags), 0, "train/ tags should still exist")

            # No overlap between eval/ and any other namespace (train/, rollout/, time/).
            eval_set = {t for t in scalar_tags if t.startswith("eval/")}
            train_set = {t for t in scalar_tags if t.startswith("train/")}
            rollout_set = {t for t in scalar_tags if t.startswith("rollout/")}
            time_set = {t for t in scalar_tags if t.startswith("time/")}

            self.assertEqual(
                len(eval_set & train_set), 0,
                "eval/ and train/ namespaces must not overlap",
            )
            self.assertEqual(
                len(eval_set & rollout_set), 0,
                "eval/ and rollout/ namespaces must not overlap",
            )
            self.assertEqual(
                len(eval_set & time_set), 0,
                "eval/ and time/ namespaces must not overlap",
            )


# ---------------------------------------------------------------------------
# Shared helpers for Task 4 / Task 5
# ---------------------------------------------------------------------------


class _FakeTextTokenizer:
    """Tokeniser double that returns small int ids for eval row conversion."""

    eos_token_id = 99
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) % 50 + 1 for ch in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        ids: list[int] = []
        for message in messages:
            ids.extend(self.encode(f"{message.get('role')}:{message.get('content')}"))
        return ids


class _FakeEvalSFTBackend:
    """Backend double for SFT eval integration tests.

    Supports ``init()``, ``close()``, ``get_tokenizer()``, ``train()``,
    ``evaluate()``, and an optional ``_metrics`` attribute.
    """

    def __init__(self, metrics=None):
        self.closed = False
        self.train_calls = 0
        self.eval_calls = 0
        self._metrics = metrics
        self._saved: list[str] = []

    def init(self):
        return None

    def close(self):
        self.closed = True

    def get_tokenizer(self):
        return _FakeTextTokenizer()

    def train(self, _batch, _loss_fn, *, mini_bs, gradient_accumulation_steps):
        del mini_bs, gradient_accumulation_steps
        self.train_calls += 1
        return {"sft_loss": 1.0, "sft_target_tokens": 4.0}

    def evaluate(self, batch_data, _loss_fn, *, mini_bs, gradient_accumulation_steps=None):
        del mini_bs, gradient_accumulation_steps
        self.eval_calls += 1
        return {"sft_loss": 0.5, "sft_target_tokens": float(len(batch_data) * 4)}

    def save_checkpoint(self, _ctx, path):
        self._saved.append(path)
        return path


_FIXTURE_EVAL_DATA: list[dict] = [
    {"prompt": "q1", "response": "a1"},
    {"prompt": "q2", "response": "a2"},
    {"prompt": "q3", "response": "a3"},
    {"prompt": "q4", "response": "a4"},
    {"prompt": "q5", "response": "a5"},
]

_FIXTURE_TRAIN_DATA: list[dict] = [
    {"prompt": "q1", "response": "a1"},
    {"prompt": "q2", "response": "a2"},
]


def _sft_eval_config(**overrides):
    """Return a SimpleNamespace with the fields SFTTrainer reads."""
    from types import SimpleNamespace

    defaults = {
        "batch_size": 2,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "max_new_tokens": 10,
        "max_prompt_tokens": 10,
        "mini_bs": 1,
        "save_interval": 100,
        "save_path": None,
        "max_steps": None,
        "eval_dataset_path": None,
        "eval_interval": 0,
        "eval_batches": 0,
        "model_hub": "hf",
        "dataset_loader_fn": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sft_trainer(config, *, train_data=None, backend=None, _metrics=None):
    """Build an SFTTrainer with a controlled backend for eval testing."""
    from areno.api.trainers.sft import SFTTrainer

    if backend is None:
        backend = _FakeEvalSFTBackend(metrics=_metrics)
    if train_data is None:
        train_data = list(_FIXTURE_TRAIN_DATA)
    loss_fn = lambda _pack, _logprobs: None
    trainer = SFTTrainer(config, instance=backend, dataset=train_data, reward_fn=None, loss_fn=loss_fn)
    return trainer, backend


# ---------------------------------------------------------------------------
# Task 4: SFT training evaluation integration
# ---------------------------------------------------------------------------


class SFTEvalIntegrationTest(unittest.TestCase):
    """Tests for SFTTrainer eval integration: _run_eval + _fit_initialized hooks."""

    # -- _eval_enabled ---------------------------------------------------------

    def test_eval_disabled_by_default(self):
        """_eval_enabled is False when eval_dataset_path is None."""
        config = _sft_eval_config(eval_dataset_path=None)
        trainer, backend = _make_sft_trainer(config)
        self.assertFalse(trainer._eval_enabled)
        self.assertEqual(backend.train_calls, 0)
        self.assertEqual(backend.eval_calls, 0)

    # -- _run_eval (direct call) ----------------------------------------------

    def test_eval_sft_success(self):
        """SFT eval returns correct metrics when called directly."""
        from areno.api.metrics import MetricsRecorder

        with tempfile.TemporaryDirectory() as log_dir:
            metrics = MetricsRecorder(log_dir)
            config = _sft_eval_config(
                eval_dataset_path="dummy",
                eval_interval=2,
                eval_batches=0,
                batch_size=2,
            )
            trainer, backend = _make_sft_trainer(config, _metrics=metrics)
            trainer._eval_dataset = list(_FIXTURE_EVAL_DATA)

            trainer._run_eval(step=10)

            # All 5 fixture rows should be processed (batch_size=2 => 3 batches).
            self.assertEqual(backend.eval_calls, 3)
            # Verify metrics were recorded.
            import tensorboard.backend.event_processing.event_accumulator as ea

            metrics.close()
            acc = ea.EventAccumulator(log_dir)
            acc.Reload()
            eval_tags = [t for t in acc.Tags()["scalars"] if t.startswith("eval/")]
            self.assertIn("eval/sft_loss", eval_tags)
            self.assertIn("eval/sft_logprob_mean", eval_tags)
            self.assertIn("eval/sft_target_tokens", eval_tags)
            self.assertIn("eval/sample_count", eval_tags)
            self.assertIn("eval/duration_s", eval_tags)

    def test_eval_batches_limit(self):
        """eval_batches=2 processes at most 2 batches."""
        config = _sft_eval_config(
            eval_dataset_path="dummy",
            eval_batches=2,
            batch_size=1,
        )
        trainer, backend = _make_sft_trainer(config)
        trainer._eval_dataset = list(_FIXTURE_EVAL_DATA)

        trainer._run_eval(step=5)

        self.assertEqual(backend.eval_calls, 2)

    def test_eval_batches_full(self):
        """eval_batches=0 traverses the full eval dataset."""
        config = _sft_eval_config(
            eval_dataset_path="dummy",
            eval_batches=0,
            batch_size=1,
        )
        trainer, backend = _make_sft_trainer(config)
        trainer._eval_dataset = list(_FIXTURE_EVAL_DATA)

        trainer._run_eval(step=5)

        # 5 rows / batch_size=1 => 5 batches.
        self.assertEqual(backend.eval_calls, 5)

    def test_eval_dataset_empty(self):
        """Empty eval dataset raises ValueError."""
        config = _sft_eval_config(eval_dataset_path="dummy")
        trainer, backend = _make_sft_trainer(config)
        trainer._eval_dataset = []

        with self.assertRaisesRegex(ValueError, "empty"):
            trainer._run_eval(step=0)

    # -- _fit_initialized integration -----------------------------------------

    def test_eval_interval_trigger(self):
        """eval_interval=2 triggers eval at steps 2, 4, ..."""
        train_data = [_FIXTURE_TRAIN_DATA[0].copy() for _ in range(10)]
        config = _sft_eval_config(
            eval_dataset_path="dummy",
            eval_interval=2,
            eval_batches=1,
            batch_size=1,
            epochs=1,
        )
        trainer, backend = _make_sft_trainer(config, train_data=train_data)
        trainer._eval_dataset = list(_FIXTURE_EVAL_DATA)

        eval_steps: list[int] = []
        _orig = trainer._run_eval

        def _spy(step):
            eval_steps.append(step)
            _orig(step)

        trainer._run_eval = _spy  # type: ignore[method-assign]
        trainer.fit()

        # 10 rows at batch_size=1 => 10 train steps.
        # interval=2 => evals at steps 2, 4, 6, 8, 10 + epoch-end at step 10.
        self.assertEqual(backend.train_calls, 10)
        self.assertEqual(len(eval_steps), 6)
        self.assertEqual(eval_steps, [2, 4, 6, 8, 10, 10])
        self.assertTrue(backend.closed)

    def _count_eval_calls_from_fit(self, config_overrides, train_rows=6):
        """Helper that runs fit() and returns (train_calls, run_eval_steps).

        Uses a spy on ``_run_eval`` so we count full eval passes, not
        individual batch-evaluate calls.
        """
        train_data = [_FIXTURE_TRAIN_DATA[0].copy() for _ in range(train_rows)]
        config = _sft_eval_config(**config_overrides)
        trainer, backend = _make_sft_trainer(config, train_data=train_data)
        trainer._eval_dataset = list(_FIXTURE_EVAL_DATA)

        eval_steps: list[int] = []
        _orig = trainer._run_eval

        def _spy(step):
            eval_steps.append(step)
            _orig(step)

        trainer._run_eval = _spy  # type: ignore[method-assign]
        trainer.fit()
        return backend.train_calls, eval_steps

    def test_eval_end_of_epoch(self):
        """Epoch end triggers eval regardless of interval alignment."""
        train_calls, eval_steps = self._count_eval_calls_from_fit(
            {"eval_dataset_path": "dummy", "eval_interval": 100, "eval_batches": 0, "batch_size": 2, "epochs": 1},
            train_rows=4,
        )
        self.assertEqual(train_calls, 2)  # 4 rows / batch_size=2 = 2 batches
        # Only end-of-epoch eval should fire (interval is too large).
        self.assertEqual(len(eval_steps), 1)
        self.assertEqual(eval_steps[0], 2)  # step=2 after 2 batches

    def test_eval_end_of_max_steps(self):
        """max_steps exit triggers eval before return."""
        train_calls, eval_steps = self._count_eval_calls_from_fit(
            {"eval_dataset_path": "dummy", "eval_interval": 0, "eval_batches": 0,
             "batch_size": 1, "epochs": 2, "max_steps": 3},
            train_rows=10,
        )
        self.assertEqual(train_calls, 3)
        # max_steps exit eval fires once at step 3.
        self.assertEqual(len(eval_steps), 1)
        self.assertEqual(eval_steps[0], 3)

    def test_eval_interval_zero_skips_interval_only(self):
        """eval_interval=0 only triggers end-of-epoch eval (no interval eval)."""
        # interval=0, 4 rows, batch_size=1 => 4 train steps, 1 epoch-end eval.
        train_calls, eval_steps = self._count_eval_calls_from_fit(
            {"eval_dataset_path": "dummy", "eval_interval": 0, "eval_batches": 0, "batch_size": 1, "epochs": 1},
            train_rows=4,
        )
        self.assertEqual(train_calls, 4)
        # Only end-of-epoch eval, at step 4.
        self.assertEqual(len(eval_steps), 1)
        self.assertEqual(eval_steps[0], 4)

    def test_sft_train_unchanged_without_eval(self):
        """Training is identical to baseline when eval is disabled."""
        train_data = [_FIXTURE_TRAIN_DATA[0].copy() for _ in range(4)]
        config = _sft_eval_config(
            eval_dataset_path=None,
            eval_interval=0,
            eval_batches=0,
            batch_size=1,
            epochs=1,
        )
        trainer, backend = _make_sft_trainer(config, train_data=train_data)
        trainer.fit()

        # All 4 rows produced valid training batches; no eval calls.
        self.assertEqual(backend.train_calls, 4)
        self.assertEqual(backend.eval_calls, 0)
        self.assertTrue(backend.closed)

    def test_sft_eval_integration(self):
        """Full SFT train + eval end-to-end: verify eval/ metrics recorded."""
        from areno.api.metrics import MetricsRecorder

        with tempfile.TemporaryDirectory() as log_dir:
            metrics = MetricsRecorder(log_dir)
            train_data = [_FIXTURE_TRAIN_DATA[0].copy() for _ in range(6)]
            config = _sft_eval_config(
                eval_dataset_path="dummy",
                eval_interval=2,
                eval_batches=0,
                batch_size=2,
                epochs=1,
            )
            trainer, backend = _make_sft_trainer(config, train_data=train_data, _metrics=metrics)
            trainer._eval_dataset = list(_FIXTURE_EVAL_DATA)
            trainer.fit()
            metrics.close()

            # Verify eval/ metrics exist in TensorBoard events.
            import tensorboard.backend.event_processing.event_accumulator as ea

            acc = ea.EventAccumulator(log_dir)
            acc.Reload()
            scalar_tags = acc.Tags()["scalars"]
            eval_tags = [t for t in scalar_tags if t.startswith("eval/")]
            self.assertGreater(len(eval_tags), 0, "eval/ tags should exist")
            self.assertIn("eval/sft_loss", eval_tags)
            self.assertIn("eval/sft_logprob_mean", eval_tags)
            self.assertIn("eval/sample_count", eval_tags)
            self.assertIn("eval/duration_s", eval_tags)

            # Verify backend calls.
            self.assertEqual(backend.train_calls, 3)  # 6 rows / batch_size=2
            self.assertGreater(backend.eval_calls, 0)


# ---------------------------------------------------------------------------
# Shared helpers for Task 5 (DPO)
# ---------------------------------------------------------------------------


class _FakeEvalDPOBackend:
    """Backend double for DPO eval integration tests.

    Supports ``init()``, ``close()``, ``get_tokenizer()``, ``train()``,
    ``evaluate()``, ``save_checkpoint()``, ``score_logprobs()``, and
    ``ensure_roles()``.
    """

    def __init__(self, metrics=None):
        self.closed = False
        self.train_calls = 0
        self.eval_calls = 0
        self.ref_score_calls = 0
        self._metrics = metrics
        self._saved: list[str] = []

    def init(self):
        return None

    def close(self):
        self.closed = True

    def get_tokenizer(self):
        return _FakeTextTokenizer()

    def train(self, _batch, _loss_fn, *, mini_bs, gradient_accumulation_steps):
        del mini_bs, gradient_accumulation_steps
        self.train_calls += 1
        return {"dpo_loss": 0.69, "total_loss": 0.69, "dpo_accuracy": 0.75}

    def evaluate(self, batch_data, _loss_fn, *, mini_bs, gradient_accumulation_steps=None):
        del mini_bs, gradient_accumulation_steps
        self.eval_calls += 1
        return {"dpo_loss": 0.5, "dpo_accuracy": 0.8, "dpo_margin": 0.3}

    def save_checkpoint(self, _ctx, path):
        self._saved.append(path)
        return path

    def score_logprobs(self, role, token_rows, microbatch_size=None):
        del role, microbatch_size
        self.ref_score_calls += 1
        return [[-0.1 * float(i + 1) for i in range(len(row))] for row in token_rows]

    def ensure_roles(self, roles):
        pass


_FIXTURE_DPO_EVAL_DATA: list[dict] = [
    {"prompt": "q1", "chosen": "good_a1", "rejected": "bad_a1"},
    {"prompt": "q2", "chosen": "good_a2", "rejected": "bad_a2"},
    {"prompt": "q3", "chosen": "good_a3", "rejected": "bad_a3"},
    {"prompt": "q4", "chosen": "good_a4", "rejected": "bad_a4"},
]

_FIXTURE_DPO_TRAIN_DATA: list[dict] = [
    {"prompt": "qt1", "chosen": "gt1", "rejected": "bt1"},
    {"prompt": "qt2", "chosen": "gt2", "rejected": "bt2"},
]


def _dpo_eval_config(**overrides):
    """Return a SimpleNamespace with the fields DPOTrainer reads."""
    from types import SimpleNamespace

    defaults = {
        "algo": "dpo",
        "batch_size": 2,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "max_new_tokens": 10,
        "max_prompt_tokens": 10,
        "mini_bs": 2,
        "save_interval": 100,
        "save_path": None,
        "max_steps": None,
        "eval_dataset_path": None,
        "eval_interval": 0,
        "eval_batches": 0,
        "model_hub": "hf",
        "dataset_loader_fn": None,
        "dpo_beta": 0.1,
        "score_micro_bs": 2,
        "ref_ckpt": None,
        "ckpt": "/fake/ckpt",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_dpo_trainer(config, *, train_data=None, backend=None, _metrics=None):
    """Build a DPOTrainer with a controlled backend for eval testing."""
    from functools import partial

    from areno.api.trainers.dpo import DPOTrainer

    if backend is None:
        backend = _FakeEvalDPOBackend(metrics=_metrics)
    if train_data is None:
        train_data = list(_FIXTURE_DPO_TRAIN_DATA)

    loss_fn = partial(lambda _pack, _logprobs, *, beta: None, beta=config.dpo_beta)
    trainer = DPOTrainer(config, instance=backend, dataset=train_data, reward_fn=None, loss_fn=loss_fn)
    return trainer, backend


# ---------------------------------------------------------------------------
# Task 5: DPO training evaluation integration
# ---------------------------------------------------------------------------


class DPOEvalIntegrationTest(unittest.TestCase):
    """Tests for DPOTrainer eval integration: _run_eval + _fit_initialized hooks."""

    # -- _run_eval (direct call) -----------------------------------------------

    def test_eval_dpo_success(self):
        """DPO eval returns correct metrics when called directly."""
        from areno.api.metrics import MetricsRecorder

        with tempfile.TemporaryDirectory() as log_dir:
            metrics = MetricsRecorder(log_dir)
            config = _dpo_eval_config(
                eval_dataset_path="dummy",
                eval_interval=2,
                eval_batches=0,
                batch_size=2,
            )
            trainer, backend = _make_dpo_trainer(config, _metrics=metrics)
            trainer._eval_dataset = list(_FIXTURE_DPO_EVAL_DATA)

            # Roles must already be ensured for "ref" score_logprobs to work.
            trainer._ensure_roles()

            trainer._run_eval(step=10)

            # 4 fixture rows => 4 pairs. batch_size=2 pairs => 2 batches.
            # Each batch: 2 pairs = 4 sequences. 2 batches * 4 seq = 8 seq total.
            self.assertEqual(backend.eval_calls, 2)
            # ref_logprobs scoring: call per batch.
            self.assertEqual(backend.ref_score_calls, 2)

            # Verify metrics were recorded in TensorBoard.
            import tensorboard.backend.event_processing.event_accumulator as ea

            metrics.close()
            acc = ea.EventAccumulator(log_dir)
            acc.Reload()
            eval_tags = [t for t in acc.Tags()["scalars"] if t.startswith("eval/")]
            self.assertIn("eval/dpo_loss", eval_tags)
            self.assertIn("eval/dpo_accuracy", eval_tags)
            self.assertIn("eval/dpo_margin", eval_tags)
            self.assertIn("eval/sample_count", eval_tags)
            self.assertIn("eval/duration_s", eval_tags)

    def test_dpo_eval_integration(self):
        """Full DPO train + eval end-to-end: verify eval/ metrics recorded."""
        from areno.api.metrics import MetricsRecorder

        with tempfile.TemporaryDirectory() as log_dir:
            metrics = MetricsRecorder(log_dir)
            train_data = [_FIXTURE_DPO_TRAIN_DATA[0].copy() for _ in range(6)]
            config = _dpo_eval_config(
                eval_dataset_path="dummy",
                eval_interval=2,
                eval_batches=0,
                batch_size=2,
                epochs=1,
            )
            trainer, backend = _make_dpo_trainer(config, train_data=train_data, _metrics=metrics)
            trainer._eval_dataset = list(_FIXTURE_DPO_EVAL_DATA)
            trainer.fit()
            metrics.close()

            # Verify eval/ metrics exist in TensorBoard events.
            import tensorboard.backend.event_processing.event_accumulator as ea

            acc = ea.EventAccumulator(log_dir)
            acc.Reload()
            scalar_tags = acc.Tags()["scalars"]
            eval_tags = [t for t in scalar_tags if t.startswith("eval/")]
            self.assertGreater(len(eval_tags), 0, "eval/ tags should exist")
            self.assertIn("eval/dpo_loss", eval_tags)
            self.assertIn("eval/dpo_accuracy", eval_tags)
            self.assertIn("eval/sample_count", eval_tags)
            self.assertIn("eval/duration_s", eval_tags)

            # Verify backend calls.
            # 6 train rows at batch_size=2 pairs => 3 train batches (each is 4 rows).
            self.assertEqual(backend.train_calls, 3)
            self.assertGreater(backend.eval_calls, 0)