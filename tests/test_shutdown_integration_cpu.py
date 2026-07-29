from __future__ import annotations

import signal
from types import SimpleNamespace

from areno.api.trainers.dpo import DPOTrainer
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.api.trainers.ppo import PPOTrainer
from areno.api.trainers.sft import SFTTrainer
from areno.cli.serve import _run_uvicorn_with_graceful_shutdown
from areno.engine.shutdown import GracefulShutdown, ShutdownStage


class _FakeTrainerInstance:
    def __init__(self):
        self.train_calls = 0
        self.close_calls = []

    def init(self):
        return None

    def close(self, *, shutdown_info=None):
        self.close_calls.append(shutdown_info)

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=1)

    def load_prompt_batches(self, *_args, **_kwargs):
        return [SimpleNamespace(items=[])]

    def train(self, *_args, **_kwargs):
        self.train_calls += 1
        return {}


def test_policy_trainer_stops_after_rollout_before_materializing_new_train_work():
    instance = _FakeTrainerInstance()
    config = SimpleNamespace(
        epochs=1,
        batch_size=1,
        max_prompt_tokens=8,
        greedy=True,
        temperature=1.0,
        max_new_tokens=4,
        max_context_len=None,
        top_k=-1,
        top_p=1.0,
        max_steps=None,
        chat_template_enable_thinking=None,
    )
    trainer = PolicyOnlyTrainer(config, instance=instance, dataset=[], reward_fn=None, loss_fn=lambda *_args: None)
    trainer._agentic_enabled = lambda: False

    shutdown = GracefulShutdown(deadline_s=10)

    async def rollout(*_args):
        shutdown.set_stage(ShutdownStage.ROLLOUT)
        shutdown._simulate_signal(signal.SIGINT)
        return []

    trainer._run_prompt_rollout = rollout
    trainer._record_sample_completions = lambda *_args: None
    trainer._materialize_train_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("training work was scheduled after shutdown")
    )

    trainer.fit(shutdown=shutdown)
    shutdown.complete_shutdown()

    assert instance.train_calls == 0
    assert len(instance.close_calls) == 1
    assert instance.close_calls[0]["stage"] == "rollout"


def test_sft_honors_shutdown_before_first_train_batch_and_closes_once():
    instance = _FakeTrainerInstance()
    config = SimpleNamespace(
        epochs=1,
        max_prompt_tokens=8,
        max_new_tokens=4,
        mini_bs=2,
        gradient_accumulation_steps=1,
        max_steps=None,
        chat_template_enable_thinking=None,
    )
    trainer = SFTTrainer(config, instance=instance, dataset=[], reward_fn=None, loss_fn=lambda *_args: None)
    trainer._iter_train_batches = lambda *_args, **_kwargs: [[object()]]
    shutdown = GracefulShutdown(deadline_s=10)
    shutdown.set_stage(ShutdownStage.TRAINING)
    shutdown._simulate_signal(signal.SIGTERM)

    trainer.fit(shutdown=shutdown)
    shutdown.complete_shutdown()

    assert instance.train_calls == 0
    assert len(instance.close_calls) == 1
    assert instance.close_calls[0]["signal_number"] == signal.SIGTERM


def test_dpo_stops_after_reference_scoring_before_policy_training():
    shutdown = GracefulShutdown(deadline_s=10)

    class DPOInstance(_FakeTrainerInstance):
        def ensure_roles(self, _roles):
            return None

        def score_logprobs(self, _role, token_rows, *, microbatch_size):
            del microbatch_size
            shutdown._simulate_signal(signal.SIGINT)
            return [[0.0] * len(tokens) for tokens in token_rows]

    instance = DPOInstance()
    config = SimpleNamespace(
        ckpt="actor",
        ref_ckpt="ref",
        dpo_beta=0.1,
        epochs=1,
        max_prompt_tokens=8,
        max_new_tokens=4,
        score_micro_bs=1,
        mini_bs=2,
        gradient_accumulation_steps=1,
        max_steps=None,
        chat_template_enable_thinking=None,
    )
    trainer = DPOTrainer(config, instance=instance, dataset=[], reward_fn=None, loss_fn=lambda *_args: None)
    trainer._iter_train_batches = lambda *_args, **_kwargs: [
        [SimpleNamespace(tokens=[1, 2]), SimpleNamespace(tokens=[1, 3])]
    ]
    shutdown.set_stage(ShutdownStage.TRAINING)

    trainer.fit(shutdown=shutdown)
    shutdown.complete_shutdown()

    assert instance.train_calls == 0
    assert len(instance.close_calls) == 1
    assert instance.close_calls[0]["stage"] == "training"


def test_ppo_fit_accepts_and_propagates_shutdown():
    instance = _FakeTrainerInstance()
    trainer = object.__new__(PPOTrainer)
    trainer.areno = instance
    trainer._ensure_roles = lambda: None
    captured = []
    trainer._fit_initialized = lambda *, shutdown=None: captured.append(shutdown)
    shutdown = GracefulShutdown(deadline_s=10)

    trainer.fit(shutdown=shutdown)

    assert captured == [shutdown]
    assert instance.close_calls == [None]


def test_serving_keeps_signal_ownership_and_stops_accepting_work():
    captured = {}
    state = SimpleNamespace(closing=False, shutdown_info=None)
    app = SimpleNamespace(state=SimpleNamespace(areno_serve=state))

    class FakeConfig:
        def __init__(self, app, *, host, port):
            captured["config"] = (app, host, port)

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False
            captured["server"] = self

        def run(self):
            with self.capture_signals():
                handler = signal.getsignal(signal.SIGINT)
                captured["handler"] = handler
                handler(signal.SIGINT, None)

    uvicorn_module = SimpleNamespace(Config=FakeConfig, Server=FakeServer)
    _run_uvicorn_with_graceful_shutdown(
        app,
        host="127.0.0.1",
        port=8000,
        deadline_s=10,
        uvicorn_module=uvicorn_module,
    )

    assert captured["server"].should_exit is True
    assert state.closing is True
    assert state.shutdown_info["stage"] == "serving"
    assert getattr(captured["handler"], "__self__", None) is not captured["server"]
