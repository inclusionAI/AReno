"""SFT training with periodic evaluation — runnable example.

This example demonstrates the periodic evaluation feature for SFT training.
It creates a small synthetic dataset, runs a few training steps with eval
enabled, and logs the eval metrics to TensorBoard.

Usage::

    python examples/eval/sft_eval_example.py

Requirements: the ``datasets`` library and a tokenizer-aware model checkpoint.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

from areno.api.metrics import MetricsRecorder
from areno.api.trainers.sft import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class _ExampleTokenizer:
    """Minimal tokeniser for the offline example."""

    eos_token_id = 0
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


class _ExampleBackend:
    """Backend double for the offline example."""

    def __init__(self, metrics=None):
        self.train_calls = 0
        self.eval_calls = 0
        self._metrics = metrics

    def init(self):
        pass

    def close(self):
        pass

    def get_tokenizer(self):
        return _ExampleTokenizer()

    def train(self, _batch, _loss_fn, *, mini_bs, gradient_accumulation_steps):
        del mini_bs, gradient_accumulation_steps
        self.train_calls += 1
        return {"sft_loss": 1.0, "sft_target_tokens": 4.0}

    def evaluate(self, batch_data, _loss_fn, *, mini_bs, gradient_accumulation_steps=None):
        del mini_bs, gradient_accumulation_steps
        self.eval_calls += 1
        return {"sft_loss": 0.5, "sft_target_tokens": float(len(batch_data) * 4)}

    def save_checkpoint(self, _ctx, path):
        return path


def main() -> None:
    log_dir = tempfile.mkdtemp(prefix="sft_eval_example_")
    metrics = MetricsRecorder(log_dir)

    # Build a synthetic training dataset (short texts to fit token budgets).
    train_data = [
        {"prompt": "2+2?", "response": "4"},
        {"prompt": "1+1?", "response": "2"},
        {"prompt": "3+3?", "response": "6"},
        {"prompt": "4+4?", "response": "8"},
    ]

    # Build a synthetic evaluation dataset.
    eval_data = [
        {"prompt": "5+5?", "response": "10"},
        {"prompt": "6+6?", "response": "12"},
    ]

    config = SimpleNamespace(
        batch_size=2,
        epochs=1,
        gradient_accumulation_steps=1,
        max_new_tokens=10,
        max_prompt_tokens=10,
        mini_bs=1,
        save_interval=100,
        save_path=None,
        max_steps=None,
        eval_dataset_path="synthetic_eval",  # non-None enables eval
        eval_interval=2,
        eval_batches=0,
        model_hub="hf",
        dataset_loader_fn=None,
    )

    backend = _ExampleBackend(metrics=metrics)
    loss_fn = lambda _pack, _logprobs: None
    trainer = SFTTrainer(config, instance=backend, dataset=train_data, reward_fn=None, loss_fn=loss_fn)

    # Inject eval dataset directly to skip file loading.
    trainer._eval_dataset = eval_data

    print("=== SFT training with periodic evaluation ===")
    trainer.fit()

    metrics.close()

    print(f"\nTrain calls : {backend.train_calls}")
    print(f"Eval calls  : {backend.eval_calls}")
    print(f"TensorBoard logs written to: {log_dir}")
    print("Run: tensorboard --logdir", log_dir)


if __name__ == "__main__":
    main()