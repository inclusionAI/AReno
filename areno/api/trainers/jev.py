"""Offline algorithm loop; the backend owns model-specific execution."""

import json
from pathlib import Path

from areno.api.jev.data import QuestionIterator, encode_question, normalize_questions


class JevTrainer:
    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        del reward_fn
        self.config, self.instance, self.loss_fn = config, instance, loss_fn
        self.dataset = normalize_questions(dataset)

    def fit(self) -> None:
        try:
            self.instance.init()
            self._fit()
        finally:
            self.instance.close()

    def _fit(self) -> None:
        if not self.dataset or self.config.batch_size < 1:
            raise ValueError("JEV requires a nonempty question dataset and positive batch_size")
        destination = Path(self.config.save_path)
        destination.mkdir(parents=True, exist_ok=True)
        counts = self.instance._backend.decision_parameters(self.instance._ctx)
        backbone = "lora" if self.config.lora is not None else "backbone"
        if not counts["head"] or not counts.get(backbone):
            raise RuntimeError(f"both the decision head and {backbone} must be trainable")
        if self.config.lora is not None and counts.get("backbone", 0):
            raise RuntimeError("LoRA JEV requires the base backbone parameters to remain frozen")
        (destination / "trainable_parameters.json").write_text(json.dumps(counts, indent=2) + "\n")
        checkpoint = self.config.resume_checkpoint()
        state_path = Path(checkpoint) / "jev_training_state.json" if checkpoint else None
        state = json.loads(state_path.read_text()) if state_path and state_path.is_file() else None
        iterator = QuestionIterator(self.dataset, seed=self.config.seed, batch_size=self.config.batch_size, state=state)
        tokenizer = self.instance.get_tokenizer()
        run_steps = 0

        def save(path):
            self.instance.save_checkpoint(str(path))
            (path / "jev_training_state.json").write_text(json.dumps(iterator.state_dict(), indent=2) + "\n")

        with (destination / "metrics.jsonl").open("w") as metrics:
            while iterator.epoch < self.config.epochs:
                epoch, start = iterator.epoch, iterator.next_index
                selected = iterator.take()
                rows = [row for q in selected for row in encode_question(q, tokenizer, self.config.max_seq_len)]
                result = self.instance.train(rows, self.loss_fn, mini_bs=self.config.mini_bs)
                run_steps += 1
                step = iterator.completed_steps
                result.update(step=step, epoch=epoch, dataset_start=start)
                metrics.write(json.dumps(result) + "\n")
                metrics.flush()
                print(json.dumps(result), flush=True)
                if step % self.config.save_interval == 0:
                    save(destination / f"step_{step:06d}")
                if self.config.max_steps is not None and run_steps >= self.config.max_steps:
                    break
        save(destination / "final")
