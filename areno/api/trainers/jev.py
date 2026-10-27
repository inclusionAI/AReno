"""Offline algorithm loop; the backend owns model-specific execution."""

import json
from pathlib import Path

from areno.api.jev.data import QuestionIterator, encode_question


class JevTrainer:
    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        del reward_fn
        self.config, self.instance, self.loss_fn = config, instance, loss_fn
        self.dataset = dataset["train"] if isinstance(dataset, dict) else dataset

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
        counts = trainable_parameter_counts(self.instance, self.config.backend, self.config.lora is not None)
        (destination / "trainable_parameters.json").write_text(json.dumps(counts, indent=2) + "\n")
        checkpoint = self.config.lora.adapter_path if self.config.lora is not None else None
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
                if self.config.backend == "cuda":
                    # Each candidate is one packed row; rows of a question share a
                    # trajectory id (the question index) so the engine keeps the
                    # whole question together. One microbatch holds every selected
                    # question, so each question's grouped softmax is complete.
                    #
                    # The engine averages this microbatch's gradient across DP
                    # ranks, so each question needs `dp_size / question_count`
                    # weight to make the step gradient the mean over questions.
                    question_count = len(selected)
                    if question_count < 1:
                        raise ValueError("JEV step selected no questions")
                    weight = self.instance.dp_size() / question_count
                    rows = [
                        row
                        for question_index, question in enumerate(selected)
                        for row in encode_question(
                            question,
                            tokenizer,
                            self.config.max_seq_len,
                            group=question_index,
                            weight=weight,
                        )
                    ]
                    mini_bs = len(rows)
                else:
                    rows = [
                        row for question in selected for row in encode_question(question, tokenizer, self.config.max_seq_len)
                    ]
                    mini_bs = self.config.mini_bs
                result = self.instance.train(rows, self.loss_fn, mini_bs=mini_bs)
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


def trainable_parameter_counts(instance, backend: str, lora: bool) -> dict[str, int]:
    """Describe the JEV trainable surface for the diagnostics artifact.

    MLX exposes its trainable parameters in-process, so the counts are exact.
    The CUDA engine owns the model inside worker processes; the head and
    backbone there are enforced by `RuntimeConfig.score_head` and the optimizer
    LR group, so this records the configured surface instead of probing it.
    """

    counts = {"head": 0, "lora": 0, "backbone": 0}
    if backend == "cuda":
        settings = instance._ctx.custom_config
        if settings is None or not settings.runtime.get("score_head"):
            raise RuntimeError("CUDA JEV requires runtime.score_head; build the trainer through JevTrainerConfig")
        counts["head"] = 1
        # Native LoRA freezes the backbone and trains the adapters next to head.
        counts["lora" if lora else "backbone"] = 1
        return counts
    from mlx.utils import tree_flatten

    if not lora:
        raise RuntimeError("MLX JEV requires LoRA so the backbone and head train together")
    trainable = dict(tree_flatten(instance._backend.model.trainable_parameters()))
    for name, value in trainable.items():
        if name.startswith("score_head."):
            counts["head"] += value.size
        elif name.endswith(("lora_a", "lora_b")):
            counts["lora"] += value.size
        else:
            raise RuntimeError(f"unexpected trainable base parameter: {name}")
    if not counts["head"] or not counts["lora"]:
        raise RuntimeError("both the decision head and backbone LoRA must be trainable")
    return counts
