"""Offline raw-question materialization and public SDK label training loop."""

import json
import random
from collections.abc import Mapping
from pathlib import Path

from areno.api.jev.labels.protocol import encode_question, save_protocol, training_row
from areno.api.models import TrainSequence


class LabelJevTrainer:
    @staticmethod
    def load_training_dataset(dataset_path, *, default_loader):
        path = Path(dataset_path)
        if path.is_dir() and (path / "train.jsonl").is_file():
            path = path / "train.jsonl"
        if path.is_file() and path.suffix == ".jsonl":
            # Mixed question criteria can be lists or objects. Read raw JSON
            # before Arrow attempts to unify those incompatible column types.
            with path.open() as handle:
                return [json.loads(line) for line in handle if line.strip()]
        return default_loader(dataset_path)

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        self.config, self.instance, self.loss_fn = config, instance, loss_fn
        if isinstance(dataset, Mapping):
            dataset = dataset["train"]
        self.questions = []
        for record in dataset:
            if isinstance(record, TrainSequence):
                self.questions.append(record)
                continue
            if record.get("split", "train") != "train":
                continue
            if "kind" in record and "options" in record and "target" in record:
                from areno.api.jev.labels.protocol import corpus_record

                record = corpus_record(record)
            if "request" not in record or "targets" not in record:
                raise ValueError(
                    "--sgl-jev requires raw request/targets records; scalar-head candidate rows are not compatible"
                )
            for qid, question in record["request"]["questions"].items():
                if qid in record["targets"]:
                    self.questions.append((record["request"]["state"], question, record["targets"][qid]))
        if not self.questions:
            raise ValueError("label JEV requires a nonempty question dataset")
        if config.data_start_index > len(self.questions):
            raise ValueError("data_start_index exceeds the question count")

    def fit(self):
        config = self.config
        destination = Path(config.save_path)
        destination.mkdir(parents=True, exist_ok=True)
        if config.backend == "cuda":
            import torch

            torch.manual_seed(config.seed)
        cursor = {"epoch": 0, "next_index": config.data_start_index}
        try:
            self.instance.init()
            tokenizer = self.instance.get_tokenizer()

            def save(path):
                self.instance.save_checkpoint(str(path))
                save_protocol(path, tokenizer, temperature=config.label_temperature, max_seq_len=config.max_seq_len)
                (Path(path) / "label_jev_cursor.json").write_text(
                    json.dumps(
                        {
                            **cursor,
                            "seed": config.seed,
                            "question_count": len(self.questions),
                            "resume": "data position and adapter weights only; optimizer is fresh",
                        },
                        indent=2,
                    )
                    + "\n"
                )

            step = 0
            rng = random.Random(config.seed)
            with (destination / "metrics.jsonl").open("w") as metrics:
                for epoch in range(config.epochs):
                    order = list(range(len(self.questions)))
                    rng.shuffle(order)
                    first = config.data_start_index if epoch == 0 else 0
                    for start in range(first, len(order), config.batch_size):
                        rows = []
                        for index in order[start : start + config.batch_size]:
                            question = self.questions[index]
                            if isinstance(question, TrainSequence):
                                rows.append(question)
                            else:
                                state, prompt, target = question
                                rows.append(
                                    training_row(
                                        encode_question(
                                            state, prompt, tokenizer, target=target, max_seq_len=config.max_seq_len
                                        ),
                                        temperature=config.label_temperature,
                                    )
                                )
                        result = self.instance.train(
                            rows,
                            self.loss_fn,
                            mini_bs=config.mini_bs,
                            gradient_accumulation_steps=config.gradient_accumulation_steps,
                        )
                        step += 1
                        cursor.update(epoch=epoch, next_index=min(start + config.batch_size, len(order)))
                        result.update(step=step, epoch=epoch, dataset_start=start)
                        metrics.write(json.dumps(result) + "\n")
                        metrics.flush()
                        print(json.dumps(result), flush=True)
                        if step % config.save_interval == 0:
                            save(destination / f"step_{step:06d}")
                        if config.max_steps is not None and step >= config.max_steps:
                            break
                    if config.max_steps is not None and step >= config.max_steps:
                        break
            save(destination / "final")
        finally:
            self.instance.close()
