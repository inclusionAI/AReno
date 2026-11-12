import pytest
import torch

from areno.api.algorithms import get_algorithm
from areno.api.jev.labels.loss import label_distribution, label_jev_loss_fn
from areno.api.jev.labels.protocol import encode_question, render_question, training_row


def test_label_jev_reference_cpu():
    logits = torch.tensor([[1.0, -2.0, 3.0, 0.5], [-1.0, 2.0, 0.2, 3.0]], requires_grad=True)
    pack = {
        "_label_questions": [
            {"ids": [0, 2], "target": [0.25, 0.75], "temperature": 0.7},
            {"ids": [1, 0, 3], "target": [0.0, 1.0, 0.0], "temperature": 1.3},
        ]
    }
    loss, stats = label_jev_loss_fn(pack, logits)
    reference = (
        -(torch.tensor([0.25, 0.75]) * torch.log_softmax(logits[0, [0, 2]] / 0.7, 0)).sum()
        - torch.log_softmax(logits[1, [1, 0, 3]] / 1.3, 0)[1]
    ) / 2
    torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-4)
    (actual,) = torch.autograd.grad(loss, logits, retain_graph=True)
    (expected,) = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    assert 0 < stats["label_mass"] < 1
    assert get_algorithm("label_jev").experimental is False


@pytest.mark.parametrize("ids", [[0, 0], [0, 9], [-1, 2], [1]])
def test_bad_label_ids_cpu(ids):
    with pytest.raises(ValueError):
        label_distribution(torch.ones(4), ids)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_temperature_cpu(temperature):
    with pytest.raises(ValueError):
        label_distribution(torch.ones(4), [0, 1], temperature)


def test_nonfinite_cpu():
    with pytest.raises(ValueError):
        label_distribution(torch.tensor([float("nan"), 0.0]), [0, 1])
    with pytest.raises(ValueError):
        label_jev_loss_fn(
            {"_label_questions": [{"ids": [0, 1], "target": [0.2, 0.3], "temperature": 1.0}]}, torch.ones(1, 3)
        )


class Tokenizer:
    chat_template = "test"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[-1]["content"] + "\nASSISTANT:"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def test_protocol_order_boundary_cpu():
    question = {"type": "choice", "instructions": "Choose.", "criteria": {"z": "last", "a": "first"}}
    encoded = encode_question("state", question, Tokenizer(), target={"z": 0.7, "a": 0.3})
    assert encoded["names"] == ["z", "a"] and encoded["labels"] == ["A", "B"]
    assert training_row(encoded).features["label_jev"]["target"] == [0.7, 0.3]
    with pytest.raises(ValueError, match="truncation"):
        encode_question("state", question, Tokenizer(), max_seq_len=3)

    class Merged(Tokenizer):
        def encode(self, text, **kwargs):
            return [0] if text.endswith("A") else super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="boundary"):
        encode_question("state", question, Merged())
    text, names, labels = render_question("", {"type": "noul", "instructions": "True?"})
    assert names == ["true", "false"] and labels == ["yes", "no"]
    assert text.endswith("Answer with yes or no only.")
    for count in [2, 10]:
        assert len(render_question("", {"type": "score", "criteria": ["level"] * count})[1]) == count
    with pytest.raises(ValueError):
        render_question("", {"type": "score", "criteria": ["level"] * 11})


def test_answer_head_dtype_and_last_position_cpu():
    from areno.api.jev.labels.cuda import answer_logits

    class Head(torch.nn.Linear):
        def forward(self, states):
            assert states.ndim == 3
            assert states.dtype == self.weight.dtype
            return super().forward(states)

    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(1, 6, 4).requires_grad_()
    head = Head(4, 3, bias=False, dtype=torch.float32)
    positions = torch.tensor([1, 5])
    actual = answer_logits(hidden, positions, head)
    expected = torch.nn.functional.linear(hidden.reshape(6, 4)[positions].float(), head.weight)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert hidden.grad[:, [1, 5]].abs().sum() > 0
    assert hidden.grad[:, [0, 2, 3, 4]].abs().sum() == 0


def test_label_trainer_raw_loader_and_checkpoint_protocol_cpu(tmp_path):
    import json

    from areno.api.algorithms import label_jev_loss_fn
    from areno.api.jev.labels.config import LabelJevTrainerConfig
    from areno.api.jev.labels.trainer import LabelJevTrainer

    records = [
        {
            "id": "one",
            "request": {
                "state": "State",
                "questions": {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}},
            },
            "targets": {"q": {"a": 0.4, "b": 0.6}},
        },
        {
            "id": "two",
            "request": {
                "state": "State",
                "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]}},
            },
            "targets": {"q": {"0": 1.0, "1": 0.0}},
        },
    ]
    source = tmp_path / "raw.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))

    def unexpected_arrow(path):
        pytest.fail("heterogeneous raw criteria must not pass through Arrow")

    dataset = LabelJevTrainer.load_training_dataset(str(source), default_loader=unexpected_arrow)

    class SDK:
        closed = False
        calls = []

        def init(self):
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def train(self, rows, loss_fn, **kwargs):
            assert loss_fn is label_jev_loss_fn
            self.calls.append(rows)
            return {"loss": 1.0}

        def save_checkpoint(self, path):
            from pathlib import Path

            Path(path).mkdir(parents=True, exist_ok=True)

        def close(self):
            self.closed = True

    sdk = SDK()
    config = LabelJevTrainerConfig(
        algo="label_jev",
        ckpt="unused",
        dataset_path=str(source),
        backend="cuda",
        world_size=1,
        tp_size=1,
        save_path=str(tmp_path / "output"),
        epochs=2,
        max_steps=2,
        batch_size=2,
        save_interval=1,
    )
    LabelJevTrainer(config, instance=sdk, dataset=dataset, reward_fn=None, loss_fn=label_jev_loss_fn).fit()
    assert len(sdk.calls) == 2 and sdk.closed
    assert all(row.features.get("label_jev") for rows in sdk.calls for row in rows)
    metadata = json.loads((tmp_path / "output/final/label_jev_config.json").read_text())
    assert metadata["prompt_version"] == 1 and metadata["thinking"] is False


def test_label_cursor_preserves_suffix_after_batch_size_change_cpu(tmp_path):
    import json
    import random

    from areno.api.jev.labels.config import LabelJevTrainerConfig
    from areno.api.jev.labels.trainer import LabelJevTrainer
    from areno.api.models import TrainSequence

    class SDK:
        def __init__(self):
            self.seen = []

        def init(self):
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def train(self, rows, loss_fn, **kwargs):
            self.seen.extend(row.tokens[0] for row in rows)
            return {"loss": 1.0}

        def save_checkpoint(self, path):
            from pathlib import Path

            Path(path).mkdir(parents=True, exist_ok=True)

        def close(self):
            pass

    order = list(range(11))
    random.Random(17).shuffle(order)
    # Two completed batches of three; now consume the suffix in batches of two.
    sdk = SDK()
    config = LabelJevTrainerConfig(
        algo="label_jev",
        ckpt="unused",
        dataset_path="unused",
        backend="cuda",
        world_size=1,
        tp_size=1,
        seed=17,
        save_path=str(tmp_path),
        epochs=1,
        batch_size=2,
        data_start_index=6,
        save_interval=1,
    )
    dataset = [TrainSequence(tokens=[i]) for i in range(11)]
    LabelJevTrainer(config, instance=sdk, dataset=dataset, reward_fn=None, loss_fn=None).fit()
    assert sdk.seen == order[6:]
    assert order[:6] + sdk.seen == order
    first = json.loads((tmp_path / "step_000001/label_jev_cursor.json").read_text())
    assert first["next_index"] == 8
    final = json.loads((tmp_path / "final/label_jev_cursor.json").read_text())
    assert final["next_index"] == 11 and final["question_count"] == 11
    config.data_start_index = 12
    with pytest.raises(ValueError, match="question count"):
        LabelJevTrainer(config, instance=sdk, dataset=dataset, reward_fn=None, loss_fn=None)
