"""CPU contracts and exact small-model mathematics for generic JEV training."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from areno.api.algorithms import get_algorithm
from areno.api.jev.data import encode_question, group_questions


def test_jev_is_builtin_and_offline():
    spec = get_algorithm("jev")
    assert not spec.experimental and not spec.requires_rollout
    assert spec.resolve_trainer_cls().__name__ == "JevTrainer"


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) % 31 + 1 for c in text]


def test_candidates_and_answer_suffix_survive_truncation():
    question = {
        "prompt": "a" * 100,
        "candidates": ["yes\nANSWER", "no\nANSWER"],
        "target": [0.7, 0.3],
        "question_id": "q",
    }
    rows = encode_question(question, Tokenizer(), 16)
    assert all(len(row.tokens) <= 16 for row in rows)
    assert rows[0].tokens[-len("yes\nANSWER") :] == Tokenizer().encode("yes\nANSWER")
    assert len(group_questions(rows)) == 1
    with pytest.raises(ValueError, match="incomplete"):
        group_questions(rows[:-1])
    with pytest.raises(ValueError, match="complete candidate"):
        encode_question(question, Tokenizer(), 5)
    for target in ([0.5, -0.5], [0.5, float("nan")], [0.2, 0.3]):
        with pytest.raises(ValueError, match="distribution"):
            encode_question({**question, "target": target}, Tokenizer(), 16)


def test_loss_matches_numpy_and_is_candidate_permutation_equivariant():
    mx = pytest.importorskip("mlx.core")
    from areno.api.jev.loss import jev_loss_fn

    with mx.stream(mx.cpu):
        logits, target = mx.array([2.0, -0.4, 0.7]), mx.array([0.1, 0.6, 0.3])
        loss, stats = jev_loss_fn(target, logits)
        scores, probabilities = np.array([2.0, -0.4, 0.7]), np.array([0.1, 0.6, 0.3])
        prediction = np.exp(scores - scores.max())
        prediction /= prediction.sum()
        ce = -(probabilities * np.log(prediction)).sum()
        brier = ((prediction - probabilities) ** 2).sum()
        assert float(loss.item()) == pytest.approx(ce + 0.5 * brier, abs=1e-6)
        permutation = mx.array([2, 0, 1])
        permuted, _ = jev_loss_fn(target[permutation], logits[permutation])
        assert float(permuted.item()) == pytest.approx(float(loss.item()), abs=1e-6)
        assert float(stats["jev_brier"].item()) == pytest.approx(brier, abs=1e-6)


@pytest.mark.parametrize("sizes", [(3,), (2, 3, 2)])
@pytest.mark.parametrize("mini_bs", [1, 2, 3, 8])
@pytest.mark.parametrize("alternate_layout", [False, True])
def test_streaming_gradient_matches_joint_gradient_and_updates_head_and_lora(
    tmp_path, alternate_layout, mini_bs, sizes
):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear

    from areno.api.backend.mlx.decision import initialize_head, question_logits, save_head, train_questions
    from areno.api.backend.mlx.provider import MlxModelProvider
    from areno.api.jev.loss import jev_loss_fn

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(40, 8)
            self.projection = nn.Linear(8, 8)
            self.freeze()
            self.projection = LoRALinear.from_base(self.projection, r=4, scale=2.0, dropout=0.0)

        def __call__(self, tokens):
            return self.projection(mx.cumsum(self.embedding(tokens), axis=1))

    class OtherLayout(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Backbone()

    class Provider(MlxModelProvider):
        calls = 0

        def forward_hidden_states(self, batch):
            self.calls += 1
            body = self.model.encoder if alternate_layout else self.model
            return body(batch["input_ids"])

    with mx.stream(mx.cpu):
        mx.random.seed(42)
        model = OtherLayout() if alternate_layout else Backbone()
        provider = Provider(model, None, None, {"dim": 8})
        optimizer = optim.AdamW(learning_rate=1e-3)
        backend = SimpleNamespace(
            model=model,
            provider=provider,
            optimizer=optimizer,
            model_config={"dim": 8},
            _policy_version=0,
            config=SimpleNamespace(
                head_lr=1e-3,
                decision_checkpoint=None,
                base_model_name_or_path="test",
                model_path="test",
                decision_head=True,
                optimizer={"grad_clip_norm": None},
            ),
            _optimizer_groups=[],
        )
        initialize_head(backend)
        # SGD gives an exact, transparent relationship between reference gradients and updated weights.
        backend.optimizer = optim.SGD(learning_rate=1e-3)
        backend._optimizer_groups = [
            ("model", backend.optimizer, {"lr": 1e-3}),
            ("head", backend.optimizer, {"lr": 1e-3}),
        ]
        rows = []
        targets = []
        for index, size in enumerate(sizes):
            target = [(i + 1) / sum(range(1, size + 1)) for i in range(size)]
            targets.append(mx.array(target))
            rows.extend(
                encode_question(
                    {
                        "prompt": "state",
                        "candidates": ["abc"[: i + 1] for i in range(size)],
                        "question_id": str(index),
                        "target": target,
                    },
                    Tokenizer(),
                    20,
                )
            )
        before = dict(tree_flatten(model.parameters()))
        trainable = dict(tree_flatten(model.trainable_parameters()))

        def objective(module):
            logits = question_logits(provider, rows)
            offset, total = 0, 0
            for target, size in zip(targets, sizes, strict=True):
                total = total + jev_loss_fn(target, logits[offset : offset + size])[0]
                offset += size
            return total / len(sizes)

        loss, grads = nn.value_and_grad(model, objective)(model)
        mx.eval(loss, grads)
        flat_grads = dict(tree_flatten(grads))
        provider.calls = 0
        result = train_questions(backend, SimpleNamespace(global_step=0), rows, mini_bs=mini_bs)
        expected_calls = {(3,): {1: 6, 2: 4, 3: 1, 8: 1}, (2, 3, 2): {1: 14, 2: 6, 3: 3, 8: 1}}
        assert provider.calls == expected_calls[sizes][mini_bs]
        after = dict(tree_flatten(model.parameters()))
        assert result["loss"] == pytest.approx(float(loss.item()), abs=1e-5)
        for key, value in after.items():
            expected = before[key] - 1e-3 * flat_grads[key] if key in trainable else before[key]
            np.testing.assert_allclose(np.array(value), np.array(expected), atol=3e-6, rtol=3e-5)
        assert result["head_update_norm"] > 0 and result["lora_update_norm"] > 0
        assert result["head_grad_norm"] > 0 and result["lora_grad_norm"] > 0
        expected_logits = np.array(question_logits(provider, rows))
        save_head(backend, tmp_path)
        model.score_head.load_weights(str(tmp_path / "score_head.safetensors"))
        np.testing.assert_array_equal(np.array(question_logits(provider, rows)), expected_logits)
        assert json.loads((tmp_path / "jev_config.json").read_text())["pooling"] == "last_valid_token"


@pytest.mark.parametrize("family", ["qwen3", "qwen3_5", "bailing_hybrid"])
def test_provider_shared_prefix_matches_full_paths_for_different_backbones(family):
    mx = pytest.importorskip("mlx.core")
    from areno.api.backend.mlx.provider import MlxLmProvider, MlxModelProvider

    with mx.stream(mx.cpu):
        if family == "qwen3":
            from mlx_lm.models.qwen3 import Model, ModelArgs

            args = ModelArgs(
                model_type="qwen3",
                hidden_size=16,
                num_hidden_layers=2,
                intermediate_size=24,
                num_attention_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=40,
                num_key_value_heads=1,
                max_position_embeddings=128,
                head_dim=8,
                tie_word_embeddings=True,
                rope_theta=10000.0,
            )
        elif family == "qwen3_5":
            from mlx_lm.models.qwen3_5 import Model, ModelArgs

            args = ModelArgs(
                model_type="qwen3_5",
                text_config={
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "intermediate_size": 24,
                    "num_attention_heads": 2,
                    "vocab_size": 40,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "tie_word_embeddings": True,
                    "full_attention_interval": 2,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 2,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                },
            )
        else:
            from mlx_lm.models.bailing_moe_v3 import Model, ModelArgs

            args = ModelArgs(
                vocab_size=40,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=4,
                num_attention_heads=2,
                head_dim=8,
                q_lora_rank=8,
                kv_lora_rank=8,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                n_group=2,
                topk_group=1,
                moe_intermediate_size=16,
                moe_shared_expert_intermediate_size=16,
            )
        mx.random.seed(5)
        model = Model(args)
        provider = MlxLmProvider(model, None, None, {})
        model.eval()
        token_rows = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7], [1, 2, 3, 4, 8, 9, 10]]
        cached = provider.pooled_candidate_hidden_states(token_rows)
        full = MlxModelProvider.pooled_candidate_hidden_states(provider, token_rows)
        np.testing.assert_allclose(np.array(cached), np.array(full), atol=2e-5, rtol=2e-5)
        # Repeating inference cannot leak a previous question's cache state.
        again = provider.pooled_candidate_hidden_states(token_rows)
        np.testing.assert_array_equal(np.array(cached), np.array(again))


def test_jev_sdk_config_builds_typed_runtime_and_binds_loss():
    from areno import LoraConfig
    from areno.api import JevMlxConfig, JevTrainerConfig

    config = JevTrainerConfig(
        algo="jev",
        ckpt="unused",
        dataset_path="unused",
        backend="mlx",
        world_size=1,
        tp_size=1,
        lora=LoraConfig(rank=32, alpha=64),
        save_path="unused",
        brier_weight=0.25,
        head_lr=3e-4,
        max_context_len=512,
    )
    runtime = config.backend_config()
    assert isinstance(runtime, JevMlxConfig)
    assert runtime.lora.rank == 32 and runtime.max_seq_len == 512 and runtime.head_lr == 3e-4
    assert get_algorithm("jev").make_loss_fn(config).keywords == {"brier_weight": 0.25}
    with pytest.raises(ValueError, match="requires backend"):
        JevTrainerConfig(algo="jev", ckpt="unused", dataset_path="unused", backend="npu", save_path="unused")


def test_jev_cli_config_restores_head_alongside_a_saved_lora(tmp_path):
    from areno import LoraConfig
    from areno.api.jev import JevTrainerConfig

    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj"],
            }
        )
    )
    (tmp_path / "score_head.safetensors").touch()
    config = JevTrainerConfig(
        algo="jev",
        ckpt="base",
        dataset_path="data",
        backend="mlx",
        world_size=1,
        tp_size=1,
        save_path="out",
        lora=LoraConfig(adapter_path=str(tmp_path)),
    )
    assert config.backend_config().decision_checkpoint == str(tmp_path)


def test_training_normalizes_cli_records_without_including_test_split(tmp_path):
    from areno.api.jev.data import load_parquet_questions, normalize_questions
    from areno.api.jev.schema import record_to_questions
    from areno.api.trainers.jev import JevTrainer

    question = {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}
    record = {
        "id": "train-1",
        "source_group": "test-workflow",
        "request": {"state": "state", "questions": {"q": question}},
        "targets": {"q": {"a": 0.25, "b": 0.75}},
    }
    expected = record_to_questions(record)
    test = {**record, "id": "held-out", "split": "test"}
    assert normalize_questions([record, test]) == expected
    assert normalize_questions({"train": expected, "test": record_to_questions(test)}) == expected
    trainer = JevTrainer(None, instance=None, dataset=[record, test], reward_fn=None, loss_fn=None)
    assert trainer.dataset == expected

    typed = {
        "id": record["id"],
        "workflow": record["source_group"],
        "state": "state",
        "questions": json.dumps({"q": question}),
        "gold": json.dumps({"q": {"probabilities": {"a": 1, "b": 3}}}),
    }
    assert normalize_questions([typed]) == expected
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([typed]), path)
    assert load_parquet_questions(path) == expected
    assert normalize_questions(pq.read_table(path).to_pylist()) == expected
