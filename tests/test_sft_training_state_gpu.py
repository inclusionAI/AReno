"""CUDA training-state parity with a small native Qwen3 model."""

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")

from areno.api.algorithms import sft_loss_fn
from areno.api.backend.cuda.training import make_train_pack
from areno.api.models import TrainSequence
from areno.engine.api import ArenoEngine
from areno.engine.config import EngineConfig, ModelConfig, OptimizerConfig, RuntimeConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_single_worker_training_state_continues_model_and_optimizer(tmp_path):
    model = ModelConfig(
        model_type="qwen3",
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        dtype=torch.float32,
        attn_backend="native",
    )
    config = EngineConfig(
        model=model,
        devices=[0],
        tp_size=1,
        dp_size=1,
        train_loss_fn=sft_loss_fn,
        optimizer=OptimizerConfig(lr=1e-3, min_lr=1e-4, lr_decay_steps=10, lr_decay_style="cosine"),
        runtime=RuntimeConfig(attn_backend="native", compile_model=False, activation_checkpointing=False),
    )
    rows = [
        TrainSequence(
            tokens=[1, 2, 3, 4],
            prompt_mask=[True, True, False, False],
            logprobs=[0.0] * 4,
            advantages=[0.0] * 4,
            eos_token_id=0,
        )
    ]
    pack = make_train_pack(rows)
    checkpoint = tmp_path / "step1.pt"
    expected_file = tmp_path / "expected.pt"
    actual_file = tmp_path / "actual.pt"
    engine = ArenoEngine(config)
    try:
        first = engine.step([pack])[0]
        engine.save_training_state(str(checkpoint))
        expected = engine.step([pack])[0]
        engine.save_training_state(str(expected_file))
    finally:
        engine.close()
    resumed = ArenoEngine(config)
    try:
        resumed.load_training_state(str(checkpoint))
        actual = resumed.step([pack])[0]
        resumed.save_training_state(str(actual_file))
    finally:
        resumed.close()
    assert math.isfinite(first.loss) and math.isfinite(expected.loss) and math.isfinite(actual.loss)
    assert math.isfinite(first.metrics["grad_norm"]) and math.isfinite(actual.metrics["grad_norm"])
    assert actual.metrics["grad_norm"] == pytest.approx(expected.metrics["grad_norm"], rel=1e-5, abs=1e-6)
    assert actual.loss == pytest.approx(expected.loss, rel=1e-5, abs=1e-6)
    assert actual.metrics["lr"] == expected.metrics["lr"]
    expected_state = torch.load(expected_file, weights_only=True)
    actual_state = torch.load(actual_file, weights_only=True)
    assert actual_state["global_step"] == expected_state["global_step"] == 2
    for name in expected_state["model"]:
        torch.testing.assert_close(actual_state["model"][name], expected_state["model"][name], rtol=1e-5, atol=1e-6)
    for actual_bucket, expected_bucket in zip(
        actual_state["optimizer"]["state"], expected_state["optimizer"]["state"], strict=True
    ):
        assert actual_bucket["step"] == expected_bucket["step"]
        for name in ("exp_avg", "exp_avg_sq"):
            if expected_bucket[name] is None:
                assert actual_bucket[name] is None
            else:
                torch.testing.assert_close(actual_bucket[name], expected_bucket[name], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("ARENO_SFT_RESUME_MODEL"),
    reason="requires a CUDA GPU and an explicit local Qwen3 checkpoint",
)
def test_pretrained_qwen3_lora_sft_resume(tmp_path):
    """Compare a real checkpoint after continuous and resumed single-GPU SFT."""
    from areno import Trainer
    from areno.adapters import LoraConfig
    from areno.api.trainer_config import SFTTrainerConfig
    from areno.api.trainers.sft import SFTTrainer

    source = Path(os.environ["ARENO_SFT_RESUME_MODEL"])
    model = tmp_path / "fp32_model"
    model.mkdir()
    for path in source.iterdir():
        if path.name == "config.json":
            model_config = json.loads(path.read_text())
            model_config["dtype"] = "float32"
            model_config["torch_dtype"] = "float32"
            (model / path.name).write_text(json.dumps(model_config))
        elif path.is_file():
            (model / path.name).symlink_to(path.resolve())
    rows = [{"prompt": f"Add one to {i}.", "response": str(i + 1)} for i in range(8)]
    config = SFTTrainerConfig(
        algo="sft",
        ckpt=str(model),
        dataset_path="in-memory",
        backend="cuda",
        tp_size=1,
        world_size=1,
        batch_size=2,
        mini_bs=2,
        epochs=1,
        max_steps=4,
        seed=31,
        shuffle=True,
        max_prompt_tokens=128,
        max_new_tokens=32,
        attn_backend="native",
        activation_checkpointing=False,
        save_path=str(tmp_path / "full"),
        save_interval=2,
        save_training_state=True,
        lora=LoraConfig(rank=4, alpha=8),
    )

    class ObservedTrainer(Trainer):
        def __init__(self, sft_config):
            backend_config = sft_config.cuda_config()
            backend_config.runtime["compile_model"] = False
            super().__init__(1, str(model), backend_type=sft_config.backend_type(), custom_config=backend_config)
            self.results = []

        def train(self, *args, **kwargs):
            result = super().train(*args, **kwargs)
            self.results.append(result)
            return result

    def fit(sft_config):
        trainer = ObservedTrainer(sft_config)
        SFTTrainer(sft_config, instance=trainer, dataset=rows, reward_fn=None, loss_fn=sft_loss_fn).fit()
        return trainer.results

    expected_results = fit(config)
    resumed_config = replace(
        config,
        save_path=str(tmp_path / "resumed"),
        resume_from=str(tmp_path / "full" / "step_000002.training-state"),
    )
    actual_results = fit(resumed_config)
    assert len(expected_results) == 4 and len(actual_results) == 2
    for actual, expected in zip(actual_results, expected_results[2:], strict=True):
        for key in ("loss", "lr", "grad_norm"):
            assert math.isfinite(actual[key])
            assert actual[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-6)
    expected_state = torch.load(tmp_path / "full" / "step_000004.training-state" / "worker.pt", weights_only=True)
    actual_state = torch.load(tmp_path / "resumed" / "step_000004.training-state" / "worker.pt", weights_only=True)
    assert actual_state["global_step"] == expected_state["global_step"] == 4
    for name in expected_state["model"]:
        torch.testing.assert_close(actual_state["model"][name], expected_state["model"][name], rtol=1e-5, atol=1e-6)
    for actual_bucket, expected_bucket in zip(
        actual_state["optimizer"]["state"], expected_state["optimizer"]["state"], strict=True
    ):
        assert actual_bucket["step"] == expected_bucket["step"]
        for name in ("exp_avg", "exp_avg_sq"):
            if expected_bucket[name] is None:
                assert actual_bucket[name] is None
            else:
                torch.testing.assert_close(actual_bucket[name], expected_bucket[name], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("use_lora", [False, True])
def test_sft_trainer_snapshot_continues_on_cuda(tmp_path, use_lora):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from areno import Trainer
    from areno.adapters import LoraConfig
    from areno.api.trainer_config import SFTTrainerConfig
    from areno.api.trainers.sft import SFTTrainer

    model_path = tmp_path / "model"
    model_path.mkdir()
    torch.manual_seed(17)
    hf_config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    Qwen3ForCausalLM(hf_config).save_pretrained(model_path)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(
        model_path
    )
    rows = [{"tokens": [2, i + 4, 1], "prompt_mask": [True, False, False]} for i in range(5)]
    config = SFTTrainerConfig(
        algo="sft",
        ckpt=str(model_path),
        dataset_path="in-memory",
        backend="cuda",
        tp_size=1,
        world_size=1,
        batch_size=2,
        mini_bs=1,
        gradient_accumulation_steps=1,
        epochs=2,
        max_steps=3,
        seed=31,
        shuffle=True,
        max_prompt_tokens=16,
        max_new_tokens=16,
        attn_backend="native",
        activation_checkpointing=False,
        save_path=str(tmp_path / "full"),
        save_interval=1,
        save_training_state=True,
        lora=LoraConfig(rank=4, alpha=8) if use_lora else None,
    )

    def fit(sft_config):
        backend_config = sft_config.cuda_config()
        backend_config.runtime["compile_model"] = False
        api = Trainer(1, str(model_path), backend_type=sft_config.backend_type(), custom_config=backend_config)
        SFTTrainer(sft_config, instance=api, dataset=rows, reward_fn=None, loss_fn=sft_loss_fn).fit()

    fit(config)
    resume = replace(
        config,
        save_path=str(tmp_path / "resumed"),
        resume_from=str(tmp_path / "full" / "step_000001.training-state"),
    )
    fit(resume)
    expected = torch.load(tmp_path / "full" / "step_000003.training-state" / "worker.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed" / "step_000003.training-state" / "worker.pt", weights_only=True)
    assert expected["global_step"] == actual["global_step"] == 5
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=1e-5, atol=1e-6)
    for actual_bucket, expected_bucket in zip(
        actual["optimizer"]["state"], expected["optimizer"]["state"], strict=True
    ):
        assert actual_bucket["step"] == expected_bucket["step"]
        for name in ("exp_avg", "exp_avg_sq"):
            if expected_bucket[name] is None:
                assert actual_bucket[name] is None
            else:
                torch.testing.assert_close(actual_bucket[name], expected_bucket[name], rtol=1e-5, atol=1e-6)
