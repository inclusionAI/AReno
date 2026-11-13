"""Single-GPU SFT recovery after an actual process-group kill."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")


def _fit(model_path, save_path, *, resume_from=None, stage=None, marker=None):
    from areno import Trainer
    from areno.adapters import LoraConfig
    from areno.api.algorithms import sft_loss_fn
    from areno.api.trainer_config import SFTTrainerConfig
    from areno.api.trainers.sft import SFTTrainer

    # CUDA LoRA initialization uses torch.initial_seed() in the driver process.
    # Independent baseline and interrupted processes must start identically.
    torch.manual_seed(31)
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
        save_path=str(save_path),
        save_interval=1,
        save_training_state=True,
        lora=LoraConfig(rank=4, alpha=8),
        resume_from=str(resume_from) if resume_from else None,
    )
    rows = [{"tokens": [2, i + 4, 1], "prompt_mask": [True, False, False]} for i in range(5)]

    class PausingTrainer(Trainer):
        def save_training_state(self, path):
            super().save_training_state(path)
            if stage == "worker" and ".step_000002.training-state." in path:
                Path(marker).write_text("worker saved")
                signal.pause()

    class PausingSFTTrainer(SFTTrainer):
        def _maybe_save(self, epoch, step):
            super()._maybe_save(epoch, step)
            if stage == "published" and step == 1:
                Path(marker).write_text("snapshot published")
                signal.pause()

    backend_config = config.cuda_config()
    backend_config.runtime["compile_model"] = False
    api = PausingTrainer(1, str(model_path), backend_type=config.backend_type(), custom_config=backend_config)
    PausingSFTTrainer(config, instance=api, dataset=rows, reward_fn=None, loss_fn=sft_loss_fn).fit()


@pytest.mark.skipif(not torch.cuda.is_available() or os.name != "posix", reason="requires CUDA and POSIX signals")
@pytest.mark.parametrize("stage", ["worker", "published"])
def test_sft_recovers_after_driver_and_worker_are_killed(tmp_path, stage):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from areno.api.trainers.sft_snapshot import load_snapshot

    model_path = tmp_path / "model"
    model_path.mkdir()
    torch.manual_seed(17)
    config = Qwen3Config(
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
    Qwen3ForCausalLM(config).save_pretrained(model_path)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(
        model_path
    )
    _fit(model_path, tmp_path / "baseline")

    marker = tmp_path / "ready"
    output = tmp_path / "interrupted"
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.test_sft_crash_gpu import _fit; "
        "_fit(sys.argv[1], sys.argv[2], stage=sys.argv[3], marker=sys.argv[4])",
        str(model_path),
        str(output),
        stage,
        str(marker),
    ]
    log_path = tmp_path / "interrupted.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if not marker.exists():
                log.flush()
                pytest.fail(f"training did not reach {stage}; log: {log_path.read_text()}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)
    assert process.returncode == -signal.SIGKILL

    if stage == "worker":
        incomplete = next(output.glob(".step_000002.training-state.*"))
        with pytest.raises(ValueError, match="incomplete|unpublished"):
            load_snapshot(str(incomplete), object(), "unused", lambda state: None)
        checkpoint = output / "step_000001.training-state"
        assert not (output / "step_000002.training-state").exists()
    else:
        checkpoint = output / "step_000002.training-state"
    _fit(model_path, tmp_path / "resumed", resume_from=checkpoint)
    expected = torch.load(tmp_path / "baseline" / "step_000003.training-state" / "worker.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed" / "step_000003.training-state" / "worker.pt", weights_only=True)
    assert actual["global_step"] == expected["global_step"] == 5
    assert actual["optimizer"]["lr"] == expected["optimizer"]["lr"]
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
