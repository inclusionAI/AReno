import errno
import json
import multiprocessing
import os
import shutil
import signal
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from areno import Trainer
from areno.api.algorithms import sft_loss_fn
from areno.api.context import Context
from areno.api.trainer_config import SFTTrainerConfig
from areno.api.trainers.sft import SFTTrainer
from areno.api.trainers.sft_snapshot import load_snapshot, save_snapshot


class Tokenizer:
    eos_token_id = 99
    chat_template = None
    special_tokens_map = {}

    def get_vocab(self):
        return {"token": 1}


class FakeTrainer:
    def __init__(self, *, fail_after=None):
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.optimizer = torch.optim.AdamW([self.weight], lr=0.01)
        self.calls = 0
        self.fail_after = fail_after
        self.seen = []

    def init(self):
        pass

    def close(self):
        pass

    def get_tokenizer(self):
        return Tokenizer()

    def get_processor(self):
        return None

    def train(self, batch, loss_fn, *, mini_bs, gradient_accumulation_steps):
        if self.calls == self.fail_after:
            raise RuntimeError("simulated interruption")
        self.seen.append([row.tokens[1] for row in batch])
        self.optimizer.zero_grad()
        loss = sum((self.weight - row.tokens[1] * 0.01).square() for row in batch) + self.weight * torch.rand(())
        loss.backward()
        self.optimizer.step()
        self.calls += 1
        return {"loss": float(loss.detach())}

    def score_logprobs(self, role, tokens, **kwargs):
        return [[0.0] * len(row) for row in tokens]

    def record_validation_stats(self, stats, *, step):
        pass

    def save_checkpoint(self, path):
        Path(path).mkdir(parents=True)
        return path

    def save_training_state(self, path):
        torch.save({"weight": self.weight.detach(), "optimizer": self.optimizer.state_dict()}, path)

    def load_training_state(self, path, *, completed_batches):
        state = torch.load(path, weights_only=True)
        with torch.no_grad():
            self.weight.copy_(state["weight"])
        self.optimizer.load_state_dict(state["optimizer"])


def _crash_during_snapshot(path, ready, stage):
    import areno.api.trainers.sft_snapshot as snapshot_module

    class PausingTrainer(FakeTrainer):
        def save_training_state(self, target):
            super().save_training_state(target)
            if stage == "worker":
                ready.set()
                signal.pause()

    if stage == "manifest":

        def pause_before_manifest(source, destination):
            ready.set()
            signal.pause()

        snapshot_module.os.replace = pause_before_manifest
    save_snapshot(path, PausingTrainer(), {"completed_batches": 2}, "fingerprint")


def _fit(config, trainer, rows, validation):
    SFTTrainer(
        config, instance=trainer, dataset=rows, validation_dataset=validation, reward_fn=None, loss_fn=sft_loss_fn
    ).fit()


def test_sft_resume_matches_uninterrupted_with_shuffle_validation_and_filtered_rows(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    rows = [{"tokens": [1, i, 99], "prompt_mask": [True, False, False]} for i in range(1, 10)]
    rows.insert(2, {"tokens": [1], "prompt_mask": [True]})
    config = SFTTrainerConfig(
        algo="sft",
        ckpt=str(model),
        dataset_path="rows",
        backend="cuda",
        tp_size=1,
        world_size=1,
        batch_size=2,
        mini_bs=1,
        epochs=2,
        max_steps=6,
        seed=9,
        shuffle=True,
        validation_dataset_path="validation",
        eval_interval=1,
        save_interval=2,
        save_path=str(tmp_path / "full"),
        save_training_state=True,
    )
    torch.manual_seed(31)
    full = FakeTrainer()
    _fit(config, full, rows, rows[:3])

    torch.manual_seed(31)
    partial = FakeTrainer(fail_after=2)
    interrupted_config = replace(config, save_path=str(tmp_path / "partial"))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _fit(interrupted_config, partial, rows, rows[:3])
    snapshot = tmp_path / "partial" / "step_000002.training-state"
    state = torch.load(snapshot / "driver.pt", weights_only=True)
    assert state["completed_batches"] == 2
    assert state["supervised_tokens"] == 8
    resumed = FakeTrainer()
    resumed_config = replace(config, save_path=str(tmp_path / "resumed"), resume_from=str(snapshot))
    _fit(resumed_config, resumed, rows, rows[:3])
    assert partial.seen + resumed.seen == full.seen
    torch.testing.assert_close(resumed.weight, full.weight, atol=0, rtol=0)
    assert resumed.optimizer.state_dict()["state"][0]["step"] == full.optimizer.state_dict()["state"][0]["step"]

    finished = FakeTrainer()
    finished_config = replace(config, resume_from=str(tmp_path / "full" / "step_000006.training-state"))
    _fit(finished_config, finished, rows, rows[:3])
    assert finished.seen == []
    torch.testing.assert_close(finished.weight, full.weight, atol=0, rtol=0)

    (snapshot / "worker.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        _fit(resumed_config, FakeTrainer(), rows, rows[:3])


def test_sft_snapshot_rejects_changed_data(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    rows = [{"tokens": [1, 2], "prompt_mask": [True, False]}]
    config = SFTTrainerConfig(
        algo="sft",
        ckpt=str(model),
        dataset_path="rows",
        backend="cuda",
        tp_size=1,
        world_size=1,
        batch_size=1,
        mini_bs=1,
        epochs=1,
        max_steps=1,
        save_interval=1,
        save_path=str(tmp_path / "first"),
        save_training_state=True,
    )
    _fit(config, FakeTrainer(), rows, None)
    resumed = replace(config, resume_from=str(tmp_path / "first" / "step_000001.training-state"))
    with pytest.raises(ValueError, match="configuration mismatch"):
        _fit(resumed, FakeTrainer(), [{"tokens": [1, 3], "prompt_mask": [True, False]}], None)


def test_sft_snapshot_rejects_custom_loss(tmp_path):
    config = SFTTrainerConfig(
        algo="sft",
        ckpt=str(tmp_path),
        dataset_path="rows",
        backend="cuda",
        tp_size=1,
        world_size=1,
        save_path=str(tmp_path / "out"),
        save_training_state=True,
    )
    trainer = SFTTrainer(config, instance=FakeTrainer(), dataset=[], reward_fn=None, loss_fn=lambda *_: None)
    with pytest.raises(ValueError, match="built-in SFT loss"):
        trainer.fit()


def test_failed_snapshot_does_not_publish_or_overwrite_existing_snapshot(tmp_path):
    class FailingTrainer:
        def save_training_state(self, path):
            Path(path).write_bytes(b"partial")
            raise RuntimeError("simulated save failure")

    snapshot = tmp_path / "snapshot"
    with pytest.raises(RuntimeError, match="simulated save failure"):
        save_snapshot(str(snapshot), FailingTrainer(), {}, "fingerprint")
    assert not snapshot.exists()
    assert list(tmp_path.iterdir()) == []
    snapshot.mkdir()
    (snapshot / "complete").write_text("preserved")
    with pytest.raises(FileExistsError):
        save_snapshot(str(snapshot), FailingTrainer(), {}, "fingerprint")
    assert (snapshot / "complete").read_text() == "preserved"


def test_unpublished_snapshot_is_rejected_before_directory_publication(tmp_path, monkeypatch):
    import areno.api.trainers.sft_snapshot as snapshot_module

    class StopBeforePublish(Exception):
        pass

    def stop_before_publish(source, destination):
        with pytest.raises(ValueError, match="incomplete|unpublished"):
            load_snapshot(str(source), FakeTrainer(), "fingerprint", lambda state: None)
        raise StopBeforePublish

    monkeypatch.setattr(snapshot_module.os, "rename", stop_before_publish)
    with pytest.raises(StopBeforePublish):
        save_snapshot(
            str(tmp_path / "step_000001.training-state"), FakeTrainer(), {"completed_batches": 1}, "fingerprint"
        )


def test_snapshot_interrupted_after_directory_rename_is_not_loadable(tmp_path, monkeypatch):
    import areno.api.trainers.sft_snapshot as snapshot_module

    target = tmp_path / "step_000002.training-state"

    def fail_before_manifest(source, destination):
        raise OSError("simulated failure publishing manifest")

    monkeypatch.setattr(snapshot_module.os, "replace", fail_before_manifest)
    with pytest.raises(OSError, match="simulated failure"):
        save_snapshot(str(target), FakeTrainer(), {"completed_batches": 2}, "fingerprint")
    assert target.is_dir()
    with pytest.raises(ValueError, match="incomplete"):
        load_snapshot(str(target), FakeTrainer(), "fingerprint", lambda state: None)


def test_snapshot_sync_failure_does_not_publish(tmp_path, monkeypatch):
    import areno.api.trainers.sft_snapshot as snapshot_module

    def fail_sync(descriptor):
        raise OSError("simulated disk sync failure")

    monkeypatch.setattr(snapshot_module.os, "fsync", fail_sync)
    target = tmp_path / "step_000002.training-state"
    with pytest.raises(OSError, match="simulated disk sync failure"):
        save_snapshot(str(target), FakeTrainer(), {"completed_batches": 2}, "fingerprint")
    assert not target.exists()


def test_disk_full_while_writing_keeps_previous_snapshot(tmp_path):
    previous = tmp_path / "step_000001.training-state"
    target = tmp_path / "step_000002.training-state"
    save_snapshot(str(previous), FakeTrainer(), {"completed_batches": 1}, "fingerprint")

    class FullDiskTrainer:
        def save_training_state(self, path):
            Path(path).write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(OSError) as error:
        save_snapshot(str(target), FullDiskTrainer(), {"completed_batches": 2}, "fingerprint")
    assert error.value.errno == errno.ENOSPC
    assert not target.exists()
    state = load_snapshot(str(previous), FakeTrainer(), "fingerprint", lambda state: None)
    assert state["completed_batches"] == 1


@pytest.mark.parametrize("stage", ["worker", "manifest"])
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process signals")
def test_killed_writer_leaves_previous_snapshot_recoverable(tmp_path, stage):
    previous = tmp_path / "step_000001.training-state"
    target = tmp_path / "step_000002.training-state"
    save_snapshot(str(previous), FakeTrainer(), {"completed_batches": 1}, "fingerprint")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=_crash_during_snapshot, args=(str(target), ready, stage))
    process.start()
    try:
        assert ready.wait(timeout=30), f"writer did not reach {stage}: exitcode={process.exitcode}"
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=10)
    assert process.exitcode == -signal.SIGKILL
    restored = FakeTrainer()
    state = load_snapshot(str(previous), restored, "fingerprint", lambda state: None)
    assert state["completed_batches"] == 1
    if stage == "worker":
        assert not target.exists()
        incomplete = next(tmp_path.glob(".step_000002.training-state.*"))
    else:
        incomplete = target
    with pytest.raises(ValueError, match="incomplete|unpublished"):
        load_snapshot(str(incomplete), FakeTrainer(), "fingerprint", lambda state: None)


def test_legacy_completed_manifest_is_accepted_but_hidden_temporary_copy_is_rejected(tmp_path):
    previous = tmp_path / "step_000001.training-state"
    save_snapshot(str(previous), FakeTrainer(), {"completed_batches": 1}, "fingerprint")
    manifest_path = previous / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    assert load_snapshot(str(previous), FakeTrainer(), "fingerprint", lambda state: None)["completed_batches"] == 1
    hidden = tmp_path / ".step_000001.training-state.interrupted"
    shutil.copytree(previous, hidden)
    with pytest.raises(ValueError, match="unpublished"):
        load_snapshot(str(hidden), FakeTrainer(), "fingerprint", lambda state: None)


def test_public_trainer_restores_metric_step_after_backend_load():
    class Backend:
        def load_training_state(self, ctx, path):
            assert ctx.global_step == -1
            assert path == "state.pt"

    trainer = Trainer(world_size=1, model_path="model", metrics_log_dir=None)
    trainer._ctx = Context(1, "model", object())
    trainer._backend = Backend()
    trainer._initialized = True
    trainer.load_training_state("state.pt", completed_batches=7)
    assert trainer._ctx.global_step == 6
    trainer._begin_step()
    assert trainer._ctx.global_step == 7
