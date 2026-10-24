"""Stage 6 CPU verification of the actual OPD trainer/CLI without GPU weights."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "opd"))
import check_models  # noqa: E402
import run_smoke  # noqa: E402
import verify_run  # noqa: E402

from areno.api.data import PromptBatch, PromptItem
from areno.api.models import RolloutResult, RolloutSequence
from areno.api.trainer_config import OPDTrainerConfig


def _config(tmp_path: Path, **kw):
    opts = dict(
        algo="opd", ckpt="student", teacher_ckpt="teacher",
        dataset_path=str(ROOT / "examples/opd/prompts.jsonl"),
        model_hub="hf", epochs=1, max_steps=2,
        tp_size=1, world_size=1, batch_size=1, n_samples=1, mini_bs=1,
        score_micro_bs=1, max_prompt_tokens=128, max_new_tokens=16,
        save_path=str(tmp_path / "checkpoints"), save_interval=1,
        metrics_log_dir=str(tmp_path / "metrics"),
    )
    opts.update(kw)
    return OPDTrainerConfig(**opts)


class FakeStudentTeacherBackend:
    """Simulates GPU/backend boundary; OPD loss/backward are real PyTorch."""
    def __init__(self, *_args, **_kwargs):
        self.events = []
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.SGD([self.param], lr=0.1)
        self.trained_rows = []
        self.saved_parameters = []
        self.loss_metrics = []

    def init(self):
        self.events.append("student_init")

    def ensure_roles(self, roles):
        self.events.append("teacher_init")
        assert list(roles) == ["ref"]
        assert roles["ref"].trainable is False
        assert roles["ref"].path == "teacher"

    def close(self):
        self.events.append("close")

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=151645, decode=lambda ids: " ".join(map(str, ids)))

    def get_processor(self):
        return None

    def record_rollout_sample(self, entry):
        self.events.append("record_sample")

    def load_prompt_batches(self, dataset, *, batch_size, max_prompt_tokens):
        for idx, record in enumerate(dataset[:2]):
            yield PromptBatch(items=[PromptItem(
                prompt=record["prompt"], solutions=None,
                input_tokens=[100 + idx, 200 + idx], record=dict(record),
            )], scanned=1, skipped_long=0, total_skipped_long=0)

    @asynccontextmanager
    async def rollout_session(self, **kwargs):
        self.events.append("rollout_start")
        yield None
        self.events.append("rollout_end")

    async def rollout_token_batch_async(self, prompts, n_samples, sampling_params, *, prompt_features):
        self.events.append("student_generate")
        return [RolloutResult(sequences=[RolloutSequence(
            resp_tokens=[301, 302], resp_logprobs=[-1.2, -0.9],
        )]) for _ in prompts]

    def score_logprobs(self, role, token_rows, *, features, microbatch_size):
        self.events.append("teacher_score")
        assert role == "ref" and microbatch_size == 1
        return [[-0.4, -0.4, -0.2, -0.1] for _ in token_rows]

    def train(self, rows, loss_fn, *, mini_bs, gradient_accumulation_steps):
        self.events.append("train")
        self.trained_rows.extend(rows)
        from areno.api.backend.cuda.training import make_train_pack as _make_train_pack
        pack = _make_train_pack(rows)
        logprobs = pack["logprobs"][:, 1:].detach().clone() + self.param
        loss, stats = loss_fn(pack, logprobs)
        assert loss.requires_grad
        self.optimizer.zero_grad()
        loss.backward()
        assert self.param.grad is not None and torch.isfinite(self.param.grad)
        self.optimizer.step()
        metric = {k: float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
                  for k, v in stats.items()}
        self.loss_metrics.append(metric)
        return metric

    def save_checkpoint(self, path):
        self.events.append("save")
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        (dest / "model.safetensors").write_bytes(b"fake-weights-for-integration-test")
        self.saved_parameters.append(float(self.param.detach().item()))
        return str(dest)


def _write_fake_log(path, stats, steps=2):
    log = []
    for _ in range(steps):
        log.extend([
            "role=student stage=rollout_start",
            "role=ref stage=logprob_score_end",
            "role=student stage=train_end",
            f"train_stats={stats!r}",
            "stage=save_checkpoint_end",
        ])
    (path / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")


def test_default_data_are_short_prompt_only_jsonl():
    lines = (ROOT / "examples/opd/prompts.jsonl").read_text().splitlines()
    assert len(lines) >= 4
    assert all(set(json.loads(row)) == {"prompt"} and json.loads(row)["prompt"] for row in lines)


def test_build_command_preserves_student_teacher_and_checkpoint_cadence(tmp_path):
    args = run_smoke.parse_args(["--output-dir", str(tmp_path), "--steps", "2"])
    cmd = run_smoke.build_command(args, python_executable="python")
    assert cmd[:3] == ["python", "-m", "areno.cli.train"]
    for flag in ("--algo", "--teacher-ckpt", "--opd-coef", "--save-path",
                 "--save-interval", "--max-steps", "--score-micro-bs"):
        assert flag in cmd
    assert cmd[cmd.index("--algo") + 1] == "opd"
    assert cmd[cmd.index("--max-steps") + 1] == "2"
    assert cmd[cmd.index("--save-interval") + 1] == "1"
    assert "--reward-ckpt" not in cmd and "--reward-fn-path" not in cmd


@pytest.mark.parametrize("values", [
    ["--steps", "0"], ["--tp-size", "2", "--world-size", "1"],
    ["--dataset", "/does/not/exist.jsonl"],
])
def test_runner_rejects_bad_arguments(values):
    args = run_smoke.parse_args(values)
    with pytest.raises(ValueError):
        run_smoke.validate_args(args)


def test_dry_run_does_not_require_gpu_or_download(capsys, tmp_path):
    code = run_smoke.main(["--dry-run", "--output-dir", str(tmp_path / "run")])
    assert code == 0
    assert "areno.cli.train" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def _tok(vocab, eos=3, pad=0, bos=1):
    return SimpleNamespace(get_vocab=lambda: vocab, eos_token_id=eos, bos_token_id=bos, pad_token_id=pad)


def _cfg(vocab_size=5, model_type="qwen3"):
    return SimpleNamespace(vocab_size=vocab_size, model_type=model_type)


def test_model_preflight_equal_tokenizer_mapping():
    x = check_models.validate_compatibility(_tok({"a": 2, "b": 3}), _tok({"a": 2, "b": 3}), _cfg(), _cfg())
    assert x["compatible"] and x["vocab_size"] == 5


@pytest.mark.parametrize("student,teacher,scfg,tcfg", [
    (_tok({"a": 2, "b": 3}), _tok({"a": 3, "b": 2}), _cfg(), _cfg()),
    (_tok({"a": 2}), _tok({"a": 2}, eos=4), _cfg(), _cfg()),
    (_tok({"a": 2}), _tok({"a": 2}), _cfg(), _cfg(model_type="llama")),
    (_tok({"a": 2}), _tok({"a": 2}), _cfg(), _cfg(vocab_size=6)),
    (_tok({"a": 6}), _tok({"a": 6}), _cfg(), _cfg()),
])
def test_model_preflight_catches_unsafe_model_pairs(student, teacher, scfg, tcfg):
    with pytest.raises(ValueError):
        check_models.validate_compatibility(student, teacher, scfg, tcfg)


def test_checkpoint_verifier_rejects_missing_or_empty_weights(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    assert not verify_run.checkpoint_is_nonempty(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"")
    assert not verify_run.checkpoint_is_nonempty(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"abc")
    assert verify_run.checkpoint_is_nonempty(ckpt)


def test_checkpoint_verifier_checks_weight_index_members(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "part-1.safetensors").write_bytes(b"abc")
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    assert not verify_run.checkpoint_is_nonempty(ckpt)


def test_verify_run_success_and_failure_modes(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    for step in (1, 2):
        ckpt = root / "checkpoints" / f"step_{step:06d}"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text("{}")
        (ckpt / "model.safetensors").write_bytes(b"123")
    data = {"opd_loss": -0.25, "opd_response_token_count": 2.0}
    _write_fake_log(root, data)
    assert verify_run.verify_run(root)["ok"]
    _write_fake_log(root, {"opd_loss": float("nan"), "opd_response_token_count": 2.0})
    with pytest.raises(ValueError, match="finite OPD loss"):
        verify_run.verify_run(root)
    _write_fake_log(root, data)
    (root / "checkpoints/step_000002/model.safetensors").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        verify_run.verify_run(root)


def test_full_cli_run_real_loss_gradient_checkpoint_with_fake_backend(monkeypatch, tmp_path):
    """Run cli.run() -> model roles -> rollout -> teacher -> PyTorch update -> HF-shaped save."""
    from areno.cli import train as cli
    import areno.api
    backend = FakeStudentTeacherBackend()
    monkeypatch.setattr(areno.api, "Trainer", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(cli, "resolve_model_refs_for_config", lambda config: config)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda *_args, **kwargs: [json.loads(line) for line in
            (ROOT / "examples/opd/prompts.jsonl").read_text().splitlines()],
        load_from_disk=lambda *_args: None,
    ))
    config = _config(tmp_path)
    cli.run(config)
    assert backend.events.count("student_init") == 1
    assert backend.events.count("teacher_init") == 1
    assert backend.events.count("student_generate") == 2
    assert backend.events.count("teacher_score") == 2
    assert backend.events.count("train") == 2
    assert backend.events.count("save") == 2
    assert backend.events[-1] == "close"
    assert len(backend.trained_rows) == 2
    assert backend.saved_parameters[0] != 0.0
    assert backend.saved_parameters[1] != backend.saved_parameters[0]
    assert all(verify_run.checkpoint_is_nonempty(
        tmp_path / "checkpoints" / f"step_{step:06d}") for step in (1, 2))
    assert all(torch.isfinite(torch.tensor(m["opd_loss"])) for m in backend.loss_metrics)
    _write_fake_log(tmp_path, {**backend.loss_metrics[0], "opd_response_token_count": 2.0})
    report = verify_run.verify_run(tmp_path, steps=2)
    assert report["ok"] and len(report["checkpoint_paths"]) == 2
    assert all(entry["response_tokens"] == 2 for entry in report["train_steps"])


def test_real_script_is_dry_run_executable_from_repo_root():
    process = subprocess.run([sys.executable, "examples/opd/run_smoke.py", "--dry-run"],
                             cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert "areno.cli.train" in process.stdout
