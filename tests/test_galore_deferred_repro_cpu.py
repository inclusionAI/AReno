"""CPU-only contract checks for the portable runner; no native model import."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deferred_repro", ROOT / "scripts/galore_deferred/run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def helper_functions():
    tree = ast.parse((ROOT / "scripts/galore_sft/common.py").read_text())
    nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("batch_indices", "evaluate")
    ]
    namespace = {"random": random, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual common.py functions>", "exec"), namespace)
    return namespace


def arguments(tmp_path, mode="performance", method="deferred", steps=None, fused=False):
    return RUN.configure(
        SimpleNamespace(
            model=tmp_path / "model",
            data_dir=tmp_path / "data",
            output_dir=tmp_path,
            mode=mode,
            method=method,
            seed=0,
            steps=steps,
            fused_scale_writeback=fused,
        )
    )


class Engine:
    def __init__(self, args):
        self.args, self._ctx, self.calls = args, SimpleNamespace(global_step=-1), 0
        self.closed = False

    def init(self):
        pass

    def get_tokenizer(self):
        return None

    def score_logprobs(self, role, tokens, microbatch_size):
        return [[-1.0] * len(sample) for sample in tokens]

    def train(self, batch, loss_fn, mini_bs):
        index = self.calls
        self.calls += 1
        self._ctx.global_step = index
        return {
            "loss": 1.0,
            "logp_abs_diff_mean": 0,
            "update_optimizer_steps": 1,
            "update_worker_step": index + 1,
            "update_refresh_any": int(self.args.optimizer == "galore" and index % 200 == 0),
            "optimizer_deferred_projection_enabled": int(self.args.method == "deferred"),
            "optimizer_fused_scale_writeback_enabled": int(self.args.fused_scale_writeback),
            "optimizer_model_owned_master_enabled": 0,
            "galore_master_bytes": 1,
            "deferred_owner_count": 226,
            "deferred_owners_once": 1,
            "deferred_barrier_versions_unchanged": 1,
            "deferred_full_leaf_bytes_after_backward": 0,
            "deferred_cache_empty_after_step": 1,
            "sft_pack_count": 1,
            "sft_dp_shards_per_pack": 1,
            "sft_total_target_tokens": sum(len(s.tokens) - s.prompt_len for s in batch),
            "sft_grad_scale": 1,
            "update_cuda_time_s": 1,
            "update_peak_allocated_bytes": 10000 if index < 2 else 20,
            "update_peak_reserved_bytes": 20000 if index < 2 else 30,
        }

    def save_checkpoint(self, *args):
        raise AssertionError("runner must never save")

    def close(self):
        self.closed = True


def fake_common(args):
    engine, encoded, evaluations = Engine(args), [], []
    funcs = helper_functions()

    def encode(tokenizer, supplied, split):
        encoded.append((split, engine.calls))
        if split == "train":
            return [SimpleNamespace(tokens=[1, 2, 3], prompt_len=1)] * 13544, {"rows": 13544, "targets": 27088}
        assert split == "validation"
        # 707*73 +570 =52181 targets, exactly the declared coverage.
        sizes = [73] * 707 + [570]
        return [SimpleNamespace(tokens=[0] * (n + 1), prompt_len=1) for n in sizes], {"rows": 708, "targets": 52181}

    def evaluate(eng, samples, micro):
        evaluations.append(eng.calls)
        return funcs["evaluate"](eng, samples, micro)

    common = SimpleNamespace(
        trainer=lambda *a, **k: engine,
        encode=encode,
        evaluate=evaluate,
        batch_indices=funcs["batch_indices"],
        write=lambda path, value: path.write_text(json.dumps(value)),
    )
    return common, engine, encoded, evaluations


@pytest.mark.parametrize("method", ["adamw", "adam8", "strict", "deferred"])
def test_performance_budget_warm_exclusion_and_refreshes_cpu(tmp_path, method):
    args = arguments(tmp_path, method=method)
    common, engine, encoded, evaluated = fake_common(args)
    ticks = iter(range(2000))
    summary = RUN.execute(args, common, None, clock=lambda: next(ticks))
    assert engine.calls == 402 and engine.closed
    assert summary["all"]["updates"] == 400
    assert summary["all"]["allocated_peak_bytes"] == 20
    assert summary["all"]["score_plus_train_seconds"] == 800
    assert summary["refresh"]["updates"] == (2 if method in ("strict", "deferred") else 0)
    assert encoded == [("train", 0)] and evaluated == []
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if r["update_refresh_any"] and not r["warmup"]] == (
        [200, 400] if method in ("strict", "deferred") else []
    )


@pytest.mark.parametrize("steps", [1024, 4096])
def test_quality_exact_budget_final_only_response_nll_cpu(tmp_path, steps):
    args = arguments(tmp_path, "quality", steps=steps)
    common, engine, encoded, evaluated = fake_common(args)
    summary = RUN.execute(args, common, None)
    assert engine.calls == steps and engine.closed
    assert summary["all"]["updates"] == steps and summary["warmup_updates"] == 0
    assert encoded == [("train", 0), ("validation", steps)] and evaluated == [steps]
    assert summary["validation"] == {"loss": 1.0, "examples": 708, "target_tokens": 52181}
    assert not summary["read_test"] and not summary["saved_checkpoint"]


@pytest.mark.parametrize("mode,steps", [("quality", None), ("quality", 8), ("performance", 1024)])
def test_invalid_budget_rejected_cpu(tmp_path, mode, steps):
    with pytest.raises(ValueError):
        arguments(tmp_path, mode, steps=steps)


def test_native_guard_failure_closes_engine_cpu(tmp_path):
    args = arguments(tmp_path)
    common, engine, _, _ = fake_common(args)
    original = engine.train

    def wrong(*a, **k):
        result = original(*a, **k)
        result["update_optimizer_steps"] = 2
        return result

    engine.train = wrong
    with pytest.raises(RuntimeError, match="update_optimizer_steps"):
        RUN.execute(args, common, None)
    assert engine.closed and engine.calls == 1


def test_original_input_index_prefix_cpu():
    prefix = json.loads((ROOT / "scripts/galore_deferred/input-prefix.json").read_text())
    orders = {}
    batch_indices = helper_functions()["batch_indices"]
    for row in prefix["rows"]:
        assert batch_indices(row["step"], prefix["seed"], prefix["train_rows"], 4, orders) == row["sample_indices"]


def test_batch_hash_uses_formal_serialization_cpu():
    batch = [SimpleNamespace(tokens=[1, 2, 3], prompt_len=1)]
    assert RUN.batch_identity(batch) == RUN.hashlib.sha256(b"[[[1,2,3],1]]").hexdigest()


def test_unregistered_experiment_rejected_before_cuda_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENO_EXPERIMENT_UNKNOWN", "0")
    with pytest.raises(ValueError, match="unregistered"):
        RUN.process_environment(arguments(tmp_path))


@pytest.mark.parametrize("enabled", [False, True])
def test_fusion_option_environment_native_guard_and_budget_cpu(tmp_path, monkeypatch, enabled):
    for key in (
        "ARENO_BENCH_SEED",
        "ARENO_EXPERIMENT_DEFERRED_PROJECTION",
        "ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK",
        "ARENO_DEFERRED_REPRO_METHOD",
    ):
        monkeypatch.setenv(key, "stale")
    args = arguments(tmp_path, fused=enabled)
    RUN.process_environment(args)
    assert RUN.os.environ["ARENO_BENCH_SEED"] == "0"
    assert RUN.os.environ["ARENO_EXPERIMENT_DEFERRED_PROJECTION"] == "1"
    assert RUN.os.environ["ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK"] == str(int(enabled))
    common, engine, _, _ = fake_common(args)
    summary = RUN.execute(args, common, None)
    assert engine.calls == 402 and summary["all"]["updates"] == 400
    result = engine.train([SimpleNamespace(tokens=[1, 2], prompt_len=1)], None, mini_bs=4)
    result["optimizer_fused_scale_writeback_enabled"] = 1 - int(enabled)
    with pytest.raises(RuntimeError, match="optimizer_fused_scale_writeback_enabled"):
        RUN.check_update(result, engine, args, 402, 1)


@pytest.mark.parametrize("method", ["adamw", "adam8", "strict"])
def test_fusion_rejects_other_methods_cpu(tmp_path, method):
    with pytest.raises(ValueError, match="requires --method deferred"):
        arguments(tmp_path, method=method, fused=True)


def test_fusion_cli_is_explicit_opt_in_cpu():
    base = [
        "--model",
        "model",
        "--data-dir",
        "data",
        "--output-dir",
        "output",
        "--method",
        "deferred",
        "--mode",
        "performance",
    ]
    assert RUN.parser().parse_args(base).fused_scale_writeback is False
    assert RUN.parser().parse_args(base + ["--fused-scale-writeback"]).fused_scale_writeback is True


def test_worker_rejects_missing_fusion_before_native_update_cpu(monkeypatch):
    calls = []

    class Manager:
        def train(self, payload):
            calls.append(payload)

    monkeypatch.setitem(sys.modules, "areno.engine.training", SimpleNamespace(TrainingManager=Manager))
    monkeypatch.setenv("ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK", "1")
    RUN.install_observer()
    manager = Manager()
    manager.worker = SimpleNamespace(optimizer=SimpleNamespace(_fused_scale_writeback=False))
    payload = SimpleNamespace(data_packs_by_dp=[[{"_sft_total_target_tokens": 1, "_sft_grad_scale": 1}]])
    with pytest.raises(RuntimeError, match="policy does not match"):
        manager.train(payload)
    assert calls == []
