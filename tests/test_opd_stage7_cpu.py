"""Stage 7 CPU-only: real report algorithms, fake model interface, tiny real safetensors."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/opd"))
import eval_core  # noqa: E402
import evaluate_models  # noqa: E402
import compare_eval  # noqa: E402
import runtime_report  # noqa: E402
import compare_weights  # noqa: E402


def fixture_data(tmp_path, answers=("21", "54")):
    fp = tmp_path / "eval.jsonl"
    records = [dict(id=f"e{i}", prompt=f"Prompt {i}?", answer=answer)
               for i, answer in enumerate(answers)]
    fp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return fp, records


def fake_result(dataset_file, *, completions, nlls=None, model="model"):
    records, h = eval_core.load_eval_set(dataset_file)
    nlls = nlls or [1.0] * len(records)
    rows = [{**r, "completion": reply, "correct": eval_core.exact_match(reply, r),
             "answer_tokens": 1, "answer_nll_sum": nll, "generated_tokens": 1,
             "generation_wall_time_s": 0.1}
            for r, reply, nll in zip(records, completions, nlls, strict=True)]
    return {"schema_version": 1, "model": model, "dataset_sha256": h,
            "max_new_tokens": 32, "decoding": "greedy", "disable_thinking": True,
            "rows": rows}


def test_eval_jsonl_and_fingerprint(tmp_path):
    path, records = fixture_data(tmp_path)
    rows, h = eval_core.load_eval_set(path)
    assert rows[0] == {**records[0], "aliases": []}
    assert len(h) == 64
    assert eval_core.load_eval_set(path)[1] == h


@pytest.mark.parametrize("bad", ['{}', '{"id":"e","prompt":"x"}', '{"id":"e","prompt":"x","answer":""}', 'oops'])
def test_reject_bad_eval_data(tmp_path, bad):
    path = tmp_path / "eval.jsonl"
    path.write_text(bad + "\n")
    with pytest.raises(ValueError):
        eval_core.load_eval_set(path)


def test_reject_duplicate_ids(tmp_path):
    path, _ = fixture_data(tmp_path)
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        eval_core.load_eval_set(path)


def test_normalize_and_alias():
    assert eval_core.normalize_answer("  Tokyo! ") == "tokyo"
    assert eval_core.exact_match("United States.", {"answer": "USA", "aliases": ["United States"]})
    assert not eval_core.exact_match("Tokyo is nice", {"answer": "Tokyo"})


def test_holdout_leakage(tmp_path):
    path, _ = fixture_data(tmp_path)
    data, _ = eval_core.load_eval_set(path)
    training = tmp_path / "train.jsonl"
    training.write_text('{"prompt":"prompt 0?"}\n')
    with pytest.raises(ValueError, match="overlap"):
        eval_core.validate_train_holdout_disjoint(training, data)


def test_stage6_heldout_set_is_distinct():
    data, _ = eval_core.load_eval_set(ROOT / "examples/opd/heldout_eval.jsonl")
    assert len(data) == 12
    eval_core.validate_train_holdout_disjoint(ROOT / "examples/opd/prompts.jsonl", data)


def test_weighted_nll_across_response_lengths():
    rows = [dict(correct=True, answer_nll_sum=2., answer_tokens=2, generation_wall_time_s=0.1),
            dict(correct=False, answer_nll_sum=9., answer_tokens=3, generation_wall_time_s=0.3)]
    summary = eval_core.summary_from_rows(rows)
    assert summary["exact_match"] == 0.5
    assert summary["answer_nll_per_token"] == pytest.approx(11 / 5)
    assert summary["answer_perplexity"] == pytest.approx(math.exp(11 / 5))
    assert summary["mean_generation_wall_time_s"] == pytest.approx(0.2)


@pytest.mark.parametrize("key,value", [("correct", 1), ("answer_tokens", 0), ("answer_nll_sum", float("nan")), ("answer_nll_sum", -1)])
def test_bad_summary_rows_rejected(key, value):
    row = dict(correct=True, answer_nll_sum=1., answer_tokens=1)
    row[key] = value
    with pytest.raises(ValueError):
        eval_core.summary_from_rows([row])


def test_paired_delta_and_bootstrap_repeatable(tmp_path):
    path, _ = fixture_data(tmp_path)
    b = fake_result(path, completions=["wrong", "54"])
    a = fake_result(path, completions=["21", "54"], nlls=[0.2, 0.3])
    result = compare_eval.compare(b, a, bootstrap_replicates=500)
    assert result["delta"]["delta_exact_match"] == 0.5
    assert result["before"]["exact_match"] == 0.5
    assert result["after"]["exact_match"] == 1.
    assert result["no_update_control"]["kind"] == "identity_reference"
    assert result == compare_eval.compare(b, a, bootstrap_replicates=500)
    assert "not a matched" in compare_eval.render_markdown(result)


def test_control_is_labeled_separate_inference(tmp_path):
    path, _ = fixture_data(tmp_path)
    b = fake_result(path, completions=["21", "54"])
    a = fake_result(path, completions=["wrong", "54"])
    c = fake_result(path, completions=["21", "54"], model="unchanged")
    result = compare_eval.compare(b, a, control=c)
    assert result["no_update_control"]["kind"] == "independent_unchanged_checkpoint_evaluation"
    assert result["no_update_control"]["delta_exact_match_vs_before"] == 0


def test_mismatched_fingerprint_fails(tmp_path):
    path, _ = fixture_data(tmp_path)
    b = fake_result(path, completions=["21", "54"])
    a = fake_result(path, completions=["21", "54"])
    a["dataset_sha256"] = "another"
    with pytest.raises(ValueError, match="fingerprints"):
        compare_eval.compare(b, a)


def test_different_settings_fail(tmp_path):
    path, _ = fixture_data(tmp_path)
    b = fake_result(path, completions=["21", "54"])
    a = fake_result(path, completions=["21", "54"])
    a["max_new_tokens"] = 12
    with pytest.raises(ValueError, match="settings"):
        compare_eval.compare(b, a)


def test_missing_prediction_ids_fail(tmp_path):
    path, _ = fixture_data(tmp_path)
    b = fake_result(path, completions=["21", "54"])
    a = fake_result(path, completions=["21", "54"])
    a["rows"] = a["rows"][:1]
    with pytest.raises(ValueError, match="IDs"):
        compare_eval.compare(b, a)


def test_incorrect_saved_correct_flag_fail(tmp_path):
    path, _ = fixture_data(tmp_path)
    b = fake_result(path, completions=["21", "54"])
    b["rows"][0]["correct"] = False
    outfile = tmp_path / "pred.json"
    outfile.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="incorrectly scored"):
        eval_core.read_prediction_file(outfile)


def test_runtime_extract_two_real_log_entries():
    stat = {"opd_loss": -0.5, "opd_response_token_count": 5.,
            "opd_teacher_score_time_s": 0.2, "student_train_wall_time_s": 0.1,
            "step_e2e_time_s": 0.5, "step_rollout_time_s": 0.15}
    lines = "\n".join([f"2026-10-20 INFO train_stats={stat!r}" for _ in range(2)])
    rows = runtime_report.extract_runtime_steps(lines)
    summary = runtime_report.summarize_runtime(rows)
    assert summary["steps"] == 2
    assert summary["response_tokens"] == 10
    assert summary["end_to_end_response_tokens_per_s"] == pytest.approx(10.)
    assert summary["timings"]["opd_teacher_score_time_s"]["total_observed_s"] == pytest.approx(0.4)


def test_runtime_no_e2e_reports_null():
    rows = runtime_report.extract_runtime_steps("train_stats={'opd_loss': -0.2, 'opd_response_token_count': 4.0}")
    result = runtime_report.summarize_runtime(rows)
    assert result["end_to_end_response_tokens_per_s"] is None
    assert result["timings"]["step_rollout_time_s"]["mean_s"] is None


@pytest.mark.parametrize("value", ["nan", "inf", "-1.0"])
def test_runtime_reject_invalid_tokens(value):
    with pytest.raises(ValueError):
        runtime_report.extract_runtime_steps("train_stats={'opd_loss': 0.1, 'opd_response_token_count': " + value + "}")


def test_runtime_real_stage6_cpu_log_is_labeled_fake():
    stage6_path = Path("/mnt/data/opd_stage6_cpu_evidence/simulated_run")
    if not stage6_path.is_dir():
        pytest.skip("Local CPU evidence not included in user repository")
    assert runtime_report.classify_run(stage6_path) == "cpu_fake_backend_simulation"
    rows = runtime_report.extract_runtime_steps((stage6_path / "run.log").read_text())
    assert len(rows) == 2
    assert rows[1]["opd_loss"] == pytest.approx(-0.984756827, abs=1e-5)
    assert runtime_report.summarize_runtime(rows, kind="cpu_fake_backend_simulation")["end_to_end_response_tokens_per_s"] is None


def test_synthetic_real_safetensors_diff_and_identical(tmp_path):
    from safetensors.torch import save_file
    before, after, same = [tmp_path / d for d in ("before", "after", "same")]
    for path in (before, after, same):
        path.mkdir()
        (path / "config.json").write_text('{"model_type":"fake"}')
    p = dict(x=torch.tensor([1., 2.]), y=torch.tensor([[3., 4.]]))
    q = dict(x=torch.tensor([1., 3.]), y=p["y"].clone())
    save_file(p, str(before / "model.safetensors"))
    save_file(q, str(after / "model.safetensors"))
    save_file(p, str(same / "model.safetensors"))
    d = compare_weights.compare_weight_directories(before, after)
    assert d["complete_scan"]
    assert d["changed_tensors"] == 1
    assert d["l2_parameter_difference"] == pytest.approx(1.)
    assert d["nonzero_update_verified"] is True
    d2 = compare_weights.compare_weight_directories(before, same)
    assert d2["nonzero_update_verified"] is False
    small = compare_weights.compare_weight_directories(before, after, max_tensors=1)
    assert small["complete_scan"] is False
    assert small["nonzero_update_verified"] is None


def test_fake_safetensors_payload_rejected(tmp_path):
    a = tmp_path / "fake"
    a.mkdir()
    (a / "config.json").write_text('{"model_type":"qwen3"}')
    (a / "model.safetensors").write_bytes(b"fake-weights-for-integration-test")
    with pytest.raises(Exception):
        compare_weights.compare_weight_directories(a, tmp_path / "something")


def test_dry_run_is_network_free(tmp_path, capsys):
    path, _ = fixture_data(tmp_path)
    train = tmp_path / "train.jsonl"
    train.write_text('{"prompt":"completely different"}\n')
    out = tmp_path / "out.json"
    assert evaluate_models.main(["--model", "no-weights", "--dataset", str(path), "--training-dataset", str(train), "--output", str(out), "--dry-run"]) == 0
    assert "no_model_loaded" in capsys.readouterr().out
    assert not out.exists()


class ToyTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0
    def encode(self, text, add_special_tokens=True):
        if text == "ok":
            return [2]
        return [1]
    def decode(self, token_ids, skip_special_tokens=True):
        return "ok" if token_ids == [2] else "?"


class ToyModel:
    def generate(self, ids, **kwargs):
        return torch.cat([ids, torch.tensor([[2]], dtype=torch.long, device=ids.device)], 1)
    def __call__(self, input_ids):
        batch, steps = input_ids.shape
        logits = torch.zeros((batch, steps, 4), device=input_ids.device)
        logits[..., 2] = 1.5
        return SimpleNamespace(logits=logits)


def test_actual_torch_generation_scoring_alignment():
    record = {"id": "e1", "prompt": "Go", "answer": "ok", "aliases": []}
    result = evaluate_models.evaluate_one(ToyModel(), ToyTokenizer(), torch, "cpu", record, max_new_tokens=3)
    assert result["correct"] is True
    assert result["answer_tokens"] == 1
    assert result["generated_tokens"] == 1
    expected_nll = -torch.nn.functional.log_softmax(torch.tensor([0., 0., 1.5, 0.]), -1)[2].item()
    assert result["answer_nll_sum"] == pytest.approx(expected_nll)


def test_prompt_generation_uses_no_thinking():
    class FakeChat:
        chat_template = "template"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return [12, 13, 14]
    assert evaluate_models.encode_prompt(FakeChat(), "hello") == [12, 13, 14]


def test_comparison_is_honest_about_provenance(tmp_path):
    path, _ = fixture_data(tmp_path)
    before = fake_result(path, completions=["21", "wrong"])
    after = fake_result(path, completions=["21", "54"])
    r = compare_eval.compare(before, after)
    assert r["evidence_kind"] == "synthetic_or_unverified_comparison"
    before["evidence_kind"] = after["evidence_kind"] = "real_model_inference"
    assert compare_eval.compare(before, after)["evidence_kind"] == "real_model_comparison"


def test_compare_rejects_tokenizer_mismatch(tmp_path):
    path, _ = fixture_data(tmp_path)
    before = fake_result(path, completions=["21", "54"])
    after = fake_result(path, completions=["21", "54"])
    before["tokenizer"] = "student-tokenizer"
    after["tokenizer"] = "different"
    with pytest.raises(ValueError, match="tokenizers differ"):
        compare_eval.compare(before, after)


def test_runtime_training_signal_is_observed_not_inferred():
    rows = runtime_report.extract_runtime_steps(
        "train_stats={'opd_loss': -0.2, 'opd_response_token_count': 2.0, 'ratio_mean': 1.0}\n"
        "train_stats={'opd_loss': -0.4, 'opd_response_token_count': 3.0, 'ratio_mean': 1.1}")
    signal = runtime_report.summarize_runtime(rows)["training_signal"]
    assert signal["opd_loss"]["first"] == -0.2
    assert signal["opd_loss"]["last"] == -0.4
    assert signal["opd_advantage_mean"]["mean"] is None
