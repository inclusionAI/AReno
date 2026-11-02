"""Reject misleading P1 endpoints without CUDA or executing archived code."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/reports/galore-p1-quality-evidence"
SPEC = importlib.util.spec_from_file_location("p1_quality_verifier", EVIDENCE / "verify.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture(scope="module")
def bundle():
    return VERIFIER.load_bundle(EVIDENCE)


def test_all_six_raw_endpoints_cpu(bundle):
    result = VERIFIER.audit_bundle(*bundle)
    assert result["training_updates"] == 24576
    assert result["raw_response_nll_recomputed"] is True
    assert result["all_completed_worker_score_train_records_checked"] is True
    assert result["old_completed3_unchanged"] is True
    assert -1.33 < result["relative_mean_nll_difference_percent"] < -1.32
    assert all(row["relative_nll_difference_percent"] < 0 for row in result["paired_seeds"])
    assert result["fresh_8bit_quality_comparison"] is False
    assert result["untouched_PR640_end_to_end"] is False


@pytest.mark.parametrize("mutation", ["runtime", "native", "protocol", "original640"])
def test_changed_scientific_source_rejected_cpu(bundle, mutation):
    _, original = bundle
    files = dict(original)
    if mutation in ("runtime", "native"):
        name = "records/queue/harness/PREPARED.json"
        value = json.loads(files[name])
        value["runtime_sha256" if mutation == "runtime" else "native_sha256"] = "0" * 64
        files[name] = json.dumps(value).encode()
    elif mutation == "protocol":
        name = "records/queue/PREREGISTERED_QUALITY_PROTOCOL.json"
        files[name] += b"\n"
    else:
        name = "sources/runtime/areno/experimental/controlled_galore640/original.py.txt"
        files[name] += b"\n"
    with pytest.raises(ValueError):
        VERIFIER.audit_sources(files)


@pytest.mark.parametrize(
    "mutation", ["missing_seed", "replace_reused", "different_gpu", "extra_retry", "old_raw_changed"]
)
def test_case_selection_or_reuse_corruption_rejected_cpu(bundle, mutation):
    _, original = bundle
    files = dict(original)
    if mutation == "extra_retry":
        name = "records/queue/control/STATUS.json"
        value = json.loads(files[name])
        value["queue"]["1"]["dispatches"] = 2
    elif mutation == "old_raw_changed":
        name = "cases/port640_s0/worker-score-0000.json"
        files[name] += b"\n"
        with pytest.raises(ValueError, match="old completed evidence changed"):
            VERIFIER.audit_selection(files)
        return
    else:
        name = "records/cases.json"
        value = json.loads(files[name])
        if mutation == "missing_seed":
            value.pop()
        elif mutation == "replace_reused":
            value[0]["reused"] = not value[0]["reused"]
        else:
            value[0]["gpu"] = 0
    files[name] = json.dumps(value).encode()
    with pytest.raises(ValueError):
        VERIFIER.audit_selection(files)


def test_changed_raw_logprob_rejected_even_after_stream_rehash_cpu(bundle, tmp_path):
    _, files = bundle
    prefix = "cases/deferred647_s0/"
    name = "validation-response-logprobs.jsonl"
    rows = [json.loads(line) for line in files[prefix + name].splitlines()]
    rows[0]["response_logprobs"][0] += 1
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    (tmp_path / name).write_bytes(raw)
    metadata = json.loads(files[prefix + "validation-capture-metadata.json"])
    metadata["stream_sha256"] = hashlib.sha256(raw).hexdigest()
    metadata["stream_bytes"] = len(raw)
    (tmp_path / "validation-capture-metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="NLL reconstruction differs"):
        VERIFIER.AUDIT.recompute_capture(tmp_path)


def test_native_gradient_barrier_violation_rejected_cpu(bundle):
    _, files = bundle
    row = json.loads(files["cases/deferred647_s0/steps.jsonl"].splitlines()[0])
    row["deferred_barrier_versions_unchanged"] = 0
    with pytest.raises(ValueError, match="deferred_barrier_versions_unchanged"):
        VERIFIER.AUDIT.audit_update(row, method="deferred647", index=0, targets=row["target_tokens"], sdk_step=0)


def test_static_schedule_matches_frozen_helper_ast_cpu(bundle):
    _, files = bundle
    tree = ast.parse(files["sources/quality_helper.py.txt"].decode())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "batch_indices")
    trusted = ast.parse((EVIDENCE / "audit.py").read_text())
    copied = next(n for n in trusted.body if isinstance(n, ast.FunctionDef) and n.name == "batch_indices")
    assert ast.dump(original, include_attributes=False) == ast.dump(copied, include_attributes=False)


def test_corrupt_archive_rejected_cpu(bundle, tmp_path):
    manifest, _ = bundle
    changed = copy.deepcopy(manifest)
    changed["parts"][0]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(changed))
    part = manifest["parts"][0]["archive"]
    (tmp_path / part).write_bytes((EVIDENCE / part).read_bytes())
    with pytest.raises(ValueError, match="archive identity mismatch"):
        VERIFIER.load_bundle(tmp_path)


def test_isolated_standard_library_cli_cpu(tmp_path):
    output = tmp_path / "analysis.json"
    process = subprocess.run(
        [sys.executable, "-I", "-S", str(EVIDENCE / "verify.py"), str(EVIDENCE), "--json-output", str(output)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text())["status"] == "passed"


def test_optimized_python_cannot_disable_checks_cpu():
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-O",
            "-c",
            f"import runpy; m=runpy.run_path({str(EVIDENCE / 'verify.py')!r}); m['audit_bundle']({{}}, {{}})",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "requires assertions" in process.stderr
