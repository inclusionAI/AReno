"""Final-source quality evidence checks; no CUDA or archived code execution."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/reports/galore-final-quality-evidence"
SPEC = importlib.util.spec_from_file_location("final_quality_verifier", EVIDENCE / "verify.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture(scope="module")
def bundle():
    return VERIFIER.load_bundle(EVIDENCE)


def test_real_final_source_quality_and_raw_logprobs_cpu(bundle):
    result = VERIFIER.audit_bundle(*bundle, repo=REPO)
    assert result["training_updates"] == 36864
    assert result["primary_quality_gate"]["passed"] is True
    assert result["raw_response_nll_recomputed"] is True
    assert result["token_payload_replayed_offline"] is False
    assert result["current_repo_checked"] is True
    assert result["response_nll_comparisons"]["adam8"]["mean_relative_nll_increase"] > 0
    assert all(28 < row["memory_reduction_percent"] < 31 for row in result["same_card_training_observations"])


@pytest.mark.parametrize("mutation", ["loss", "count", "guard", "input", "capture", "schedule", "source"])
def test_corrupted_scientific_records_rejected_cpu(bundle, mutation):
    manifest, original = bundle
    files = dict(original)
    prefix = "stage3/outputs/deferred_s0/"
    name = prefix + "summary.json"
    if mutation in ("loss", "count"):
        value = json.loads(files[name])
        value["validation"]["loss" if mutation == "loss" else "target_tokens"] += 1
    elif mutation in ("guard", "input"):
        name = prefix + "steps.jsonl"
        rows = [json.loads(line) for line in files[name].splitlines()]
        rows[0]["logp_abs_diff_mean" if mutation == "guard" else "batch_sha256"] = (
            1 if mutation == "guard" else "0" * 64
        )
        files[name] = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    elif mutation == "capture":
        name = prefix + "validation-response-logprobs.jsonl"
        rows = [json.loads(line) for line in files[name].splitlines()]
        rows[0]["response_logprobs"][0] += 1
        files[name] = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        metadata_name = prefix + "validation-capture-metadata.json"
        metadata = json.loads(files[metadata_name])
        metadata["stream_sha256"] = hashlib.sha256(files[name]).hexdigest()
        metadata["stream_bytes"] = len(files[name])
        files[metadata_name] = json.dumps(metadata).encode()
    elif mutation == "schedule":
        name = "stage3/CASE_RESULTS.json"
        value = json.loads(files[name])[:-1]
    else:
        name = prefix + "metadata.json"
        value = json.loads(files[name])
        value["source_sha256"] = "0" * 64
    if mutation not in ("guard", "input", "capture"):
        files[name] = json.dumps(value).encode()
    with pytest.raises((AssertionError, ValueError)):
        VERIFIER.audit_bundle(manifest, files, repo=REPO)


def test_archive_identity_rejected_cpu(bundle, tmp_path):
    manifest, _ = bundle
    changed = copy.deepcopy(manifest)
    changed["parts"][0]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(changed))
    part = manifest["parts"][0]["archive"]
    (tmp_path / part).write_bytes((EVIDENCE / part).read_bytes())
    with pytest.raises(ValueError, match="archive identity"):
        VERIFIER.load_bundle(tmp_path)


def test_cli_needs_only_standard_library_cpu(tmp_path):
    output = tmp_path / "analysis.json"
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(EVIDENCE / "verify.py"),
            str(EVIDENCE),
            "--repo",
            str(REPO),
            "--json-output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text())["status"] == "passed"


def test_optimized_python_cannot_disable_audit_cpu():
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-O", str(EVIDENCE / "verify.py"), str(EVIDENCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "do not use -O" in process.stderr
