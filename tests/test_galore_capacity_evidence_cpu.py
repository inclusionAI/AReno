"""Portable capacity evidence contracts; no runtime, CUDA or archived code import."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/reports/galore-capacity-evidence"
SPEC = importlib.util.spec_from_file_location("capacity_evidence_auditor", EVIDENCE / "verify.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture(scope="module")
def bundle():
    return VERIFIER.load_bundle(EVIDENCE)


def json_file(files, name):
    return json.loads(files[name])


def changed_json(files, name, value):
    changed = dict(files)
    changed[name] = (json.dumps(value, allow_nan=False) + "\n").encode()
    return changed


def boundary_record(files, method="deferred"):
    return next(
        record
        for record in json_file(files, "repair/CASE_RESULTS.json")
        if record["stage"] == "boundary" and record["method"] == method
    )


def case_path(record, name):
    # Boundary cases are fresh repair evidence; never resolve original host paths.
    return f"repair/{record['output_dir']}/{name}"


def archive_bytes(members, *, compressed=False):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:xz" if compressed else "w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return result.getvalue()


def test_real_bundle_recomputes_counts_gates_and_same_card_ranges_cpu(bundle):
    manifest, files = bundle
    result = VERIFIER.audit_bundle(manifest, files, REPO)
    assert result["status"] == "passed"
    assert (result["valid_cases"], result["complete_cases"], result["typed_cuda_oom_cases"]) == (69, 47, 22)
    assert (result["reused_cases"], result["step_rows"], result["monitor_samples"]) == (49, 4128, 26448)
    assert result["current_repo_checked"] is True
    assert result["token_payload_replayed_offline"] is False
    assert result["primary_adamw_capacity_gate"]["passed"] is True
    assert result["stronger_8bit_capacity_gate"]["passed"] is False
    for axis in ("batch", "context"):
        assert result["primary_adamw_capacity_gate"]["axes"][axis]["confirmed_grid_ratio"] == 2
        assert result["stronger_8bit_capacity_gate"]["axes"][axis]["confirmed_grid_ratio"] == 1
    assert sum(case["rows"] for case in result["excluded_interrupted_cases"]) == 370
    assert len(result["boundary_comparisons"]) == 6
    for comparison in result["boundary_comparisons"]:
        assert 1.89 < comparison["vs8bit_memory_reduction_percent"] < 1.93
        assert 0.57 < comparison["vs8bit_wall_overhead_percent"] < 1.89


def test_cli_runs_with_only_python_standard_library_cpu(tmp_path):
    output = tmp_path / "audit.json"
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
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "passed" and result["current_repo_checked"] is True
    assert result["token_payload_replayed_offline"] is False


def test_package_omits_training_and_tensor_payloads_and_bounds_files_cpu(bundle):
    _, files = bundle
    assert all(path.stat().st_size < 1_000_000 for path in EVIDENCE.iterdir() if path.is_file())
    assert all(not name.endswith((".u32", ".pt", ".bin", ".so", ".safetensors", ".py")) for name in files)
    assert all(not name.endswith(("train.jsonl", "validation.jsonl", "test.jsonl", "env.sh")) for name in files)
    # Harnesses remain inert text. Source hashes attest identity; they are never imported.
    assert "screen/harness/train.py.txt" in files and "repair/harness/repair.py.txt" in files


@pytest.mark.parametrize("mutation", ["runtime_error", "missing_trace", "wrong_stage", "embedded_message"])
def test_actual_oom_record_requires_specific_native_type_and_trace_cpu(bundle, mutation):
    _, files = bundle
    failure = copy.deepcopy(json_file(files, case_path(boundary_record(files, "adamw"), "failure.json")))
    assert VERIFIER.typed_oom(failure)
    native = failure.get("worker_failure") or failure
    if mutation == "runtime_error":
        native["exception_type"] = "RuntimeError"
    elif mutation == "missing_trace":
        native["traceback"] = ""
    elif mutation == "wrong_stage":
        native["stage"] = "disk_admission"
    else:
        native["exception"] = "worker reported: " + native["exception"]
    assert not VERIFIER.typed_oom(failure)


def test_untyped_oom_text_cannot_establish_capacity_boundary_cpu(bundle):
    manifest, files = bundle
    name = case_path(boundary_record(files, "adamw"), "failure.json")
    failure = json_file(files, name)
    failure["exception_type"] = "RuntimeError"
    failure["worker_failure"]["exception_type"] = "RuntimeError"
    with pytest.raises(ValueError, match="not actual typed CUDA OOM"):
        VERIFIER.audit_bundle(manifest, changed_json(files, name, failure))


def test_one_missing_boundary_card_prevents_completed_gate_cpu(bundle):
    manifest, files = bundle
    records = json_file(files, "repair/CASE_RESULTS.json")
    selected = boundary_record(files)
    records = [record for record in records if record["case_id"] != selected["case_id"]]
    with pytest.raises(ValueError, match="69 unique valid cases required"):
        VERIFIER.audit_bundle(manifest, changed_json(files, "repair/CASE_RESULTS.json", records))


def test_refresh200_cannot_be_replaced_with_ordinary_update_cpu(bundle):
    manifest, files = bundle
    name = case_path(boundary_record(files), "steps.jsonl")
    rows = [json.loads(line) for line in files[name].splitlines()]
    assert len(rows) == 203 and rows[200]["update_refresh_any"] == 1
    rows[200]["update_refresh_any"] = 0
    changed = dict(files)
    changed[name] = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    with pytest.raises(ValueError, match="native lifecycle/normalization mismatch: update_refresh_any"):
        VERIFIER.audit_bundle(manifest, changed)


def test_valid_hex_input_digest_must_match_paired_methods_cpu(bundle):
    manifest, files = bundle
    name = case_path(boundary_record(files), "steps.jsonl")
    rows = [json.loads(line) for line in files[name].splitlines()]
    rows[50]["batch_sha256"] = "0" * 64
    changed = dict(files)
    changed[name] = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    with pytest.raises(ValueError, match="paired batch identities differ"):
        VERIFIER.audit_bundle(manifest, changed)


def test_all49_reused_records_preserve_even_nonmeasurement_fields_cpu(bundle):
    manifest, files = bundle
    records = json_file(files, "repair/CASE_RESULTS.json")
    reused = next(record for record in records if "evidence_root" in record)
    reused["timestamp"] = "changed after evidence reuse"
    with pytest.raises(ValueError, match="reused record was changed"):
        VERIFIER.audit_bundle(manifest, changed_json(files, "repair/CASE_RESULTS.json", records))


def test_original_timeout_attempt_cannot_be_relabeled_as_cuda_oom_cpu(bundle):
    manifest, files = bundle
    original = json_file(files, "screen/CASE_RESULTS.json")
    interrupted = next(record for record in original if record["outcome"] == "error")
    interrupted["outcome"] = "cuda_oom"
    with pytest.raises(ValueError, match="original49/2 separation mismatch"):
        VERIFIER.audit_bundle(manifest, changed_json(files, "screen/CASE_RESULTS.json", original))


def test_wrong_current_repository_cannot_claim_measured_source_cpu(bundle, tmp_path):
    manifest, files = bundle
    (tmp_path / "areno").mkdir()
    (tmp_path / "areno/__init__.py").write_text("# Unrelated source tree\n")
    with pytest.raises(ValueError, match="current repository runtime differs"):
        VERIFIER.audit_bundle(manifest, files, tmp_path)


def test_archive_bit_corruption_is_rejected_before_evidence_audit_cpu(tmp_path):
    copy_dir = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copy_dir)
    manifest = json.loads((copy_dir / "manifest.json").read_text())
    archive = copy_dir / manifest["parts"][0]["archive"]
    damaged = bytearray(archive.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    archive.write_bytes(damaged)
    with pytest.raises(ValueError, match="archive identity mismatch"):
        VERIFIER.load_bundle(copy_dir)


def test_changed_chunk_is_rejected_even_with_rehashed_outer_archive_cpu(tmp_path):
    copy_dir = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copy_dir)
    manifest_path = copy_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    part = manifest["parts"][0]
    path = copy_dir / part["archive"]
    with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode="r:xz") as archive:
        members = [(member.name, archive.extractfile(member).read()) for member in archive]
    name, original = members[0]
    altered = bytearray(original)
    altered[-1] ^= 1
    members[0] = (name, bytes(altered))
    raw = archive_bytes(members, compressed=True)
    path.write_bytes(raw)
    part.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="chunk identity mismatch"):
        VERIFIER.load_bundle(copy_dir)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "dir\\escape"])
def test_archive_paths_cannot_escape_without_extracting_cpu(name):
    with pytest.raises(ValueError, match="unsafe archive path"):
        VERIFIER.read_tar(archive_bytes([(name, b"untrusted")]))


def test_duplicate_members_and_symlinks_are_rejected_cpu():
    with pytest.raises(ValueError, match="nonregular or duplicate archive member"):
        VERIFIER.read_tar(archive_bytes([("duplicate", b"first"), ("duplicate", b"second")]))
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "outside"
        archive.addfile(info)
    with pytest.raises(ValueError, match="nonregular or duplicate archive member"):
        VERIFIER.read_tar(stream.getvalue())
