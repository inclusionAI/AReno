"""CPU tests for the shared skill SDK (issue #275).

Covers the SDK's core logic, malformed input, boundary values, deterministic
output, and backward compatibility of migrated scripts. All tests run on CPU
without GPU or network. Mirrors the subprocess-based pattern in
``test_agent_skills_cpu.py`` for the migrated-script regression tests.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SDK_PATH = str(ROOT / ".agents/scripts")

# Make the SDK importable as `areno_skill_sdk`.
sys.path.insert(0, SDK_PATH)

import areno_skill_sdk as sdk  # noqa: E402
from areno_skill_sdk import (  # noqa: E402
    JsonLinesSink,
    ProgressEvent,
    Result,
    SkillError,
    build_parser,
    emit,
    envelope,
    exit_code,
    skill_main,
    validate_positive,
)


# ---------------------------------------------------------------------------
# Result objects + exit codes
# ---------------------------------------------------------------------------


def test_result_success_dict_has_only_ok_and_data():
    result = Result(ok=True, data={"rollout_demand": 64, "waves": 8}).to_dict()
    assert result == {"ok": True, "rollout_demand": 64, "waves": 8}


def test_result_failure_with_errors_list_preserves_field_name():
    # Option A: multi-error scripts keep the `errors` list field name.
    result = Result(ok=False, errors=["batch_size must be positive"]).to_dict()
    assert result == {"ok": False, "errors": ["batch_size must be positive"]}
    assert "error" not in result  # distinct from single-error `error`


def test_result_stage_only_present_when_set():
    assert "stage" not in Result(ok=True, data={"x": 1}).to_dict()
    out = Result(ok=False, stage="validate").to_dict()
    assert out["stage"] == "validate"


def test_exit_code_semantics_match_legacy():
    assert exit_code({"ok": True}) == 0
    assert exit_code({"ok": False}) == 1
    assert exit_code({}) == 1  # missing ok treated as failure


# ---------------------------------------------------------------------------
# Exception envelopes
# ---------------------------------------------------------------------------


def test_skill_error_carries_stage():
    err = SkillError("bad value", stage="validate")
    assert err.stage == "validate"
    assert str(err) == "bad value"


def test_skill_error_default_stage_is_execute():
    assert SkillError("oops").stage == "execute"


def test_envelope_format_matches_legacy_convention():
    env = envelope(ValueError("nope"), stage="load")
    assert env == {"ok": False, "error": "ValueError: nope", "stage": "load"}


def test_skill_main_wraps_skill_error_with_stage(capsys):
    @skill_main
    def main():
        raise SkillError("count must be positive", stage="validate")

    code = main()
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out == {"ok": False, "error": "count must be positive", "stage": "validate"}


def test_skill_main_wraps_unexpected_exception(capsys):
    @skill_main
    def main():
        raise RuntimeError("boom")

    code = main()
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out == {"ok": False, "error": "RuntimeError: boom", "stage": "execute"}


def test_skill_main_emits_and_returns_zero_on_success(capsys):
    @skill_main
    def main():
        return Result(ok=True, data={"count": 3})

    code = main()
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out == {"ok": True, "count": 3}


def test_skill_main_accepts_plain_dict(capsys):
    @skill_main
    def main():
        return {"ok": True, "value": 7}

    assert main() == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "value": 7}


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_build_parser_disables_abbrev():
    parser = build_parser("test")
    parser.add_argument("--batch-size", type=int, required=True)
    # `--batch` must NOT be accepted as an abbreviation.
    with pytest.raises(SystemExit):
        parser.parse_args(["--batch", "4"])


def test_build_parser_preserves_required_flags():
    parser = build_parser("test")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args(["--count", "5"])
    assert args.count == 5


def test_validate_positive_passes_for_positive_values():
    parser = build_parser("test")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args(["--count", "5"])
    validate_positive(args)  # should not raise


def test_validate_positive_raises_for_zero_boundary():
    parser = build_parser("test")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args(["--count", "0"])
    with pytest.raises(SkillError) as exc_info:
        validate_positive(args)
    assert exc_info.value.stage == "validate"
    assert "count must be positive" in str(exc_info.value)


def test_validate_positive_raises_for_negative():
    parser = build_parser("test")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args(["--count", "-3"])
    with pytest.raises(SkillError):
        validate_positive(args)


def test_validate_positive_excludes_named_keys():
    parser = build_parser("test")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    args = parser.parse_args(["--count", "4", "--memory-fraction", "0.0"])
    # memory_fraction is excluded, so 0.0 must not trigger an error.
    validate_positive(args, exclude=("memory_fraction",))


# ---------------------------------------------------------------------------
# Rendering: JSON mode stdout-clean, human mode to stderr
# ---------------------------------------------------------------------------


def test_emit_json_mode_writes_only_json_to_stdout(capsys):
    emit({"ok": True, "count": 3}, json_mode=True)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ok": True, "count": 3}
    assert captured.err == ""  # stderr clean in JSON mode


def test_emit_json_mode_sorts_keys(capsys):
    emit({"ok": True, "z": 1, "a": 2}, json_mode=True)
    out = capsys.readouterr().out
    # `a` must appear before `z` in the sorted output.
    assert out.index('"a"') < out.index('"z"')


def test_emit_human_mode_keeps_stdout_clean(capsys):
    emit({"ok": True, "count": 3}, json_mode=False)
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays machine-clean
    assert "OK" in captured.err  # human text on stderr


def test_emit_human_mode_failure_to_stderr(capsys):
    emit({"ok": False, "errors": ["bad"]}, json_mode=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "FAIL" in captured.err


def test_emit_to_custom_stream():
    buf = io.StringIO()
    emit({"ok": True}, json_mode=True, stream=buf)
    assert json.loads(buf.getvalue()) == {"ok": True}


def test_emit_ensure_ascii_false_preserves_non_ascii(capsys):
    emit({"ok": True, "text": "中文"}, json_mode=True, ensure_ascii=False)
    out = capsys.readouterr().out
    assert "中文" in out  # not escaped as \uXXXX
    assert json.loads(out) == {"ok": True, "text": "中文"}


def test_emit_sort_keys_false_preserves_insertion_order(capsys):
    emit({"ok": True, "z": 1, "a": 2}, json_mode=True, sort_keys=False)
    out = capsys.readouterr().out
    # Insertion order: ok, z, a — z must appear before a.
    assert out.index('"z"') < out.index('"a"')


# ---------------------------------------------------------------------------
# Progress: deterministic JSONL sink
# ---------------------------------------------------------------------------


def test_progress_jsonl_sink_emits_one_json_per_line():
    buf = io.StringIO()
    sink = JsonLinesSink(buf)
    sink.emit(ProgressEvent(stage="load", fraction=0.0, message="start"))
    sink.emit(ProgressEvent(stage="load", fraction=1.0, message="done"))
    sink.close()

    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "type": "progress",
        "stage": "load",
        "fraction": 0.0,
        "message": "start",
    }
    assert json.loads(lines[1])["fraction"] == 1.0


def test_progress_event_fields():
    event = ProgressEvent(stage="train", fraction=0.5)
    assert event.stage == "train"
    assert event.fraction == 0.5
    assert event.message == ""  # default


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface_exports_all_symbols():
    expected = {
        "skill_main",
        "build_parser",
        "validate_positive",
        "Result",
        "exit_code",
        "emit",
        "SkillError",
        "envelope",
        "ProgressEvent",
        "ProgressSink",
        "JsonLinesSink",
    }
    assert expected <= set(dir(sdk))
    assert expected <= set(sdk.__all__)


# ---------------------------------------------------------------------------
# Migrated-script regression tests (subprocess, like test_agent_skills_cpu.py)
# ---------------------------------------------------------------------------


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


CHECK_CAPACITY = ROOT / ".agents/skills/areno-tune-capacity/scripts/check_capacity.py"


def test_check_capacity_migrated_success():
    proc = _run(
        CHECK_CAPACITY,
        "--batch-size", "4", "--n-samples", "16", "--max-running-prompts", "8",
        "--mini-bs", "2", "--world-size", "4", "--tp-size", "2",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["ok"] is True
    assert result["rollout_demand"] == 64
    assert result["minimum_admission_waves"] == 8
    assert result["data_parallel_size"] == 2


def test_check_capacity_migrated_invalid_negative():
    proc = _run(
        CHECK_CAPACITY,
        "--batch-size", "-1", "--n-samples", "16", "--max-running-prompts", "8",
        "--mini-bs", "2", "--world-size", "4", "--tp-size", "2",
    )
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["ok"] is False
    assert "batch_size must be positive" in result["errors"]


def test_check_capacity_migrated_invalid_divisibility():
    proc = _run(
        CHECK_CAPACITY,
        "--batch-size", "4", "--n-samples", "16", "--max-running-prompts", "8",
        "--mini-bs", "2", "--world-size", "4", "--tp-size", "3",
    )
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["ok"] is False
    assert "world_size must be divisible by tp_size" in result["errors"]


def test_check_capacity_backward_compat_flags_unchanged():
    """Old flags must work identically: same JSON shape, same exit codes."""
    success = _run(
        CHECK_CAPACITY,
        "--batch-size", "2", "--n-samples", "4", "--max-running-prompts", "4",
        "--mini-bs", "1", "--world-size", "2", "--tp-size", "1",
    )
    assert success.returncode == 0
    payload = json.loads(success.stdout)
    # Same top-level fields as the pre-migration script.
    assert set(payload) == {
        "ok", "errors", "rollout_demand", "minimum_admission_waves",
        "data_parallel_size", "settings",
    }
    assert payload["settings"]["memory_fraction"] == 0.9  # default preserved


def test_check_capacity_help_still_works():
    proc = _run(CHECK_CAPACITY, "--help")
    assert proc.returncode == 0
    assert "--batch-size" in proc.stdout


# ---------------------------------------------------------------------------
# P2/P3 migrated scripts: build_parser preserves flags + validation behavior
# ---------------------------------------------------------------------------

MONITOR_GPU = ROOT / ".agents/skills/areno-profile-performance/scripts/monitor_gpu.py"
COMPARE_CKPT = ROOT / ".agents/skills/areno-model-adaptation/scripts/compare_ckpt_diff.py"


def test_monitor_gpu_help_preserves_flags():
    proc = _run(MONITOR_GPU, "--help")
    assert proc.returncode == 0
    for flag in ("--pid", "--duration", "--interval", "--output", "--no-children"):
        assert flag in proc.stdout
    # The description is now wired through build_parser.
    assert "Sample NVIDIA GPU" in proc.stdout


def test_monitor_gpu_invalid_duration_uses_parser_error_exit_code():
    """Pre-migration behavior: parser.error -> exit 2, message on stderr."""
    proc = _run(MONITOR_GPU, "--duration", "-1")
    assert proc.returncode == 2
    assert "must be positive" in proc.stderr


def test_compare_ckpt_diff_help_preserves_positional_and_flags():
    proc = _run(COMPARE_CKPT, "--help")
    assert proc.returncode == 0
    for token in ("base", "other", "--top-k", "--pattern", "--device", "--max-elements"):
        assert token in proc.stdout
    assert "Compare same-name tensors" in proc.stdout


# ---------------------------------------------------------------------------
# Additional migrated-script regression tests
# ---------------------------------------------------------------------------

VALIDATE_TRANSCRIPT = ROOT / ".agents/skills/areno-build-agentic-workflow/scripts/validate_transcript.py"
INSPECT_DATASET = ROOT / ".agents/skills/areno-run-training/scripts/inspect_dataset.py"


def test_validate_transcript_migrated_rejects_unmatched_tool_result(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "guess_code", "arguments": {"code": "0123"}},
                            }
                        ],
                    },
                    # tool result with a DIFFERENT call id -> unmatched
                    {"role": "tool", "tool_call_id": "call-999", "content": "{}"},
                ]
            }
        ),
        encoding="utf-8",
    )
    proc = _run(VALIDATE_TRANSCRIPT, str(transcript))
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["ok"] is False
    assert any("unmatched tool result" in e for e in result["errors"])


def test_validate_transcript_migrated_success(tmp_path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "guess_code", "arguments": {"code": "0123"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": '{"solved":true}'},
                ]
            }
        ),
        encoding="utf-8",
    )
    proc = _run(VALIDATE_TRANSCRIPT, str(transcript))
    assert proc.returncode == 0
    result = json.loads(proc.stdout)
    assert result["ok"] is True
    assert result["tool_calls"] == 1
    assert result["errors"] == []


def test_inspect_dataset_migrated_preserves_non_ascii_and_unsorted(tmp_path):
    """inspect_dataset must keep ensure_ascii=False and unsorted keys."""
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "解释这段代码", "response": "这是一个函数"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    proc = _run(INSPECT_DATASET, "--dataset-path", str(dataset), "--algo", "sft", "--model-hub", "hf")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Non-ASCII content must be preserved verbatim, not escaped.
    assert "解释这段代码" in proc.stdout
    assert "\\u" not in proc.stdout
    result = json.loads(proc.stdout)
    assert result["ok"] is True
    # Insertion order preserved (ok before count before sample_keys ...).
    assert list(result.keys())[0] == "ok"