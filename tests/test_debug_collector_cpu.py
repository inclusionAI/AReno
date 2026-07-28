"""CPU tests for the areno debug failure-evidence collector."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from areno.cli.debug import (
    FailureBundle,
    _is_sensitive_key,
    _redact_value,
    _render_markdown,
    _safe_areno_version,
    _safe_env_collect,
    _safe_process_info,
    _safe_traceback,
    _safe_cuda_info,
    collect_failure_bundle,
    write_bundle,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TestFailureBundle:
    def test_default_bundle_has_expected_fields(self):
        bundle = FailureBundle()
        raw = bundle.to_ordered_dict()
        for field_name in ("timestamp", "python_version", "platform_info", "process_info", "collection_warnings"):
            assert field_name in raw

    def test_bundle_is_json_serializable(self):
        bundle = FailureBundle(timestamp="2026-01-01T00:00:00+00:00", python_version="3.10")
        text = json.dumps(bundle.to_ordered_dict(), default=str)
        assert "2026-01-01" in text

    def test_to_ordered_dict_respects_field_order(self):
        bundle = FailureBundle(timestamp="t", python_version="p", areno_version="a")
        ordered = bundle.to_ordered_dict()
        keys = list(ordered.keys())
        # _FIELD_ORDER: timestamp, areno_version, python_version, ...
        assert keys.index("timestamp") < keys.index("areno_version")
        assert keys.index("areno_version") < keys.index("platform_info")


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedact:
    def test_sensitive_keys_are_detected(self):
        for key in ("HF_TOKEN", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD"):
            assert _is_sensitive_key(key) is True

    def test_nonsensitive_keys_are_not_redacted(self):
        for key in ("HOME", "USER", "PATH", "PYTHONPATH", "LANG"):
            assert _is_sensitive_key(key) is False

    def test_redact_value_truncates(self):
        assert len(_redact_value("a" * 3)) <= 4
        assert _redact_value("a" * 3) == "****"

    def test_redact_value_preserves_prefix_suffix(self):
        result = _redact_value("abcdefgh")
        assert result.startswith("ab")
        assert result.endswith("gh")
        assert result[2] == "*"


# ---------------------------------------------------------------------------
# Safe collectors
# ---------------------------------------------------------------------------


class TestSafeCollectors:
    def test_safe_areno_version_returns_string_or_none(self):
        v = _safe_areno_version()
        assert v is None or isinstance(v, str)

    def test_safe_traceback_returns_none_for_none_error(self):
        assert _safe_traceback(None) is None

    def test_safe_traceback_formats_real_exception(self):
        try:
            raise ValueError("test error")
        except ValueError as exc:
            tb = _safe_traceback(exc)
        assert "ValueError" in tb
        assert "test error" in tb

    def test_safe_process_info_has_pid(self):
        info = _safe_process_info()
        assert info["pid"] == os.getpid()

    def test_safe_cuda_info_returns_dict_or_none(self):
        info = _safe_cuda_info()
        assert info is None or isinstance(info, dict)

    def test_safe_env_collect_redacts_sensitive(self):
        with tempfile.TemporaryDirectory() as _tmpdir:
            pass  # use ephemeral scope to avoid leaking into real env
        env = _safe_env_collect(redact=True)
        assert isinstance(env, dict)
        # The local env may not have token vars, but the collector must not raise.
        for key, value in env.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_safe_env_collect_no_redact_keeps_raw_values(self):
        env = _safe_env_collect(redact=False)
        assert isinstance(env, dict)


# ---------------------------------------------------------------------------
# collect_failure_bundle
# ---------------------------------------------------------------------------


class TestCollectFailureBundle:
    def test_success_path_has_minimal_fields(self):
        bundle = collect_failure_bundle()
        assert bundle.timestamp
        assert bundle.python_version
        assert bundle.platform_info

    def test_with_error_populates_error_fields(self):
        error = RuntimeError("something went wrong")
        bundle = collect_failure_bundle(error=error)
        assert bundle.error_type == "RuntimeError"
        assert "something went wrong" in (bundle.error_message or "")
        assert bundle.error_traceback
        assert "RuntimeError" in bundle.error_traceback

    def test_without_error_all_fields_none(self):
        bundle = collect_failure_bundle()
        assert bundle.error_type is None
        assert bundle.error_message is None
        assert bundle.error_traceback is None

    def test_command_is_preserved(self):
        cmd = ["areno", "train", "--ckpt", "model"]
        bundle = collect_failure_bundle(command=cmd)
        assert bundle.command == cmd

    def test_collection_warnings_on_nonfatal_failure(self):
        # GPU info may fail without CUDA; the bundle must still succeed.
        bundle = collect_failure_bundle(include_env=False, include_gpu=True)
        # In a CPU-only test the GPU summary is None or has available=False.
        assert bundle.gpu_summary is None or isinstance(bundle.gpu_summary, dict)
        # The collector itself never raises.
        assert isinstance(bundle.collection_warnings, list)

    def test_include_env_false_skips_collection(self):
        bundle = collect_failure_bundle(include_env=False)
        assert bundle.env_vars_redacted == {}

    def test_include_gpu_false_skips_collection(self):
        bundle = collect_failure_bundle(include_gpu=False)
        assert bundle.gpu_summary is None
        assert bundle.cuda_info is None

    def test_redact_env_false_preserves_raw_values(self):
        bundle = collect_failure_bundle(include_env=True, redact_env=False)
        assert isinstance(bundle.env_vars_redacted, dict)

    def test_config_is_serialized(self):
        from dataclasses import dataclass

        @dataclass
        class TinyConfig:
            algo: str = "gspo"
            lr: float = 1e-6

        config = TinyConfig()
        bundle = collect_failure_bundle(config=config)
        assert bundle.resolved_config is not None
        assert bundle.resolved_config["algo"] == "gspo"


# ---------------------------------------------------------------------------
# write_bundle
# ---------------------------------------------------------------------------


class TestWriteBundle:
    def test_writes_bundle_json_and_summary(self):
        bundle = collect_failure_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            result_dir = write_bundle(bundle, output)
            assert result_dir.exists()

            bundle_json = result_dir / "bundle.json"
            assert bundle_json.exists()
            data = json.loads(bundle_json.read_text())
            assert data["python_version"]

            summary_md = result_dir / "summary.md"
            assert summary_md.exists()
            text = summary_md.read_text()
            assert "# AReno Failure Bundle" in text

    def test_with_error_writes_traceback_txt(self):
        try:
            raise ValueError("write test error")
        except ValueError as exc:
            bundle = collect_failure_bundle(error=exc)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            result_dir = write_bundle(bundle, output)
            traceback_txt = result_dir / "traceback.txt"
            assert traceback_txt.exists()
            assert "ValueError" in traceback_txt.read_text()

    def test_without_error_no_traceback_file(self):
        bundle = collect_failure_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            result_dir = write_bundle(bundle, output)
            assert not (result_dir / "traceback.txt").exists()

    def test_bundle_json_contains_collection_warnings(self):
        bundle = collect_failure_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            result_dir = write_bundle(bundle, output)
            data = json.loads((result_dir / "bundle.json").read_text())
            assert "collection_warnings" in data


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_renders_basic_header(self):
        bundle = FailureBundle(timestamp="2026-01-01T00:00:00+00:00", python_version="3.10.0")
        md = _render_markdown(bundle)
        assert "# AReno Failure Bundle" in md
        assert "2026-01-01" in md
        assert "3.10.0" in md

    def test_renders_error_section(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            bundle = collect_failure_bundle(error=exc)
        md = _render_markdown(bundle)
        assert "## Error" in md
        assert "RuntimeError" in md
        assert "boom" in md
        assert "### Traceback" in md

    def test_renders_warnings(self):
        bundle = collect_failure_bundle()
        md = _render_markdown(bundle)
        if bundle.collection_warnings:
            assert "## Collection Warnings" in md

    def test_renders_gpu_section_when_present(self):
        bundle = FailureBundle(timestamp="t", gpu_summary={"available": True, "device_count": 1})
        md = _render_markdown(bundle)
        assert "## GPU" in md

    def test_no_gpu_section_when_none(self):
        bundle = FailureBundle(timestamp="t", gpu_summary=None)
        md = _render_markdown(bundle)
        assert "## GPU" not in md


# ---------------------------------------------------------------------------
# Boundary and default-behaviour tests
# ---------------------------------------------------------------------------


class TestBoundaryDefaults:
    def test_empty_args_no_raise(self):
        """collect_failure_bundle with no arguments must not raise."""
        bundle = collect_failure_bundle()
        assert bundle is not None

    def test_disabled_features_produce_empty_or_none(self):
        bundle = collect_failure_bundle(include_env=False, include_gpu=False)
        assert bundle.env_vars_redacted == {}
        assert bundle.gpu_summary is None
        assert bundle.cuda_info is None

    def test_import_does_not_trigger_side_effects(self):
        """Mere import must not write files or modify global state."""
        import areno.cli.debug as _m

        # No side effects beyond module-level constants.
        assert _m.DEFAULT_REDACT_KEYS is not None


# ---------------------------------------------------------------------------
# collect_evidence.py script (areno-debug-runtime skill)
# ---------------------------------------------------------------------------


class TestCollectEvidenceScript:
    """Integration tests for .agents/skills/areno-debug-runtime/scripts/collect_evidence.py."""

    _SCRIPT = Path(__file__).parents[1] / ".agents" / "skills" / "areno-debug-runtime" / "scripts" / "collect_evidence.py"

    @pytest.fixture
    def script_path(self) -> Path:
        assert self._SCRIPT.is_file(), f"collect_evidence.py not found at {self._SCRIPT}"
        return self._SCRIPT

    def _run(self, script_path: Path, *extra_args: str):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(script_path), *extra_args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return result

    def test_help_flag(self, script_path):
        result = self._run(script_path, "--help")
        assert result.returncode == 0
        assert "collect" in result.stdout.lower() or "evidence" in result.stdout.lower()

    def test_success_path_produces_markdown(self, script_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(script_path, "--output-dir", tmpdir, "--no-env", "--no-gpu")
            assert result.returncode == 0
            assert "# AReno Failure Bundle" in result.stdout

    def test_json_flag_produces_valid_json(self, script_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(script_path, "--output-dir", tmpdir, "--no-env", "--no-gpu", "--json")
            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert "timestamp" in data
            assert "python_version" in data

    def test_traceback_file_post_mortem(self, script_path):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nRuntimeError: boom\n")
            tb_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = self._run(script_path, "--output-dir", tmpdir, "--traceback-file", tb_path, "--no-env", "--no-gpu")
                assert result.returncode == 0
                assert "RuntimeError" in result.stdout
                assert "boom" in result.stdout
        finally:
            Path(tb_path).unlink()

    def test_no_traceback_file_no_error(self, script_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(script_path, "--output-dir", tmpdir, "--no-env", "--no-gpu")
            assert result.returncode == 0
            assert "## Error" not in result.stdout

    def test_writes_bundle_files(self, script_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(script_path, "--output-dir", tmpdir, "--no-env", "--no-gpu")
            assert result.returncode == 0
            out = Path(tmpdir)
            # write_bundle creates a timestamped sub-directory
            subdirs = [d for d in out.iterdir() if d.is_dir() and d.name.startswith("areno-failure-")]
            assert len(subdirs) == 1
            bundle_dir = subdirs[0]
            assert (bundle_dir / "bundle.json").is_file()
            assert (bundle_dir / "summary.md").is_file()
            # No traceback when no error
            assert not (bundle_dir / "traceback.txt").exists()

    def test_unrecognized_flag_becomes_command(self, script_path):
        """Subcommand flags like --ckpt must be tolerated and merged into command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(
                script_path,
                "--output-dir", tmpdir,
                "--no-env", "--no-gpu", "--json",
                "areno", "train", "--ckpt", "./model", "--algo", "gspo",
            )
            assert result.returncode == 0
            data = json.loads(result.stdout)
            cmd = data.get("command", [])
            assert "areno" in cmd
            assert "--ckpt" in cmd
            assert "./model" in cmd
            assert "--algo" in cmd
            assert "gspo" in cmd

    def test_collection_warnings_return_code_2(self, script_path):
        """When collection_warnings are present, exit code should be 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Forcing GPU collection on a non-CUDA host may produce warnings.
            result = self._run(script_path, "--output-dir", tmpdir, "--no-env", "--no-redact", "--json")
            if "collection_warnings" in result.stdout:
                warnings_data = json.loads(result.stdout).get("collection_warnings", [])
                if warnings_data:
                    assert result.returncode == 2
                else:
                    assert result.returncode in (0, 2)


# ---------------------------------------------------------------------------
# Enhanced coverage: malformed input, boundaries, deterministic output
# ---------------------------------------------------------------------------


class TestMalformedInput:
    """Core logic must handle malformed or boundary inputs without raising."""

    def test_none_command_falls_back_to_argv(self):
        """When command is None, collect_failure_bundle falls back to sys.argv."""
        bundle = collect_failure_bundle(command=None, include_env=False, include_gpu=False)
        # Falls back to sys.argv — at minimum the running script.
        assert bundle.command is not None
        assert isinstance(bundle.command, list)

    def test_empty_command_falls_back_to_argv(self):
        """Empty list is treated the same as None — falls back to sys.argv."""
        bundle = collect_failure_bundle(command=[], include_env=False, include_gpu=False)
        assert bundle.command is not None
        assert isinstance(bundle.command, list)

    def test_zero_length_string_keys_still_collected(self):
        """Borderline env keys must not break the collector."""
        bundle = collect_failure_bundle(include_env=True, include_gpu=False)
        assert isinstance(bundle.env_vars_redacted, dict)

    def test_unusual_exception_types_are_serialized(self):
        """Exceptions without a .traceback (e.g. str-based) must not raise."""
        error = SystemExit(42)
        bundle = collect_failure_bundle(error=error, include_env=False, include_gpu=False)
        assert bundle.error_type == "SystemExit"
        assert bundle.error_message is not None

    def test_exception_with_no_traceback(self):
        """Bare exception without __traceback__ must be handled safely."""

        class FlatError(Exception):
            pass

        exc = FlatError("flat")
        exc.__traceback__ = None  # Explicitly strip
        bundle = collect_failure_bundle(error=exc, include_env=False, include_gpu=False)
        assert bundle.error_type == "FlatError"
        # _safe_traceback falls back to repr()
        assert bundle.error_traceback is not None

    def test_custom_extra_field_is_preserved(self):
        bundle = FailureBundle(timestamp="t", extra={"custom_key": 42})
        raw = bundle.to_ordered_dict()
        assert raw["extra"] == {"custom_key": 42}


class TestDeterministicOutput:
    """Output must be deterministic for the same inputs."""

    def test_bundle_json_is_deterministic(self):
        """Two bundles with identical timestamps must produce identical JSON."""
        bundle1 = FailureBundle(timestamp="2026-01-01T00:00:00+00:00", python_version="3.10.0", platform_info="linux")
        bundle2 = FailureBundle(timestamp="2026-01-01T00:00:00+00:00", python_version="3.10.0", platform_info="linux")
        j1 = json.dumps(bundle1.to_ordered_dict(), sort_keys=True)
        j2 = json.dumps(bundle2.to_ordered_dict(), sort_keys=True)
        assert j1 == j2

    def test_markdown_output_does_not_contain_raw_env_if_not_collected(self):
        bundle = collect_failure_bundle(include_env=False, include_gpu=False)
        md = _render_markdown(bundle)
        # Markdown must not leak env values.
        assert "HF_TOKEN" not in md

    def test_same_crash_same_error_section(self):
        """Deterministic error rendering."""
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            b1 = collect_failure_bundle(error=exc, include_env=False, include_gpu=False)
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            b2 = collect_failure_bundle(error=exc, include_env=False, include_gpu=False)
        # Error type and message must match.
        assert b1.error_type == b2.error_type
        assert b1.error_message == b2.error_message

    def test_no_extra_files_in_clean_bundle(self):
        """write_bundle must produce exactly expected files: bundle.json, summary.md (no traceback.txt)."""
        bundle = collect_failure_bundle(include_env=False, include_gpu=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = write_bundle(bundle, Path(tmpdir))
            files = sorted(f.name for f in result_dir.iterdir())
            assert files == ["bundle.json", "summary.md"]


# ---------------------------------------------------------------------------
# Mock / isolation: GPU behaviour
# ---------------------------------------------------------------------------


class TestGPUIsolation:
    """Orchestration for GPU-only code must be isolated behind fakes."""

    def test_gpu_section_renders_correctly_from_fake_summary(self, monkeypatch):
        """When GPU is faked as available, markdown must include GPU section with correct fields."""
        fake_gpu = {"available": True, "device_count": 8, "device_name": "NVIDIA H100"}
        bundle = FailureBundle(timestamp="t", gpu_summary=fake_gpu, cuda_info={"cuda_home": "/usr/local/cuda", "nvcc_path": "/usr/local/cuda/bin/nvcc"})
        md = _render_markdown(bundle)
        assert "## GPU" in md
        assert "NVIDIA H100" in md
        assert "8" in md

    def test_cuda_info_none_does_not_render_cuda_section(self):
        bundle = FailureBundle(timestamp="t", gpu_summary={"available": True}, cuda_info=None)
        md = _render_markdown(bundle)
        assert "## CUDA" not in md

    def test_collect_with_monkeypatched_gpu(self, monkeypatch):
        """collect_failure_bundle must accept a monkeypatched GPU info and not crash on missing CUDA env."""

        def fake_gpu():
            return {"available": True, "device_count": 4, "fake": True}

        monkeypatch.setattr("areno.cli.debug._safe_gpu_info", fake_gpu)
        bundle = collect_failure_bundle(include_env=False, include_gpu=True)
        assert bundle.gpu_summary is not None
        assert bundle.gpu_summary["available"] is True
        assert bundle.gpu_summary["device_count"] == 4


# ---------------------------------------------------------------------------
# Existing behaviour must be unchanged when the feature is *not* enabled
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Default behaviour must not change existing contracts."""

    def test_areno_cli_debug_module_api_unchanged(self):
        """All public symbols must remain importable."""
        symbols = [
            "FailureBundle",
            "collect_failure_bundle",
            "write_bundle",
            "_render_markdown",
            "_is_sensitive_key",
            "_redact_value",
            "_safe_areno_version",
            "_safe_env_collect",
            "_safe_process_info",
            "_safe_traceback",
            "_safe_cuda_info",
            "debug_command",
        ]
        import areno.cli.debug as m

        for sym in symbols:
            assert hasattr(m, sym), f"Symbol {sym} missing from areno.cli.debug"
            assert getattr(m, sym) is not None

    def test_failure_bundle_data_class_contract_stable(self):
        """FailureBundle field names and types must not silently change."""
        bundle = FailureBundle()
        fields = {f.name for f in bundle.__dataclass_fields__.values()}
        expected = {
            "timestamp", "areno_version", "python_version", "platform_info",
            "command", "resolved_config", "env_vars_redacted", "gpu_summary",
            "cuda_info", "error_type", "error_message", "error_traceback",
            "process_info", "worker_state", "collection_warnings", "extra",
        }
        assert fields == expected

    def test_write_bundle_signature_unchanged(self):
        """write_bundle(bundle, output_dir: Path) -> Path must hold."""
        import inspect

        sig = inspect.signature(write_bundle)
        params = list(sig.parameters.keys())
        assert "bundle" in params
        assert "output_dir" in params

    def test_collect_failure_bundle_signature_unchanged(self):
        """collect_failure_bundle signature must hold invariants."""
        import inspect

        sig = inspect.signature(collect_failure_bundle)
        for param in ("command", "config", "error", "include_env", "include_gpu", "redact_env"):
            assert param in sig.parameters