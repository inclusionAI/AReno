"""CPU tests for the model-adaptation scaffold generator (issue #274).

Tests cover:
- Config inference (dense, MoE, missing fields, invalid JSON).
- File generation (dense scaffold, MoE scaffold, all expected files present).
- Rerun safety (preserve user-edited files, report conflicts, --overwrite).
- No writes outside destination.
- Invalid inputs and boundary values.
- Deterministic output.
- CLI invocation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add the scripts directory to path so we can import the generator.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "areno-model-adaptation" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from generate_adapter_scaffold import (  # noqa: E402
    GenerationResult,
    _to_pascal_case,
    format_result,
    generate_scaffold,
    infer_config,
    main,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_hf_config(tmpdir: Path, config: dict) -> str:
    """Write a config.json to tmpdir and return the path."""

    path = tmpdir / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


_DENSE_CONFIG = {
    "model_type": "mydense",
    "architectures": ["MyDenseForCausalLM"],
    "hidden_size": 1024,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 3072,
    "vocab_size": 32000,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "torch_dtype": "bfloat16",
}

_MOE_CONFIG = {
    "model_type": "mymoe",
    "architectures": ["MyMoEForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 4096,
    "vocab_size": 64000,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "torch_dtype": "bfloat16",
}


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestToPascalCase(unittest.TestCase):
    """_to_pascal_case should produce correct class names."""

    def test_simple(self):
        self.assertEqual(_to_pascal_case("mymodel"), "Mymodel")

    def test_with_underscore(self):
        self.assertEqual(_to_pascal_case("my_model"), "MyModel")

    def test_with_hyphen(self):
        self.assertEqual(_to_pascal_case("my-model"), "MyModel")

    def test_empty_parts(self):
        self.assertEqual(_to_pascal_case("_my_model_"), "MyModel")


# ---------------------------------------------------------------------------
# Config inference
# ---------------------------------------------------------------------------


class TestInferConfig(unittest.TestCase):
    """infer_config should correctly parse HF configs."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dense_config(self):
        path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        config = infer_config(path, "mydense", "/tmp/mydense")
        self.assertEqual(config.model_type, "mydense")
        self.assertFalse(config.is_moe)
        self.assertEqual(config.class_name, "MydenseAdapter")
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.vocab_size, 32000)
        self.assertIsNone(config.num_experts)

    def test_moe_config(self):
        path = _write_hf_config(self.tmpdir, _MOE_CONFIG)
        config = infer_config(path, "mymoe", "/tmp/mymoe")
        self.assertEqual(config.model_type, "mymoe")
        self.assertTrue(config.is_moe)
        self.assertEqual(config.class_name, "MymoeAdapter")
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_tok, 2)

    def test_moe_detected_by_architecture_name(self):
        cfg = {
            **_DENSE_CONFIG,
            "architectures": ["MyModelMoEForCausalLM"],
            "num_experts": 8,
            "num_experts_per_tok": 2,
        }
        path = _write_hf_config(self.tmpdir, cfg)
        config = infer_config(path, "mymodel", "/tmp/mymodel")
        self.assertTrue(config.is_moe)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            infer_config("/nonexistent/config.json", "test", "/tmp/test")

    def test_missing_model_type(self):
        path = _write_hf_config(self.tmpdir, {"hidden_size": 1024})
        with self.assertRaises(ValueError):
            infer_config(path, "test", "/tmp/test")

    def test_invalid_json(self):
        path = self.tmpdir / "config.json"
        path.write_text("not json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            infer_config(str(path), "test", "/tmp/test")

    def test_non_object_json(self):
        path = self.tmpdir / "config.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(ValueError):
            infer_config(str(path), "test", "/tmp/test")

    def test_invalid_adapter_name(self):
        """adapter_name with spaces or special chars should raise."""
        path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        with self.assertRaises(ValueError):
            infer_config(path, "my model", "/tmp/test")
        with self.assertRaises(ValueError):
            infer_config(path, "my@model", "/tmp/test")

    def test_empty_adapter_name(self):
        path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        with self.assertRaises(ValueError):
            infer_config(path, "", "/tmp/test")

    def test_adapter_name_must_be_python_module(self):
        path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        with self.assertRaises(ValueError):
            infer_config(path, "123model", "/tmp/test")

    def test_hyphenated_adapter_name_is_normalized(self):
        path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        config = infer_config(path, "my-model", "/tmp/test")
        self.assertEqual(config.adapter_name, "my_model")

    def test_required_dimensions_must_be_positive_integers(self):
        for value in (None, 0, -1, True, 1.5, "1024"):
            with self.subTest(value=value):
                path = _write_hf_config(self.tmpdir, {**_DENSE_CONFIG, "hidden_size": value})
                with self.assertRaises(ValueError):
                    infer_config(path, "test", "/tmp/test")

    def test_attention_dimensions_must_be_compatible(self):
        path = _write_hf_config(self.tmpdir, {**_DENSE_CONFIG, "num_key_value_heads": 3})
        with self.assertRaises(ValueError):
            infer_config(path, "test", "/tmp/test")


# ---------------------------------------------------------------------------
# Dense scaffold generation
# ---------------------------------------------------------------------------


class TestDenseScaffold(unittest.TestCase):
    """generate_scaffold should produce correct dense adapter files."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        self.dest_dir = self.tmpdir / "my_adapter"
        self.config = infer_config(self.config_path, "mydense", str(self.dest_dir))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_all_expected_files(self):
        result = generate_scaffold(self.config)
        self.assertIn("__init__.py", result.created_files)
        self.assertIn("model.py", result.created_files)
        self.assertIn("checkpoint.py", result.created_files)
        self.assertIn("example.py", result.created_files)

    def test_no_moe_files_in_dense(self):
        result = generate_scaffold(self.config)
        self.assertNotIn("model_moe.py", result.created_files)

    def test_init_py_contains_class_name(self):
        generate_scaffold(self.config)
        content = (self.dest_dir / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("MydenseAdapter", content)
        self.assertIn("def register()", content)
        self.assertIn("register_adapter(MydenseAdapter())", content)

    def test_model_py_contains_model_type(self):
        generate_scaffold(self.config)
        content = (self.dest_dir / "model.py").read_text(encoding="utf-8")
        self.assertIn("mydense", content)
        self.assertIn("MydenseAdapter", content)

    def test_checkpoint_py_contains_placeholders(self):
        generate_scaffold(self.config)
        content = (self.dest_dir / "checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("CheckpointSpec", content)
        self.assertIn("TODO", content)

    def test_example_py_contains_adapter_name(self):
        generate_scaffold(self.config)
        content = (self.dest_dir / "example.py").read_text(encoding="utf-8")
        self.assertIn("mydense", content)

    def test_files_contain_sentinel(self):
        generate_scaffold(self.config)
        for f in self.dest_dir.glob("*.py"):
            content = f.read_text(encoding="utf-8")
            self.assertIn("GENERATED BY", content)

    def test_generated_python_files_compile(self):
        import py_compile

        generate_scaffold(self.config)
        for path in self.dest_dir.glob("*.py"):
            py_compile.compile(str(path), doraise=True)

    def test_does_not_write_outside_dest(self):
        """Generated files must only appear in the destination directory."""
        before = set()
        for root, dirs, files in os.walk(self.tmpdir):
            for fname in files:
                if fname.endswith(".py"):
                    before.add(Path(root) / fname)

        # Only dest_dir should have new files.
        generate_scaffold(self.config)
        after = set()
        for root, dirs, files in os.walk(self.tmpdir):
            for fname in files:
                if fname.endswith(".py"):
                    after.add(Path(root) / fname)

        new_files = after - before
        for f in new_files:
            self.assertTrue(str(f).startswith(str(self.dest_dir)))


# ---------------------------------------------------------------------------
# MoE scaffold generation
# ---------------------------------------------------------------------------


class TestMoEScaffold(unittest.TestCase):
    """generate_scaffold should produce correct MoE adapter files."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_path = _write_hf_config(self.tmpdir, _MOE_CONFIG)
        self.dest_dir = self.tmpdir / "my_moe_adapter"
        self.config = infer_config(self.config_path, "mymoe", str(self.dest_dir))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_moe_model_file(self):
        result = generate_scaffold(self.config)
        self.assertIn("model_moe.py", result.created_files)

    def test_no_dense_model_file_in_moe(self):
        result = generate_scaffold(self.config)
        self.assertNotIn("model.py", result.created_files)

    def test_model_moe_contains_moe_config(self):
        generate_scaffold(self.config)
        content = (self.dest_dir / "model_moe.py").read_text(encoding="utf-8")
        self.assertIn("enable_moe_block", content)
        self.assertIn("num_experts", content)

    def test_checkpoint_contains_moe_note(self):
        generate_scaffold(self.config)
        content = (self.dest_dir / "checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("MoeSpec", content)


# ---------------------------------------------------------------------------
# Rerun safety
# ---------------------------------------------------------------------------


class TestRerunSafety(unittest.TestCase):
    """Reruns should preserve user-edited files and report conflicts."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        self.dest_dir = self.tmpdir / "rerun_adapter"
        self.config = infer_config(self.config_path, "mydense", str(self.dest_dir))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rerun_preserves_unchanged_files(self):
        generate_scaffold(self.config)
        result = generate_scaffold(self.config)
        # All files already exist and are unchanged.
        self.assertEqual(len(result.created_files), 0)
        self.assertEqual(len(result.preserved_files), 4)
        self.assertEqual(len(result.conflicted_files), 0)

    def test_rerun_reports_edited_files_as_conflicts(self):
        generate_scaffold(self.config)
        # Edit a generated file.
        model_file = self.dest_dir / "model.py"
        content = model_file.read_text(encoding="utf-8")
        model_file.write_text(content + "\n# User edit\n", encoding="utf-8")

        result = generate_scaffold(self.config)
        self.assertIn("model.py", result.conflicted_files)
        self.assertNotIn("model.py", result.created_files)

    def test_overwrite_replaces_conflicted_files(self):
        generate_scaffold(self.config)
        # Edit a generated file.
        model_file = self.dest_dir / "model.py"
        content = model_file.read_text(encoding="utf-8")
        model_file.write_text(content + "\n# User edit\n", encoding="utf-8")

        result = generate_scaffold(self.config, overwrite=True)
        self.assertIn("model.py", result.created_files)
        self.assertNotIn("model.py", result.conflicted_files)

    def test_user_created_file_not_overwritten(self):
        # Pre-create a file that is NOT generated (no sentinel).
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        user_file = self.dest_dir / "model.py"
        user_file.write_text("# My custom code\n", encoding="utf-8")

        result = generate_scaffold(self.config)
        self.assertIn("model.py", result.preserved_files)
        self.assertNotIn("model.py", result.created_files)
        # Content must be unchanged.
        self.assertEqual(user_file.read_text(encoding="utf-8"), "# My custom code\n")

    def test_overwrite_does_not_follow_symlink(self):
        outside = self.tmpdir / "outside.py"
        outside.write_text("# outside\n", encoding="utf-8")
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            (self.dest_dir / "model.py").symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        result = generate_scaffold(self.config, overwrite=True)
        self.assertIn("model.py", result.conflicted_files)
        self.assertEqual(outside.read_text(encoding="utf-8"), "# outside\n")


# ---------------------------------------------------------------------------
# Format result
# ---------------------------------------------------------------------------


class TestFormatResult(unittest.TestCase):
    """format_result should produce readable output."""

    def test_shows_created_files(self):
        result = GenerationResult(created_files=["a.py", "b.py"], dest_dir="/tmp/test")
        text = format_result(result)
        self.assertIn("Created files:", text)
        self.assertIn("a.py", text)
        self.assertIn("b.py", text)

    def test_shows_conflicts(self):
        result = GenerationResult(conflicted_files=["c.py"], dest_dir="/tmp/test")
        text = format_result(result)
        self.assertIn("Conflicted", text)
        self.assertIn("c.py", text)
        self.assertIn("--overwrite", text)

    def test_shows_next_steps(self):
        result = GenerationResult(dest_dir="/tmp/test")
        text = format_result(result)
        self.assertIn("Next steps:", text)
        self.assertIn("checkpoint.py", text)


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------


class TestDeterministic(unittest.TestCase):
    """Same config should produce identical file contents."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_deterministic_output(self):
        dest1 = self.tmpdir / "dest1"
        dest2 = self.tmpdir / "dest2"
        config1 = infer_config(self.config_path, "mydense", str(dest1))
        config2 = infer_config(self.config_path, "mydense", str(dest2))
        generate_scaffold(config1)
        generate_scaffold(config2)

        for fname in ["__init__.py", "model.py", "checkpoint.py", "example.py"]:
            c1 = (dest1 / fname).read_text(encoding="utf-8")
            c2 = (dest2 / fname).read_text(encoding="utf-8")
            self.assertEqual(c1, c2)


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    """main() should work as a CLI entrypoint."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config_path = _write_hf_config(self.tmpdir, _DENSE_CONFIG)
        self.dest_dir = str(self.tmpdir / "cli_adapter")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cli_with_yes_flag(self):
        ret = main(
            [
                "--hf-config",
                self.config_path,
                "--adapter-name",
                "mydense",
                "--dest-dir",
                self.dest_dir,
                "--yes",
            ]
        )
        self.assertEqual(ret, 0)
        self.assertTrue((Path(self.dest_dir) / "model.py").exists())

    def test_cli_file_not_found(self):
        ret = main(
            [
                "--hf-config",
                "/nonexistent/config.json",
                "--adapter-name",
                "test",
                "--dest-dir",
                self.dest_dir,
                "--yes",
            ]
        )
        self.assertNotEqual(ret, 0)

    def test_cli_missing_model_type(self):
        path = _write_hf_config(self.tmpdir, {"hidden_size": 1024})
        ret = main(
            [
                "--hf-config",
                path,
                "--adapter-name",
                "test",
                "--dest-dir",
                self.dest_dir,
                "--yes",
            ]
        )
        self.assertNotEqual(ret, 0)

    def test_cli_yes_does_not_overwrite(self):
        """--yes should NOT implicitly overwrite user-edited files."""
        # First generate.
        ret = main(
            [
                "--hf-config",
                self.config_path,
                "--adapter-name",
                "mydense",
                "--dest-dir",
                self.dest_dir,
                "--yes",
            ]
        )
        self.assertEqual(ret, 0)
        # Edit a file.
        model_file = Path(self.dest_dir) / "model.py"
        original = model_file.read_text(encoding="utf-8")
        model_file.write_text(original + "\n# User edit\n", encoding="utf-8")
        # Re-generate with --yes (should NOT overwrite).
        ret = main(
            [
                "--hf-config",
                self.config_path,
                "--adapter-name",
                "mydense",
                "--dest-dir",
                self.dest_dir,
                "--yes",
            ]
        )
        self.assertEqual(ret, 0)
        # File should still have the user edit.
        content = model_file.read_text(encoding="utf-8")
        self.assertIn("# User edit", content)

    def test_cli_json_output(self):
        """--json should emit JSON with created_files."""
        import contextlib
        import io

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ret = main(
                [
                    "--hf-config",
                    self.config_path,
                    "--adapter-name",
                    "mydense",
                    "--dest-dir",
                    self.dest_dir,
                    "--yes",
                    "--json",
                ]
            )
        self.assertEqual(ret, 0)
        payload = json.loads(output.getvalue())
        self.assertIn("model.py", payload["created_files"])

    def test_generation_result_to_dict(self):
        """GenerationResult.to_dict should return all fields."""
        from generate_adapter_scaffold import GenerationResult

        result = GenerationResult(created_files=["a.py"], dest_dir="/tmp")
        d = result.to_dict()
        self.assertEqual(d["created_files"], ["a.py"])
        self.assertEqual(d["dest_dir"], "/tmp")
        self.assertEqual(d["preserved_files"], [])
        self.assertEqual(d["conflicted_files"], [])


if __name__ == "__main__":
    unittest.main()
