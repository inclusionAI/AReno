"""`areno dataset-cache` inspect/clean 子命令的 CPU 测试（Issue #206）。

镜像 `test_train_cli_config_cpu.py` 的 `CliRunner` 风格；子命令本身不依赖引擎，
因此不会加载分词器或 torch。
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from areno.api.dataset_cache import DatasetCache, compute_cache_key
from areno.cli.dataset_cache import dataset_cache_command


def _populate(cache_dir, tokenizer_dir) -> str:
    """写入一条缓存条目并返回其指纹哈希。"""
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tokenizer = type("T", (), {"chat_template": None})()
    key = compute_cache_key(
        tokenizer_dir,
        tokenizer,
        [{"prompt": "hi"}],
        max_prompt_tokens=8,
        prompt_key="prompt",
        solutions_key="solutions",
    )
    DatasetCache(cache_dir, mode="auto").save(
        key,
        [{"prompt": "hi", "input_tokens": [1], "record": {"prompt": "hi"}}],
        {"max_prompt_tokens": 8, "prompt_key": "prompt", "solutions_key": "solutions", "skipped_long": 0},
    )
    return key.fingerprint_hash


def test_inspect_json_reports_artifacts(tmp_path):
    cache_dir = tmp_path / "cache"
    fp = _populate(cache_dir, tmp_path / "tok")

    result = CliRunner().invoke(dataset_cache_command, ["inspect", "--cache-path", str(cache_dir), "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["cache_path"] == str(cache_dir)
    assert report["count"] == 1
    entry = report["entries"][0]
    assert entry["fingerprint_hash"] == fp
    assert entry["count"] == 1
    assert entry["valid"] is True
    assert entry["size_bytes"] > 0


def test_inspect_human_summary(tmp_path):
    cache_dir = tmp_path / "cache"
    _populate(cache_dir, tmp_path / "tok")

    result = CliRunner().invoke(dataset_cache_command, ["inspect", "--cache-path", str(cache_dir)])

    assert result.exit_code == 0, result.output
    assert "dataset cache:" in result.output
    assert "artifacts: 1" in result.output
    assert "valid" in result.output


def test_clean_requires_a_target(tmp_path):
    result = CliRunner().invoke(dataset_cache_command, ["clean", "--cache-path", str(tmp_path / "cache")])

    assert result.exit_code != 0
    assert "Pass --all or --fingerprint" in result.output


def test_clean_all_removes_every_artifact(tmp_path):
    cache_dir = tmp_path / "cache"
    _populate(cache_dir, tmp_path / "tok")

    result = CliRunner().invoke(
        dataset_cache_command, ["clean", "--cache-path", str(cache_dir), "--all", "--json"]
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["removed"] == 1
    assert report["bytes_freed"] > 0
    assert list(cache_dir.glob("*.json")) == []


def test_clean_fingerprint_removes_one_artifact(tmp_path):
    cache_dir = tmp_path / "cache"
    fp = _populate(cache_dir, tmp_path / "tok")

    result = CliRunner().invoke(
        dataset_cache_command, ["clean", "--cache-path", str(cache_dir), "--fingerprint", fp, "--json"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["removed"] == 1
    assert not (cache_dir / f"{fp}.json").is_file()


def test_inspect_requires_cache_path_or_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ARENO_DATASET_CACHE_DIR", raising=False)
    result = CliRunner().invoke(dataset_cache_command, ["inspect"])

    assert result.exit_code != 0
    assert "cache directory is required" in result.output


def test_inspect_falls_back_to_env_var(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    _populate(cache_dir, tmp_path / "tok")
    monkeypatch.setenv("ARENO_DATASET_CACHE_DIR", str(cache_dir))

    result = CliRunner().invoke(dataset_cache_command, ["inspect", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["count"] == 1
