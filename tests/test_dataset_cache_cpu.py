"""分词/数据集预处理缓存的 CPU 单元测试（Issue #206）。

直接测试 `areno/api/dataset_cache.py`：指纹确定性与失效、原子写+读取往返、
损坏或不兼容条目拒绝、不可序列化记录的安全跳过、inspect/clean 报告辅助函数。
无需分词器或 torch。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from areno.api.dataset_cache import (
    ARENO_DATASET_CACHE_VERSION,
    DatasetCache,
    DatasetCacheError,
    compute_cache_key,
    validate_cache_config,
)


def _tokenizer_dir(tmp_path: Path) -> Path:
    """创建一个微型分词器目录，使资产文件哈希有内容可读。"""

    tok_dir = tmp_path / "tok"
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "tokenizer_config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    return tok_dir


class _StubTokenizer:
    """最小分词器替身，仅暴露指纹计算所需的属性。"""

    def __init__(self, chat_template: str | None = "<|im_start|>user", enable_thinking=None):
        self.chat_template = chat_template
        if enable_thinking is not None:
            self._areno_chat_template_enable_thinking = enable_thinking


def _make_key(tmp_path, dataset=None, *, max_prompt_tokens=8, prompt_key="prompt", solutions_key="solutions"):
    return compute_cache_key(
        _tokenizer_dir(tmp_path),
        _StubTokenizer(),
        dataset or [{"prompt": "hi"}],
        max_prompt_tokens=max_prompt_tokens,
        prompt_key=prompt_key,
        solutions_key=solutions_key,
    )


def _save_metadata(**overrides):
    meta = {"max_prompt_tokens": 8, "prompt_key": "prompt", "solutions_key": "solutions", "skipped_long": 0}
    meta.update(overrides)
    return meta


# --- fingerprint determinism + invalidation ----------------------------------


def test_fingerprint_is_deterministic(tmp_path):
    a = _make_key(tmp_path, dataset=[{"prompt": "hi"}, {"prompt": "yo"}])
    b = _make_key(tmp_path, dataset=[{"prompt": "hi"}, {"prompt": "yo"}])

    assert a.fingerprint_hash == b.fingerprint_hash
    assert a.filename == b.filename


def test_fingerprint_invalidates_on_each_relevant_input(tmp_path):
    base = _make_key(tmp_path, dataset=[{"prompt": "hi"}])
    different_prompt = _make_key(tmp_path, dataset=[{"prompt": "bye"}])
    different_max = _make_key(tmp_path, dataset=[{"prompt": "hi"}], max_prompt_tokens=16)
    different_key = _make_key(tmp_path, dataset=[{"prompt": "hi"}], prompt_key="question")

    assert base.fingerprint_hash != different_prompt.fingerprint_hash
    assert base.fingerprint_hash != different_max.fingerprint_hash
    assert base.fingerprint_hash != different_key.fingerprint_hash


def test_fingerprint_invalidates_when_tokenizer_asset_changes(tmp_path):
    tok_dir = _tokenizer_dir(tmp_path)
    tokenizer = _StubTokenizer(chat_template=None)
    base = compute_cache_key(tok_dir, tokenizer, [{"prompt": "x"}], max_prompt_tokens=4, prompt_key="prompt", solutions_key="solutions")
    (tok_dir / "tokenizer_config.json").write_text('{"model_type":"changed"}', encoding="utf-8")
    changed = compute_cache_key(tok_dir, tokenizer, [{"prompt": "x"}], max_prompt_tokens=4, prompt_key="prompt", solutions_key="solutions")

    assert base.fingerprint_hash != changed.fingerprint_hash


def test_fingerprint_invalidates_when_chat_template_changes(tmp_path):
    tok_dir = _tokenizer_dir(tmp_path)
    without_thinking = compute_cache_key(
        tok_dir, _StubTokenizer(enable_thinking=None), [{"prompt": "x"}], max_prompt_tokens=4, prompt_key="prompt", solutions_key="solutions"
    )
    with_thinking = compute_cache_key(
        tok_dir, _StubTokenizer(enable_thinking=True), [{"prompt": "x"}], max_prompt_tokens=4, prompt_key="prompt", solutions_key="solutions"
    )
    different_template = compute_cache_key(
        tok_dir, _StubTokenizer(chat_template="<changed>"), [{"prompt": "x"}], max_prompt_tokens=4, prompt_key="prompt", solutions_key="solutions"
    )

    assert without_thinking.fingerprint_hash != with_thinking.fingerprint_hash
    assert without_thinking.fingerprint_hash != different_template.fingerprint_hash


# --- save / load round-trip + rejection --------------------------------------


def test_save_then_load_roundtrips_items(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    items = [{"prompt": "hi", "solutions": ["a"], "input_tokens": [1, 2], "record": {"prompt": "hi", "answer": "2"}}]

    size_bytes = cache.save(key, items, _save_metadata())

    assert size_bytes and size_bytes > 0
    assert cache.artifact_path(key).is_file()
    loaded_items, meta = cache.try_load(key)
    assert loaded_items == items
    assert meta["count"] == 1
    assert meta["skipped_long"] == 0
    assert meta["prompt_key"] == "prompt"


def test_save_is_byte_identical_for_identical_inputs(tmp_path):
    """Same inputs must produce a byte-identical artifact for reproducibility."""

    cache_a = DatasetCache(tmp_path / "cache_a", mode="auto")
    cache_b = DatasetCache(tmp_path / "cache_b", mode="auto")
    key_a = _make_key(tmp_path / "cache_a")
    key_b = _make_key(tmp_path / "cache_b")
    items = [{"prompt": "hi", "solutions": ["a"], "input_tokens": [1, 2], "record": {"prompt": "hi"}}]
    cache_a.save(key_a, items, _save_metadata())
    cache_b.save(key_b, items, _save_metadata())

    assert cache_a.artifact_path(key_a).read_bytes() == cache_b.artifact_path(key_b).read_bytes()


def test_try_load_returns_none_when_artifact_missing(tmp_path):
    assert DatasetCache(tmp_path / "cache").try_load(_make_key(tmp_path)) is None


def test_try_load_rejects_corrupt_json(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    cache.save(key, [{"prompt": "hi", "input_tokens": [1], "record": {}}], _save_metadata())
    cache.artifact_path(key).write_text("{not valid json", encoding="utf-8")

    assert cache.try_load(key) is None


def test_try_load_rejects_version_mismatch(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    cache.save(key, [{"prompt": "hi", "input_tokens": [1], "record": {}}], _save_metadata())
    path = cache.artifact_path(key)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["version"] = ARENO_DATASET_CACHE_VERSION + 999
    path.write_text(json.dumps(envelope), encoding="utf-8")

    assert cache.try_load(key) is None


def test_try_load_rejects_fingerprint_mismatch(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    cache.save(key, [{"prompt": "hi", "input_tokens": [1], "record": {}}], _save_metadata())
    path = cache.artifact_path(key)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["fingerprint_hash"] = "0" * 64
    path.write_text(json.dumps(envelope), encoding="utf-8")

    assert cache.try_load(key) is None


def test_save_skips_non_serializable_record_without_writing(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    items = [{"record": {"unserializable": {1, 2, 3}}}]  # a set is not JSON-serializable

    assert cache.save(key, items, _save_metadata()) is None
    assert not cache.artifact_path(key).is_file()


def test_atomic_write_leaves_no_temp_file(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    cache.save(key, [{"prompt": "hi", "input_tokens": [1], "record": {}}], _save_metadata())

    assert list(cache.cache_dir.glob("*.tmp")) == []
    assert list(cache.cache_dir.glob("*.json")) == [cache.artifact_path(key)]


# --- modes -------------------------------------------------------------------


def test_readonly_mode_is_constructable_and_reads(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="readonly")
    assert cache.mode == "readonly"
    # The gate that prevents writes lives in `Trainer`; the cache itself just reads.
    assert DatasetCache(tmp_path / "ro").try_load(_make_key(tmp_path)) is None


def test_invalid_mode_raises():
    with pytest.raises(DatasetCacheError):
        DatasetCache("ignored-path", mode="bogus")


# --- inspect / clean ---------------------------------------------------------


def test_inspect_reports_valid_artifact(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    cache.save(key, [{"prompt": "hi", "input_tokens": [1], "record": {}}], _save_metadata(skipped_long=2))

    entries = cache.inspect()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["valid"] is True
    assert entry["count"] == 1
    assert entry["skipped_long"] == 2
    assert entry["size_bytes"] > 0
    assert entry["fingerprint_hash"] == key.fingerprint_hash


def test_inspect_reports_invalid_file(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "deadbeef.json").write_text("{not json", encoding="utf-8")

    entries = DatasetCache(cache_dir).inspect()
    assert len(entries) == 1
    assert entries[0]["valid"] is False


def test_clean_removes_one_artifact_by_fingerprint(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    cache.save(key, [{"prompt": "hi", "input_tokens": [1], "record": {}}], _save_metadata())
    size = cache.artifact_path(key).stat().st_size

    result = cache.clean(fingerprint_hash=key.fingerprint_hash)

    assert result == {"removed": 1, "bytes_freed": size}
    assert not cache.artifact_path(key).is_file()


def test_clean_all_removes_every_artifact(tmp_path):
    cache = DatasetCache(tmp_path / "cache", mode="auto")
    key = _make_key(tmp_path)
    cache.save(key, [{"prompt": "hi", "input_tokens": [1], "record": {}}], _save_metadata())

    result = cache.clean(remove_all=True)

    assert result["removed"] == 1
    assert result["bytes_freed"] > 0
    assert list(cache.cache_dir.glob("*.json")) == []


def test_clean_missing_artifact_is_idempotent(tmp_path):
    result = DatasetCache(tmp_path / "cache").clean(remove_all=True)
    assert result == {"removed": 0, "bytes_freed": 0}


# --- pre-flight config validation -------------------------------------------


def test_validate_cache_config_accepts_unset_path():
    validate_cache_config(None, "auto")  # no raise


def test_validate_cache_config_rejects_bad_mode():
    with pytest.raises(DatasetCacheError, match="dataset cache mode must be one of"):
        validate_cache_config(None, "bogus")


def test_validate_cache_config_rejects_uncreatable_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    bad_path = str(blocker / "cache")  # parent is a file, not a directory

    with pytest.raises(DatasetCacheError, match="dataset cache path is not creatable"):
        validate_cache_config(bad_path, "auto")
