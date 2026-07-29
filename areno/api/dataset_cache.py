"""原子化、基于指纹的分词数据集样本磁盘缓存。

`Trainer.load_prompt_batches` 会在每个 epoch 对每条数据集记录重新分词。
本模块将该过程产生的 prompt 样本缓存到磁盘，使用内容/分词器/模板/选项的组合指纹作为键，
后续的 epoch —— 以及后续针对相同输入的运行 —— 可以直接复用缓存样本，无需重复分词。

设计要点：

- 内容寻址键：指纹由数据集内容、分词器资产文件、当前 chat template + thinking 开关、
  预处理选项以及缓存格式版本号共同决定。任何相关输入的变化都会导致生成不同的键，
  因此永远不会复用过期的缓存条目。
- 原子写入：每个缓存条目是单个 JSON 文件，先写入同目录的临时文件，然后通过
  ``os.replace`` 原子替换，镜像了 ``areno/api/metrics.py`` 和
  ``areno/cli/dashboard_registry.py`` 的实现方式。跨进程的并发读者要么读到完整的
  旧文件，要么读到完整的新文件，永远不会看到部分写入。
- 严格校验：读取时校验版本号、指纹哈希和数据结构；损坏或不兼容的缓存条目会被记录
  并视为未命中，而不是被使用。
- 安全默认值在别处：本模块永不静默改变行为；调用方需显式构造 ``DatasetCache`` 并
  传入才会启用缓存。

本模块仅使用标准库（``hashlib``、``json``、``os``、``tempfile``、``pathlib``），
避免引入重量级依赖，确保 ``areno train --help`` 保持快速。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARENO_DATASET_CACHE_VERSION = 1
"""缓存格式版本号；递增后旧代码生成的缓存条目将失效。"""

DATASET_CACHE_MODES = ("auto", "refresh", "readonly")
"""``auto`` 未命中时读取并写入；``refresh`` 忽略已有缓存并覆盖；``readonly`` 只读，永不写入。"""

# 影响 ``tokenize()`` / ``apply_chat_template()`` 输出的 HF 分词器文件。
# 仅将模型路径下实际存在的文件计入资产哈希；缺失的文件会被忽略而非报错，
# 因此没有 chat template 的基础分词器也能正确缓存。
_TOKENIZER_ASSET_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
)

logger = logging.getLogger("areno.api.dataset_cache")


class DatasetCacheError(ValueError):
    """无效的缓存配置（错误的模式或不可写的路径）时抛出。"""


@dataclass(frozen=True)
class CacheKey:
    """指纹组件与其内容寻址哈希的封装。"""

    fingerprint: dict[str, Any]
    fingerprint_hash: str

    @property
    def filename(self) -> str:
        """由指纹哈希推导出的缓存文件名。"""

        return f"{self.fingerprint_hash}.json"


def _sha256_text(text: str) -> str:
    """对短字符串（模板、JSON 投影）计算 SHA256 哈希，返回十六进制摘要。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    """流式计算文件的 SHA256 哈希，避免一次性加载大文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_tokenizer(model_path: str | Path, tokenizer: Any) -> dict[str, Any]:
    """计算分词器资产文件 + 当前 chat template + thinking 开关的指纹。

    chat template 和 ``enable_thinking`` 开关被显式计入哈希（而非仅依赖磁盘上的配置），
    因为调用方可能在运行时设置它们（见 ``configure_chat_template_enable_thinking``），
    这会影响分词结果但不改变磁盘文件。
    """

    base = Path(model_path)
    asset_hashes: dict[str, str] = {}
    for name in _TOKENIZER_ASSET_FILES:
        candidate = base / name
        if candidate.is_file():
            asset_hashes[name] = _hash_file(candidate)
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template = chat_template if isinstance(chat_template, str) else ""
    enable_thinking = getattr(tokenizer, "_areno_chat_template_enable_thinking", None)
    return {
        "asset_files": asset_hashes,
        "chat_template_hash": _sha256_text(chat_template),
        "enable_thinking": enable_thinking,
    }


def fingerprint_dataset(dataset: Any, prompt_key: str, solutions_key: str) -> dict[str, Any]:
    """计算训练器实际看到的已规范化数据集的身份指纹。

    遍历一次并对每行数据的 JSON 投影（prompt 字段 + 排序后的记录键）计算哈希，
    使缓存具备内容寻址能力：数据集中任何 prompt 或 schema 的变化都会导致缓存条目失效，
    而数据集路径字面值不会阻碍另一台机器上相同内容的复用。
    ``default=str`` 保持流式哈希对非 JSON 行值的鲁棒性；
    严格的序列化校验发生在 ``DatasetCache.save`` 中。
    """

    digest = hashlib.sha256()
    schema_keys: set[str] = set()
    row_count = 0
    for record in dataset:
        row_count += 1
        keys = sorted(record.keys()) if hasattr(record, "keys") else []
        schema_keys.update(keys)
        projection = {key: record.get(key) for key in keys}
        digest.update(json.dumps(projection, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return {
        "row_count": row_count,
        "schema_keys": sorted(schema_keys),
        "content_hash": digest.hexdigest(),
        "prompt_key": prompt_key,
        "solutions_key": solutions_key,
    }


def compute_cache_key(
    model_path: str | Path,
    tokenizer: Any,
    dataset: Any,
    *,
    max_prompt_tokens: int,
    prompt_key: str,
    solutions_key: str,
) -> CacheKey:
    """为一次分词配置构建内容寻址的缓存键。"""

    components = {
        "version": ARENO_DATASET_CACHE_VERSION,
        "tokenizer": fingerprint_tokenizer(model_path, tokenizer),
        "dataset": fingerprint_dataset(dataset, prompt_key, solutions_key),
        "preprocessing": {
            "max_prompt_tokens": max_prompt_tokens,
            "prompt_key": prompt_key,
            "solutions_key": solutions_key,
        },
    }
    fingerprint_hash = hashlib.sha256(
        json.dumps(components, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    return CacheKey(fingerprint=components, fingerprint_hash=fingerprint_hash)


def validate_cache_config(cache_path: str | None, mode: str) -> None:
    """数据集缓存配置的预校验。

    若配置无效则抛出 ``DatasetCacheError``，错误信息中包含有问题的输入项名称，
    以便 CLI 在昂贵的模型/worker 初始化之前快速失败。
    此函数与运行时构造分离，因此无效路径在配置构建阶段即被拒绝，而非训练中途。
    """

    if mode not in DATASET_CACHE_MODES:
        raise DatasetCacheError(
            f"dataset cache mode must be one of {DATASET_CACHE_MODES}, got {mode!r}; "
            "stage=dataset_cache_config input=dataset_cache_mode"
        )
    if not cache_path:
        return
    parent = Path(cache_path).expanduser().parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatasetCacheError(
            f"dataset cache path is not creatable: {cache_path} ({exc}); "
            "stage=dataset_cache_config input=dataset_cache_path"
        ) from exc
    if not os.access(parent, os.W_OK):
        raise DatasetCacheError(
            f"dataset cache path is not writable: {cache_path}; "
            "stage=dataset_cache_config input=dataset_cache_path"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    """通过同目录临时文件 + rename 将 ``text`` 原子写入 ``path``。

    临时文件位于目标目录中，因此最终的 ``os.replace`` 是同文件系统的原子重命名；
    读者永远不会看到半写入的文件。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False, encoding="utf-8"
    )
    tmp_path = Path(handle.name)
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    except BaseException:
        handle.close()
        _silent_remove(tmp_path)
        raise
    os.replace(tmp_path, path)


def _silent_remove(path: Path) -> None:
    """静默删除临时文件，若文件已不存在则不报错。"""

    try:
        path.unlink()
    except OSError:
        pass


class DatasetCache:
    """以 ``CacheKey`` 为键的已分词 prompt 样本磁盘缓存。

    缓存目录下存放一个 ``<fingerprint_hash>.json`` 条目，对应唯一的分词配置。
    ``mode`` 控制未命中时是否持久化：

    * ``auto`` —— 命中时读取，未命中时分词并持久化（默认）。
    * ``refresh`` —— 忽略已有条目，重新分词并覆盖。
    * ``readonly`` —— 命中时读取，未命中时仅在内存中分词，永不持久化
      （适用于只读文件系统或共享的不可变缓存）。
    """

    def __init__(self, cache_dir: str | Path, *, mode: str = "auto") -> None:
        if mode not in DATASET_CACHE_MODES:
            raise DatasetCacheError(f"dataset cache mode must be one of {DATASET_CACHE_MODES}, got {mode!r}")
        self.cache_dir = Path(cache_dir)
        self.mode = mode

    def artifact_path(self, key: CacheKey) -> Path:
        """解析某个缓存键对应的磁盘路径。"""

        return self.cache_dir / key.filename

    def try_load(self, key: CacheKey) -> tuple[list[dict], dict[str, Any]] | None:
        """加载并校验缓存条目，成功返回 ``(items, meta)``，失败返回 ``None``。

        任何读取错误、版本不匹配、指纹不匹配或结构问题都会被记录并视为未命中，
        因此损坏或不兼容的缓存条目永远不会被使用；调用方重新分词并在可写模式下覆盖。
        """

        path = self.artifact_path(key)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.info("stage=dataset_cache_rejected reason=read_error path=%s", path)
            return None
        if not isinstance(envelope, dict):
            logger.info("stage=dataset_cache_rejected reason=corrupt_envelope path=%s", path)
            return None
        if envelope.get("version") != ARENO_DATASET_CACHE_VERSION:
            logger.info("stage=dataset_cache_rejected reason=version_mismatch path=%s", path)
            return None
        if envelope.get("fingerprint_hash") != key.fingerprint_hash:
            logger.info("stage=dataset_cache_rejected reason=fingerprint_mismatch path=%s", path)
            return None
        items = envelope.get("items")
        if not isinstance(items, list):
            logger.info("stage=dataset_cache_rejected reason=corrupt_items path=%s", path)
            return None
        meta = {
            "max_prompt_tokens": envelope.get("max_prompt_tokens"),
            "prompt_key": envelope.get("prompt_key"),
            "solutions_key": envelope.get("solutions_key"),
            "count": envelope.get("count"),
            "skipped_long": int(envelope.get("skipped_long", 0)),
        }
        return items, meta

    def save(self, key: CacheKey, items: list[dict], meta: dict[str, Any]) -> int | None:
        """原子持久化 ``items``，成功返回缓存条目字节数。

        若任意记录不可 JSON 序列化则返回 ``None`` 且不写入文件，
        因此包含不可序列化字段（例如含二进制载荷的多模态行）的数据集会优雅降级为不缓存，
        而不是留下不完整或损坏的缓存条目。``items`` 直接原样存储；
        调用方需确保传入纯 JSON 兼容的 dict。
        """

        envelope = {
            "version": ARENO_DATASET_CACHE_VERSION,
            "fingerprint": key.fingerprint,
            "fingerprint_hash": key.fingerprint_hash,
            "max_prompt_tokens": meta.get("max_prompt_tokens"),
            "prompt_key": meta.get("prompt_key"),
            "solutions_key": meta.get("solutions_key"),
            "count": len(items),
            "skipped_long": int(meta.get("skipped_long", 0)),
            "items": items,
        }
        try:
            payload = json.dumps(envelope, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            # 拒绝不完整的缓存：绝不向磁盘写入不完整的文件。
            # 截断错误信息以避免暴露完整样本内容。
            logger.info("stage=dataset_cache_skip reason=non_serializable_record error=%s", str(exc)[:160])
            return None
        path = self.artifact_path(key)
        _atomic_write_text(path, payload)
        return len(payload.encode("utf-8"))

    def inspect(self) -> list[dict[str, Any]]:
        """扫描缓存目录下的所有条目并返回摘要报告。

        读取每个文件以展示其 manifest 字段（count、skipped_long、fingerprint_hash）；
        仅需文件大小的调用方可以忽略解析字段。无效文件以 ``valid=False`` 报告而非丢弃，
        以便运维人员能看到并清理损坏条目。
        """

        results: list[dict[str, Any]] = []
        if not self.cache_dir.is_dir():
            return results
        for entry in sorted(self.cache_dir.glob("*.json")):
            stat = entry.stat()
            record: dict[str, Any] = {
                "path": str(entry),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
            try:
                envelope = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record["valid"] = False
                results.append(record)
                continue
            record.update(
                {
                    "valid": envelope.get("version") == ARENO_DATASET_CACHE_VERSION,
                    "fingerprint_hash": envelope.get("fingerprint_hash"),
                    "count": envelope.get("count"),
                    "skipped_long": envelope.get("skipped_long"),
                    "mode_hint": "refresh" if envelope.get("version") != ARENO_DATASET_CACHE_VERSION else None,
                }
            )
            results.append(record)
        return results

    def clean(self, *, fingerprint_hash: str | None = None, remove_all: bool = False) -> dict[str, Any]:
        """删除一个缓存条目（按 ``fingerprint_hash``）或全部条目。

        返回 ``{"removed": int, "bytes_freed": int}``。若指定的条目不存在则不会报错，
        以保证命令幂等。
        """

        removed = 0
        bytes_freed = 0
        if not self.cache_dir.is_dir():
            return {"removed": removed, "bytes_freed": bytes_freed}
        if remove_all:
            targets = list(self.cache_dir.glob("*.json"))
        elif fingerprint_hash:
            target = self.cache_dir / f"{fingerprint_hash}.json"
            targets = [target] if target.is_file() else []
        else:
            targets = []
        for entry in targets:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            try:
                entry.unlink()
            except OSError:
                continue
            removed += 1
            bytes_freed += size
        return {"removed": removed, "bytes_freed": bytes_freed}
