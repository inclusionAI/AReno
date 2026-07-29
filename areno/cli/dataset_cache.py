"""`areno dataset-cache` CLI —— 检查与清理分词缓存。

本命令用于展示 Issue #206 分词缓存写入磁盘的条目（位置 + 大小报告 + 显式移除），
同时支持人类可读与 `--json` 形式，镜像 `areno diagnostics --json` 的约定。
命令不执行分词、不初始化引擎，保持轻量。
"""

from __future__ import annotations

import json
import os

import click

from areno.api.dataset_cache import DatasetCache

_DEFAULT_CACHE_ENV = "ARENO_DATASET_CACHE_DIR"


def _resolve_cache_path(cache_path: str | None) -> str:
    """从命令行参数或环境变量解析缓存目录路径。"""

    path = cache_path or os.environ.get(_DEFAULT_CACHE_ENV)
    if not path:
        raise click.UsageError(
            "A cache directory is required; pass --cache-path or set "
            f"{_DEFAULT_CACHE_ENV}. stage=dataset_cache_config input=cache_path"
        )
    return path


@click.group(name="dataset-cache", context_settings={"help_option_names": ["-h", "--help"]})
def dataset_cache_command() -> None:
    """检查或清理已缓存的分词条目（Issue #206）。"""


@dataset_cache_command.command(name="inspect", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--cache-path", default=None, help="待检查的缓存目录；默认取 $ARENO_DATASET_CACHE_DIR。")
@click.option("--json", "as_json", is_flag=True, help="输出机器可读的 JSON 报告。")
def inspect_command(cache_path: str | None, as_json: bool) -> None:
    """列出缓存条目，包含大小、条目数、有效性及指纹。"""

    path = _resolve_cache_path(cache_path)
    entries = DatasetCache(path, mode="readonly").inspect()
    report = {"cache_path": path, "count": len(entries), "entries": entries}
    if as_json:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    click.echo(f"dataset cache: {path}")
    click.echo(f"  artifacts: {len(entries)}")
    for entry in entries:
        fingerprint = entry.get("fingerprint_hash") or "?"
        status = "valid" if entry.get("valid") else "INVALID"
        click.echo(
            f"  {fingerprint[:12]}  count={entry.get('count')}  "
            f"size_bytes={entry.get('size_bytes')}  {status}"
        )


@dataset_cache_command.command(name="clean", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--cache-path", default=None, help="待清理的缓存目录；默认取 $ARENO_DATASET_CACHE_DIR。")
@click.option("--fingerprint", default=None, help="仅删除指定指纹哈希对应的缓存条目。")
@click.option("--all", "remove_all", is_flag=True, help="删除缓存目录下的所有条目。")
@click.option("--json", "as_json", is_flag=True, help="输出机器可读的 JSON 报告。")
def clean_command(cache_path: str | None, fingerprint: str | None, remove_all: bool, as_json: bool) -> None:
    """删除缓存条目（显式移除）。"""

    if not remove_all and not fingerprint:
        raise click.UsageError(
            "Pass --all or --fingerprint <hash>; refusing to remove nothing. stage=dataset_cache_clean"
        )
    path = _resolve_cache_path(cache_path)
    result = DatasetCache(path, mode="auto").clean(
        fingerprint_hash=fingerprint, remove_all=bool(remove_all and not fingerprint)
    )
    if as_json:
        click.echo(json.dumps({"cache_path": path, **result}, indent=2, sort_keys=True))
        return
    click.echo(
        f"dataset cache clean: removed {result['removed']} artifact(s), "
        f"freed {result['bytes_freed']} bytes from {path}"
    )
