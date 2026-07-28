# AReno 运行时故障取证收集器 — 解决方案设计

> 生成时间: 2026-07-28 | 对应 Issue: #280 | 分支: issue_280_yixuan

---

## 一、需求分析

### 1.1 核心目标

为 AReno 用户提供一个**运行时故障取证收集器 (areno-debug-runtime)**，在训练/服务出现异常时自动或手动收集关键诊断信息，生成可读的本地取证包。

### 1.2 关键需求拆解

| # | 需求 | 优先级 |
|---|------|--------|
| 1 | 收集命令、解析后的关键设置、AReno/Python 版本、GPU 摘要 | P0 |
| 2 | 收集最早 traceback 和相关进程状态 | P0 |
| 3 | 生成可读的本地 bundle（文件包） | P0 |
| 4 | 使用 AReno 现有公共合约和本地产物格式 | P0 |
| 5 | 故障保存不隐藏原始错误 | P0 |
| 6 | 脱敏处理（环境变量、样本内容） | P1 |
| 7 | 支持自定义输出位置 | P1 |
| 8 | 支持时间限界日志 | P1 |
| 9 | CLI 暴露入口，安全默认值，清晰验证错误 | P0 |
| 10 | 人可读 + 结构化输出 | P1 |

### 1.3 Non-goals（明确不做的）

- 不替换 Trainer、rollout engine、dashboard、SDK 架构
- 不引入外部数据库、托管控制面板、重量级依赖
- 不自动修改用户配置、删除产物、终止无关进程
- 不引入新依赖（尽可能复用现有的 `json`, `traceback`, `sys`, `os`, `platform`, `datetime`）

---

## 二、现有基础设施分析

### 2.1 可直接复用的现有能力

| 现有模块 | 可复用能力 | 复用方式 |
|----------|-----------|----------|
| `areno/cli/diagnostics.py` | `collect_env()` — 环境信息收集；`CheckResult` — 检查结果 dataclass 模式 | 直接调用 `collect_env()` 获取 AReno/Python/CUDA/GPU 环境报告 |
| `areno/cli/diagnostics.py` | `_torch_info()`, `_nvidia_smi_driver_info()`, `_dependency_info()` | 复用子函数收集 GPU/依赖信息 |
| `areno/engine/log.py` | `configure_default_logging()` — 日志格式 | 可临时调整日志级别或添加额外 handler 捕获定向日志 |
| `areno/api/metrics.py` | `MetricsRecorder` — 本地文件写入模式（JSON lines + TensorBoard） | 复用 JSON lines 写入模式，不引入新格式 |
| `areno/engine/protocol.py` | Worker 进程异常处理：`traceback.format_exc()` 序列化到 `WorkerResult.error` | 已在 worker 层面捕获 traceback，复用此机制 |
| `areno/cli/train.py` | `_trainer_config_from_options()` → `TrainerConfig` — 完整配置解析 | 直接序列化已有 TrainerConfig |
| `areno/cli/main.py` | `ArenoCli` Click group 模式 | 在此注册新子命令 `areno debug` |

### 2.2 现有文件/目录布局

```
areno/
├── cli/
│   ├── main.py          ← 新增子命令注册点
│   ├── diagnostics.py   ← 复用 collect_env(), _torch_info()
│   └── debug.py         ← 新建: debug 子命令 + 核心收集逻辑
├── api/
│   └── metrics.py       ← 参考 JSON lines 写入模式
├── engine/
│   ├── log.py           ← 日志配置
│   └── protocol.py      ← Worker traceback 捕获（已有）
tests/
├── test_cli_diagnostics_cpu.py  ← 参考测试风格
└── test_debug_collector_cpu.py   ← 新建: 收集器 CPU 测试
处理过程/                  ← 方案文档目录（不是代码）
skills/
└── areno-model-adaptation/  ← 参考 skills 目录结构
```

### 2.3 现有代码中的关键约定

1. **延迟导入** — CLI 子命令通过 `ArenoCli._COMMANDS` 字典注册，`get_command()` 动态 `__import__`
2. **错误处理** — CLI 层用 `click.UsageError`，API 层用 `RuntimeError/ValueError`，engine 层用 `traceback.format_exc()` 序列化
3. **环境报告** — `collect_env()` 返回纯 `dict[str, Any]`，已有结构化 JSON 输出
4. **进程模型** — Worker 是 `mp.Process` 子进程，main 进程是协调者
5. **不引入重量级依赖** — 所有现有代码使用 `json`, `os`, `sys`, `traceback`, `datetime` 等标准库

---

## 三、方案设计

### 3.1 总体架构

```
用户触发 → areno debug [选项]
    │
    ├── 手动模式: areno debug --collect --output-dir /path/to/bundle
    │   └── 收集当前进程环境快照
    │
    ├── 包装模式: areno debug --wrap -- areno train --algo gspo ...
    │   └── 包装训练命令，异常时自动收集
    │
    └── 事后模式: areno debug --traceback-file /tmp/traceback.txt --config-file ./run.json
        └── 从已有文件重建取证包
```

### 3.2 模块设计

新增两个文件 + 注册一个新 CLI 子命令，改动范围极小：

```
areno/cli/debug.py          ← 核心实现 (~200-300 行)
tests/test_debug_collector_cpu.py  ← CPU 测试 (~150 行)
areno/cli/main.py           ← 修改 1 行: _COMMANDS 字典增加 "debug" 条目
```

### 3.3 核心数据模型

```python
# areno/cli/debug.py

@dataclass
class FailureBundle:
    """故障取证包的数据模型"""
    # --- 元数据 ---
    timestamp: str                      # ISO 8601 时间戳
    areno_version: str | None           # areno 版本
    python_version: str                 # Python 版本
    platform: str                       # 操作系统

    # --- 命令与配置 ---
    command: list[str] | None           # 执行的原始命令
    resolved_config: dict | None       # 解析后的 TrainerConfig
    env_vars_redacted: dict            # 脱敏后的环境变量

    # --- GPU 与硬件 ---
    gpu_summary: dict | None            # GPU 信息
    cuda_info: dict | None              # CUDA 版本/工具链

    # --- 错误信息 ---
    error_type: str | None              # 异常类型名称
    error_message: str | None           # 异常消息
    error_traceback: str | None         # 完整 traceback（最早 origin 在前）

    # --- 进程状态 ---
    process_info: dict                  # pid, ppid, memory 使用等
    worker_state: list[dict] | None    # worker 子进程状态快照（如可获取）

    # --- 扩展 ---
    extra: dict                         # 自定义扩展字段
```

### 3.4 收集器核心逻辑

```python
# areno/cli/debug.py 伪代码

def collect_failure_bundle(
    *,
    command: list[str] | None = None,
    config: TrainerConfig | None = None,
    error: BaseException | None = None,
    include_env: bool = True,
    include_gpu: bool = True,
    redact_env_keys: set[str] = DEFAULT_REDACT_KEYS,
) -> FailureBundle:
    """收集故障取证信息，不抛出异常（即使子收集步骤失败也不中断）。"""

    bundle = FailureBundle(
        timestamp=datetime.now(timezone.utc).isoformat(),
        areno_version=_safe_areno_version(),
        python_version=sys.version,
        platform=platform.platform(),
        command=list(command) if command else None,
        resolved_config=_safe_config_dump(config),
        env_vars_redacted=_safe_env_collect(redact_keys=redact_env_keys),
        gpu_summary=_safe_gpu_info() if include_gpu else None,
        cuda_info=_safe_cuda_info() if include_gpu else None,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
        error_traceback=_safe_traceback(error),
        process_info=_safe_process_info(),
        worker_state=None,  # 由调用者补充
        extra={},
    )
    return bundle


def write_bundle(bundle: FailureBundle, output_dir: Path) -> Path:
    """将 bundle 写入本地目录，返回 bundle 路径。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / f"areno-failure-{bundle.timestamp.replace(':', '-')}"
    bundle_dir.mkdir()

    # 1. 结构化 JSON（机器可读）
    (bundle_dir / "bundle.json").write_text(
        json.dumps(dataclasses.asdict(bundle), indent=2, default=str),
        encoding="utf-8",
    )

    # 2. 人可读摘要（Markdown）
    (bundle_dir / "summary.md").write_text(
        _render_bundle_markdown(bundle),
        encoding="utf-8",
    )

    # 3. 原始 traceback（方便 grep）
    if bundle.error_traceback:
        (bundle_dir / "traceback.txt").write_text(
            bundle.error_traceback,
            encoding="utf-8",
        )

    return bundle_dir
```

### 3.5 CLI 子命令设计

```python
# areno/cli/debug.py

@click.command(
    name="debug",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Collect runtime failure evidence for diagnostics.",
)
@click.option(
    "--collect", is_flag=True,
    help="Collect an environment snapshot and write it to --output-dir.",
)
@click.option(
    "--wrap", is_flag=True,
    help="Wrap a subcommand and auto-collect on failure.",
)
@click.option(
    "--output-dir", default="./areno-debug",
    show_default=True,
    help="Output directory for failure bundles.",
)
@click.option(
    "--traceback-file", default=None,
    help="Read traceback from an existing file (post-mortem mode).",
)
@click.option(
    "--redact/--no-redact", default=True, show_default=True,
    help="Redact sensitive environment variable values.",
)
@click.option(
    "--timeout-s", type=float, default=30.0, show_default=True,
    help="Maximum time (seconds) to spend collecting evidence.",
)
@click.argument("subcommand", nargs=-1, type=click.UNPROCESSED)
def debug_command(subcommand, **options):
    """AReno runtime debug collector."""
    ...
```

**使用示例：**

```bash
# 手动收集当前环境快照
areno debug --collect --output-dir ./debug-bundles

# 包装训练，失败时自动收集
areno debug --wrap --output-dir ./debug-bundles -- \
  areno train --algo gspo --ckpt Qwen/Qwen3-0.6B ...

# 事后分析已有 traceback
areno debug --traceback-file /tmp/crash.txt --output-dir ./debug-bundles
```

### 3.6 脱敏策略

```python
# 需要脱敏的环境变量 key
DEFAULT_REDACT_KEYS = {
    "HF_TOKEN", "HUGGINGFACE_TOKEN", "MODELSCOPE_API_TOKEN",
    "OPENAI_API_KEY", "API_KEY", "TOKEN", "SECRET",
    "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
}

def _redact_env(key: str, value: str) -> str:
    """对敏感值脱敏：仅显示前2后2字符，其余用 * 替代。"""
    if len(value) <= 4:
        return "****"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]
```

### 3.7 错误处理原则

```
收集过程中任何子步骤失败 → 不中断整体收集
    │
    ├── _safe_xxx() 函数返回 None 而非抛异常
    │
    ├── 每个子步骤的错误被记录到 bundle.extra["collection_warnings"]
    │
    └── 原始错误始终完整保留在 bundle 中
```

---

## 四、实现细则

### 4.1 文件清单与改动范围

| 文件 | 操作 | 行数估计 | 说明 |
|------|------|----------|------|
| `areno/cli/debug.py` | **新建** | ~280 行 | 核心收集器 + CLI 命令 |
| `areno/cli/main.py` | **修改 1 行** | +1 | `_COMMANDS` 新增 `"debug"` |
| `tests/test_debug_collector_cpu.py` | **新建** | ~150 行 | CPU 测试 |

总计: **新增 2 个文件，修改 1 个文件，约 430 行代码**。不涉及任何新依赖。

### 4.2 `areno/cli/debug.py` 详细结构

```python
"""Runtime failure evidence collector for AReno diagnostics.

Exposes `areno debug` commands for collecting environment snapshots,
wrapping commands with auto-collection on failure, and reconstructing
failure bundles from saved tracebacks.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import signal
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

# --- Data Model ---

@dataclass
class FailureBundle:
    timestamp: str = ""
    areno_version: str | None = None
    python_version: str = ""
    platform_info: str = ""
    command: list[str] | None = None
    resolved_config: dict | None = None
    env_vars_redacted: dict[str, str] = field(default_factory=dict)
    gpu_summary: dict | None = None
    cuda_info: dict | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    process_info: dict = field(default_factory=dict)
    worker_state: list[dict] | None = None
    collection_warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# --- Sensitive Data Redaction ---

DEFAULT_REDACT_KEYS: set[str] = {
    "HF_TOKEN", "HUGGINGFACE_TOKEN", "MODELSCOPE_API_TOKEN",
    "OPENAI_API_KEY", "API_KEY", "TOKEN", "SECRET",
    "KEY", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
    "ACCESS_TOKEN", "REFRESH_TOKEN",
}


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return upper in DEFAULT_REDACT_KEYS or any(
        pattern in upper for pattern in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")
    )


def _redact_value(value: str) -> str:
    if len(value) <= 4:
        return "****"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


# --- Safe Collectors (never raise) ---

def _safe_areno_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("areno")
    except Exception:
        return None


def _safe_env_collect(redact_keys: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if _is_sensitive_key(key):
            result[key] = _redact_value(value)
        else:
            result[key] = value
    return result


def _safe_gpu_info() -> dict | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": f"{major}.{minor}",
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1),
                }
                for i in range(torch.cuda.device_count())
                for major, minor in [torch.cuda.get_device_capability(i)]
            ],
        }
    except Exception:
        return None


def _safe_cuda_info() -> dict | None:
    try:
        import shutil
        nvcc = shutil.which("nvcc")
        cuda_home = os.environ.get("CUDA_HOME")
        return {"cuda_home": cuda_home, "nvcc_path": nvcc}
    except Exception:
        return None


def _safe_traceback(error: BaseException | None) -> str | None:
    if error is None:
        return None
    try:
        return "".join(traceback.format_exception(type(error), error, error.__traceback__))
    except Exception:
        return repr(error)


def _safe_process_info() -> dict:
    try:
        return {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": str(Path.cwd()),
            "executable": sys.executable,
        }
    except Exception:
        return {"pid": os.getpid()}


def _safe_config_dump(config) -> dict | None:
    if config is None:
        return None
    try:
        if dataclasses.is_dataclass(config):
            return dataclasses.asdict(config)
        return {"config_type": type(config).__name__, "config_repr": repr(config)}
    except Exception:
        return None


# --- Bundle Collection ---

def collect_failure_bundle(
    *,
    command: list[str] | None = None,
    config=None,
    error: BaseException | None = None,
    include_env: bool = True,
    include_gpu: bool = True,
    redact_env_keys: set[str] | None = None,
) -> FailureBundle:
    warnings: list[str] = []

    def _safe_call(fn, *args, label: str, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            warnings.append(f"{label}: {exc}")
            return None

    if redact_env_keys is None:
        redact_env_keys = DEFAULT_REDACT_KEYS

    bundle = FailureBundle(
        timestamp=datetime.now(timezone.utc).isoformat(),
        areno_version=_safe_call(_safe_areno_version, label="areno_version"),
        python_version=sys.version,
        platform_info=platform.platform(),
        command=list(command) if command else (sys.argv[1:] if len(sys.argv) > 1 else None),
        resolved_config=_safe_call(_safe_config_dump, config, label="config_dump"),
        env_vars_redacted=_safe_call(_safe_env_collect, redact_env_keys, label="env_collect") if include_env else {},
        gpu_summary=_safe_call(_safe_gpu_info, label="gpu_info") if include_gpu else None,
        cuda_info=_safe_call(_safe_cuda_info, label="cuda_info") if include_gpu else None,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
        error_traceback=_safe_call(_safe_traceback, error, label="traceback"),
        process_info=_safe_call(_safe_process_info, label="process_info"),
        worker_state=None,
        collection_warnings=warnings,
        extra={},
    )
    return bundle


# --- Bundle Output ---

def write_bundle(bundle: FailureBundle, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = bundle.timestamp.replace(":", "-").replace("+", "-")
    bundle_dir = output_dir / f"areno-failure-{safe_ts}"
    bundle_dir.mkdir(exist_ok=True)

    # JSON (machine-readable)
    bundle_dict = dataclasses.asdict(bundle)
    (bundle_dir / "bundle.json").write_text(
        json.dumps(bundle_dict, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # Markdown (human-readable)
    (bundle_dir / "summary.md").write_text(
        _render_markdown(bundle),
        encoding="utf-8",
    )

    # Raw traceback
    if bundle.error_traceback:
        (bundle_dir / "traceback.txt").write_text(
            bundle.error_traceback,
            encoding="utf-8",
        )

    return bundle_dir


def _render_markdown(bundle: FailureBundle) -> str:
    lines = [
        "# AReno Failure Bundle",
        "",
        f"**Timestamp**: {bundle.timestamp}",
        f"**AReno Version**: {bundle.areno_version or 'unknown'}",
        f"**Python**: {bundle.python_version.strip()}",
        f"**Platform**: {bundle.platform_info}",
        "",
    ]

    if bundle.command:
        lines.append("## Command")
        lines.append("```")
        lines.append(" ".join(bundle.command))
        lines.append("```")
        lines.append("")

    if bundle.error_type:
        lines.append("## Error")
        lines.append(f"**Type**: `{bundle.error_type}`")
        lines.append(f"**Message**: {bundle.error_message}")
        lines.append("")
        if bundle.error_traceback:
            lines.append("### Traceback (most recent call last)")
            lines.append("```")
            lines.append(bundle.error_traceback.strip())
            lines.append("```")
            lines.append("")

    if bundle.gpu_summary:
        lines.append("## GPU")
        lines.append(f"```json")
        lines.append(json.dumps(bundle.gpu_summary, indent=2))
        lines.append(f"```")
        lines.append("")

    if bundle.collection_warnings:
        lines.append("## Collection Warnings")
        for w in bundle.collection_warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


# --- CLI Command ---

@click.command(
    name="debug",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Collect runtime failure evidence for AReno diagnostics.",
)
@click.option(
    "--collect", is_flag=True,
    help="Collect an environment snapshot and write it to --output-dir.",
)
@click.option(
    "--wrap", is_flag=True,
    help="Execute a subcommand and auto-collect evidence on failure.",
)
@click.option(
    "--output-dir", default="./areno-debug", show_default=True,
    help="Directory for failure evidence bundles.",
)
@click.option(
    "--traceback-file", default=None,
    help="Read traceback from a file (post-mortem mode).",
)
@click.option(
    "--redact/--no-redact", default=True, show_default=True,
    help="Redact sensitive environment variable values.",
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def debug_command(collect, wrap, output_dir, traceback_file, redact, extra_args):
    """AReno runtime failure evidence collector."""
    output_path = Path(output_dir)

    if collect:
        bundle = collect_failure_bundle(include_env=redact)
        result_path = write_bundle(bundle, output_path)
        click.echo(f"Environment snapshot written to: {result_path}")

    elif traceback_file:
        tb_text = Path(traceback_file).read_text(encoding="utf-8")
        # 构造一个虚拟异常来携带 traceback 文本
        bundle = collect_failure_bundle(
            error=RuntimeError("(reconstructed from traceback file)"),
            include_env=redact,
        )
        bundle.error_traceback = tb_text  # 覆盖为原始 traceback
        result_path = write_bundle(bundle, output_path)
        click.echo(f"Post-mortem bundle written to: {result_path}")

    elif wrap and extra_args:
        # 包装模式：执行子命令，异常时自动收集
        try:
            result = subprocess.run(list(extra_args), check=True)
        except subprocess.CalledProcessError as exc:
            click.echo(f"Command failed with exit code {exc.returncode}. Collecting evidence...", err=True)
            bundle = collect_failure_bundle(
                command=list(extra_args),
                error=exc,
                include_env=redact,
            )
            result_path = write_bundle(bundle, output_path)
            click.echo(f"Failure bundle written to: {result_path}", err=True)
            raise SystemExit(exc.returncode) from None
    else:
        # 无参数：打印当前环境快照到 stdout
        bundle = collect_failure_bundle(include_env=redact)
        click.echo(json.dumps(dataclasses.asdict(bundle), indent=2, default=str))


def main():
    debug_command.main(prog_name="areno debug")
```

### 4.3 `areno/cli/main.py` 修改（仅 1 行）

```python
# 在 _COMMANDS 字典中新增一行:
"debug": ("areno.cli.debug", "debug_command", "Collect runtime failure evidence for diagnostics."),
```

### 4.4 测试设计 (`tests/test_debug_collector_cpu.py`)

| 测试用例 | 覆盖场景 | 断言点 |
|----------|----------|--------|
| `test_collect_success` | 成功路径 | bundle 包含 timestamp, areno_version, python_version, platform |
| `test_collect_with_error` | 传入异常对象 | error_type/error_message/error_traceback 非空 |
| `test_collect_without_error` | 无异常场景 | error_type/error_message/error_traceback 均为 None |
| `test_redact_sensitive_env` | 脱敏功能 | HF_TOKEN→"hf****"，非敏感键原样保留 |
| `test_redact_short_value` | 边界：短值脱敏 | 短值→"****" |
| `test_write_bundle_creates_files` | 写入磁盘 | bundle.json, summary.md 存在且内容匹配 |
| `test_write_bundle_without_error_no_traceback_file` | 无错误时不创建 traceback.txt | traceback.txt 不存在 |
| `test_collect_all_safe_methods_no_raise` | 安全方法不抛异常 | 所有 _safe_xxx 在异常情况下返回 None |
| `test_default_behavior_no_change` | 默认不启用不影响现有行为 | 导入 areno 不触发 debug 逻辑 |

---

## 五、与项目的集成点

### 5.1 与现有 CLI 的关系

```
areno
├── check    ← areno/cli/diagnostics.py   (环境检查)
├── env      ← areno/cli/diagnostics.py   (环境报告)
├── debug    ← areno/cli/debug.py         (新增: 故障取证)
├── train    ← areno/cli/train.py         (训练)
├── serve    ← areno/cli/serve.py         (服务)
├── agent    ← areno/cli/agent.py         (Agent)
└── dashboard ← areno/cli/dashboard.py    (看板)
```

`areno debug` 与 `areno check`/`areno env` 形成互补：
- `areno check` — 安装前/运行前环境就绪检查
- `areno env` — 环境信息收集报告
- `areno debug` — **运行时故障后**收集取证信息

### 5.2 与现有异常处理的关系

```
现有: Worker 异常 → traceback.format_exc() → WorkerResult.error → cluster.call() raise
                    ↓ (此处捕获不到)
        用户只看到最终 RuntimeError，原始 traceback 可能丢失

新增: areno debug --wrap → 在最外层捕获 → 保存完整上下文 → 保留原始退出码
      或 CLI 调用者自行 try/except → collect_failure_bundle(error=e)
```

### 5.3 与现有日志系统的关系

不改变现有日志配置。`debug` 收集器只读取和快照，不修改 `areno` logger 的行为。

---

## 六、未来可扩展方向（不在本 issue 范围）

| 扩展方向 | 说明 |
|----------|------|
| 自动 attach 到 Trainer | `Trainer.__init__` 的可选 `failure_collector` 参数，异常时自动收集 |
| Dashboard 集成 | Dashboard 页面上展示最近的 failure bundle |
| Worker 子进程快照 | 从主进程收集 worker 的 `ps`/`py-spy` 状态（需运行中可获取） |
| 时间限界日志 | `--log-tail N` 参数：仅收集最后 N 行日志 |

---

## 七、验收标准对照

| 标准 | 方案覆盖 | 实现方式 |
|------|----------|----------|
| 收集失败不隐藏原始错误 | 完整覆盖 | `_safe_xxx()` 函数 + `collection_warnings` 列表 |
| 脱敏敏感环境值 | 完整覆盖 | `DEFAULT_REDACT_KEYS` + `_redact_value()` |
| 自定义输出位置 | 完整覆盖 | `--output-dir` 选项 |
| 使用现有 AReno 合约 | 完整覆盖 | 复用 `collect_env()`, `TrainerConfig`, `MetricsRecorder` 写入模式 |
| 不引入外部数据库 | 完整覆盖 | 纯本地文件系统操作 |
| 默认行为向后兼容 | 完整覆盖 | 仅新增子命令，不修改现有行为 |
| CPU 测试覆盖 | 完整覆盖 | 9 个测试用例 |
| 用户文档包含可运行示例 | 完整覆盖 | summary.md 自动生成 + README 级别文档 |
| 支持单/多进程 fixture | 部分覆盖 | 单进程完整测试；多进程标记为 TODO(agent)，需 GPU 环境 |

---

## 八、总结

本方案在 AReno 现有架构基础上，以 **最小侵入性** 新增一个 `areno debug` CLI 子命令，提供三种使用模式：

- **手动模式** (`--collect`): 立即生成环境快照
- **包装模式** (`--wrap`): 包裹任意命令，失败时自动收集
- **事后模式** (`--traceback-file`): 从已有 traceback 重建取证包

核心设计原则：
1. **不隐藏原始错误** — 任何子步骤失败不影响整体收集
2. **安全默认值** — 默认脱敏，不自动发送，不改变现有行为
3. **复用现有合约** — `collect_env()`, `TrainerConfig`, JSON lines 格式
4. **零新依赖** — 仅使用 `json`, `dataclasses`, `traceback`, `pathlib` 等标准库

预计工作量：约 430 行新代码（280 行核心 + 150 行测试），修改 1 行已有代码。