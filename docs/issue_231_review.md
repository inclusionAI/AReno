# PR Self-Review: Preflight output-directory writability and atomic writes (#231)

## 1. 我对这个 Issue 的理解

读完 issue #231 后，我认为这个需求要解决的是两件事：

- **训练启动前的输出目录可用性验证** —— 别等跑到一半才发现目录不可写
- **写入过程的原子性保证** —— 别因为中断留下半个坏文件

### 实际痛点场景

| 场景 | 后果 |
|------|------|
| `--save-path` 指向只读目录 | 训练跑到 `save_interval` 步才崩溃，算力全白费 |
| `--metrics-log-dir` 指向满盘路径 | TensorBoard writer 初始化就挂 |
| checkpoint `index.json` 写一半被 OOM kill | 留下损坏 JSON，后续加载报莫名其妙的解析错误 |

### Issue 中的关键约束

- **"Do not overwrite user files"** —— 探测文件必须独占创建，不能覆盖已有文件
- **"always remove probe files"** —— 探测后必须清理
- **"use existing public contracts"** —— 不能引入外部依赖
- **"safe default that preserves current behavior"** —— 默认开启但不改变现有行为
- **"keep the change narrow"** —— 改动要小而聚焦

---

## 2. 设计决策与思考过程

### 2.1 为什么分成 atomic_io 和 preflight_io 两个模块

我让 AI 通读项目后发现：仓库中已有 3 处手动实现了 write-to-temp + rename（`metrics.py`、`dashboard_registry.py`、`server.py`），但都是重复代码且没有异常清理。

**决策**：先提取公共工具，再在 preflight 中复用。

**分模块的理由**：职责不同——
- `atomic_io`：解决"怎么写才安全"
- `preflight_io`：解决"能不能写"

各自可以独立测试，互不依赖。

### 2.2 探测链为什么是 create→write→flush→rename→cleanup

Issue 明确要求检查这 5 个操作，而不是简单的 `os.access`。现有 `diagnostics.py` 中的 `_writable_path_check` 就用的 `os.access`，但它有盲区：

| 盲区 | `os.access` 表现 | 实际情况 |
|------|-----------------|---------|
| NFS 挂载 | 返回 True | 实际写入可能被拒绝 |
| 磁盘满 | 返回 True | 写入抛 ENOSPC |
| SELinux | 返回 True | 策略阻止写入 |

**决策**：实现完整 I/O 探测链，每步独立 try/except。

**一个取舍**：fsync 失败要不要算探测失败？我选择**不算**——tmpfs 等文件系统上 fsync 会抛 OSError 但实际写入成功，算硬失败会误报。

### 2.3 为什么 safetensors 权重文件不做原子写入

| 因素 | 分析 |
|------|------|
| 技术限制 | `save_file()` 是 C 扩展，Python 层无法包装 |
| 改动成本 | 多 rank 并行写入，做临时目录+rename 需要分布式协调，太大 |
| 风险评估 | preflight 已验证目录可写；safetensors 格式"写完才可读" |
| Issue 约束 | "keep the change narrow" |

**决策**：只对 JSON 文件做原子写入，safetensors 保持不变。

### 2.4 diagnostics 为什么要特殊处理

直接把 `_writable_path_check` 改为调用 `probe_directory_writability` 会有副作用：`areno check` 会在检查时**创建目录**。

这不合理——诊断命令不应该改系统状态。

**决策**：
- 已存在的目录 → 用真实 I/O 探测（不会创建新东西）
- 不存在的路径 → 回退到 `os.access` 检查父目录（不创建任何目录）

### 2.5 为什么 metrics.py 改用 lazy import

最初在模块顶层写了 `from areno.cli.atomic_io import atomic_write_text`，导致 `areno.api`（SDK 层）在导入时依赖 `areno.cli`（CLI 层）。

虽然当前不会循环导入（`areno.cli.__init__.py` 是空的），但 SDK 层不应知道 CLI 层存在——如果未来 `__init__.py` 加了导入逻辑就可能出问题。

**决策**：改为函数内 lazy import，依赖从"导入时硬绑定"变成"调用时临时借用"。

### 2.6 renamed_name 的构建方式

原来的代码：
```python
renamed_name = probe_name.replace(".tmp", ".renamed")
```

`str.replace` 会替换**所有**出现的 `.tmp`。如果用户自定义前缀含 `.tmp`（如 `.tmp_probe_`），会产生错误文件名。

**决策**：改为 f-string 显式构建，无歧义。

---

## 3. 自审：发现的问题与修复

### 第一轮自查（代码生成后立即 review）

| # | 位置 | 问题 | 严重性 | 修复 |
|---|------|------|--------|------|
| 3.1 | `preflight_io.py:18` | `field` 导入未使用 | 低 | 删除 |
| 3.2 | `diagnostics.py:255` | `_writable_path_check` 在 `areno check` 时会创建目录 | 中 | 已存在目录用 probe；不存在的用 `os.access` |
| 3.3 | `preflight_io.py:85-183` | `KeyboardInterrupt` 时 `fh` 文件句柄泄漏 | 低 | 添加 `fh = None` 跟踪，`finally` 中关闭 |
| 3.4 | `preflight_io.py:45` | `# noqa: RUF100` 无对应 lint 规则 | 微 | 删除 |

### 第二轮自查（项目负责人建议后的深度 review）

| # | 位置 | 问题 | 严重性 | 修复 |
|---|------|------|--------|------|
| 3.5 | `atomic_io.py:13` | `import os` 未使用 | 低 | 删除 |
| 3.6 | `metrics.py:20` | 顶层 import 造成 SDK→CLI 跨层依赖 | 中 | 改为 lazy import（详见 2.5） |
| 3.7 | `preflight_io.py:81` | `str.replace` 隐患 | 中 | 改为 f-string（详见 2.6） |
| 3.8 | `test_metrics_cpu.py:127` | 修复 3.6 后 mock.patch 目标失效，测试 silently 通过但没测到 | **高** | 改为 `mock.patch.object(atomic_io_mod, ...)` |
| 3.9 | `test_preflight_io_cpu.py:190` | 注释自相矛盾 | 低 | 重写注释 |

> **3.8 是最危险的 bug**：测试看起来通过了但其实什么都没 patch 到。如果不做第二轮 review，这个问题不会被发现。

### 确认无问题的部分

- **`atomic_io.py` 的 `except BaseException`**：故意捕获 `BaseException` 让 `KeyboardInterrupt` 也触发清理；清理本身用 `try/except OSError` 包裹不掩盖原始异常
- **`train.py` 集成位置**：`_preflight_output_directories` 在 config 构建后、`run()` 之前调用，时序正确
- **`dashboard_registry.py` / `server.py`**：保留原有 `try/except Exception: pass` 策略，只换了内部写入方式
- **`io.py` / `common.py`**：lazy import 不影响分布式 checkpoint 流程

---

## 4. 测试运行记录

### 4.1 原子写入单元测试

```bash
python3 -m unittest tests.test_atomic_io_cpu -v
```

**结果：8 passed**

```
test_atomic_write_bytes_creates_file ... ok
test_atomic_write_json_ensure_ascii ... ok
test_atomic_write_json_roundtrip ... ok
test_atomic_write_no_partial_file_on_failure ... ok
test_atomic_write_text_cleans_temp_on_failure ... ok
test_atomic_write_text_cleans_temp_on_replace_failure ... ok
test_atomic_write_text_creates_file ... ok
test_atomic_write_text_overwrites_existing ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.013s
OK
```

> 特别关注 `test_atomic_write_no_partial_file_on_failure`：验证写入失败时目标文件保持原有内容且无 `.tmp` 残留——这是原子写入最核心的保证。

### 4.2 Preflight 探测单元测试

```bash
python3 -m unittest tests.test_preflight_io_cpu -v
```

**结果：19 passed**

```
test_probe_success_on_writable_dir ... ok
test_probe_success_on_nested_missing_dir ... ok
test_probe_results_deterministic ... ok
test_probe_fails_on_readonly_dir ... ok
test_probe_fails_on_existing_file ... ok
test_probe_fails_on_disk_full ... ok
test_probe_fails_on_quota_exceeded ... ok
test_probe_cleans_up_on_success ... ok
test_probe_cleans_up_on_failure ... ok
test_probe_cleans_up_on_keyboard_interrupt ... ok
test_probe_does_not_overwrite_user_files ... ok
test_probe_concurrent_creation ... ok
test_probe_disabled_returns_skipped ... ok
test_probe_none_path_is_skipped ... ok
test_probe_empty_path_is_skipped ... ok
test_probe_with_custom_prefix ... ok
test_probe_paths_multiple ... ok
test_format_probe_results_contains_stage_and_operation ... ok
test_format_probe_results_json_is_valid_json ... ok

----------------------------------------------------------------------
Ran 19 tests in 0.030s
OK
```

> `test_probe_cleans_up_on_keyboard_interrupt` 是 review 时特别关注的——最初代码中断时不关闭文件句柄，修复后此测试验证中断场景下探测文件也被清理。

### 4.3 合并运行

```bash
python3 -m unittest tests.test_atomic_io_cpu tests.test_preflight_io_cpu -v
```

**结果：27 passed in 0.041s**

### 4.4 边界情况验证

| 场景 | 验证方式 | 结果 |
|------|---------|------|
| 可写目录通过 | `test_probe_success_on_writable_dir` | 通过 |
| 只读目录拦截 | `os.chmod(dir, 0o444)` | 通过，报告 `operation=create` |
| 磁盘满 (ENOSPC) | mock `fh.write` 抛 `OSError(28)` | 通过，报告 `operation=write` |
| 配额超限 (EDQUOT) | mock `fh.write` 抛 `OSError(122)` | 通过 |
| 并发文件冲突 | 预创建同名前缀文件 | 通过，UUID 重试成功 |
| Ctrl+C 中断 | mock `Path.replace` 抛 `KeyboardInterrupt` | 通过，文件被清理 |
| 嵌套缺失目录 | 探测 `tmp/a/b/c` | 通过，目录自动创建 |
| 路径是文件 | 指向已有文件 | 通过，报告 "exists but is a file" |
| 不覆盖用户文件 | 目录中放用户文件后探测 | 通过，内容不变 |
| 探测文件清理 | 成功/失败后检查残留 | 通过，无 `.areno_preflight_*` |
| 禁用时不探测 | `enabled=False` + `--no-preflight-io` | 通过 |

### 4.5 Python 3.10+ 测试

当前环境 Python 3.9，项目用 `@dataclass(slots=True)`（3.10+），涉及 `areno.api` import 链的测试无法运行。通过 `ast.parse` 验证语法：

```
OK  tests/test_metrics_cpu.py          (+2 测试)
OK  tests/test_train_cli_config_cpu.py (+8 测试)
OK  tests/test_cli_diagnostics_cpu.py  (+2 测试, 1 更新)
```

---

## 5. 验收标准对照

| Issue 验收标准 | 如何满足 |
|---------------|---------|
| 只读目录、配额错误、并发创建、中断探测、嵌套缺失目录 | 探测链每步独立 try/except + `PreflightProbeResult` 记录 operation/path/error；7 个测试覆盖 |
| 报告失败的 operation 和 path | `format_probe_results` 格式化输出含 stage/operation/path/error |
| 使用现有 AReno 契约，无外部数据库 | 纯标准库（json/os/uuid/dataclasses/pathlib），无新依赖 |
| 默认行为向后兼容 | `save_path=None` 跳过；`--no-preflight-io` 逃生阀门；成功时无输出；3 个测试验证 |
| 测试覆盖成功、无效输入、边界/失败 | 27 个可运行测试 + 14 个 3.10+ 测试语法验证 |
| 文档含可运行示例 | `docs/cli/training.rst` 新增 Preflight I/O checks 章节 |

---

## 6. 改动文件

### 新增（4 个）

| 文件 | 内容 |
|------|------|
| `areno/cli/atomic_io.py` | `atomic_write_text`/`bytes`/`json`：write-to-temp + rename + 异常清理 |
| `areno/cli/preflight_io.py` | `probe_directory_writability` 完整探测链 + 配置/结果/格式化 |
| `tests/test_atomic_io_cpu.py` | 8 个测试 |
| `tests/test_preflight_io_cpu.py` | 19 个测试 |

### 修改（13 个）

| 文件 | 改了什么 |
|------|---------|
| `areno/cli/train.py` | +`_preflight_output_directories()` + CLI 选项；dashboard config 改原子写入 |
| `areno/cli/diagnostics.py` | `_writable_path_check` 已存在目录用探测，不存在用 `os.access` |
| `areno/api/metrics.py` | `record_dashboard_state` 改用 `atomic_write_text`（lazy import） |
| `areno/cli/dashboard_registry.py` | `_write_registry` 改用 `atomic_write_json` |
| `areno/dashboard/server.py` | `_save_state` 改用 `atomic_write_json` |
| `areno/engine/checkpoints/io.py` | `index.json` 改用 `atomic_write_json` |
| `areno/engine/checkpoints/common.py` | passthrough `index.json` 改用 `atomic_write_json` |
| `tests/test_metrics_cpu.py` | +2 测试 |
| `tests/test_train_cli_config_cpu.py` | +8 测试 |
| `tests/test_cli_diagnostics_cpu.py` | +2 测试，1 更新 |
| `docs/cli/training.rst` | +Preflight I/O checks 章节 |
| `docs/cli/diagnostics.rst` | 更新 writable 检查描述 |
| `CODEMAP.md` | +preflight_io / atomic_io 条目 |