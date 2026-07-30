# Issue #238 需求分析与系统分析文档

> **Issue**: https://github.com/inclusionAI/AReno/issues/238
> **标题**: 生产可操作的训练非有限值报告（NaN/Inf）
> **作者**: 鸿寰 (honghuan.tb)
> **日期**: 2026-07-30

---

## 一、需求分析

### 1.1 Issue 原文核心诉求

Issue #238 要求实现一个**聚焦的、独立可审查的非有限值训练报告能力**。核心需求可拆解为以下几个维度：

| 维度 | Issue 原文要求 | 当前状态 |
|------|---------------|---------|
| **检测覆盖** | loss、gradients、advantages、rewards 中的 NaN/Inf | ✅ loss/grad/param/optimizer 已检测；❌ advantages/rewards 未检测 |
| **Skip unsafe update** | 检测到非有限值时跳过不安全的参数更新 | ❌ 当前 NaN 后仍执行 `optimizer.step()` |
| **报告内容** | 报告 update、stage、offending metric、bounded recent metric context | ✅ 已有 step/phase/loss/grad_norm/recent_losses |
| **受控终止** | controlled termination（报告后受控终止） | ❌ 当前仅 print 报告，训练继续 |
| **跨 rank 保持** | preserve the original failure across ranks | ❌ 当前各 rank 独立检测，未跨 rank 协调 |
| **用户显式启用** | The user enables or invokes the feature with explicit inputs | ❌ 当前无条件启用，无配置开关 |
| **输入验证** | validates those inputs before expensive model or worker initialization | ❌ 无输入验证 |
| **默认向后兼容** | Default behavior remains backward compatible | ⚠️ 当前检测无条件启用（改变了默认行为） |
| **不泄露数据** | avoid dumping full samples or tensors | ✅ 报告只含聚合统计 |
| **人读+结构化输出** | both human-readable and structured output | ✅ terminal report + JSON file |
| **CLI 暴露** | output through existing logs, metrics, artifacts, CLI output | ❌ 无 CLI flag |

### 1.2 需求分解为功能点

根据 Issue 的 Acceptance Criteria，需要实现的功能点如下：

#### F1. Skip Unsafe Update（核心缺失）

**需求**: 当 loss、gradient、advantages 或 rewards 检测到 NaN/Inf 时，跳过当前 optimizer step，不更新参数。

**分析**:
- 现有代码中 `TrainStats.stepped: bool` 已是合法协议字段，`merge_train_stats` 用 AND 逻辑聚合跨 rank 结果
- `TrainingManager._train_step` 中 `stepped = allow_step`，NaN 检测不会改变它
- 需要在 `optimizer.step()` 前判断：若检测到非有限值，设 `stepped=False`，跳过 step，不递增 `_global_step`
- 梯度已经被 `backward()` 计算但需被 `zero_grad()` 清除，不能 retry 当前 step

#### F2. Controlled Termination（核心缺失）

**需求**: 报告后受控终止训练，而非继续跑下一步。

**分析**:
- 两种策略可选：
  - **策略 A**: 检测到 NaN/Inf 立即终止（raise exception）
  - **策略 B**: 可配置容忍次数（如连续 N 步 NaN 才终止）
- Issue 原文说 "controlled termination"，倾向于 NaN 后受控停止
- 需要一个配置开关控制行为：`skip_update`（跳过但继续）vs `terminate`（终止训练）

#### F3. Advantages / Rewards 检测（缺失）

**需求**: 除了 loss 和 gradient，还需检测 advantages 和 rewards 中的 NaN/Inf。

**分析**:
- advantages 和 rewards 不在 `TrainingManager` 中，它们在 trainer 层（`policy_only.py` 等）计算
- rewards 由外部 `reward_fn` 计算，结果传入 trainer
- advantages 在 trainer 的 `_materialize_train_batch` 中从 rewards 计算
- 检测点应在 trainer 层，在 `train_batch` 传入 backend 之前

#### F4. 跨 Rank 一致性（缺失）

**需求**: 跨 rank 保持原始失败，即所有 rank 对"是否跳过/终止"达成一致。

**分析**:
- 当前每个 rank 独立检测自己的 loss/grad，可能不同 rank 结果不同
- 需要一个 `all_reduce` 操作让所有 rank 对 "是否有非有限值" 达成一致
- 合适的位置：grad_norm 计算后（已有 `all_reduce`）、skip update 决策前

#### F5. 用户可配置 + CLI 暴露（缺失）

**需求**: 用户通过显式输入启用功能，有 safe default 保持向后兼容。

**分析**:
- 需要在 `TrainerConfig` 增加配置字段
- 需要在 `areno/cli/train.py` 增加 CLI flag
- 默认值应保持向后兼容（Issue 说 "safe default that preserves current behavior"）
- 当前代码无条件启用检测，需改为默认行为可关闭或可配置

#### F6. 测试规范（部分缺失）

**需求**: CPU 测试覆盖成功路径、无效输入、边界值、禁用/默认行为、确定性输出；集成测试跨模块；断言字段和错误消息。

**分析**:
- 现有测试在 `test_non_finite_report.py`（根目录），不符合 `tests/test_*_cpu.py` 命名约定
- 现有测试是裸 pytest 函数，不使用 `unittest.TestCase`
- 缺少 skip update 行为测试、默认行为兼容性测试、跨模块集成测试
- 缺少 fault-injection 测试（Issue 明确要求）

#### F7. 文档（部分完成）

**需求**: 文档包括用户选项、输入契约、默认值、输出字段、限制、可复制示例。

**分析**:
- 已有 `docs/non_finite_detection.rst`（设计文档 + 验证过程）
- 缺少 CLI 选项文档、用户操作手册、troubleshooting 页面链接

---

## 二、系统分析

### 2.1 系统架构定位

```
用户层
  areno train --non-finite-skip-update ...     CLI flag
       │
配置层
  TrainerConfig.non_finite_skip_update: bool   顶层配置
  TrainerConfig.non_finite_terminate: bool     顶层配置
       │
Trainer 层 (areno/api/trainers/)
  PolicyOnlyTrainer._materialize_train_batch()
    ↓ 检测 advantages/rewards NaN/Inf          ← F3 新增检测点
  self.areno.train(batch, loss_fn, ...)         调用 backend
       │
Backend 层 (areno/api/backend/)
  ArenoBackend.train()                          透传
       │
Engine 层 (areno/engine/api.py)
  ArenoEngine.train() → cluster.call(Op.TRAIN)
       │
Worker 层 (areno/engine/worker.py)
  ArenoWorker.train(payload) → TrainingManager.train()
       │
TrainingManager._train_step()                   ← F1/F2/F4 核心改动点
  1. check_loss_non_finite(loss)                快速检查
  2. backward() + accumulate_grads()
  3. grad_norm = _grad_norm(...)                已有 all_reduce
  4. detect_non_finite(...)                      深度检查
  5. ★ if non_finite: skip optimizer.step()     ← F1 新增
  6. ★ if non_finite and terminate: raise        ← F2 新增
  7. ★ all_reduce(non_finite_flag) for TP/DP     ← F4 新增
```

### 2.2 现有代码评估

#### 已完成部分（质量评估）

| 组件 | 文件 | 完成度 | 质量评价 |
|------|------|--------|---------|
| 检测核心 | `areno/engine/runtime/non_finite.py` | 90% | 设计完整，双层检测 + 报告 + JSON + 原因推断 |
| Actor 注入 | `areno/engine/training.py:96-119` | 50% | 检测+报告已接入，缺 skip update 和 terminate |
| Critic 注入 | `areno/engine/roles.py:476-494` | 40% | 检测+报告已接入，缺 skip update、metrics 注入、定期检查 |
| 模块导出 | `areno/engine/runtime/__init__.py` | 100% | 正确导出 |
| 单元测试 | `test_non_finite_report.py` | 40% | 位置不规范，缺 skip/terminate/默认行为测试 |
| 文档 | `docs/non_finite_detection.rst` | 70% | 验证文档详尽，缺用户操作文档 |

#### 缺失部分（对照 Issue 要求）

| 缺失项 | 影响范围 | 严重程度 |
|--------|---------|---------|
| Skip unsafe update | `training.py`, `roles.py` | **高** — Issue 核心要求 |
| Controlled termination | `training.py`, `roles.py`, `trainer` | **高** — Issue 核心要求 |
| Advantages/rewards 检测 | `areno/api/trainers/policy_only.py` | **中** — Issue 要求检测 |
| 跨 rank 一致性 | `training.py` | **高** — 分布式正确性 |
| 配置项 + CLI flag | `trainer_config.py`, `cli/train.py` | **高** — 向后兼容性 |
| Fault-injection 测试 | `tests/` | **高** — Issue 明确要求 |
| 测试规范迁移 | `tests/` | **中** — CI 可见性 |
| 日志统一 | `non_finite.py`, `training.py` | **低** — 用 `print(stderr)` 而非 `logging` |

### 2.3 关键约束分析

#### C1. `_merge_metrics` 契约

`_merge_metrics` 通过 `_metrics_to_float` 把所有值强制转为 `float`。任何注入 metrics dict 的值必须是：
- `float` / `int` — 直接转换
- `torch.Tensor` — `float(value.detach().float().cpu())`
- **不能是** `str`、`None`、`list`、`dict` — 会抛 `ValueError`

当前 `NonFiniteReport.to_dict()` 已严格遵守此契约。新增的 metrics 字段（如 `non_finite_skipped_update`）也必须是 float。

#### C2. 分布式 All-Reduce 约束

要让所有 TP/DP rank 对 "是否跳过/终止" 达成一致：

- 需要在 `detect_non_finite` 之后、`optimizer.step()` 之前插入一个 `all_reduce` 操作
- 用一个 `torch.tensor([0或1], device=worker.device)` 做布尔 OR
- `dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=ctx.group)` — TP group
- 若 `ctx.dp_size > 1`：再对 DP group 做一次
- 任一 rank 检测到 NaN → 所有 rank 都跳过

#### C3. Skip Update 的语义细节

当检测到 NaN 并决定 skip 时：

1. **不调用** `optimizer.step()`
2. **不递增** `worker._global_step`
3. **调用** `optimizer.zero_grad(set_to_none=True)` — 清除被污染的梯度
4. **设置** `stepped = False`
5. **不调用** `_lr_for_step` — LR 不变
6. 返回的 metrics 中标记 `non_finite_skipped_update = 1.0`

trainer 层收到 `stepped=False` 后，`merge_train_stats` 的 AND 逻辑会保证跨 rank 一致。

#### C4. Controlled Termination 的实现选择

```
方案 A: raise NonFiniteTrainingError(...)
  → 后端 worker 的异常处理捕获
  → 通过 cluster 通信返回到 trainer
  → trainer 传播到 CLI
  → CLI 打印报告 + 退出码非零

方案 B: 返回特殊 metrics 标记
  → trainer 检查 metrics 中的 non_finite_terminate 字段
  → trainer 主动 break 训练循环
  → 正常关闭资源
```

方案 A 更直接但需要确保资源清理（`finally` 块中的 `close()`）。
方案 B 更优雅但需要 trainer 层感知非有限值语义。

推荐**方案 A**：定义 `NonFiniteTrainingError(RuntimeError)`，在 `TrainingManager._train_step` 中 raise，确保 `Trainer.fit()` 的 `finally: self.areno.close()` 正常执行。

#### C5. 配置传递链路

```
CLI flag: --non-finite-skip-update / --non-finite-terminate
    ↓
TrainerConfig.non_finite_skip_update: bool = False   ← 默认 False 保持兼容
TrainerConfig.non_finite_terminate: bool = False     ← 默认 False 保持兼容
    ↓
TrainerConfig.areno_config() → ArenoConfig(runtime={...})
    ↓
RuntimeConfig.non_finite_skip_update: bool = False
    ↓
EngineConfig.runtime → ArenoWorker.config.runtime
    ↓
TrainingManager._train_step 读取 worker.config.runtime.non_finite_skip_update
```

需要在 `RuntimeConfig`（`areno/engine/config.py`）增加字段，因为 `TrainerConfig.areno_config()` 通过 `runtime={}` dict 传递运行时配置。

### 2.4 模块影响范围

| 模块 | 文件 | 改动类型 | 改动内容 |
|------|------|---------|---------|
| 检测核心 | `areno/engine/runtime/non_finite.py` | 扩展 | 增加 `NonFiniteTrainingError`；`detect_non_finite` 增加 advantages/rewards 检测 |
| Runtime 导出 | `areno/engine/runtime/__init__.py` | 扩展 | 导出新异常类 |
| Actor 训练 | `areno/engine/training.py` | 修改 | 增加 skip update 逻辑、terminate 逻辑、跨 rank all_reduce |
| Critic 训练 | `areno/engine/roles.py` | 修改 | 同 actor，加 skip/terminate/all_reduce、metrics 注入 |
| Engine 配置 | `areno/engine/config.py` | 扩展 | `RuntimeConfig` 增加 `non_finite_skip_update` 和 `non_finite_terminate` 字段 |
| Trainer 配置 | `areno/api/trainer_config.py` | 扩展 | `TrainerConfig` 增加配置字段 + `areno_config()` 传递 |
| CLI 入口 | `areno/cli/train.py` | 扩展 | 增加 `--non-finite-skip-update` / `--non-finite-terminate` flag |
| Advantages 检测 | `areno/api/trainers/policy_only.py` | 扩展 | 在 `_materialize_train_batch` 中检测 advantages/rewards |
| 测试 | `tests/test_non_finite_detection_cpu.py` | 新建 | 迁移+扩展为 unittest 风格 CPU 测试 |
| 集成测试 | `tests/test_non_finite_integration_cpu.py` | 新建 | fault-injection 跨模块测试 |
| 文档 | `docs/non_finite_detection.rst` | 扩展 | 增加 CLI 选项、用户流程、troubleshooting 链接 |

### 2.5 数据流分析

#### 正常训练流（无 NaN）

```
train_step:
  forward → loss (finite) → backward → grad_norm → detect (None) → clip → step → +1 → return metrics
```

#### NaN 检测流（当前实现）

```
train_step:
  forward → loss (NaN) → backward → grad_norm (NaN) → detect (report) → print(stderr) → clip → step(!!!) → +1 → return metrics
                                                    ↑ 问题：NaN 参数被更新了
```

#### NaN 检测流（目标实现 - skip update）

```
train_step:
  forward → loss (NaN) → backward → grad_norm (NaN) → detect (report) → print(stderr)
    → all_reduce(non_finite_flag)  ← 跨 rank 一致
    → if skip_update: zero_grad → stepped=False → 不 step → 不 +1 → return metrics{non_finite_skipped=1.0}
    → if terminate: zero_grad → raise NonFiniteTrainingError
```

#### NaN 检测流（advantages/rewards 层）

```
trainer._materialize_train_batch:
  compute rewards → check NaN/Inf → if bad: report + skip batch or terminate
  compute advantages → check NaN/Inf → if bad: report + skip batch or terminate
  pass clean batch to backend.train()
```

### 2.6 测试策略分析

#### Issue 要求的测试矩阵

| 测试类别 | 覆盖场景 | 实现方式 |
|---------|---------|---------|
| 成功路径 | 正常训练不产生报告 | `test_normal_no_report` (已有) |
| 无效输入 | NaN loss / Inf loss | `test_loss_nan` / `test_loss_inf` (已有) |
| 边界值 | 参数 Inf、梯度爆炸 | `test_param_inf` / `test_grad_explosion` (已有) |
| 默认行为兼容 | 配置关闭时不检测、不 skip | **新增** |
| Skip update | NaN 时参数不更新 | **新增** fault-injection |
| Terminate | NaN 时训练终止 | **新增** fault-injection |
| 跨 rank 一致 | 模拟 TP 多 rank | **新增** 用 fake dist 上下文 |
| 确定性输出 | 报告字段断言 | **新增** 断言 JSON 字段 |
| 跨模块集成 | trainer → engine → training | **新增** stub backend |

#### Fault-Injection 测试设计

Issue 明确要求："Inject each non-finite category, verify parameters do not update, preserve the original failure across ranks, and avoid dumping full samples or tensors."

```
test_inject_nan_loss_skip_update:
  1. 创建 tiny model + optimizer
  2. 注入 NaN 到 loss 计算结果
  3. 调用 detect_non_finite → 得到 report
  4. 验证 optimizer.step() 未被调用（参数值不变）
  5. 验证 metrics 包含 non_finite_skipped_update=1.0
  6. 验证 report 不含 raw tensor dump

test_inject_nan_grad_skip_update:
  类似但注入梯度 NaN

test_default_behavior_unchanged:
  1. non_finite_skip_update=False (默认)
  2. NaN loss → 不 skip，不 terminate
  3. 参数被更新（保持现有行为）
  4. 仍产生报告（检测仍启用）
```

### 2.7 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Skip update 改变训练步数 | 影响 LR schedule 对齐 | 不递增 `_global_step`，LR schedule 不受影响 |
| Terminate 异常未清理资源 | GPU 内存泄漏 | `Trainer.fit()` 的 `finally: close()` 保证清理 |
| 跨 rank all_reduce 死锁 | TP/DP rank 不一致 | 在 `stepped` 块内做（所有 rank 都到达此点） |
| backward() 后 NaN 梯度 | 已计算的梯度被污染 | `zero_grad(set_to_none=True)` 清除 |
| advantages NaN 在 trainer 层 | 需要修改 trainer 不只是 engine | 在 `_materialize_train_batch` 中检测，batch 级跳过 |
| 向后兼容 | 现有用户训练行为改变 | 默认 `skip_update=False`，仅检测+报告 |

---

## 三、实施建议优先级

### Phase 1: 核心功能（Issue 必须）

1. **Skip unsafe update** — `training.py` + `roles.py` 中 NaN 时跳过 `optimizer.step()`
2. **Controlled termination** — 定义异常，config 控制是否终止
3. **跨 rank 一致性** — `all_reduce` non_finite flag
4. **配置项 + CLI** — `RuntimeConfig` + `TrainerConfig` + CLI flag
5. **Advantages/rewards 检测** — trainer 层检测

### Phase 2: 测试与验证（Issue 必须）

6. **迁移测试** — `test_non_finite_report.py` → `tests/test_non_finite_detection_cpu.py`
7. **Fault-injection 测试** — skip update、terminate、默认行为
8. **集成测试** — 跨模块 CPU 测试

### Phase 3: 文档与完善（Issue 必须）

9. **文档更新** — CLI 选项、用户流程、troubleshooting
10. **日志统一** — `print(stderr)` → `logger.warning()`

---

## 四、总结

当前 PR（#316）已完成 Issue #238 的**约 50%**：

- ✅ **已完成**: 检测核心模块、双层检测策略、终端报告、JSON 文件输出、原因推断、actor/critic 注入点、验证文档
- ❌ **未完成**: Skip unsafe update、Controlled termination、跨 rank 一致性、advantages/rewards 检测、配置项 + CLI flag、fault-injection 测试、测试规范迁移

Issue 的 Acceptance Criteria 中最关键的三个要求：
1. "Inject each non-finite category, verify parameters do not update" — **未实现 skip update**
2. "Default behavior remains backward compatible" — **当前无条件启用检测，无法关闭**
3. "Focused automated tests cover success, invalid input, and one boundary/failure path" — **测试不规范、不完整**

下一步应优先实现 Phase 1（skip update + terminate + config + CLI），然后补充 Phase 2（测试），最后完成 Phase 3（文档）。
