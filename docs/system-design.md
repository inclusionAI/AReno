# 系统分析文档 — 按长度分组组成 Batch 系统

> **文档版本**: V3.0（AReno 对齐版）
> **创建日期**: 2026 年 7 月 28 日
> **创建人**: 549221 (李航/舟枢)
> **审核状态**: 待审核

## 文档修订记录

| 版本 | 日期 | 作者 | 修订内容 | 审核状态 |
|------|------|------|----------|----------|
| V1.0 | 2026-07-28 | 549221 | 初稿 | 待审核 |
| V2.0 | 2026-07-28 | 549221 | 评审修订：统一 Java 版本为 17；明确系统形态为"库 + CLI 入口"；修正分桶边界重叠/遗漏；缓存键改为文本哈希；统一 BucketStrategy 接口；明确样本完整性边界；补充 batch shuffle 机制；修正 OOM 处理策略；性能测试改为基准对比；补全风险评估 | 待审核 |
| V3.0 | 2026-07-28 | 549221 | AReno 对齐修订：技术栈从 Java 17 改为 Python 3.10+；系统形态从独立 Java 项目改为 AReno 内嵌模块；实体类对齐 `TrainSequence`/`PromptItem`；Tokenizer 对齐 HF tokenizer 与 `encode_generation_prompt`；padding 发生点对齐 `pad_rows`/`pad_rollout_rows`；配置入口对齐 `TrainerConfig` 与 `train.py` CLI；集成点明确为 `SFTTrainer._iter_train_batches` 与 `Trainer.load_prompt_batches` | 待审核 |

---

## 目录

1. [项目概述](#1-项目概述)
2. [需求分析](#2-需求分析)
3. [系统设计](#3-系统设计)
4. [详细设计](#4-详细设计)
5. [接口设计](#5-接口设计)
6. [数据设计](#6-数据设计)
7. [性能设计](#7-性能设计)
8. [异常处理](#8-异常处理)
9. [测试策略](#9-测试策略)
10. [部署方案](#10-部署方案)
11. [风险评估](#11-风险评估)
12. [附录](#12-附录)

---

## 1. 项目概述

### 1.1 项目背景

在大模型训练过程中，输入样本的长度通常差异较大。AReno 现有的 batch 组成逻辑是**顺序切片**：

- `SFTTrainer._iter_train_batches`（`areno/api/trainers/sft.py`）按 `batch_size` 顺序累积 `TrainSequence` 后整批 yield；
- `Trainer.load_prompt_batches`（`areno/api/trainer.py`）同样按数据集游标顺序填充 `PromptBatch`，仅按 `max_prompt_tokens` 跳过长样本，不做长度分组。

随后 `pad_rows` / `pad_rollout_rows`（`areno/engine/runtime/common.py`）按 batch 内最长样本做右侧 padding，导致：

- **Padding 浪费**: batch 内样本按最长样本 padding，短样本填充大量无效 token
- **计算资源浪费**: GPU 需要处理大量 padding token，降低训练效率
- **训练成本增加**: 额外的计算导致训练时间延长

> **说明**: AReno 已支持 packed varlen 布局（`_pack_train_data`，`areno/engine/runtime/train_step.py`），可在 forward 阶段去除 pad token。但该机制只在**注意力内核层**消除 padding 计算，**batch 组成阶段**仍按顺序填充，无法保证 batch 内长度同质，因此 padding 比例仍由 batch 组成策略决定。本系统聚焦 batch 组成阶段的长度分组优化。

**解决方案**: 在 batch 组成阶段按 token 长度分组，将长度相近的样本组成 batch，从源头降低 padding 比例。

### 1.2 项目目标

| 目标类型 | 目标描述 | 衡量指标 |
|----------|----------|----------|
| 功能目标 | 实现按长度分组的 batch 组成系统 | 支持 100K+ 样本处理 |
| 性能目标 | 降低 padding 比例 | 从 45% 降至 20% 以下 |
| 效率目标 | 减少训练时间 | 训练速度提升 15-25% |
| 质量目标 | 保证数据完整性 | 样本丢失率 < 0.01%（默认配置下为 0%） |

> **修订说明**: 质量目标明确"默认配置下为 0%"，因为 `drop_last_batch=true` 或 `truncate_strategy=DROP` 时会有意丢弃样本，此时丢失率由用户配置决定，不属于"丢失"。

### 1.3 适用范围

| 适用场景 | 说明 | 现有入口 |
|----------|------|----------|
| 大模型 SFT 训练 | 指令微调数据 batching | `SFTTrainer._iter_train_batches` |
| DPO 偏好训练 | chosen/rejected 对 batching | `DPOTrainer._iter_train_batches` |
| RLHF 在线 rollout | rollout 前 prompt batching | `Trainer.load_prompt_batches` |
| 预训练数据处理 | 语料库 batching | 同 SFT 路径 |

> **修订说明**: "多轮对话训练"并入 RLHF/在线 rollout 场景（对应 `areno/api/agentic.py` 的 `AgentBatch`）。适用场景列新增"现有入口"，指明本系统需嵌入的 AReno 代码位置。

### 1.4 系统形态

> **V3.0 修订**（对齐 AReno）

本系统是 **AReno 的内嵌模块**，而非独立项目：

- **核心库**: 新增 `areno/api/length_grouped.py`，提供长度分组 batching API，供各 trainer 调用。
- **配置入口**: 在 `TrainerConfig`（`areno/api/trainer_config.py`）新增长度分组配置字段；在 `train.py` CLI 新增对应开关。
- **集成方式**: 改造 `SFTTrainer._iter_train_batches` 与 `Trainer.load_prompt_batches`，在 batch 组成阶段调用本模块；**不改动** `pad_rows` / `_pack_train_data` 等下游 padding/packing 逻辑。
- **复用现有能力**: Tokenizer 直接使用 AReno 已加载的 HF tokenizer（`Trainer.get_tokenizer()`）；数据集加载复用 `train.py` 的 `_load_dataset_for_training`。

本系统 **不是** 一个独立可执行工具，不单独发布 artifact，部署方式随 AReno 主流程。

### 1.5 名词解释

| 术语 | 定义 |
|------|------|
| Token | 文本最小处理单元，由 HF tokenizer 切分 |
| Batch | 一次训练迭代处理的样本集合，对应 `list[TrainSequence]` 或 `PromptBatch` |
| Padding | 将短样本填充至与最长样本等长，由 `pad_rows` 执行 |
| Bucket | 长度区间，用于样本分组 |
| Grouping ID | 分组标识，用于区分不同分组策略 |
| Batch Shuffle | 组 batch 完成后对 batch 列表做随机打乱，保留桶内长度相近但打乱 batch 间顺序 |
| Packed varlen | AReno 现有的去除 pad token 的 packed 布局，由 `_pack_train_data` 生成 |

---

## 2. 需求分析

### 2.1 功能需求

#### FR-001: 样本长度计算

| 需求项 | 描述 |
|--------|------|
| 需求 ID | FR-001 |
| 需求名称 | 样本长度计算 |
| 优先级 | P0 |
| 需求描述 | 系统应能计算每个样本的 token 长度 |
| 输入 | 样本文本（SFT 的 prompt+response，或 rollout 的 prompt） |
| 输出 | 样本的 token 长度 |
| 处理逻辑 | 复用 `encode_generation_prompt`（`areno/api/tokenizer.py`）对文本编码，返回 `len(token_ids)` |
| 性能要求 | 10K 样本/秒 |
| 验收标准 | 长度计算准确率 100% |
| Null 处理 | 输入为 None 或空字符串时返回 0，不抛异常（见 5.1 接口契约） |

> **V3.0 修订**: 处理逻辑明确复用 `encode_generation_prompt`，与 `SFTTrainer` / `Trainer.load_prompt_batches` 现有 tokenization 路径一致，不引入新的 tokenizer 调用方式。

#### FR-002: 长度分桶

| 需求项 | 描述 |
|--------|------|
| 需求 ID | FR-002 |
| 需求名称 | 长度分桶 |
| 优先级 | P0 |
| 需求描述 | 系统应能按 token 长度将样本分配到不同的桶中 |
| 输入 | 带有 token 长度的样本列表 |
| 输出 | 按桶 ID 分组的样本 dict |
| 处理逻辑 | 遍历样本，根据长度匹配对应的桶区间 |
| 配置要求 | 支持自定义桶边界 |
| 验收标准 | 每个样本必须且只能分配到一个桶；桶区间无缝衔接且不重叠 |

> **V3.0 修订**: 输出类型从 Java `Map<Integer, List<Sample>>` 改为 Python `dict[int, list[...]]`，具体 value 类型由集成点决定（SFT 为 `list[TrainSequence]`，rollout 为 `list[PromptItem]`）。

#### FR-003: Batch 组成

| 需求项 | 描述 |
|--------|------|
| 需求 ID | FR-003 |
| 需求名称 | Batch 组成 |
| 优先级 | P0 |
| 需求描述 | 系统应能在每个桶内按 batch_size 组成 batch |
| 输入 | 按桶分组的样本 dict |
| 输出 | Batch 列表（`list[list[TrainSequence]]` 或 `list[PromptBatch]`） |
| 处理逻辑 | 每个桶内按顺序切片，每 batch_size 个样本组成一个 batch |
| 配置要求 | batch_size 可配置（复用 `TrainerConfig.batch_size`） |
| 验收标准 | 除最后一个 batch 外，所有 batch 大小等于 batch_size |

> **V3.0 修订**: 输出类型对齐 AReno 现有 trainer 消费的 batch 形态——SFT 路径消费 `list[TrainSequence]`，rollout 路径消费 `PromptBatch`。本系统不引入新的 batch 容器类型。

#### FR-004: 桶内排序

| 需求项 | 描述 |
|--------|------|
| 需求 ID | FR-004 |
| 需求名称 | 桶内排序 |
| 优先级 | P1 |
| 需求描述 | 系统应能在桶内按长度排序，进一步减少 padding |
| 输入 | 桶内样本列表 |
| 输出 | 按长度排序的样本列表 |
| 处理逻辑 | 按 token 长度升序排序 |
| 配置要求 | 可开关 |
| 验收标准 | 排序后 batch 内长度标准差降低 |

#### FR-005: 长度缓存

| 需求项 | 描述 |
|--------|------|
| 需求 ID | FR-005 |
| 需求名称 | 长度缓存 |
| 优先级 | P1 |
| 需求描述 | 系统应能缓存长度计算结果，避免重复计算 |
| 输入 | 文本内容 |
| 输出 | 缓存的长度值 |
| 处理逻辑 | 先查缓存，未命中则计算并写入缓存 |
| 存储要求 | 支持文件持久化（JSON） |
| 验收标准 | 缓存命中率 > 80% 时，处理时间减少 50% |
| 缓存键 | 使用文本的 SHA-256 哈希值作为缓存键（见 4.2） |

> **V3.0 修订**: 持久化格式从 Java JSON 改为 Python `json` 模块序列化；缓存实现使用 `functools.lru_cache` 或 `cachetools.LRUCache`，避免引入 Guava 等 Java 依赖。具体依赖选择需在实现前确认（见 2.3 约束）。

#### FR-006: 统计报告

| 需求项 | 描述 |
|--------|------|
| 需求 ID | FR-006 |
| 需求名称 | 统计报告 |
| 优先级 | P2 |
| 需求描述 | 系统应能生成 batching 统计报告 |
| 输入 | batching 结果 |
| 输出 | 统计报告（日志 + 可选 CSV） |
| 处理逻辑 | 计算各项指标并通过 AReno 现有 `logging` 输出 |
| 报告内容 | batch 数、padding 比例、桶分布、跳过样本数等 |
| 验收标准 | 报告包含所有关键指标 |

> **V3.0 修订**: 输出方式从"控制台 + CSV 文件"改为通过 AReno 现有 `logging.getLogger` 输出，与 `SFTTrainer` 的 `self.logger.info(...)` 风格一致；CSV 导出作为可选能力。

#### FR-007: Batch Shuffle

| 需求项 | 描述 |
|--------|------|
| 需求 ID | FR-007 |
| 优先级 | P1 |
| 需求描述 | 组 batch 完成后，对 batch 列表做随机打乱 |
| 输入 | Batch 列表 |
| 输出 | 打乱顺序后的 Batch 列表 |
| 处理逻辑 | 保留每个 batch 内部样本不变，随机打乱 batch 间顺序 |
| 配置要求 | 可开关，可配置随机种子 |
| 验收标准 | 打乱后每个 batch 内样本不变，顺序与原列表不同 |
| 设计理由 | 按长度排序后组 batch 会导致训练顺序高度规律（从短到长），影响 SGD 收敛 |

### 2.2 非功能需求

#### NFR-001: 性能需求

| 指标 | 目标值 | 测试条件 | 测量口径 |
|------|--------|----------|----------|
| 处理速度 | ≥ 10K 样本/秒 | 100K 样本，batch_size=32 | 包含 tokenizer 编码 + 分桶 + 组 batch，不含磁盘 I/O |
| 内存占用 | ≤ 2GB | 100K 样本 | 进程 RSS 峰值，通过 `resource.getrusage` 或 `tracemalloc` 测量 |
| Padding 降低 | ≥ 50% | 对比顺序 batching | 同一数据集两种策略的 avgPaddingRatio 对比 |
| 缓存命中率 | ≥ 80% | 重复数据处理 | 缓存统计 hitCount / (hitCount + missCount) |

> **V3.0 修订**: 内存测量口径从 JMX `MemoryMXBean` 改为 Python `resource.getrusage`/`tracemalloc`；性能测试对比基准从"随机 batching"改为"顺序 batching"（即 AReno 现有默认行为），更贴合实际收益评估。

#### NFR-002: 可靠性需求

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 样本完整性 | 100%（默认配置） | `drop_last_batch=false` 且 `truncate_strategy=KEEP` 时无样本丢失 |
| 数据一致性 | 100% | 样本与长度一一对应 |
| 异常恢复 | 支持断点续跑 | 可选；AReno 现有 checkpoint 机制在训练步级保存，本系统的断点续跑指 batching 阶段的进度文件 |

> **V3.0 修订**: 断点续跑与 AReno 现有 checkpoint（`areno/engine/checkpoints/`，步级模型 checkpoint）区分说明，避免概念混淆。本系统的断点续跑仅针对大数据集 batching 阶段的样本处理进度，优先级降为 P2。

#### NFR-003: 可扩展性需求

| 维度 | 要求 |
|------|------|
| 数据规模 | 支持 1M+ 样本 |
| 并行处理 | 支持多线程（tokenizer 编码阶段） |
| 配置灵活 | 支持动态调整桶边界和 batch_size |

### 2.3 约束条件

| 约束类型 | 描述 |
|----------|------|
| 技术约束 | 使用 **Python 3.10+** 开发，与 AReno `pyproject.toml` 的 `requires-python = ">=3.10"` 对齐 |
| 依赖约束 | 复用 AReno 已加载的 HF tokenizer；**新增第三方依赖需先征得同意**（AGENTS.md 规定 `pyproject.toml` 改动需 ask first） |
| 集成约束 | 不改动 `pad_rows` / `pad_rollout_rows` / `_pack_train_data` 等下游 padding/packing 逻辑；不改动 `TrainerConfig` 现有字段语义，只新增字段 |
| 配置约束 | 修改 `TrainerConfig` 与 `train.py` CLI 需先 ask first（AGENTS.md 规定） |
| 测试约束 | 新增 CPU 测试 under `tests/`，命名遵循 `*_cpu.py` 后缀（AGENTS.md 规定） |
| 资源约束 | 单机运行，不依赖分布式框架 |

> **V3.0 修订**: 技术约束从 Java 17 改为 Python 3.10+；新增集成约束、配置约束、测试约束，全部对齐 AGENTS.md 的硬性规定。

---

## 3. 系统设计

### 3.1 系统架构

```text
┌─────────────────────────────────────────────────────────────────┐
│                        输入层 (Input Layer)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  HF Dataset │  │  JSON/JSONL │  │  内存 list  │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         └─────────────────┴─────────────────┘                    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  复用 train.py: _load_dataset_for_training                  ││
│  └────────────────────────┬────────────────────────────────────┘│
└───────────────────────────┼──────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                       处理层 (Processing Layer)                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  areno/api/length_grouped.py  (新增模块)                      ││
│  │                                                              ││
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  ││
│  │  │ compute_len  │ →  │ Length       │ →  │ Batch        │  ││
│  │  │ (复用 encode │    │ Bucketer     │    │ Grouper      │  ││
│  │  │ _generation_ │    │              │    │              │  ││
│  │  │  prompt)     │    │              │    │              │  ││
│  │  └──────────────┘    └──────────────┘    └──────────────┘  ││
│  │         ↓                   ↓                   ↓          ││
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  ││
│  │  │ TokenLength  │    │ Bucket       │    │ Batch        │  ││
│  │  │ Cache        │    │ Strategy     │    │ Shuffler     │  ││
│  │  └──────────────┘    └──────────────┘    └──────────────┘  ││
│  │                                                              ││
│  └──────────────────────────────────────────────────────────────┘│
└───────────────────────────┼──────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                        集成层 (Integration Layer)                │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │ SFTTrainer          │  │ Trainer             │               │
│  │ _iter_train_batches │  │ load_prompt_batches │               │
│  │ (改造: 调用本模块)   │  │ (改造: 调用本模块)   │               │
│  └──────────┬──────────┘  └──────────┬──────────┘               │
│             ↓                        ↓                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  现有下游 (不改动):                                          ││
│  │  pad_rows / pad_rollout_rows → _pack_train_data → loss_fn   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

> **V3.0 修订**: 架构图重画为三层（输入/处理/集成），明确标注"复用 `_load_dataset_for_training`"、"新增模块 `areno/api/length_grouped.py`"、"改造 `_iter_train_batches`/`load_prompt_batches`"、"下游不改动"。

### 3.2 模块划分

| 模块名称 | 职责 | 关键类/函数 | 归属 |
|----------|------|-------------|------|
| Length Grouped Batcher | 长度分组主入口 | `LengthGroupedBatcher` | `areno/api/length_grouped.py`（新增） |
| Length Calculator | 长度计算 | `compute_token_length` | `areno/api/length_grouped.py`（新增，调用 `encode_generation_prompt`） |
| Length Bucketer | 长度分桶 | `LengthBucketer`, `LengthBucket`, `BucketStrategy`, `BucketContext` | `areno/api/length_grouped.py`（新增） |
| Batch Grouper | 组 batch | `BatchGrouper`, `BatchShuffler` | `areno/api/length_grouped.py`（新增） |
| TokenLength Cache | 长度缓存 | `TokenLengthCache` | `areno/api/length_grouped.py`（新增） |
| Statistics Reporter | 统计报告 | `BatchingMetrics`, `log_batching_report` | `areno/api/length_grouped.py`（新增） |
| Progress Tracker | 断点续跑（可选） | `ProgressTracker` | `areno/api/length_grouped.py`（新增，P2） |
| 配置字段 | 长度分组开关与参数 | `TrainerConfig` 新增字段 | `areno/api/trainer_config.py`（扩展） |
| CLI 开关 | 命令行入口 | `--length-grouped` 等 | `areno/cli/train.py`（扩展） |

> **V3.0 修订**: 模块归属全部指向 AReno 现有文件路径或明确的新增文件；"关键类"从 Java 类名改为 Python 类/函数名。

### 3.3 类设计

> **V3.0 修订**: 以下代码示例改为 Python，类型注解对齐 AReno 风格（`from __future__ import annotations`，`dataclass(slots=True)`）。实体类复用 AReno 现有的 `TrainSequence` / `PromptItem`，不新建 `Sample` 类。

#### 3.3.1 长度桶实体

```python
# areno/api/length_grouped.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LengthBucket:
    """长度桶实体。

    区间为 [min_len, max_len)，左闭右开。
    相邻桶的 max_len == 下一桶 min_len，保证无缝衔接且不重叠。
    """

    bucket_id: int
    min_len: int          # 含
    max_len: int          # 不含

    def contains(self, length: int) -> bool:
        return self.min_len <= length < self.max_len
```

> **V3.0 修订**: `LengthBucket` 改为 Python `dataclass(frozen=True)`；不再持有 `samples` 列表，样本分组结果由 `dict[int, list[...]]` 承载。不新建 `Sample`/`Batch` 实体类——SFT 路径直接操作 `TrainSequence`，rollout 路径直接操作 `PromptItem`。

#### 3.3.2 核心处理类

```python
# areno/api/length_grouped.py
from areno.api.tokenizer import encode_generation_prompt


def compute_token_length(tokenizer, text: str | None) -> int:
    """计算文本的 token 长度。

    契约：None 或空字符串返回 0，不抛异常。
    复用 encode_generation_prompt 与 SFTTrainer/Trainer.load_prompt_batches 一致。
    """
    if not text:
        return 0
    return len(encode_generation_prompt(tokenizer, text))


class LengthBucketer:
    """长度分桶器，通过 BucketStrategy 创建桶并分配样本。"""

    def __init__(self, strategy: BucketStrategy) -> None:
        self.strategy = strategy
        self.buckets: list[LengthBucket] = []

    def bucketize(self, samples: list, get_length) -> dict[int, list]:
        """将样本分配到桶，返回 {bucket_id: [samples]}。

        get_length: 从样本提取 token 长度的回调，适配 TrainSequence/PromptItem。
        """
        ctx = BucketContext(dataset=samples)
        self.buckets = self.strategy.create_buckets(ctx)

        result: dict[int, list] = {}
        for sample in samples:
            length = get_length(sample)
            for bucket in self.buckets:
                if bucket.contains(length):
                    result.setdefault(bucket.bucket_id, []).append(sample)
                    break
        return result


class BatchGrouper:
    """桶内组 batch，支持桶内排序与 drop_last。"""

    def __init__(self, batch_size: int, sort_within_bucket: bool, drop_last: bool) -> None:
        self.batch_size = batch_size
        self.sort_within_bucket = sort_within_bucket
        self.drop_last = drop_last

    def group_by_bucket(
        self, bucketed_data: dict[int, list], get_length
    ) -> list[list]:
        batches: list[list] = []
        for samples in bucketed_data.values():
            if self.sort_within_bucket:
                samples = sorted(samples, key=get_length)
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        return batches


class BatchShuffler:
    """随机打乱 batch 间顺序，保持每个 batch 内部样本不变。"""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def shuffle(self, batches: list[list]) -> list[list]:
        import random
        shuffled = list(batches)
        random.Random(self.seed).shuffle(shuffled)
        return shuffled
```

> **V3.0 修订**:
> - 全部改为 Python 类，类型注解对齐 AReno 风格。
> - `bucketize` / `group_by_bucket` 通过 `get_length` 回调适配不同样本类型（`TrainSequence` 用 `len(seq.tokens)`，`PromptItem` 用 `len(item.input_tokens)`），避免为每种样本类型写一个分支。
> - `BatchShuffler` 使用 `random.Random(seed)` 而非 `java.util.Random`。
> - 不新建 `Batch` 实体类——batch 就是 `list[TrainSequence]` 或 `list[PromptItem]`，与现有 trainer 消费的形态一致。

#### 3.3.3 辅助类

```python
# areno/api/length_grouped.py
import hashlib
import json
from collections import OrderedDict


class TokenLengthCache:
    """长度缓存，键为文本的 SHA-256 哈希十六进制串。"""

    def __init__(self, cache_file_path: str | None = None, max_size: int = 100_000) -> None:
        self.cache_file_path = cache_file_path
        # OrderedDict 实现 LRU；生产实现可选用 cachetools.LRUCache（需确认依赖）
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        if cache_file_path:
            self._load_from_file()

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> int | None:
        key = self._cache_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hit_count += 1
            return self._cache[key]
        self.miss_count += 1
        return None

    def put(self, text: str, length: int) -> None:
        key = self._cache_key(text)
        self._cache[key] = length
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total else 0.0

    def save_to_file(self) -> None: ...
    def _load_from_file(self) -> None: ...
```

> **V3.0 修订**: 缓存实现从 Guava `CacheBuilder` 改为 `OrderedDict` 实现 LRU；哈希计算从 `DigestUtils.sha256Hex` 改为 `hashlib.sha256`；持久化使用 Python `json` 模块。是否引入 `cachetools` 需在实现前确认（见 2.3 依赖约束）。

### 3.4 流程设计

#### 3.4.1 主流程（SFT 路径集成示例）

```text
SFTTrainer._fit_initialized
  ↓
for epoch in range(epochs):
  ↓
  _iter_train_batches(tokenizer, ...)   ← 改造点
    ↓
    遍历 dataset 生成 TrainSequence（复用现有 _record_to_train_sequence）
    ↓
    是否启用 length-grouped？──否──→ 现有顺序切片逻辑（保持不变）
    ↓是
    compute_token_length（复用 encode_generation_prompt，可选缓存）
    ↓
    LengthBucketer.bucketize（get_length = lambda seq: len(seq.tokens)）
    ↓
    BatchGrouper.group_by_bucket（桶内排序 + 切片）
    ↓
    BatchShuffler.shuffle（可选）
    ↓
    yield 每个 batch（list[TrainSequence]）  ← 下游不变
  ↓
  self.areno.train(train_batch, loss_fn, mini_bs=...)
    ↓
  pad_rows / _pack_train_data / sft_loss_fn  ← 全部不改动
```

> **V3.0 修订**: 主流程图改为展示与 `SFTTrainer._fit_initialized` 的集成路径，明确标注"改造点"与"下游不变"。rollout 路径（`Trainer.load_prompt_batches`）的集成流程类似，`get_length = lambda item: len(item.input_tokens)`。

#### 3.4.2 分桶流程

```text
开始分桶
  ↓
通过 BucketStrategy.create_buckets(BucketContext) 创建桶列表
  ↓
遍历所有样本
  ↓
get_length(sample) 获取样本 token 长度
  ↓
遍历所有桶，找到第一个 contains(length) == True 的桶
  ↓
将样本添加到该桶
  ↓
是否还有样本？──是──→ 继续处理下一个样本
  ↓否
未分配样本记录到 warning 日志
  ↓
返回分桶结果 dict[bucket_id, list[samples]]
```

#### 3.4.3 组 Batch 流程

```text
开始组 Batch
  ↓
遍历所有桶
  ↓
是否启用桶内排序？──是──→ 按 get_length(sample) 升序排序
  ↓
从索引 0 开始，步长 batch_size 遍历桶内样本
  ↓
切片获取当前 batch 的样本 [i, i+batch_size)
  ↓
是否需要丢弃最后一个不完整 batch？
  ↓   ──是──→ 样本数 < batch_size 则跳过，记录跳过样本数
  ↓   ──否──→ 保留
  ↓
是否启用 batch shuffle？──是──→ BatchShuffler.shuffle()
  ↓
返回 Batch 列表 list[list[sample]]
```

---

## 4. 详细设计

### 4.1 分桶策略设计

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


class BucketStrategy(Protocol):
    """分桶策略接口，所有策略通过 BucketContext 获取所需信息。"""

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]: ...


@dataclass(slots=True)
class BucketContext:
    """分桶上下文，携带策略所需的可选信息。"""

    dataset: list | None = None     # 百分位分桶需要
    max_length: int = 0             # 固定间隔分桶需要（可从 dataset 推导）
```

> **V3.0 修订**: 接口从 Java `interface` 改为 Python `typing.Protocol`；`BucketContext` 改为 `dataclass(slots=True)`，与 AReno 现有 dataclass 风格一致。

#### 策略 1: 固定间隔分桶（默认）

```python
class FixedIntervalBucketStrategy:
    def __init__(self, interval: int = 32) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]:
        buckets: list[LengthBucket] = []
        bucket_id = 0
        max_length = ctx.max_length
        min_len = 0
        while min_len < max_length:
            buckets.append(LengthBucket(bucket_id, min_len, min_len + self.interval))
            bucket_id += 1
            min_len += self.interval
        # 最后一个桶：[max_length, inf)，处理超长样本
        buckets.append(LengthBucket(bucket_id, max_length, 2**31 - 1))
        return buckets
```

> **V3.0 修订**: `Integer.MAX_VALUE` 改为 `2**31 - 1`（Python 无内置 MAX_VALUE 常量）。

#### 策略 2: 百分位分桶

```python
class PercentileBucketStrategy:
    def __init__(self, num_buckets: int = 8) -> None:
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        self.num_buckets = num_buckets

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]:
        if not ctx.dataset:
            raise ValueError("dataset required for percentile strategy")
        lengths = sorted(self._extract_lengths(ctx.dataset))
        step = max(len(lengths) // self.num_buckets, 1)
        buckets: list[LengthBucket] = []
        for i in range(self.num_buckets):
            min_len = 0 if i == 0 else lengths[i * step]
            if i == self.num_buckets - 1:
                max_len = 2**31 - 1
            else:
                max_len = lengths[min((i + 1) * step, len(lengths) - 1)]
            if min_len >= max_len and i > 0:
                continue  # 跳过重复值导致的退化桶
            buckets.append(LengthBucket(i, min_len, max_len))
        return buckets

    def _extract_lengths(self, dataset: list) -> list[int]:
        # 由调用方通过 get_length 提取后传入，或在此处统一处理
        raise NotImplementedError("lengths 应由调用方提取，避免策略层耦合样本类型")
```

> **V3.0 修订**: `_extract_lengths` 标记为 `NotImplementedError`，实际实现需由调用方传入长度列表或通过 `get_length` 回调，避免策略层耦合 `TrainSequence`/`PromptItem` 具体类型。

#### 策略 3: 自定义边界分桶

```python
class CustomBucketStrategy:
    def __init__(self, boundaries: list[int]) -> None:
        self.boundaries = boundaries

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]:
        return [
            LengthBucket(i, self.boundaries[i], self.boundaries[i + 1])
            for i in range(len(self.boundaries) - 1)
        ]

# 使用示例
boundaries = [0, 64, 128, 256, 512, 1024, 2**31 - 1]
strategy = CustomBucketStrategy(boundaries)
```

### 4.2 缓存设计

缓存键为文本的 SHA-256 哈希十六进制串（固定 64 字符），不使用原文作为键，避免大文本的内存开销和 JSON 文件膨胀。实现见 3.3.3 `TokenLengthCache`。

**缓存键设计对比**:

| 方案 | 键内容 | 内存开销 | 文件体积 | 哈希冲突风险 |
|------|--------|----------|----------|-------------|
| 原方案 | 文本原文 | 高（存储完整文本） | 极大 | 无 |
| 修订方案 | SHA-256 哈希 | 低（固定 64 字符） | 小 | 极低（2^~128 碰撞概率） |

### 4.3 边界情况处理

#### 场景 1: 最后一个 batch 样本不足

由 `BatchGrouper.group_by_bucket` 处理：`drop_last=True` 时丢弃不足 `batch_size` 的尾部 batch，丢弃样本数计入 `BatchingMetrics.dropped_samples`；`drop_last=False`（默认）时保留，保证 100% 样本完整性。实现见 3.3.2。

> **V3.0 修订**: 不再新建 `Batch` 实体类记录 paddingRatio 等指标——这些指标在统计阶段由 `log_batching_report` 基于 `len(seq.tokens)` 计算，避免引入与现有 `TrainSequence` 并存的冗余容器。

#### 场景 2: 某个桶样本太少

```python
class BucketMerger:
    """合并相邻小桶，仅合并长度区间相邻的桶，避免 padding 增大。"""

    def merge_small_buckets(
        self, bucketed_data: dict[int, list], min_samples: int
    ) -> dict[int, list]:
        sorted_ids = sorted(bucketed_data.keys())
        merged: dict[int, list] = {}
        pending: list = []
        pending_id = -1
        for bucket_id in sorted_ids:
            samples = bucketed_data[bucket_id]
            if len(samples) < min_samples:
                if not pending:
                    pending_id = bucket_id
                pending.extend(samples)
            else:
                if pending:
                    if len(pending) >= min_samples:
                        merged[pending_id] = pending
                    else:
                        samples = [*samples, *pending]
                    pending = []
                merged[bucket_id] = samples
        if pending:
            merged[pending_id] = pending
        return merged
```

#### 场景 3: 超长样本处理

```python
class TruncateStrategy:
    KEEP = "keep"        # 保留
    TRUNCATE = "truncate"  # 截断到 max_length
    DROP = "drop"        # 丢弃


@dataclass(slots=True)
class FilterResult:
    kept: list
    dropped_count: int


class SampleFilter:
    def __init__(self, max_length: int, truncate_strategy: str) -> None:
        self.max_length = max_length
        self.truncate_strategy = truncate_strategy

    def handle_over_long_samples(self, samples: list, get_length) -> FilterResult:
        kept: list = []
        dropped = 0
        for sample in samples:
            if get_length(sample) <= self.max_length:
                kept.append(sample)
                continue
            if self.truncate_strategy == TruncateStrategy.DROP:
                dropped += 1
            elif self.truncate_strategy == TruncateStrategy.TRUNCATE:
                # 截断逻辑取决于样本类型，这里保留样本引用，由下游处理
                kept.append(sample)
            else:  # KEEP
                kept.append(sample)
        return FilterResult(kept, dropped)
```

> **V3.0 修订**: `TruncateStrategy` 从 Java enum 改为字符串常量；`SampleFilter` 通过 `get_length` 回调适配不同样本类型。注意 AReno 现有 `SFTTrainer` 已在 `_record_to_train_sequence` 中按 `max_prompt_tokens` / `max_new_tokens` 过滤，本系统的 `SampleFilter` 仅在启用 length-grouped 时作为 batching 前的预过滤，避免与现有过滤逻辑重复——集成时需确认两者边界。

---

## 5. 接口设计

### 5.1 内部接口

> **V3.0 修订**: 本系统是 AReno 内嵌模块，不提供外部 REST/RPC 接口。以下为模块间 Python 接口契约。

#### 长度计算接口

```python
def compute_token_length(tokenizer, text: str | None) -> int:
    """计算文本的 token 长度。

    契约：None 或空字符串返回 0，不抛异常。
    复用 encode_generation_prompt，与 SFTTrainer 一致。
    """
```

> **V3.0 修订**: 不新建 `ITokenizer`/`TokenSequence` 接口——直接使用 AReno 已加载的 HF tokenizer 和 `encode_generation_prompt`。线程安全由 HF tokenizer 自身保证（`encode` 线程安全），无需 `isThreadSafe()` 方法。

#### 样本长度回调接口

```python
# SFT 路径
get_length_sft = lambda seq: len(seq.tokens)          # TrainSequence
# rollout 路径
get_length_rollout = lambda item: len(item.input_tokens)  # PromptItem
```

> **V3.0 修订**: 通过 `get_length` 回调适配 `TrainSequence`/`PromptItem`，避免为每种样本类型写一个分桶器分支。

### 5.2 配置接口

> **V3.0 修订**: 配置载体从独立 `BatcherConfig` 改为扩展 `TrainerConfig`，在 `areno/api/trainer_config.py` 新增字段。修改 `TrainerConfig` 需先 ask first（AGENTS.md 规定）。

```python
# areno/api/trainer_config.py — 新增字段（需 ask first）
@dataclass(slots=True)
class TrainerConfig:
    # ... 现有字段保持不变 ...
    batch_size: int = 32
    max_prompt_tokens: int = 1024
    # ... 现有字段 ...

    # ===== 长度分组配置（新增） =====
    length_grouped: bool = False                  # 是否启用长度分组 batching
    bucket_strategy: str = "fixed_interval"       # fixed_interval | percentile | custom
    bucket_interval: int = 32                     # 固定间隔分桶的区间大小
    custom_boundaries: list[int] | None = None    # 自定义边界
    num_percentile_buckets: int = 8               # 百分位分桶桶数
    sort_within_bucket: bool = True               # 桶内是否按长度排序
    drop_last_batch: bool = False                 # 是否丢弃最后一个不完整 batch
    enable_batch_shuffle: bool = True             # 是否启用 batch 间 shuffle
    shuffle_seed: int = 42                        # shuffle 随机种子
    enable_length_cache: bool = False             # 是否启用长度缓存
    length_cache_path: str | None = None          # 缓存文件路径
    length_cache_max_size: int = 100_000          # 缓存最大条目数
    min_bucket_samples: int = 10                  # 小桶合并阈值
    max_sample_length: int = 4096                 # 超长样本阈值
    truncate_strategy: str = "keep"               # keep | truncate | drop
```

### 5.3 使用接口

```python
# areno/api/length_grouped.py — 主入口
class LengthGroupedBatcher:
    """长度分组 batch 组成器，供 SFTTrainer / Trainer.load_prompt_batches 调用。"""

    def __init__(self, config: TrainerConfig, tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.cache = (
            TokenLengthCache(config.length_cache_path, config.length_cache_max_size)
            if config.enable_length_cache else None
        )

    def make_batches(self, samples: list, get_length) -> list[list]:
        """执行完整流程：长度计算 → 过滤 → 分桶 → 组 batch → shuffle。"""
        # 1. 超长样本过滤
        # 2. 长度计算（可选缓存）
        # 3. 分桶
        # 4. 组 batch
        # 5. shuffle
        # 6. 统计报告（logging）
        ...
```

#### SFT 集成示例

```python
# areno/api/trainers/sft.py — 改造 _iter_train_batches
def _iter_train_batches(self, tokenizer, *, max_prompt_tokens, max_new_tokens):
    sequences = self._collect_train_sequences(tokenizer, max_prompt_tokens, max_new_tokens)
    if not self.config.length_grouped:
        # 现有顺序切片逻辑保持不变
        yield from self._sequential_slice(sequences)
        return
    # 长度分组路径
    batcher = LengthGroupedBatcher(self.config, tokenizer)
    for batch in batcher.make_batches(sequences, get_length=lambda seq: len(seq.tokens)):
        yield batch
```

> **V3.0 修订**: 集成示例明确展示"未启用时走原有顺序切片路径"的兼容设计，确保本系统是**可选增强**而非破坏性改动。

### 5.4 CLI 入口

> **V3.0 修订**: CLI 从独立 `BatchCli`/`java -jar` 改为 AReno `train.py` 现有 `areno train` 命令的新增 flag。修改 CLI 需先 ask first（AGENTS.md 规定）。

```bash
# 现有命令新增 --length-grouped 等开关
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --reward-fn-path examples/math/math_verify_reward.py --algo sft \
  --length-grouped --bucket-interval 32 --enable-batch-shuffle
```

```python
# areno/cli/train.py — 新增 click option（需 ask first）
@click.option("--length-grouped", is_flag=True, default=False,
              help="Enable length-grouped batching to reduce padding.")
@click.option("--bucket-strategy", type=click.Choice(["fixed_interval","percentile","custom"]),
              default="fixed_interval", show_default=True)
@click.option("--bucket-interval", type=int, default=32, show_default=True)
@click.option("--enable-batch-shuffle/--disable-batch-shuffle", default=True)
# ... 其他新增开关
```

---

## 6. 数据设计

### 6.1 输入数据格式

复用 AReno 现有数据集加载（`train.py: _load_dataset_for_training`），支持 HF Dataset、本地 JSON/JSONL/parquet 等。

#### SFT 数据格式（现有）

```json
{"prompt": "今天天气真好", "response": "是的，阳光明媚"}
{"prompt": "机器学习很有趣", "response": "确实，尤其是深度学习"}
```

#### RLHF prompt 数据格式（现有）

```json
{"prompt": "请解释梯度下降", "solutions": ["..."]}
```

> **V3.0 修订**: 输入格式对齐 AReno 现有 `SFTTrainer`（需 `prompt`+`response`）和 `Trainer.load_prompt_batches`（需 `prompt`）的字段契约，不引入新的输入格式。

### 6.2 输出数据格式

本系统不产生独立输出文件——batch 列表直接交给现有 trainer 消费，训练结果与 checkpoint 由 AReno 现有流程产出。统计报告通过 `logging` 输出。

> **V3.0 修订**: 移除独立的 Batch JSON / CSV 输出格式——本系统是内嵌模块，不单独产出文件。统计报告走 `logging`。

### 6.3 缓存数据格式

```json
{
    "cache_version": "3.0",
    "key_algorithm": "SHA-256",
    "created_at": "2026-07-28T10:00:00Z",
    "cache_size": 10000,
    "entries": {
        "e3b0c44298fc1c149afbf4c8996fb924": 5,
        "6e340b9cffb37a989ca544e6bb780a2d": 6,
        "3a7bd3e2360a3d29eea436fcfb7e44c2": 12
    }
}
```

> **V3.0 修订**: `cache_version` 升级为 "3.0" 以区分键算法与 Python 序列化格式变更。加载时版本不匹配则忽略旧缓存并重建。

### 6.4 数据统计结构

```python
@dataclass(slots=True)
class BatchingMetrics:
    total_samples: int = 0
    total_batches: int = 0
    total_buckets: int = 0
    avg_length: float = 0.0
    max_length: float = 0.0
    min_length: float = 0.0
    std_dev_length: float = 0.0
    avg_padding_ratio: float = 0.0
    max_padding_ratio: float = 0.0
    min_padding_ratio: float = 0.0
    avg_batch_size: float = 0.0
    underfull_batches: int = 0
    bucket_distribution: dict[int, int] = None
    processing_time_ms: int = 0
    samples_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    dropped_samples: int = 0
```

> **V3.0 修订**: 改为 Python `dataclass(slots=True)`；`bucket_distribution` 类型从 Java `Map<Integer,Integer>` 改为 `dict[int,int]`。

---

## 7. 性能设计

### 7.1 性能目标

| 指标 | 目标值 | 测试条件 | 测量口径 |
|------|--------|----------|----------|
| 处理速度 | ≥ 10K 样本/秒 | 100K 样本，batch_size=32 | 包含 tokenizer 编码 + 分桶 + 组 batch，不含磁盘 I/O |
| 内存占用 | ≤ 2GB | 100K 样本 | 进程 RSS 峰值，`tracemalloc`/`resource.getrusage` |
| Padding 降低 | ≥ 50% | 对比顺序 batching | 同一数据集两种策略的 avgPaddingRatio 对比 |
| 缓存命中率 | ≥ 80% | 重复数据处理 | hit_count / (hit_count + miss_count) |

> **V3.0 修订**: 对比基准从"随机 batching"改为"顺序 batching"（AReno 现有默认）；内存测量从 JMX 改为 Python 标准库工具。

### 7.2 优化策略

#### 策略 1: 并行计算

```python
# tokenizer.encode 本身线程安全，可使用并行
from concurrent.futures import ThreadPoolExecutor

def batch_compute_lengths(tokenizer, samples, get_text, cache=None, max_workers=4):
    def compute_one(sample):
        text = get_text(sample)
        if cache:
            cached = cache.get(text)
            if cached is not None:
                return cached
        length = compute_token_length(tokenizer, text)
        if cache:
            cache.put(text, length)
        return length
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(compute_one, samples))
```

> **V3.0 修订**: 从 Java `parallelStream` 改为 Python `ThreadPoolExecutor`；HF tokenizer 的 `encode` 线程安全，无需 `isThreadSafe()` 检查。注意：GIL 下 tokenizer 编码为 C 扩展释放 GIL，多线程有效。

#### 策略 2: 批量编码

```python
# HF tokenizer 支持批量编码
def batch_compute(tokenizer, texts: list[str]) -> list[int]:
    results = tokenizer.encode_batch(texts)  # 若 tokenizer 支持
    return [len(r.ids) for r in results]
```

> **V3.0 修订**: 使用 HF tokenizer 的 `encode_batch`（若可用），而非 Java `encodeBatch`。

#### 策略 3: 内存管理

AReno 现有 `SFTTrainer._iter_train_batches` 已是惰性迭代（逐行 tokenization）。本系统在启用 length-grouped 时需先收集全部 `TrainSequence` 再分桶，内存峰值会增加。对于大数据集，可通过 `min_bucket_samples` 合并小桶减少 batch 数，或在数据量超过阈值时回退为顺序切片并打印 warning。

> **V3.0 修订**: 移除原"懒加载 `LazySample`"和"流式 `StreamingBatcher`"设计——AReno 现有路径已是惰性迭代，length-grouped 本质需要全局长度信息才能分桶，无法完全流式化。改为在内存压力下回退为顺序切片的实用策略。

### 7.3 性能监控

```python
import time
import logging

class PerformanceMonitor:
    def __init__(self):
        self.timings: dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    def time_it(self, operation: str):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                self.timings[operation] = time.perf_counter() - start
                self.logger.info("%s 耗时: %.3f s", operation, self.timings[operation])
                return result
            return wrapper
        return decorator
```

> **V3.0 修订**: 从 Java `System.currentTimeMillis()` 改为 `time.perf_counter()`；日志通过 AReno 现有 `logging` 输出。

---

## 8. 异常处理

### 8.1 异常分类

| 异常类型 | 异常码 | 处理方式 | 样本去向 |
|----------|--------|----------|----------|
| 数据加载失败 | E001 | 复用 AReno 现有 `_load_dataset_for_training` 错误处理 | 未加载的样本不参与处理 |
| Tokenizer 初始化失败 | E002 | 复用 AReno 现有 tokenizer 加载错误 | — |
| 样本格式错误 | E003 | 跳过问题样本，warning 日志记录 | 跳过样本计入 `dropped_samples` |
| 内存不足 | E004 | 启动时预估，超阈值回退顺序切片 | 全部样本通过顺序切片处理 |
| 分桶失败 | E006 | 回退为顺序切片（现有默认行为） | 全部样本走顺序切片 |

> **V3.0 修订**: 异常处理对齐 AReno 现有机制——不新建 `BatcherException`，复用 Python 标准异常 + `logging.warning`。E004/E006 的回退目标从"流式模式/默认分桶"改为"顺序切片"（AReno 现有默认行为），保证降级路径可用。

### 8.2 异常处理代码

```python
class LengthGroupedError(RuntimeError):
    """长度分组 batching 相关异常。"""

    def __init__(self, code: str, message: str, suggestion: str = "") -> None:
        super().__init__(f"[{code}] {message} {suggestion}".strip())
        self.code = code
        self.suggestion = suggestion


def should_use_sequential_fallback(sample_count: int, avg_text_bytes: int) -> bool:
    """内存预估：超过 80% 堆阈值时回退顺序切片。"""
    import sys
    estimated = sample_count * (avg_text_bytes + sys.getsizeof(object()))
    import tracemalloc
    # 简化：实际实现可用 psutil.Process().memory_info().rss 与系统总内存比较
    return estimated > 2 * 1024 * 1024 * 1024  # 2GB 阈值
```

> **V3.0 修订**: `BatcherException` 改为 `LengthGroupedError`；内存预估从 Java `Runtime.getRuntime().maxMemory()` 改为 Python `sys.getsizeof` + `tracemalloc`/`psutil`。移除 `catch (OutOfMemoryError)` 后降级——Python 的 `MemoryError` 同样不应 catch 后继续分配。

### 8.3 日志设计

复用 AReno 现有 `logging` 体系，与 `SFTTrainer.self.logger` 风格一致：

```python
logger = logging.getLogger(f"{__name__}.LengthGroupedBatcher")
logger.info("stage=length_grouped_start rows=%d", len(samples))
logger.warning("stage=bucketize_skip_unassigned count=%d", unassigned)
logger.info("stage=length_grouped_end batches=%d avg_padding=%.4f", n_batches, avg_pad)
```

> **V3.0 修订**: 移除独立 `BatcherLogger` 类，直接使用 Python `logging.getLogger`，日志格式与 `SFTTrainer` 的 `self.logger.info("epoch=%d step=%d ...")` 风格对齐。

---

## 9. 测试策略

### 9.1 测试类型

| 测试类型 | 覆盖内容 | 执行频率 | 命名规范 |
|----------|----------|----------|----------|
| 单元测试 | 核心类方法 | 每次提交 | `tests/test_length_grouped_cpu.py` |
| 集成测试 | 与 SFTTrainer 集成 | 每天 | `tests/test_length_grouped_integration_cpu.py` |
| 性能测试 | 处理速度、内存 | 每周 | `tests/test_length_grouped_perf_cpu.py` |

> **V3.0 修订**: 测试文件命名遵循 AReno `*_cpu.py` 后缀规范（AGENTS.md 规定），确保 CPU 套件可独立运行。

### 9.2 测试用例

#### 单元测试用例

```python
# tests/test_length_grouped_cpu.py
import pytest
from areno.api.length_grouped import (
    LengthBucket, FixedIntervalBucketStrategy, LengthBucketer,
    BatchGrouper, BatchShuffler, compute_token_length,
)


class TestComputeTokenLength:
    def test_normal_text(self, mock_tokenizer):
        assert compute_token_length(mock_tokenizer, "你好世界") > 0

    def test_none_input(self, mock_tokenizer):
        # 契约：None 输入返回 0，不抛异常
        assert compute_token_length(mock_tokenizer, None) == 0

    def test_empty_input(self, mock_tokenizer):
        # 契约：空字符串输入返回 0，不抛异常
        assert compute_token_length(mock_tokenizer, "") == 0


class TestLengthBucketer:
    def test_bucket_assignment(self):
        strategy = FixedIntervalBucketStrategy(32)
        bucketer = LengthBucketer(strategy)
        # 使用 (length, sample) 元组简化测试
        samples = [(50, "a")]
        result = bucketer.bucketize(samples, get_length=lambda x: x[0])
        assert any("a" in lst for lst in result.values())

    def test_empty_dataset(self):
        strategy = FixedIntervalBucketStrategy(32)
        bucketer = LengthBucketer(strategy)
        result = bucketer.bucketize([], get_length=lambda x: x)
        assert not result

    def test_bucket_boundary_no_overlap(self):
        strategy = FixedIntervalBucketStrategy(32)
        buckets = strategy.create_buckets(BucketContext(max_length=128))
        for i in range(len(buckets) - 1):
            assert buckets[i].max_len == buckets[i + 1].min_len

    def test_sample_at_boundary(self):
        strategy = FixedIntervalBucketStrategy(32)
        bucketer = LengthBucketer(strategy)
        samples = [(32, "boundary")]
        result = bucketer.bucketize(samples, get_length=lambda x: x[0])
        assigned = sum(len(lst) for lst in result.values())
        assert assigned == 1


class TestBatchGrouper:
    def test_batch_size(self):
        grouper = BatchGrouper(batch_size=32, sort_within_bucket=False, drop_last=False)
        bucketed = {0: list(range(100))}
        batches = grouper.group_by_bucket(bucketed, get_length=lambda x: x)
        for batch in batches[:-1]:
            assert len(batch) == 32

    def test_padding_reduction(self):
        # 对比顺序 vs 长度分组的 padding 比例
        ...
```

> **V3.0 修订**: 测试用例改为 pytest 风格，与 AReno 现有 `tests/` 一致；使用 `mock_tokenizer` fixture 代替真实 HF tokenizer，保证 CPU 套件无模型依赖。

#### 集成测试用例

```python
# tests/test_length_grouped_integration_cpu.py
def test_sft_with_length_grouped():
    """验证 SFTTrainer 在 --length-grouped 下端到端跑通（CPU mock backend）。"""
    ...
```

> **V3.0 修订**: 集成测试验证与 `SFTTrainer` 的集成，使用 CPU mock backend，不依赖 GPU。

#### 性能测试用例

```python
# tests/test_length_grouped_perf_cpu.py
def test_padding_reduction_vs_sequential():
    """基准对比：长度分组 vs 顺序切片的 padding 比例。"""
    dataset = create_test_samples(100_000)
    grouped = process_with_length_grouping(dataset)
    sequential = process_with_sequential(dataset)
    reduction = 1.0 - grouped.avg_padding_ratio / sequential.avg_padding_ratio
    assert reduction >= 0.5
```

> **V3.0 修订**: 性能测试对比基准从"随机 batching"改为"顺序切片"（AReno 现有行为），更贴合实际收益。

### 9.3 测试数据

```python
import random
import string

def create_test_samples(count: int) -> list:
    """使用对数正态分布生成长度多样的文本。"""
    rng = random.Random(42)
    samples = []
    for i in range(count):
        target_length = max(1, min(int(rng.lognormvariate(3.0, 1.5)), 8192))
        text = "".join(rng.choices(string.ascii_letters, k=target_length))
        samples.append(text)
    return samples
```

> **V3.0 修订**: 从 Java `Random` 改为 Python `random.Random`；使用 `rng.lognormvariate` 生成对数正态分布。

---

## 10. 部署方案

### 10.1 环境要求

| 组件 | 要求 | 说明 |
|------|------|------|
| Python | **3.10+** | 与 AReno `pyproject.toml` 对齐 |
| 内存 | ≥ 4GB | 处理 100K 样本需要 2GB 进程内存 |
| 磁盘 | ≥ 10GB | 缓存 + 输出文件（沿 AReno 现有要求） |
| CPU | ≥ 4 核 | 支持并行 tokenizer 编码 |

> **V3.0 修订**: 从 JDK 17+ 改为 Python 3.10+。

### 10.2 部署步骤

本系统随 AReno 主流程部署，无独立部署步骤：

```bash
# 1. 安装 AReno（现有）
pip install -e . --no-build-isolation

# 2. 训练时启用长度分组（新增 flag）
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --reward-fn-path examples/math/math_verify_reward.py --algo sft \
  --length-grouped

# 3. 验证结果
# 检查训练日志中的 "stage=length_grouped_end" 行，确认 batches 数与 padding 比例
```

> **V3.0 修订**: 移除独立 `mvn clean package` / `java -jar` 部署步骤；移除 `curl health` 端点检查（本系统非 Web 服务）。验证方式改为检查 AReno 训练日志。

### 10.3 配置文件示例

AReno 现有配置通过 CLI flag 传递，不使用独立 YAML。长度分组的配置通过 `areno train` 的新增 flag 暴露：

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path gsm8k:main \
  --algo sft \
  --batch-size 32 \
  --length-grouped \
  --bucket-strategy fixed_interval \
  --bucket-interval 32 \
  --enable-batch-shuffle \
  --shuffle-seed 42 \
  --max-prompt-tokens 1024
```

> **V3.0 修订**: 从独立 YAML 配置文件改为 AReno CLI flag，与现有 `areno train` 配置方式一致。

---

## 11. 风险评估

### 11.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Tokenizer 性能瓶颈 | 中 | 高 | 批量编码、多线程编码（HF tokenizer 线程安全） |
| 内存溢出 | 中 | 高 | 启动时内存预估，超阈值回退顺序切片 |
| 缓存失效 | 低 | 中 | 持久化、版本化（cache_version 检查） |
| 分桶不均 | 中 | 中 | 动态分桶策略、百分位分桶 |
| 训练数据分布偏移 | 中 | 高 | Batch shuffle 打乱顺序；可选混合少量不同长度样本 |
| **与现有 packed varlen 的交互** | **中** | **中** | **本系统只改 batch 组成，不改 packing 逻辑；需集成测试验证 packed 路径下 padding 收益是否仍成立** |

> **V3.0 修订**: 新增"与现有 packed varlen 的交互"风险——AReno 已支持 packed varlen 去除 pad token，需验证长度分组在 packed 路径下是否仍有收益（理论上 packed 路径 padding 计算已消除，长度分组的收益主要体现在减少 batch 内序列长度方差，降低 packed 序列总数或提升 GPU 利用率）。

### 11.2 项目风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 开发延期 | 中 | 中 | 分批交付、优先级排序（P0 先行） |
| 需求变更 | 中 | 中 | 灵活配置、可插拔设计（BucketStrategy 接口） |
| 测试不充分 | 低 | 高 | 自动化测试、代码审查、基准对比测试 |
| **破坏现有 trainer 行为** | **中** | **高** | **未启用 `length_grouped` 时走原有顺序切片路径；集成测试覆盖回归** |

> **V3.0 修订**: 新增"破坏现有 trainer 行为"风险——改造 `SFTTrainer`/`Trainer.load_prompt_batches` 可能影响现有训练流程，缓解措施是默认关闭、回归测试覆盖。

### 11.3 数据风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 样本格式不规范 | 中 | 中 | E003 跳过坏样本并 warning 日志记录 |
| 超长样本导致分桶倾斜 | 中 | 中 | 超长样本处理策略（KEEP/TRUNCATE/DROP） |
| 缓存键哈希冲突 | 极低 | 低 | SHA-256 碰撞概率可忽略；冲突时重新计算长度 |

---

## 12. 附录

### 12.1 评审问题追踪表

| 编号 | 问题 | 涉及章节 | 修订状态 |
|------|------|----------|----------|
| C1 | Java 版本三处不一致（11+ vs 17） | 2.3 / 10.1 / pom.xml | V2.0 已修正：统一为 17+；V3.0 再次修正：改为 Python 3.10+ |
| C2 | 系统形态未定（CLI vs Web 服务） | 5.3 / 10.2 | V2.0 已修正：明确为"库 + CLI 入口"；V3.0 再次修正：改为 AReno 内嵌模块 |
| C3 | dropLast / 样本跳过与完整性 100% 冲突 | 1.2 / 4.3 / 8.1 | 已修正：限定"默认配置下 100%"，丢弃样本计入报告 |
| C4 | 缓存键用原文文本，开销过大 | 4.2 / 6.3 | 已修正：改为 SHA-256 哈希键 |
| C5 | BucketStrategy 接口签名不统一 | 4.1 / 3.3.2 | 已修正：统一为 create_buckets(BucketContext) |
| C6 | 分桶区间边界可能重叠或遗漏 | 3.3.1 / 4.1 | 已修正：明确 [min_len, max_len) 左闭右开，补测试用例 |
| C7 | Tokenizer 线程安全未在接口层声明 | 5.1 / 7.2 | V3.0 修正：HF tokenizer encode 线程安全，移除 isThreadSafe() |
| C8 | 缺少 batch shuffle 机制 | 4.3 | 已修正：新增 FR-007 和 BatchShuffler |
| C9 | OOM 后被动降级不可靠 | 8.2 | 已修正：改为启动时内存预估主动选择 |
| C10 | 性能测试硬编码绝对阈值 | 9.2 | 已修正：改为基准对比，记录打印指标 |
| — | 测试数据不够真实 | 9.3 | 已修正：改为对数正态分布生成 |
| — | 部署步骤 health 端点与 CLI 矛盾 | 10.2 | 已修正：移除 health 检查 |
| — | 配置文件 YAML 片段截断 | 10.3 | V3.0 修正：改为 AReno CLI flag |
| — | 风险评估缺少数据分布偏移风险 | 11.1 | 已修正：新增技术风险和 11.3 数据风险 |
| — | null 输入行为未定义 | 5.1 / 9.2 | 已修正：接口契约声明 null/empty 返回 0 |
| **V3.0-A** | **技术栈与 AReno 不符（Java vs Python）** | **全文** | **V3.0 已修正：改为 Python 3.10+** |
| **V3.0-B** | **实体类未复用 TrainSequence/PromptItem** | **3.3.1** | **V3.0 已修正：移除 Sample/Batch 实体，复用现有类型** |
| **V3.0-C** | **未说明与现有 packed varlen 的关系** | **1.1 / 11.1** | **V3.0 已修正：1.1 说明 packed varlen 现状，11.1 新增交互风险** |
| **V3.0-D** | **集成点未明确** | **3.1 / 5.3** | **V3.0 已修正：明确 SFTTrainer._iter_train_batches 与 Trainer.load_prompt_batches** |
| **V3.0-E** | **配置入口未对齐 TrainerConfig** | **5.2** | **V3.0 已修正：扩展 TrainerConfig 而非新建 BatcherConfig** |

### 12.2 配置项速查

| 配置项 | 默认值 | 说明 | 归属 |
|--------|--------|------|------|
| `length_grouped` | False | 是否启用长度分组 batching | TrainerConfig（新增） |
| `bucket_strategy` | fixed_interval | 分桶策略 | TrainerConfig（新增） |
| `bucket_interval` | 32 | 固定间隔分桶的区间大小 | TrainerConfig（新增） |
| `enable_length_cache` | False | 是否启用长度缓存 | TrainerConfig（新增） |
| `length_cache_max_size` | 100,000 | 缓存最大条目数 | TrainerConfig（新增） |
| `sort_within_bucket` | True | 桶内是否按长度排序 | TrainerConfig（新增） |
| `drop_last_batch` | False | 是否丢弃最后一个不完整 batch | TrainerConfig（新增） |
| `enable_batch_shuffle` | True | 是否启用 batch 间 shuffle | TrainerConfig（新增） |
| `shuffle_seed` | 42 | shuffle 随机种子 | TrainerConfig（新增） |
| `max_sample_length` | 4096 | 超长样本阈值 | TrainerConfig（新增） |
| `truncate_strategy` | keep | 超长样本处理策略 | TrainerConfig（新增） |
| `batch_size` | 32 | 每个 batch 的样本数 | TrainerConfig（现有，复用） |
| `max_prompt_tokens` | 1024 | prompt 最大 token 数 | TrainerConfig（现有，复用） |

> **V3.0 修订**: 配置项归属明确区分为"新增"与"现有复用"，避免与 `TrainerConfig` 现有字段冲突。

### 12.3 异常码速查

| 异常码 | 异常类型 | 处理方式 |
|--------|----------|----------|
| E001 | 数据加载失败 | 复用 AReno 现有 `_load_dataset_for_training` 错误处理 |
| E002 | Tokenizer 初始化失败 | 复用 AReno 现有 tokenizer 加载错误 |
| E003 | 样本格式错误 | 跳过并 warning 日志记录，计入 dropped_samples |
| E004 | 内存不足 | 启动时预估，超阈值回退顺序切片 |
| E006 | 分桶失败 | 回退为顺序切片（现有默认行为） |

> **V3.0 修订**: 移除 E005（磁盘空间不足）——本系统不独立产出文件，磁盘由 AReno 主流程管理；异常处理全部对齐 AReno 现有机制。
