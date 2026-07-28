# Issue #217: Handle empty and degenerate training samples consistently

## 系分文档

- **Issue**: [inclusionAI/AReno#217](https://github.com/inclusionAI/AReno/issues/217)
- **标题**: Handle empty and degenerate training samples consistently
- **认领人**: 夏烬 (xiajin.lcy)
- **日期**: 2026-07-28

---

## 1. Issue 概述

### 1.1 背景与动机

AReno 用户需要**统一处理空内容和退化训练样本**作为一个聚焦的、可独立审查的能力。
当前工作流要么缺少这种行为，要么需要一次性用户代码，这使得后训练运行更难操作、比较和复现。

### 1.2 目标

检测以下退化样本，并应用可配置的 error-or-skip 策略和原因计数：

| 退化类型 | 描述 |
|---------|------|
| 空内容 | prompt 或 response 为空字符串 |
| 纯空白 | prompt 或 response 仅含空白字符（空格、换行、制表符等） |
| 仅特殊 token | tokenize 后所有 token 都是 special token |
| 无可训练 token | 全 prompt 无 response token（SFT/DPO）、loss_mask 全为 0 |
| 偏好对相同 | DPO chosen 与 rejected 完全相同 |

### 1.3 验收标准

- [ ] 在 tokenize 前和 tokenize 后分别测试每种退化原因，确保所有 rank 保持相同样本集，防止全无效数据集作为一个成功的 no-op 启动
- [ ] 实现使用现有 AReno 契约，不引入外部数据库或强制沙箱
- [ ] 默认行为保持向后兼容
- [ ] 聚焦的自动化测试覆盖成功路径、无效输入和一条边界/失败路径
- [ ] 用户文档包含最小可运行示例并解释可观测输出

---

## 2. 现状分析

### 2.1 现有退化样本处理点（零散分布）

当前 AReno 中的退化样本处理散落在多个模块中，没有统一入口：

| 文件 | 行号 | 处理方式 | 问题 |
|------|------|---------|------|
| `areno/api/trainers/sft.py` | 164 | `if not response: return None` | 仅检测空 response，不检测空白/特殊 token |
| `areno/api/trainers/sft.py` | 169-172 | `len(tokens) < 2` / `response_tokens == 0` | 不区分退化原因 |
| `areno/api/trainers/dpo.py` | 194 | `not any(not item for item in prompt_mask[1:])` | 不检测 chosen==rejected |
| `areno/api/trainer.py` | 244-245 | `len > max_prompt_tokens` 跳过 | 仅长度过滤，无内容质量检测 |
| `areno/engine/data/sampling.py` | 114-118 | `nan_to_num` logits | 数值守门，非样本质量检测 |
| `areno/api/loss_fns/layout.py` | 99-105 | `valid_count.clamp(min=1)` | 防除零，非样本过滤 |
| `areno/api/rewards.py` | 60 | `std + eps` | 优势函数除法守门 |

### 2.2 核心问题

1. **无统一检测入口**：各 trainer 各自实现过滤逻辑，检测维度不一致
2. **无可配置策略**：只能 skip，不能选择 error
3. **无原因计数**：跳过时不记录具体退化原因（只记录 `skipped_long`）
4. **全无效数据集可能静默**：SFT 在 `accepted == 0` 时会报错，但 DPO 和 rollout 路径不一定

### 2.3 数据流路径

需要修改的三条数据加载路径：

```
路径 A (Rollout: GSPO/GRPO/PPO):
  CLI -> _load_dataset_for_training -> dataset (list[dict])
    -> Trainer.load_prompt_batches (areno/api/trainer.py:207)
      -> 每行 record[prompt_key] -> encode_generation_prompt(tokenizer, prompt)
      -> 长度过滤 -> PromptItem -> PromptBatch
    -> rollout_batch -> TrainSequence

路径 B (SFT):
  CLI -> _load_dataset_for_training -> dataset (list[dict])
    -> SFTTrainer._iter_train_batches (areno/api/trainers/sft.py:97)
      -> _record_to_train_sequence (sft.py:144)
        -> prompt + response -> prompt_response_to_tokens_and_mask
        -> 长度/空/退化过滤 -> TrainSequence

路径 C (DPO):
  CLI -> _load_dataset_for_training -> dataset (list[dict])
    -> DPOTrainer._iter_train_batches (areno/api/trainers/dpo.py:135)
      -> _record_to_train_pair (dpo.py:165)
        -> chosen/rejected -> _make_sequence (dpo.py:198)
        -> 长度/退化过滤 -> [chosen_seq, rejected_seq]
```

---

## 3. 设计方案

### 3.1 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                    areno/api/data.py                          │
│                                                              │
│  DegenerateReason (enum)                                     │
│  SampleQualityReport (dataclass)                             │
│  DegeneratePolicy (enum)                                     │
│  DegenerateFilterConfig (dataclass)                          │
│                                                              │
│  check_prompt_text(prompt) -> SampleQualityReport            │
│  check_response_text(response) -> SampleQualityReport        │
│  check_tokenized_prompt(token_ids, tokenizer) -> Report      │
│  check_tokenized_response(prompt_mask) -> Report             │
│  check_preference_pair(chosen, rejected) -> Report           │
│  apply_degenerate_filter(report, config) -> None | raise     │
└──────────────────────────┬───────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
  ┌────────────────┐ ┌──────────┐ ┌──────────────┐
  │ trainer.py     │ │ sft.py   │ │ dpo.py       │
  │ (Rollout 路径) │ │ (SFT)    │ │ (DPO)        │
  │                │ │          │ │              │
  │ load_prompt_   │ │ _record_ │ │ _record_to   │
  │ batches 集成   │ │ to_train │ │ _train_pair  │
  │                │ │ _sequence│ │ 集成         │
  └────────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
           ▼              ▼              ▼
  ┌──────────────────────────────────────────────────────────────┐
│              CLI 诊断输出 & metrics 上报                       │
│  skipped_empty=N skipped_whitespace=M skipped_special_only=K │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 核心数据结构

#### 3.2.1 DegenerateReason（退化原因枚举）

```python
class DegenerateReason(enum.Enum):
    """Reasons a sample is considered degenerate."""
    EMPTY = "empty"                              # 空字符串
    WHITESPACE_ONLY = "whitespace_only"          # 仅含空白字符
    SPECIAL_TOKENS_ONLY = "special_tokens_only"  # tokenize 后全是 special token
    NO_TRAINABLE_TOKENS = "no_trainable_tokens"  # 无可训练 token 位置
    IDENTICAL_PREFERENCE_BRANCHES = "identical_preference_branches"  # DPO 分支相同
```

#### 3.2.2 SampleQualityReport（样本质量报告）

```python
@dataclass(slots=True)
class SampleQualityReport:
    """Result of checking one sample for degeneracy."""
    is_degenerate: bool
    reason: DegenerateReason | None
    stage: str          # "pre_tokenization" | "post_tokenization"
    detail: str         # 人可读的描述，用于日志和错误消息

    @classmethod
    def ok(cls) -> "SampleQualityReport":
        """Construct a non-degenerate report."""
        return cls(is_degenerate=False, reason=None, stage="", detail="")

    @classmethod
    def degenerate(cls, reason: DegenerateReason, stage: str, detail: str) -> "SampleQualityReport":
        """Construct a degenerate report."""
        return cls(is_degenerate=True, reason=reason, stage=stage, detail=detail)
```

#### 3.2.3 DegeneratePolicy（策略枚举）

```python
class DegeneratePolicy(enum.Enum):
    """Policy for handling degenerate samples."""
    SKIP = "skip"    # 跳过并在计数器中记录（默认）
    ERROR = "error"  # 抛出 ValueError 终止训练
```

#### 3.2.4 DegenerateFilterConfig（过滤配置）

```python
@dataclass(slots=True)
class DegenerateFilterConfig:
    """Configuration for degenerate sample filtering."""
    policy: DegeneratePolicy = DegeneratePolicy.SKIP
    enabled: bool = True
```

### 3.3 检测函数设计

#### 3.3.1 文本层检测（tokenize 前）

```python
def check_prompt_text(prompt: str) -> SampleQualityReport:
    """Check a raw prompt string before tokenization."""
    if not prompt:
        return SampleQualityReport.degenerate(
            DegenerateReason.EMPTY, "pre_tokenization",
            "prompt is an empty string")
    if not prompt.strip():
        return SampleQualityReport.degenerate(
            DegenerateReason.WHITESPACE_ONLY, "pre_tokenization",
            "prompt contains only whitespace")
    return SampleQualityReport.ok()


def check_response_text(response: str) -> SampleQualityReport:
    """Check a raw response string before tokenization."""
    if not response:
        return SampleQualityReport.degenerate(
            DegenerateReason.EMPTY, "pre_tokenization",
            "response is an empty string")
    if not response.strip():
        return SampleQualityReport.degenerate(
            DegenerateReason.WHITESPACE_ONLY, "pre_tokenization",
            "response contains only whitespace")
    return SampleQualityReport.ok()
```

#### 3.3.2 Token 层检测（tokenize 后）

```python
def check_tokenized_prompt(
    token_ids: list[int], tokenizer
) -> SampleQualityReport:
    """Check tokenized prompt for special-token-only or zero-length degeneracy."""
    if not token_ids:
        return SampleQualityReport.degenerate(
            DegenerateReason.EMPTY, "post_tokenization",
            "prompt produced zero tokens")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if special_ids and all(tid in special_ids for tid in token_ids):
        return SampleQualityReport.degenerate(
            DegenerateReason.SPECIAL_TOKENS_ONLY, "post_tokenization",
            f"all {len(token_ids)} prompt tokens are special tokens")
    return SampleQualityReport.ok()
```

#### 3.3.3 可训练 token 检测

```python
def check_trainable_tokens(prompt_mask: list[bool]) -> SampleQualityReport:
    """Check that at least one position has a trainable (non-prompt) token."""
    # prompt_mask[1:] 因为 next-token loss 对齐：position i 预测 i+1
    if not any(not is_prompt for is_prompt in prompt_mask[1:]):
        return SampleQualityReport.degenerate(
            DegenerateReason.NO_TRAINABLE_TOKENS, "post_tokenization",
            "no trainable tokens after prompt prefix")
    return SampleQualityReport.ok()
```

#### 3.3.4 偏好对检测（DPO 专用）

```python
def check_preference_pair(chosen: Any, rejected: Any) -> SampleQualityReport:
    """Check that chosen and rejected branches are not identical."""
    if chosen == rejected:
        return SampleQualityReport.degenerate(
            DegenerateReason.IDENTICAL_PREFERENCE_BRANCHES, "pre_tokenization",
            "chosen and rejected branches are identical")
    return SampleQualityReport.ok()
```

#### 3.3.5 策略应用函数

```python
def apply_degenerate_policy(
    report: SampleQualityReport,
    config: DegenerateFilterConfig,
) -> bool:
    """Apply the configured policy to a quality report.

    Returns True if the sample should be skipped, False if it should be kept.
    Raises ValueError if the policy is ERROR and the sample is degenerate.
    """
    if not report.is_degenerate:
        return False
    if not config.enabled:
        return False
    if config.policy is DegeneratePolicy.ERROR:
        raise ValueError(
            f"degenerate sample detected ({report.stage}): {report.detail}"
        )
    return True  # SKIP policy
```

### 3.4 集成方案

#### 3.4.1 PromptBatch 扩展

在 `areno/api/data.py` 的 `PromptBatch` 中新增退化样本计数器：

```python
@dataclass(slots=True)
class PromptBatch:
    items: list[PromptItem]
    scanned: int
    skipped_long: int
    total_skipped_long: int
    # 新增：退化样本计数
    skipped_degenerate: int = 0
    total_skipped_degenerate: int = 0
    degenerate_reasons: dict[str, int] = field(default_factory=dict)
```

#### 3.4.2 Rollout 路径集成（trainer.py: `load_prompt_batches`）

在现有长度过滤之前插入退化检测：

```python
# 在 trainer.py load_prompt_batches 方法中
# 现有代码：
#   prompt = record[prompt_key]
#   input_tokens = encode_generation_prompt(self._tokenizer, prompt)
#   if len(input_tokens) > max_prompt_tokens:
#       skipped_long += 1; total_skipped_long += 1; continue
#
# 修改为：
#   prompt = record[prompt_key]
#
#   # 新增：文本层退化检测
#   report = check_prompt_text(prompt)
#   if apply_degenerate_policy(report, self._degenerate_config):
#       skipped_degenerate += 1
#       total_skipped_degenerate += 1
#       _record_degenerate_reason(degenerate_reasons, report.reason)
#       continue
#
#   input_tokens = encode_generation_prompt(self._tokenizer, prompt)
#
#   # 新增：token 层退化检测
#   report = check_tokenized_prompt(input_tokens, self._tokenizer)
#   if apply_degenerate_policy(report, self._degenerate_config):
#       skipped_degenerate += 1
#       total_skipped_degenerate += 1
#       _record_degenerate_reason(degenerate_reasons, report.reason)
#       continue
#
#   if len(input_tokens) > max_prompt_tokens:
#       skipped_long += 1; total_skipped_long += 1; continue
```

#### 3.4.3 SFT 路径集成（sft.py: `_record_to_train_sequence`）

将现有零散的 `if not response: return None` 替换为统一检测：

```python
# 现有代码（sft.py:160-173）：
#   if record["prompt"] is None or record["response"] is None:
#       return None
#   prompt = str(record["prompt"])
#   response = str(record["response"])
#   if not response:
#       return None
#
# 修改为：
#   prompt = str(record["prompt"]) if record["prompt"] is not None else ""
#   response = str(record["response"]) if record["response"] is not None else ""
#
#   report = check_prompt_text(prompt)
#   if apply_degenerate_policy(report, config): return None
#   report = check_response_text(response)
#   if apply_degenerate_policy(report, config): return None
#
#   tokens, prompt_mask = prompt_response_to_tokens_and_mask(...)
#   report = check_trainable_tokens(prompt_mask)
#   if apply_degenerate_policy(report, config): return None
#   # 现有长度过滤保持不变
```

#### 3.4.4 DPO 路径集成（dpo.py: `_record_to_train_pair`）

在 chosen/rejected 提取后加入偏好对相同检测：

```python
# 在 dpo.py _record_to_train_pair 中
# 现有代码：
#   chosen, rejected = record["chosen"], record["rejected"]
#
# 修改为：
#   chosen, rejected = record["chosen"], record["rejected"]
#   report = check_preference_pair(chosen, rejected)
#   if apply_degenerate_policy(report, config): return None
```

#### 3.4.5 全无效数据集守门

在 `load_prompt_batches` 的循环结束后（`if not items: break` 之前），检查是否整个数据集都被跳过：

```python
# 如果整个 dataset 遍历完后 items 为空，且所有跳过都是退化原因
if not items and total_skipped_degenerate > 0 and skipped_long == 0:
    raise ValueError(
        f"dataset produced no valid rows: all {total_skipped_degenerate} "
        f"rows were degenerate (reasons: {degenerate_reasons}). "
        f"Check dataset quality or disable degenerate filtering."
    )
```

### 3.5 CLI 诊断输出

在训练日志中输出退化样本统计：

```
stage=data_filter
  scanned=256
  skipped_long=10
  skipped_degenerate=8
    empty=3
    whitespace_only=2
    special_tokens_only=2
    no_trainable_tokens=1
  accepted=238
```

当 policy=ERROR 时，错误消息格式：

```
ValueError: degenerate sample detected (pre_tokenization): response is an empty string
  Hint: set --degenerate-policy skip to skip degenerate samples instead of erroring
```

### 3.6 所有 rank 一致性保证

退化检测基于确定性规则（字符串比较、token 集合比较），不涉及随机性。
只要所有 rank 使用相同的 `DegenerateFilterConfig`（由 `TrainerConfig` 传递）和相同的 tokenizer，检测结果必然一致。

关键约束：
- `DegenerateFilterConfig` 存储在 `TrainerConfig` 中，由 CLI 统一构造后广播给所有 worker
- 检测函数是纯函数（无副作用、无随机性、无状态）
- 不在检测函数中做任何基于 rank 的分支

---

## 4. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `areno/api/data.py` | 修改 | 新增退化检测数据结构和检测函数 |
| `areno/api/tokenizer.py` | 不变 | 无需修改（`all_special_ids` 已有） |
| `areno/api/trainer.py` | 修改 | `load_prompt_batches` 集成退化检测 |
| `areno/api/trainers/sft.py` | 修改 | `_record_to_train_sequence` 用统一检测替换零散过滤 |
| `areno/api/trainers/dpo.py` | 修改 | `_record_to_train_pair` 集成偏好对相同检测 |
| `areno/api/trainer_config.py` | 修改 | 新增 `degenerate_policy` 配置项 |
| `areno/cli/train.py` | 修改 | 新增 `--degenerate-policy` CLI 参数 |
| `tests/test_degenerate_sample_cpu.py` | 新增 | CPU 测试 |
| `docs/troubleshooting/data-quality.rst` | 新增 | 用户文档 |

---

## 5. 测试计划

### 5.1 测试文件

`tests/test_degenerate_sample_cpu.py`

遵循项目测试约定：`unittest.TestCase` + 每方法写 docstring + `assertRaisesRegex`。

### 5.2 测试用例

| 编号 | 测试名 | 验证内容 |
|------|--------|---------|
| T1 | `test_normal_prompt_passes` | 正常 prompt 不被标记为退化 |
| T2 | `test_empty_prompt_detected` | 空字符串 `""` → `EMPTY` |
| T3 | `test_whitespace_prompt_detected` | `"   \n\t  "` → `WHITESPACE_ONLY` |
| T4 | `test_empty_response_detected` | 空 response → `EMPTY` |
| T5 | `test_whitespace_response_detected` | 纯空白 response → `WHITESPACE_ONLY` |
| T6 | `test_special_tokens_only_detected` | tokenize 后全是 special token → `SPECIAL_TOKENS_ONLY`（mock tokenizer） |
| T7 | `test_no_trainable_tokens_detected` | `[True, True, True]` 的 prompt_mask → `NO_TRAINABLE_TOKENS` |
| T8 | `test_identical_preference_detected` | chosen == rejected → `IDENTICAL_PREFERENCE_BRANCHES` |
| T9 | `test_policy_skip_returns_true` | policy=SKIP + 退化样本 → 返回 True（跳过） |
| T10 | `test_policy_error_raises` | policy=ERROR + 退化样本 → `ValueError` |
| T11 | `test_policy_disabled_passes` | config.enabled=False + 退化样本 → 不跳过 |
| T12 | `test_normal_sample_not_skipped` | 正常样本 + 任何 policy → 不跳过 |
| T13 | `test_all_degenerate_dataset_raises` | 全退化数据集 + policy=SKIP → 不静默成功，抛错 |
| T14 | `test_backward_compatible_default` | 默认配置 → 现有行为不变（空 response 被 skip） |

### 5.3 集成测试

使用微小的本地数据集 fixture 验证跨模块行为：

- 构造含 3 条记录的数据集（1 正常 + 1 空 response + 1 纯空白）
- 调用 `load_prompt_batches` 验证只有 1 条进入 batch
- 验证 `skipped_degenerate == 2` 且原因计数正确

---

## 6. 文档计划

### 6.1 新增文档

`docs/troubleshooting/data-quality.rst`：

```rst
:orphan:

Data quality and degenerate samples
====================================

AReno detects empty, whitespace-only, special-token-only, and
no-trainable-token samples before they enter the training pipeline.

Check:

* Dataset rows contain non-empty ``prompt`` and ``response`` fields.
* Responses are not whitespace-only or composed entirely of special tokens.
* DPO ``chosen`` and ``rejected`` branches are not identical.
* Use ``--degenerate-policy skip`` (default) to skip degenerate samples
  with reason counts, or ``--degenerate-policy error`` to fail fast.

Observable output:

  stage=data_filter skipped_degenerate=N (empty=M whitespace_only=K ...)
```

---

## 7. 实施顺序

| 步骤 | 内容 | 依赖 |
|------|------|------|
| 1 | 在 `areno/api/data.py` 中新增退化检测数据结构和检测函数 | 无 |
| 2 | 在 `tests/test_degenerate_sample_cpu.py` 中编写测试 | 步骤 1 |
| 3 | 运行测试验证基础设施正确 | 步骤 2 |
| 4 | 在 `trainer_config.py` 中新增 `DegenerateFilterConfig` | 步骤 1 |
| 5 | 在 `trainer.py` `load_prompt_batches` 中集成检测 | 步骤 4 |
| 6 | 在 `sft.py` 和 `dpo.py` 中集成统一检测 | 步骤 4 |
| 7 | 在 `cli/train.py` 中新增 `--degenerate-policy` 参数 | 步骤 4 |
| 8 | 运行全部测试确保向后兼容 | 步骤 5-7 |
| 9 | 新增文档 | 步骤 8 |
