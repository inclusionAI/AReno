# PR #311 自理解、Code Review 与运行验证

> PR: https://github.com/inclusionAI/AReno/pull/311
> Issue: https://github.com/inclusionAI/AReno/issues/199
> 分支: `feat/configurable-trainable-turns`
> 日期: 2026-07-28 ~ 30

---

## 一、对 PR 任务的理解

### 当前代码存在什么问题

AReno 的 agentic 训练路径对所有 assistant turn 一视同仁地计入 policy loss。但对很多任务来说，中间那些"我想想""让我调用一个工具"的 token 训练价值不高，真正该学的是最后那个答案。之前没有开关来控制"训哪些 turn"。

代码里有个 `LossMaskPolicy.final_assistant_text` 字段（`agentic.py:53`），看着像是为此设计的，但全仓 grep 下来没人读它——是个 dead flag。实际的掩码生成函数 `_response_loss_mask_for_span` 只用了 `assistant_text` 和 `assistant_tool_calls` 两个位，逐 span 给统一 bool，**无法判断"这是不是最后一个 span"**。也不存在 trainable token 计数，没有对非法 call/result 配对的 turn 级校验。

### 本 PR 的目标

1. 实现三种 trainable-turn 选择模式 + tool-call 参数屏蔽，逐 token 生效。
2. 对非法 call/result 配对发 turn 级错误，于 worker 初始化前抛出。
3. 默认行为向后兼容（默认 `all_assistant`，效果等价于现状）。
4. 通过 `TrainerConfig` + CLI 暴露，早失败（`click.UsageError` / `__post_init__` 校验）。
5. metrics 输出 `trainable_tokens` / `masked_response_tokens`，可在测试中断言。
6. CPU 测试覆盖逐 token mask、非法输入、边界、默认关闭。
7. 文档含一个可复制示例 + 契约/默认/输出/限制说明。
8. B 档接口形状预留（不实现动态判断，仅保证字段形状不排斥将来接入）。

### 本 PR 明确不处理的内容

- 不替换 trainer / rollout engine / dashboard 存储 / SDK 架构。
- 不引入外部数据库、托管控制面或重型依赖。
- 不扩 `reward_fn` 单 scalar 契约（`rewards.py:63`）。
- 不实现 per-trajectory / per-turn 动态 credit assignment（C 档，研究空白，独立后续 issue）。
- 不修复 `_tool_call_loss_mask` 双哨兵设计（非 bug，避 scope creep）。
- 不承诺 GPU 收敛 ablation 全跑通（视算力而定）。

### 修改会影响哪些模块、接口或使用场景

- `areno/api/agentic.py`：核心逻辑——`LossMaskPolicy` 扩字段、`_AgentSample` 增 `response_spans`、新增 `_apply_trainable_turn_mode` / `_validate_call_result_pairing` / `_tool_call_arg_token_range` / `_select_trainable_span_indices` / `ResponseSpan`。移除 dead `final_assistant_text`。
- `areno/api/trainer_config.py`（ask-first）：`TrainerConfig` 加字段 + `__post_init__` 校验。
- `areno/api/trainers/policy_only.py`：`_loss_mask_policy()` 映射 config→policy；rollout 日志增字段。
- `areno/cli/train.py`（ask-first）：CLI 选项 `--trainable-turns` / `--mask-tool-call-args`。
- `tests/test_agentic_cpu.py`：21 个新 CPU 测试。
- docs + `examples/agentic/trainable_turns_demo.py`。

使用场景影响：默认行为不变；用户显式传 `--trainable-turns` 或 `--mask-tool-call-args` 时生效。两条代码路径（HTTP-proxy 与 explicit-trajectory）都经过唯一 chokepoint `_train_rows_from_samples`，不会出现路径分叉。

### 完成任务的验收标准是什么

1. 固定多工具 transcript，逐 token 断言四种配置的 mask；非法 call/result 对以 turn 级错误拒绝。
2. 复用 AReno 现有契约，无外部 DB / 强制 sandbox。
3. 默认行为向后兼容。
4. 自动测试覆盖成功 / 非法 / 边界 / 失败路径。
5. 文档含最小可运行示例 + 可观测输出说明。
6. （研究化）三模式 trainable-token 统计脚本产出可量化数据。

---

## 二、实现思路

### 修改涉及的主要文件和模块

核心逻辑全加在 `agentic.py` 的数据路径里（+211 行），没动 trainer 或 rollout 引擎。配置层加 6 行，映射层加 10 行，CLI 加 26 行，测试加 346 行，docs + demo 加 ~190 行。总共 10 文件 +781/-5，无新依赖。

### 核心流程或数据流

```
trajectory 组装期                唯一 chokepoint              输出
─────────────                   ────────────                ────
_append_sample_response         _train_rows_from_samples
  · response_spans 清单捕获  ──▶  _validate_call_result_pairing  ──▶  loss_masks
  (每轮 kind+length)              _apply_trainable_turn_mode
                                   (组合式重写)
```

1. `ResponseSpan(kind, length)` 在 `_set_sample_training_row` 初始化、`_append_sample_response` 追加。组装期每轮 kind 都在，捕获用完即弃。
2. `_apply_trainable_turn_mode` 挂在 `_train_rows_from_samples` 循环顶部——HTTP-proxy 路径（`_sample_from_pending_chat`）和 explicit-trajectory 路径（`_sample_from_trajectory_turn`，最终也调 `_sample_from_pending_chat`）都走这里。
3. mask 组合顺序铁律：已有 per-span mask（含 `_tool_call_loss_mask` 的 result 区屏蔽）→ 叠加 arg 屏蔽 → 叠加 turn selection。**绝不从零重建**，否则丢掉已有屏蔽。

### 关键数据结构、接口或算法

- `LossSelectionMode = Literal["all_assistant","last_assistant","final_answer"]`——模式枚举。
- `ResponseSpan(kind, length)`——组装期捕获的 span 清单，`_AgentSample.response_spans`。
- `_apply_trainable_turn_mode(sample)`——在 `_train_rows_from_samples` 循环顶部调用；先取已有 `loss_mask_override` 作为 base，再按 mode + mask_args 叠加修改，写回 `loss_mask_override` 和 `loss_mask_row`（通过 `response_mask_row` 逐 token 映射回写）。
- `_select_trainable_span_indices(spans, mode)`——`last_assistant` 取 `assistant_indices[-1]`；`final_answer` 找最后一个 `assistant_tool_call` 后面第一个 `assistant_text`，无 tool_call 时退化为 last assistant span，bare trailing tool_call 时返回空集（零信号）。
- `_tool_call_arg_token_range(tokenizer, span_tokens)`——decode → 搜 `"arguments"` → brace-matching JSON value → encode 定位 token 范围。标注 approximate（decode/encode 非 round-trip）。
- `_validate_call_result_pairing(samples)`——基于 `trace` 事件（`assistant_tool_call` type）配对 `tool` 消息计数。mid-trajectory call 没结果 → `ValueError`。bare trailing tool call 豁免。orphan tool result 容忍。
- `_AgentTrainRows` 增 `trainable_tokens` / `masked_response_tokens` 字段。

### 重要设计选择及理由

**为什么落 chokepoint 而非 per-span 函数。** 原计划说三模式都落在 `_response_loss_mask_for_span`。但它签名 `(response_kind, response_len)` 是单 span 视角，判不了"是不是最后一个 span"。而 `_append_sample_response` 把 `response_kind` 折叠成只剩末轮的 kind，前面几轮信息组装后就丢了。所以必须在组装期先捕获 span 清单，再在 trajectory 完整后的 `_train_rows_from_samples` 统一入口做后处理。

**为什么 mask 用组合而非重建。** 代码里已有一套 mask 在屏蔽 tool-result 区（`_tool_call_loss_mask` 用 `<|tool_response>` / `<|im_end>` 双哨兵）。如果新功能从零重建 mask，会把已有屏蔽覆盖掉。只有同时开 arg 屏蔽和 result 屏蔽才能看出差别（默认模式下两者结果一样），所以这个顺序错了不一定能被测试暴露。

**为什么容忍 orphan tool result。** 现有测试 fixture 的 messages 含 `tool` 消息但不带结构化 `tool_calls` 字段。如果严格检查会破坏现有测试，是有意取舍。

**为什么 B 档只预留不实现。** `response_spans` 清单既是 A 档基础设施也是 B 档的物理基础。未来插入 scorer 无需动奖励管线或 trainer。本期不实现 callable，仅保证字段形状不排斥其接入。

### 是否考虑过其他方案，以及没有采用的原因

- 在 `_response_loss_mask_for_span` 传 trajectory 上下文——否决：破坏其纯函数性与签名，且 proxy 路径单 turn 调用时无 trajectory 视角。
- 在 `_append_sample_response` 内即时重写 mask——否决：span 列表此时可能未完整（后续还有 turn），无法判"最后"。
- 从 messages 的 `tool_calls` 字段做校验——否决：现有 fixture 不含结构化 `tool_calls`，会把合法数据判成非法。改用 `trace` 事件配对。

### 兼容性、性能、异常处理

- 默认 `all_assistant` + `mask_tool_call_args=False` 与改前完全一致。`test_trainable_turns_all_assistant_full_row_default_parity` 断言 parity。
- 性能：`_apply_trainable_turn_mode` 在默认模式下早返回零开销。非默认模式每 sample 一次 O(n) 遍历回写 `loss_mask_row`。
- 异常处理：非法 `trainable_turns` 值在 `__post_init__` 和 `click.Choice` 双层早失败。`_tool_call_arg_token_range` 多层 `try/except` 防御 tokenizer 不支持 decode/encode。

---

## 三、对自己代码的 Review

以 reviewer 视角逐项检查：

### 正确性

- 正常输入：三种模式 + arg 屏蔽的组合有 18 个逐 token 测试覆盖。
- 边界输入：bare trailing tool_call（`final_answer` 零信号）、空响应（合法不报错）、无 tool_result 退化、orphan tool result（容忍）都有测试。
- **发现并处理**：`test_explicit_trajectory_path_last_assistant_masks_correctly` 初版断言漏算 inter-turn context token，实测后修正为 `[False, False, False, False, True, True]`。

### 可读性

- 6 个新增定义全部有 docstring，内容覆盖"做什么 + 为什么 + 关键约束"。
- 命名遵循仓库风格（`_` 前缀私有约定）。行内注释密度遵循仓库风格（重 docstring 轻行内注释）。

### 复用性

- 没有重复代码。`_apply_trainable_turn_mode` 复用 `_response_loss_mask` 取已有 mask。`ResponseSpan` 为 B 档预留复用基础。

### 兼容性

- 默认行为 parity 有测试断言。`_AgentTrainRows` 新字段用默认值 `0` 不破坏现有调用方。`TrainerConfig` 用 `str` 类型注解避免循环依赖。`policy_only` 用 `getattr` 带默认值兼容旧 config。
- **发现并处理**：移除 dead flag `final_assistant_text` 属于删公共 API（ask-first），经确认全仓无读取后安全移除。

### 异常处理

- `TrainerConfig.__post_init__` 校验 + CLI `click.Choice` 早失败，错误信息清晰。
- `_validate_call_result_pairing` 的 `ValueError` 明确说出"tool call without a matching tool result"。
- `_tool_call_arg_token_range` 定位失败返回 `None`，调用方跳过不改——不静默错误也不崩溃。
- **发现并处理**：原计划要求 orphan tool result 也报错，实现期发现现有 fixture 依赖此形态，改为容忍（spec.md 同步更新）。

### 测试

- 21 个新增 CPU 测试覆盖三模式逐 token mask、arg 屏蔽开/关、校验四场景、默认 parity、metrics、dual-path、`response_spans` 组装、log 断言、配置早失败。
- 现有 43 个回归测试全绿。总计 64 passed。

### 性能

- 默认路径早返回零开销。非默认模式每 sample 一次 O(n) 遍历。`response_spans` 是轻量 list of small dataclass，用完即弃。未引入额外内存。

### 提交范围

- 改动严格限定 10 文件，全部与任务相关。
- ruff lint / format 修复独立 commit 标注 `style:`，没混入无关格式化。
- git diff 检查确认没删原有注释（删除行仅 dead flag + 扩展替换的日志格式串，信息只增不减）。

---

## 四、遇到的问题、挑战与解决方法

### 问题 1：计划文档把 `_tool_call_loss_mask` 误判为 bug

1. **现象**：原 issue 计划 §5.1 和 §9 写"markers 两元素相同，疑似 bug"。
2. **定位过程**：`grep "markers ="` 精确定位 L862，Read 确认实际值。
3. **根因**：计划写时没实读代码，假设了两个 marker 相同。
4. **解决方法**：design.md 记校正，`mask_tool_call_args` 不复用该定位（不同区域），不"修"这个非 bug。
5. **验证方式**：确认 `("<|tool_response>", "<|im_end>")` 两个不同 sentinel；现有测试验证行为正确。
6. **经验总结**：不能盲信设计文档，到代码里验证才能下结论。

### 问题 2：落点跟计划不一致

1. **现象**：计划说三模式在 `_response_loss_mask_for_span` 落实。
2. **定位过程**：读签名 `(response_kind, response_len)` 和 `_append_sample_response` 的 `response_kind` 折叠行为。
3. **根因**：计划没追到 `response_kind` 被折叠，per-span 函数无法判"最后"。
4. **解决方法**：新增 `response_spans` 清单组装期捕获，`_apply_trainable_turn_mode` 挂 chokepoint 后处理。
5. **验证方式**：逐 token 测试断言三模式行为正确。
6. **经验总结**：落点决策要追到数据流末端，不能停在函数签名层。

### 问题 3：call/result 校验第一版写错

1. **现象**：第一版校验把现有 fixture 判成非法。
2. **定位过程**：读 fixture L693-698，messages 含 `tool` role 但不含 `tool_calls` 字段。
3. **根因**：没搞清现有数据形态就写校验。
4. **解决方法**：改用 `trace` 事件配对，容忍 orphan，bare trailing 豁免。spec.md 同步。
5. **验证方式**：`test_call_result_pairing_tolerates_orphan_tool_result` 断言不报错；原 fixture 全绿。
6. **经验总结**：写校验前先搞清楚现有数据长什么样。

### 问题 4：本机环境跑不了测试

1. **现象**：macOS 只有 Python 3.9（项目要 3.10+），无 torch / pytest。
2. **定位过程**：`pip install torch` 在 3.9 版本回溯慢；`import areno` 触发 torch 缺失。
3. **根因**：Python 版本低于项目要求 + 缺核心依赖。
4. **解决方法**：`uv` 拉独立 Python 3.12 venv + 清华镜像源装依赖。
5. **验证方式**：64 passed。
6. **经验总结**：任务开始前先确认运行环境。

### 问题 5：PR 描述超长被拒

1. **现象**：GitHub 报 "Body is too long (maximum is 65536 characters)"，但正文才 ~2KB。
2. **定位过程**：检查发现从 IDE 复制时混入了隐藏 `<rule>` / `<system-reminder>` 标签。
3. **根因**：IDE 注入隐藏上下文标签混入剪贴板。
4. **解决方法**：用纯 Markdown 文件写描述，`gh pr create --body-file` 或终端 `cat` 复制。
5. **验证方式**：GitHub API 确认 body_len = 2166 < 65536。
6. **经验总结**：从 AI 协作工具复制长文本警惕隐藏注入标签。

### 问题 6：Kaggle GPU 训练 OOM

1. **现象**：单 T4（14.5GB）跑 Qwen3-0.6B GSPO 报 OOM，在 `lm_head` 反向传播阶段。
2. **定位过程**：报错栈到 `areno/accel/linear.py`；分析为 rollout KV cache + 反向传播 + 优化器同时驻留。
3. **根因**：单卡显存不够 GSPO 完整流程。
4. **解决方法**：双卡 TP + `--drop-rollout-state --eager-decode` + 缩短序列 + `expandable_segments:True`。
5. **验证方式**：两次（`all_assistant` / `final_answer`）都跑到 `max_steps_reached`。
6. **经验总结**：小显存 GPU 跑 RL 要先估算峰值显存，用 TP/缩 batch/缩序列/drop-rollout-state 组合降峰。

---

## 五、分步骤运行结果证明

### Step 0：环境准备

**步骤目的**：建立可运行测试的 Python 环境。

**完整命令**：
```bash
cd /Users/dimlights/AReno/AReno-1
~/Library/Python/3.9/bin/uv venv .venv --python 3.12
export VIRTUAL_ENV="$(pwd)/.venv"
~/Library/Python/3.9/bin/uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch numpy pydantic pytest openai click safetensors ruff
```

**关键输出**：
```
Using CPython 3.12.13
Creating virtual environment at: .venv
 + torch==2.13.0
 + pytest==9.1.1
 + pydantic==2.13.4
```

**解释**：macOS 开发机无项目所需 Python 3.10+ 和 torch。用 uv 拉独立 Python 3.12 venv，清华镜像源加速。

---

### Step 1：全量 CPU 测试

**步骤目的**：验证核心逻辑 + 回归不破坏。

**完整命令**：
```bash
.venv/bin/python -m pytest tests/test_agentic_cpu.py -q
```

**关键输出**：
```
................................................................          [100%]
64 passed in 2.80s
```

**解释**：64 测试全绿（21 新增 + 43 回归），0 失败。

---

### Step 2：新增测试逐项验证

**步骤目的**：确认每个新增用例独立通过。

**完整命令**：
```bash
.venv/bin/python -m pytest tests/test_agentic_cpu.py -v -k "trainable_turns or mask_tool_call_args or call_result_pairing or empty_response or trainer_config_rejects or loss_mask_policy_defaults or full_row or response_spans_populated or explicit_trajectory_path or rollout_log"
```

**关键输出**：
```
test_loss_mask_policy_defaults_new_fields PASSED
test_trainable_turns_last_assistant_masks_prior_spans PASSED
test_trainable_turns_final_answer_targets_post_tool_result_text PASSED
test_trainable_turns_final_answer_degenerates_without_tool_result PASSED
test_trainable_turns_final_answer_bare_trailing_tool_call_zero_signal PASSED
test_trainable_turns_last_assistant_keeps_trailing_tool_call PASSED
test_mask_tool_call_args_masks_arguments_keeps_name PASSED
test_mask_tool_call_args_composes_with_existing_suppression PASSED
test_call_result_pairing_rejects_mid_call_without_result PASSED
test_call_result_pairing_allows_bare_trailing_tool_call PASSED
test_call_result_pairing_tolerates_orphan_tool_result PASSED
test_empty_response_tokens_not_invalid PASSED
test_trainable_turns_all_assistant_full_row_default_parity PASSED
test_trainable_turns_last_assistant_full_row_and_metrics PASSED
test_response_spans_populated_after_multi_turn_assembly PASSED
test_explicit_trajectory_path_last_assistant_masks_correctly PASSED
test_rollout_log_records_active_mode_and_mask_state PASSED
test_trainer_config_rejects_invalid_trainable_turns PASSED
```

**解释**：18 个新增测试全部 PASSED。

---

### Step 3：功能验证脚本

**步骤目的**：快速验证默认值、配置校验、导入。

**完整命令**：
```bash
.venv/bin/python -c "
from areno.api.agentic import LossMaskPolicy, ResponseSpan, LossSelectionMode
from areno.api.trainer_config import TrainerConfig
p = LossMaskPolicy()
print('1. 默认模式:', p.trainable_turns, '| arg屏蔽:', p.mask_tool_call_args, '| dead flag已移除:', not hasattr(p, 'final_assistant_text'))
try:
    TrainerConfig(algo='gspo', ckpt='x', dataset_path='y', trainable_turns='bogus')
except ValueError as e:
    print('2. 非法模式被拒:', e)
span = ResponseSpan(kind='assistant_text', length=3)
print('3. ResponseSpan:', span)
print('   LossSelectionMode:', LossSelectionMode.__args__)
print('全部验证通过')
"
```

**关键输出**：
```
1. 默认模式: all_assistant | arg屏蔽: False | dead flag已移除: True
2. 非法模式被拒: trainable_turns must be one of: all_assistant, last_assistant, final_answer
3. ResponseSpan: ResponseSpan(kind='assistant_text', length=3)
   LossSelectionMode: ('all_assistant', 'last_assistant', 'final_answer')
全部验证通过
```

**解释**：默认值正确（向后兼容）；dead flag 移除；非法值被拒；新符号可导入。

---

### Step 4：demo 脚本（Kaggle 实跑）

**步骤目的**：直观展示三模式逐 token mask 差异 + 非法输入拒绝。

**完整命令**：
```bash
python examples/agentic/trainable_turns_demo.py
```

**关键输出**：
```
Trajectory: assistant_text(2) | tool_call(2) | assistant_text(2)
tokens: [10, 11, 12, 13, 20, 21]

  all_assistant    loss_mask=[True, True, True, True, True, True]  trainable_tokens=6
  last_assistant   loss_mask=[False, False, False, False, True, True]  trainable_tokens=2
  final_answer    loss_mask=[False, False, False, False, True, True]  trainable_tokens=2

validation rejected malformed trajectory: agentic trajectory has a tool call without a matching tool result
```

**解释**：三模式效果一目了然——全训 6 个 vs 只训最后 2 个。非法 trajectory 被拒绝。

---

### Step 5：GPU 训练（Kaggle 双 T4 实跑）

**步骤目的**：验证 CLI→config 传递 + 新字段不破坏训练。

**完整命令**：
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True areno train \
  --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo --tp-size 2 --world-size 2 \
  --batch-size 2 --n-samples 2 --max-steps 2 \
  --max-prompt-tokens 384 --max-new-tokens 128 \
  --mini-bs 1 --score-micro-bs 1 \
  --gradient-accumulation-steps 4 \
  --activation-checkpointing --drop-rollout-state --eager-decode \
  --trainable-turns final_answer
```

**关键输出**（config summary 片段）：
```
Rollout
-------
  trainable_turns      final_answer
  mask_tool_call_args  no
...
epoch=0 step=2 stage=max_steps_reached
```

**解释**：config 正确显示 `final_answer`，2 步训练正常完成没崩溃。走标准 GSPO 路径（无 `--agent-fn`），mask 逻辑由 CPU 测试覆盖。

---

### Step 6：ruff lint + format

**步骤目的**：确认代码符合项目 lint 规范。

**完整命令**：
```bash
.venv/bin/python -m ruff check areno/api/agentic.py areno/api/trainer_config.py areno/api/trainers/policy_only.py areno/cli/train.py tests/test_agentic_cpu.py examples/agentic/trainable_turns_demo.py
.venv/bin/python -m ruff format areno/api/agentic.py areno/api/trainer_config.py areno/api/trainers/policy_only.py areno/cli/train.py tests/test_agentic_cpu.py examples/agentic/trainable_turns_demo.py
```

**关键输出**：
```
All checks passed!
5 files reformatted, 1 file left unchanged
```

**解释**：ruff check 零违规（修了 import 排序/未用 import/缺尾换行）。ruff format 行宽 120 合并（零逻辑改动）。

---

### Kaggle 运行截图

**截图 1：CPU 测试 + demo 脚本**
![Kaggle CPU 测试与 demo](screenshots/kaggle-cpu-test-demo.png)

**截图 2：GPU 训练日志**
![Kaggle GSPO 训练日志](screenshots/kaggle-training-log.png)

---

### 改动统计

```
 areno/api/agentic.py                     | 211 ++++++++++++++++++-
 areno/api/trainer_config.py              |   6 +
 areno/api/trainers/policy_only.py        |  10 +-
 areno/cli/train.py                       |  26 +++
 docs/cli/observability.rst               |   2 +-
 docs/cli/training.rst                    |  29 +++
 docs/sdk/trainer.rst                     |  17 +-
 docs/troubleshooting/agentic-rollout.rst |  22 ++
 examples/agentic/trainable_turns_demo.py | 117 +++++++++++
 tests/test_agentic_cpu.py                | 346 +++++++++++++++++++++++++++++++
 10 files changed, 781 insertions(+), 5 deletions(-)
```