# AReno Issue #199 — 研究进度交接文档

> **用途**：供另一台机器（有 GPU）快速了解当前已完成内容与后续待办。
>
> 仓库：`https://github.com/sliverdancer/AReno.git`
> 分支：`feat/configurable-trainable-turns`
> 上游 issue：[inclusionAI/AReno#199](https://github.com/inclusionAI/AReno/issues/199)

---

## 1. 任务背景

Issue #199 要求：让 agentic trajectory 的 trainable turns 可配置。

AReno 原先对所有 assistant turn 一视同仁地计入 policy loss，无法选择性训练
特定 turn 或屏蔽 tool-call 参数 token。本任务在现有 `LossMaskPolicy` 数据
契约上窄改，实现三种 trainable-turn 模式 + tool-call 参数屏蔽。

**双重定位**：开源工程交付 + 硕士申请（agentic RL）研究化素材。

---

## 2. 已完成内容

### 2.1 Phase 1–5（A 档：静态配置，已全部落地）

提交 `1cf414a`，以下改动均已合入分支：

**核心逻辑** — `areno/api/agentic.py`
- `LossSelectionMode = Literal["all_assistant", "last_assistant", "final_answer"]`
- `LossMaskPolicy` 增字段 `trainable_turns`（默认 `"all_assistant"`）、`mask_tool_call_args`（默认 `False`）
- 移除 dead flag `final_assistant_text`
- `_apply_trainable_turn_mode()`：在 trajectory 级别按模式选择 trainable span + 屏蔽 tool-call 参数
- `_select_trainable_span_indices()`：三模式的 span 选择逻辑
- `_tool_call_arg_token_range()`：通过 decode + brace-matching 定位 JSON arguments 值的 token range（近似值）
- `_validate_call_result_pairing()`：tool-call/tool-result 配对校验（trailing bare call 合法，orphan tool result 容忍）

**配置** — `areno/api/trainer_config.py`
- `TrainerConfig` 增 `trainable_turns: str = "all_assistant"`、`mask_tool_call_args: bool = False`
- `__post_init__` 校验字面量集合

**CLI** — `areno/cli/train.py`
- `--trainable-turns` / `--mask-tool-call-args` 选项
- 非法值 `click.UsageError`（早失败）
- 纳入 config summary

**Trainer 接入** — `areno/api/trainers/policy_only.py`
- `_loss_mask_policy()` 映射 config → policy
- metrics 输出 `trainable_tokens` / `masked_response_tokens`

**CPU 测试** — `tests/test_agentic_cpu.py`
- 16 个相关测试（三模式逐 token mask、非法输入、边界、metrics、config 校验）
- 全文件 64 个测试

**Demo** — `examples/agentic/trainable_turns_demo.py`
- 无网络无 GPU 的确定性 demo（fake tokenizer）

### 2.2 Phase 6 / G8（研究化扩展，T6.1 已完成）

提交 `ff15d8b`，以下为本次新增产出：

| 文件 | 说明 |
|---|---|
| `examples/agentic/trainable_turns_stats.py` | CPU 统计脚本：6 trajectory × 3 mode × 2 mask = 36 数据点，走真实 `RolloutSession` 代码路径 |
| `examples/agentic/trainable_turns_stats.json` | 结构化数据（JSON） |
| `examples/agentic/trainable_turns_stats.csv` | 表格数据（CSV） |
| `examples/agentic/trainable_turns_research_notes.md` | 文献定位（6 Part）：RFT/StarPO-S/MT-GRPO/PRM/STM/veRL 方法谱系 + AReno #199 定位 + 统计发现 |
| `issue-199-execution-plan.md` | 完整 spec-mode 执行计划 |
| `openspec/` | OpenSpec change artifacts |

### 2.3 统计关键发现

跨 6 个 fixture（922 total response tokens）的汇总：

| 模式 | mask_args=False | mask_args=True |
|---|---|---|
| all_assistant | 100.0% (922/922) | 79.8% (736/922) |
| last_assistant | 26.6% (245/922) | 24.0% (221/922) |
| final_answer | 20.6% (190/922) | 20.6% (190/922) |

1. Turn 选择使监督密度降 5x（100% → ~21%）
2. `mask_tool_call_args` 仅在 `all_assistant` 下显著（-20pp），在 turn-level 模式下几乎冗余
3. `bare_trailing` fixture 揭示 `last_assistant`（65.5%）vs `final_answer`（0%）的关键分歧

---

## 3. 未完成内容（需 GPU 机器执行）

### 3.1 T6.2 — 小规模 GPU Ablation（核心待办）

**目标**：对比三种 `trainable_turns` 模式在小模型 + 少步训练下的收敛趋势和最终奖励。

**推荐配置**（可按算力调整）：

```bash
# 基础环境（按 AGENTS.md）
pip install psutil flash-linear-attention
pip install -e . --no-build-isolation

# 三组对比（最小模型 + agentic 环境）
# 组1: all_assistant（baseline）
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path <agentic-dataset> \
  --reward-fn-path <reward.py> --algo gspo --tp-size <GPU数> \
  --trainable-turns all_assistant --max-steps 50

# 组2: last_assistant
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path <agentic-dataset> \
  --reward-fn-path <reward.py> --algo gspo --tp-size <GPU数> \
  --trainable-turns last_assistant --max-steps 50

# 组3: final_answer
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path <agentic-dataset> \
  --reward-fn-path <reward.py> --algo gspo --tp-size <GPU数> \
  --trainable-turns final_answer --max-steps 50
```

**可用的 agentic 环境**（`examples/agentic/` 下）：
- `tictactoe/` — 井字棋，轻量，适合快速验证
- `shopping/` — 购物任务
- `duelgrid/` — 网格对抗
- `coding/` — 代码任务

**需要记录的指标**：
- 每 step 的 `trainable_tokens` / `masked_response_tokens`（已在 metrics 输出）
- 每 step 的 reward 均值/方差
- 收敛步数（reward 不再增长的点）
- 最终 reward

**产出**：JSON/CSV 数据表 + 简要文字对比（可追加到 `trainable_turns_research_notes.md`）

### 3.2 T6.3 — 研究叙述整合（T6.2 完成后）

将 GPU ablation 数据编入 `trainable_turns_research_notes.md` 的 Part IV，
回答核心研究问题：监督密度的 5x 差异是否转化为收敛速度/质量的差异？

### 3.3 PR（可选，T6.2 后或现在）

向上游 `inclusionAI/AReno` 发起 PR：
```bash
gh pr create --repo inclusionAI/AReno \
  --head sliverdancer:feat/configurable-trainable-turns --base main \
  --title "feat: configurable trainable-turn selection for agentic trajectories (closes #199)"
```

---

## 4. 环境快速搭建

```bash
# 1. 拉取
git clone -b feat/configurable-trainable-turns https://github.com/sliverdancer/AReno.git
cd AReno

# 2. 确认 GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# 3. 安装（需要 Linux + NVIDIA GPU + CUDA + PyTorch >= 2.6）
pip install psutil flash-linear-attention
pip install -e . --no-build-isolation
# 可选: pip install flash-attn  # 仅 --attn-backend flash 时需要

# 4. 验证 CPU 测试全绿（不需要 GPU）
pytest tests/ -k cpu

# 5. 运行统计脚本验证（不需要 GPU）
python examples/agentic/trainable_turns_stats.py
```

---

## 5. 关键文件索引

| 文件 | 作用 |
|---|---|
| `areno/api/agentic.py:46` | `LossSelectionMode` 定义 |
| `areno/api/agentic.py:59-69` | `LossMaskPolicy` dataclass |
| `areno/api/agentic.py:419-471` | `_apply_trainable_turn_mode` — 核心模式逻辑 |
| `areno/api/agentic.py:473-499` | `_validate_call_result_pairing` — 配对校验 |
| `areno/api/agentic.py:1005-1030` | `_select_trainable_span_indices` — span 选择 |
| `areno/api/agentic.py:1033-1086` | `_tool_call_arg_token_range` — 参数 token 定位 |
| `areno/api/trainer_config.py:62-63` | `TrainerConfig` 新字段 |
| `areno/api/trainer_config.py:71-74` | `__post_init__` 校验 |
| `areno/cli/train.py:1318-1325` | CLI 新选项定义 |
| `areno/api/trainers/policy_only.py:179` | `_loss_mask_policy()` config→policy 映射 |
| `areno/api/trainers/policy_only.py:254-259` | metrics 输出 |
| `tests/test_agentic_cpu.py:1132-1473` | 16 个相关 CPU 测试 |
| `examples/agentic/trainable_turns_stats.py` | 统计脚本 |
| `examples/agentic/trainable_turns_research_notes.md` | 文献定位 + 研究叙述 |
| `issue-199-execution-plan.md` | 完整执行计划（spec mode） |

---

## 6. 设计要点速览

### 三模式语义

```
trajectory: text₁ → tool_call → tool_result → text₂ → tool_call → text₃

all_assistant:    [text₁] [tool_call] [text₂] [tool_call] [text₃]  ← 全训
last_assistant:                                         [text₃]      ← 仅最后一个 span
final_answer:                              [text₃]                    ← 仅最后一个 tool_call 之后的 text
```

- `final_answer` 无 tool call 时退化为 `last_assistant`
- `final_answer` 结尾是 bare tool call（无后续 text）时 → trainable = 0

### mask_tool_call_args

在 tool-call span 内，通过 decode + brace-matching 定位 `"arguments":{...}` 的 JSON
值，屏蔽这些 token，保留 tool name 可训。定位为**近似值**（decode/encode 非 round-trip），
CPU 逐 token 测试已钉死行为。

### 已知限制

1. `_tool_call_loss_mask`（agentic L983）的 markers 两元素相同（`<|tool_response>`, `<|tool_response>`），疑似 bug，不在本期修复范围
2. `_append_sample_response`（L747）折叠 turn 边界到单条 `loss_mask_override`，C 档 per-turn 判断受此阻塞
3. `reward_fn` 契约为单 scalar，无 per-turn reward hook

### A/B/C 档定位

| 档位 | 范围 | 状态 |
|---|---|---|
| A 静态配置 | 三模式 + mask_tool_call_args | ✅ 已交付 |
| B per-trajectory scorer | optional callable（每条 trajectory 打分→mask） | ⏸ 接口形状预留，未实现 |
| C per-turn credit assignment | per-turn reward/advantage | ❌ 独立后续 issue（需扩 reward 契约 + 补 turn offset） |

---

## 7. 文献锚点

| 工作 | arXiv | 与本工作的关系 |
|---|---|---|
| RFT (Yuan 2023) | [2308.01825](https://arxiv.org/abs/2308.01825) | trajectory 级过滤，不选 turn |
| StarPO-S (Wang 2025) | [2504.20073](https://arxiv.org/abs/2504.20073) | trajectory 级过滤 + 稳定性 |
| MT-GRPO (Zeng 2025) | [2505.11821](https://arxiv.org/abs/2505.11821) | turn 级 advantage，C 档锚点 |
| PRM (Lightman 2023) | [2305.20050](https://arxiv.org/abs/2305.20050) | step 级 process reward |
| STM (Wu 2025) | [2501.14315](https://arxiv.org/abs/2501.14315) | token 级 perplexity mask（SFT） |
| veRL delta-tok | (工程) | assistant/environment 二分，本工作细化到 span 级 |

详见 `examples/agentic/trainable_turns_research_notes.md`。

---

*文档生成时间：2026-07-28。分支 `feat/configurable-trainable-turns`，最新 commit `ff15d8b`。*