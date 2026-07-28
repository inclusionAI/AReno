## Why

AReno 的 agentic 训练路径对所有 assistant turn 一视同仁地计入 policy loss：既无"可训练 turn 选择"模式，也无法屏蔽 tool-call 参数 token。对照代码事实，`LossMaskPolicy.final_assistant_text`（`areno/api/agentic.py:53`）是 dead flag——全仓仅定义无读取，`_response_loss_mask_for_span`（L378）只依据 `assistant_text`/`assistant_tool_calls` 两个位。对外仅暴露 `train_tool_results: bool`（`trainer_config.py:61`）。不存在"最后一轮 / 仅最终答案"模式，无 trainable token 计数，无对非法 call/result 配对的 turn 级校验。

这是 GitHub issue [inclusionAI/AReno#199](https://github.com/inclusionAI/AReno/issues/199)（*Make trainable turns configurable for agentic trajectories*）要求填补的真实缺口。本 change 落静态可配置 mask（A 档），并预留 per-trajectory 判断的接口形状（B 档），不实现动态 credit assignment（C 档，研究空白，独立后续 issue）。

## What Changes

- 新增三种 trainable-turn 模式，**逐 token** 生效：`all_assistant`（默认，向后兼容）/ `last_assistant`（仅最后一个 assistant span）/ `final_answer`（仅最后一个 tool result 之后的 final `assistant_text` span，无 tool result 时退化为 last assistant span）。
- 新增 `mask_tool_call_args` 选项：在 tool-call turn 内屏蔽 JSON 参数 token，保留 tool-name/action token 可训。定位为**研究 ablation**（业界 ToolFormer/Gorilla/ToolACE 等一律整段训 tool-call，无公开工作做参数内屏蔽），文档须显式标注与业界惯例的偏离。
- **逻辑落点修正**：`last_assistant`/`final_answer` 需 trajectory 级 span 信息，**不能**落在单 span 视角的 `_response_loss_mask_for_span`。新增 `response_spans` 清单（append 期捕获，用完即弃）+ `_apply_trainable_turn_mode` 挂在 `_train_rows_from_samples` 唯一 chokepoint 做组合式重写。
- 新增 call/result 配对 turn 级校验（worker init 前 `ValueError`）；显式排除空 `response_tokens`（`_run_chat_request` 空 fallback 是合法路径）。
- 通过 `TrainerConfig` + CLI 暴露（**ask-first 区**，需 maintainer 认可），早失败（`click.UsageError` / `__post_init__`）。
- metrics 输出 `trainable_tokens` / `masked_response_tokens`，可在测试中断言。
- **BREAKING（内部 dead flag）**：移除/弃用 `LossMaskPolicy.final_assistant_text`（全仓无读取，含 `docs/sdk/trainer.rst` 文档声明）。非公共 API 破坏，PR 说明弃用。
- **B 档接口预留**：`LossMaskPolicy` 字段保留可扩展形态；A 档交付的 `response_spans` 清单即 B 档 per-trajectory scorer 能"无须破坏性返工"接入的物理基础。**本期不实现 callable，仅保证字段形状与数据结构不排斥其接入。**

## Capabilities

### New Capabilities

- `agentic-loss-masking`: 控制 agentic trajectory 中哪些 assistant span / token 计入 policy loss 的可配置规则——三种 trainable-turn 模式、tool-call 参数屏蔽、call/result 配对校验、trainable token 可观测输出，以及为 per-trajectory 动态判断预留的接口形状。

### Modified Capabilities

<!-- openspec/specs/ 当前为空，无现存 capability 的 requirement 变更。 -->

## Impact

- **核心逻辑**（`areno/api/agentic.py`）：`LossMaskPolicy` 扩字段、`_AgentSample` 增 `response_spans`、新增 `_apply_trainable_turn_mode`、call/result 校验函数。**不在 AGENTS.md ask-first 清单**，可先行。
- **配置 + CLI**（ask-first）：`areno/api/trainer_config.py`（`TrainerConfig` 加字段 + `__post_init__`）、`areno/cli/train.py`（`--trainable-turns` / `--mask-tool-call-args`）。
- **接入映射**：`areno/api/trainers/policy_only.py` `_loss_mask_policy()`（L179）映射 config→policy。
- **dual-path 覆盖**：`mask_tool_call_args` 须同时覆盖 `_sample_from_pending_chat`（L607，HTTP-proxy 路径）与 `_append_sample_response`（L627，explicit-trajectory 路径）。
- **测试**：`tests/test_agentic_cpu.py` 逐 token mask、非法输入、边界、metrics 断言（CPU 可验，无需 GPU）。
- **文档**：CLI 指南、`docs/sdk/trainer.rst`（弃用 `final_assistant_text` 声明）、可复制示例。
- **依赖**：无新增（复用现有 tokenizer/chat-template 契约）。
- **不替换** trainer / rollout engine / dashboard / SDK 架构；不引入外部 DB；不扩 `reward_fn` 单 scalar 契约（`rewards.py:63`）。