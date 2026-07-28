## Context

AReno 的 agentic 训练把多工具 trajectory 折叠成单条训练行：`_append_sample_response`（`areno/api/agentic.py:627`）把每轮 response tokens 拼接，`loss_mask_override` 在 L665 拍平成 `old+new`，`response_kind` 在 L666 折叠成只剩末轮。policy loss 的可训练区由 `LossMaskPolicy`（L46-55）控制，但 `_response_loss_mask_for_span`（L378）是**单 span 视角**——签名 `(response_kind, response_len)` 只能逐 span 给统一 bool，无法判断"这是不是最后一个 assistant span"或"这是不是最后一个 tool result 之后的 final text"。

两条数据路径在 `_sample_from_pending_chat`（L569）汇流：HTTP-proxy 请求与 explicit-trajectory（`_sample_from_trajectory_turn` L528→L547 调它）。tool-call turn 在 L607 经 `_tool_call_loss_mask`（L858）设 `loss_mask_override` 屏蔽 tool-**result** 区。

**核对校正（相对 issue-199-execution-plan.md 的偏差，均经代码实读验证）**：

1. **`_tool_call_loss_mask` 的 markers 非 bug**。计划 §5.1/§9 称两元素相同=疑似 bug。实况 L862 为 `("<|tool_response>", "<|im_end>")`——**两个不同 sentinel**，取最早出现者屏蔽其后的 result 区，是有意的双哨兵启发式。本 change **不修复**它（避免 scope creep），`mask_tool_call_args` 实现时**不复用**该定位（它屏蔽的是 result 区，非 tool-call 参数区），按 tool-call span 独立实现。
2. **B 档缝描述失真**。计划 §5.5 称缝在 `policy_only.py:245-246`"rewards 已算、`_train_rows_from_samples` 未调"。实况 `_run_agentic_rollout` 中 L244 算 reward_records → **L245 算 rewards → L246 就调了 `_train_rows_from_samples`**。真实 B 档缝在 **L245→L246 之间**：rewards 此时已是 per-trajectory scalar，插 scorer 用 `response_spans` 定位 span、改 `loss_mask_override`/权重即可，无需动奖励管线。
3. **dual-path 未覆盖**。计划 §5.1 只点 trajectory 侧。`mask_tool_call_args` 须同时覆盖 L607（proxy）与 L627（trajectory）两路——好在两者汇流于 L569，`_apply_trainable_turn_mode` 挂在 `_train_rows_from_samples`（L387）唯一 chokepoint 即可统一覆盖。

## Goals / Non-Goals

**Goals:**
- [G1] 三种 trainable-turn 模式 + tool-call 参数屏蔽，逐 token 生效，覆盖 dual-path。
- [G2] 非法 call/result 配对 turn 级报错，worker init 前。
- [G3] 默认 `all_assistant` 向后兼容。
- [G4] `TrainerConfig` + CLI 暴露，早失败（ask-first 区，需 maintainer 认可）。
- [G5] metrics `trainable_tokens` / `masked_response_tokens` 可断言。
- [G6] CPU 测试覆盖逐 token mask / 非法 / 边界 / 默认关闭。
- [G7] B 档接口形状预留（`response_spans` 清单 + 字段不固化成纯 bool）。

**Non-Goals:**
- 不替换 trainer / rollout engine / dashboard / SDK 架构。
- 不扩 `reward_fn` 单 scalar 契约（`rewards.py:63`）。
- 不实现 per-trajectory / per-turn 动态 credit assignment（C 档研究空白，独立后续 issue）。
- 不修复 `_tool_call_loss_mask` 双哨兵设计（非 bug，且避 scope creep）。
- 不承诺 GPU 收敛 ablation 全跑通（视算力而定；缺算力止于 mask 统计 + CPU 验证）。

## Decisions

### D1：span 清单 + trajectory 级后处理，而非扩展 per-span 函数

`last_assistant`/`final_answer` 需要"span 列表"信息，但 `_response_loss_mask_for_span`（L378）是单 span 视角，挤不进去。

**选择**：在 `_AgentSample` 增 `response_spans: list[ResponseSpan]`（`ResponseSpan(kind: Literal["assistant_text","assistant_tool_call"], length: int)`，纯静态）。span 在 append 期捕获（此时每轮 kind 都在），用完即弃。

- 首 turn：`_set_sample_training_row`（L668）初始化 `[(kind, len(response_tokens))]`。
- `_append_sample_response` L640 extend response_tokens 时，追加 `(new_sample.response_kind, len(new_sample.response_tokens))`。

`_apply_trainable_turn_mode(sample)` 挂在 `_train_rows_from_samples`（L387）循环顶部——这是 dual-path 都过的**唯一 chokepoint**，且 trajectory 此时已完整。

**替代方案**：(a) 在 `_response_loss_mask_for_span` 传 trajectory 上下文——破坏其纯函数性与签名，且 proxy 路径单 turn 调用时无 trajectory 视角；(b) 在 `_append_sample_response` 内即时重写——但 span 列表此时可能未完整（后续还有 turn），无法判"最后"。选用 chokepoint 后处理最干净。

### D2：组合式重写顺序（铁律）

`_apply_trainable_turn_mode` 必须**从已有 per-span mask 起步**，顺序叠加：

```
已有 mask (含 L607 _tool_call_loss_mask 的 result 区屏蔽)
  → 叠加 mask_tool_call_args (tool_call span 内 arg 子区收窄为 False,
     保留 name/action token,不动已被屏的 result 区)
  → 叠加 turn-selection (整 span 级置零:all_assistant=no-op / last_assistant=仅末 assistant span / final_answer=仅末 tool_result 后的 assistant_text span)
```

**绝不从零重建**——否则丢掉 `_tool_call_loss_mask` 已做的 result 区屏蔽。写回 `sample.loss_mask_row`（response 区按 span offset 覆写）+ 同步 `loss_mask_override`。

### D3：mask_tool_call_args 的 arg 子区定位为近似值

tool-call 的 name/arguments 仅字符串解析（agentic L749 `_chat_response_message_tool_calls`），无 token offset；decode→encode 非 round-trip。arg 子区定位用 chat-template 渲染边界 + tokenizer 对齐，定位为**近似值**，须 CPU 逐 token 测试钉死。业界偏离须文档标注（ToolFormer/Gorilla/ToolACE 一律整段训 tool-call）。

### D4：B 档接口预留的物理基础 = `response_spans` 清单

A 档交付的 `response_spans` 即 B 档 per-trajectory scorer 能"无须破坏性返工"接入的物理基础。B 档真实缝在 `policy_only._run_agentic_rollout` L245→L246（rewards 已算、`_train_rows_from_samples` 未调的瞬间）——scorer 用 `response_spans` 定 span、用 `rewards` 打分→改 `loss_mask_override`/权重。`LossMaskPolicy` 字段保留可扩展形态（不固化成纯 bool），未来引入 `Optional[Callable]` 时不破坏 A 档契约。**本期不实现 callable。**

### D5：call/result 配对校验在 sample 组装前

新增校验函数，在 `_train_rows_from_samples` 之前（或 `_run_agentic_rollout` samples 组装后）调用：tool_calls turn 缺匹配 tool 结果 → `ValueError`；tool 结果无前置 tool_call → `ValueError`。**不**把空 `response_tokens` 当非法（`_run_chat_request` L493 空 fallback 是合法路径）。

### D6：弃用 dead flag

`LossMaskPolicy.final_assistant_text`（L53）全仓无读取（仅 `docs/sdk/trainer.rst:443/450` 声明）。移除字段 + 同步 docs。非公共 API 破坏，PR 说明。

## Risks / Trade-offs

- **[arg 子区定位非 round-trip]** → tokenizer 一致性靠 CPU 逐 token 测试钉死；定位标注为近似值。
- **[末轮 bare tool_call，final_answer 全零]** → 显式测试 + 文档标注：无 tool_result 后的 text 时 final_answer 训练信号为零，属预期行为。
- **[GPU 缺失致收敛 ablation 不可全跑]** → 仅影响 G8 研究化扩展，止于 mask 统计；文书如实标注范围。
- **[TrainerConfig/CLI 属 ask-first 区]** → 严守 Phase 0 maintainer ack 后再动 config/CLI；core logic（agentic.py）不在 ask-first 清单，可先行。
- **[dual-path 漏覆盖]** → `_apply_trainable_turn_mode` 挂唯一 chokepoint `_train_rows_from_samples` 统一覆盖，测试须含两路径样本。
- **[误把空 response_tokens 当非法]** → 显式排除，测试覆盖。
- **[C 档受 turn offset 丢失阻塞]** → `_append_sample_response` L665 折叠边界；A 档 `response_spans` 仅组装期捕获用完即弃，C 档须 persist offset 到 reward/advantage 阶段，属独立后续 issue。

## Migration Plan

无外部数据迁移。默认 `all_assistant` 行为等价现状。弃用 `final_assistant_text` 为内部 dead flag 移除，非公共 API。回滚 = revert PR（改动局部，无 schema/存储变更）。

## Open Questions

- Phase 0：maintainer 是否认可模式名（`all_assistant`/`last_assistant`/`final_answer`）、字段名（`trainable_turns`/`mask_tool_call_args`）、`mask_tool_call_args` 是否独立选项。若改名，spec/design 全文替换。
- AGENTS.md ask-first 清单不含 `agentic.py`，但计划 Phase 0 自设"阻塞后续全部"——core logic 可否先于 maintainer ack 落地，待用户裁决。