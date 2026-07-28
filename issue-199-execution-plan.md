# Issue #199 执行计划 — Spec 模式

> 目标 issue: [inclusionAI/AReno#199](https://github.com/inclusionAI/AReno/issues/199)
> Make trainable turns configurable for agentic trajectories
> 双重定位:**开源工程交付** + **硕士申请(agentic RL)研究化素材**

---

## 1. Problem Statement(问题陈述)

AReno 当前 agentic 训练路径对所有 assistant turn 一视同仁地计入 policy loss,
既无显式的"可训练 turn 选择"模式,也无法屏蔽 tool-call 的参数 token。
对照代码事实:

- `areno/api/agentic.py:47` 的 `LossMaskPolicy` 已声明
  `assistant_text / assistant_tool_calls / tool_results / final_assistant_text / system_prompt / user_prompt`,
  但 `final_assistant_text` 是 **dead flag**(全仓仅 L53 定义,无任何读取);
  `_response_loss_mask_for_span`(L378)只依据 `assistant_text`/`assistant_tool_calls` 两个位。
- 对外仅暴露 `train_tool_results: bool = False`(`trainer_config.py:61`),
  经 `cli/train.py` 与 `policy_only._loss_mask_policy()`(L179)接入。
- 不存在"最后一轮 / 仅最终答案"模式,无可观测的 trainable token 计数,
  无对非法 call/result 配对的 turn 级校验。

结论:issue 要求的"all assistant / last assistant / final answer only / 可选 tool-call 参数"
是真实缺口,且可在现有 `LossMaskPolicy` 与 agentic 数据契约上窄改实现。

## 2. Background(背景与研究动机)

issue 踩在 agentic RL 的开放问题——**trajectory-level 选择性监督(credit assignment)**:
对多工具 agent trajectory,"训哪些 turn / 哪些 token"直接影响监督密度与最终决策质量,
是与 RFT、trajectory-level GAE、selective masking 并列的设计选择。
本计划在交付 feature 的同时,把这一选择做成可量化对比的 ablation,
作为申请文书里"工程 + 探索实践"的可验证素材。

**业界先例与研究空白**:静态"仅 assistant token 入 loss"是工业主流,代表为 veRL 多轮
delta-tokenization(逐轮 apply chat template,只标新增 assistant token),但其粒度止于
"assistant 段 vs 环境段",不分 thinking token 与 tool-call 参数 token,亦不做 turn 粒度的
选择性丢弃。基于 reward/verifier 的动态"丢弃式"判断在公开文献中属**空白**:既有工作要么
在 trajectory 级过滤(RFT,Yuan et al. 2023;RAGEN/StarPO-S,Wang et al. 2025,arXiv:2504.20073),
要么保留全部 turn 但赋不同 advantage(MT-GRPO,Zeng et al. 2025,arXiv:2505.11821),要么用
密集过程奖励(PRM,Lightman et al. 2023,arXiv:2305.20050);SFT/DPO 侧最接近"打分→mask"
的是 STM(Wu et al. 2025,arXiv:2501.14315,perplexity 阈值 token mask),但用 perplexity
而非 reward。**结论**:本计划落静态可配置 mask 属主流且可窄改;turn/token 级动态判断是研究空白,
锚定 MT-GRPO/PRM/STM 作独立后续 issue(见 §4、§9 与 §5.5 演进路径)。

## 3. Goals(目标)

- [G1] 在 agentic 数据路径实现三种 trainable-turn 模式 + tool-call 参数屏蔽,逐 token 生效。
- [G2] 对非法 call/result 配对发 turn 级错误,于 worker 初始化前抛出。
- [G3] 默认行为向后兼容(默认 `all_assistant`,效果等价于现状)。
- [G4] 通过 `TrainerConfig` + CLI 暴露,早失败(`click.UsageError` / `__post_init__` 校验)。
- [G5] metrics 输出 `trainable_tokens` / `masked_response_tokens`,可在测试中断言。
- [G6] CPU 测试覆盖逐 token mask、非法输入、边界、默认关闭。
- [G7] 文档含一个可复制示例 + 契约/默认/输出/限制说明。
- [G8](研究化扩展)三类模式 trainable-token 统计脚本 + 小规模 ablation,产出可链接 artifact。

## 4. Non-Goals(非目标)

- 不替换 trainer / rollout engine / dashboard 存储 / SDK 架构。
- 不引入外部数据库、托管控制面或重型依赖。
- 不自动改用户配置、删 artifact、终止无关进程。
- 不承诺 GPU 集成实验全跑通(见 §9 风险);收敛/奖励对比视算力而定,缺算力则止于 mask 统计 + CPU 验证。
- 不引入 per-turn / token 级 reward 驱动的动态 credit assignment —— 此为**研究空白**(业界无
  成熟先例,见 §2),且需扩 `reward_fn` 单 scalar 契约(rewards.py:63)与 advantage 计算,撞
  "不替换 trainer"边界,划为**独立后续 issue**,文献锚点 MT-GRPO(arXiv:2505.11821)/ PRM
  (arXiv:2305.20050)/ STM(arXiv:2501.14315)。本计划仅**预留 B 档接口形状**(见 §5.5),
  不实现任何动态判断逻辑。

## 5. Design Spec(设计规格)

### 5.1 数据契约(`areno/api/agentic.py`)

- 新增字面量类型 `LossSelectionMode = Literal["all_assistant","last_assistant","final_answer"]`。
- `LossMaskPolicy` 增字段 `trainable_turns: LossSelectionMode = "all_assistant"`
  与 `mask_tool_call_args: bool = False`;移除/弃用 dead `final_assistant_text`。
- 模式语义(在 `_train_rows_from_samples` / `_append_sample_response` / `_response_loss_mask_for_span` 中落实):
  - `all_assistant`:现状(所有 assistant span 计 loss)。
  - `last_assistant`:仅 trajectory 内**最后一个** assistant span trainable,前序 assistant span 置 0。
  - `final_answer`:仅"最后一个 tool result 之后的 final `assistant_text` span"trainable;
    若无 tool result,退化为最后一个 assistant span。
- `mask_tool_call_args=True`:在 tool-call turn 内,
  按 tool-call span 边界屏蔽 JSON 参数 token,保留 tool-name / action token 可训。
  **业界偏离标注**:ToolFormer(2023)/Gorilla/ToolACE(2024)/xLAM/Hermes/NexusRaven
  **一律整段训 tool-call(名+参数)**,无公开工作做参数内 token 屏蔽;防"坏参数污染训练"的
  成熟做法实为**步级/样本级 reward 过滤**(ToolFormer loss-降幅阈值、SWiRL arXiv:2504.04736
  process reward)。故本选项定位为**研究 ablation**,文档须显式标注与业界惯例的偏离,而非
  "防污染"工程标配。
  **实现注意**:`_tool_call_loss_mask`(L858)markers 字节级为
  `("<|tool_response>", "<|tool_response>")` —— **两元素相同,疑似 bug**(tool-call 段与
  tool-result 段本应用不同 marker)。参数屏蔽**不可直接复用**该定位,且 name/arguments 仅字符串
  解析(agentic L749)无 token offset,decode→encode 非 round-trip,定位为近似值,须 CPU 逐 token
  测试钉死。
- `system_prompt / user_prompt / tool_results` 维持默认 `False`;
  非 assistant 内容仍由 `[False]*prompt_len` 保证 mask。
- **接口预留(B 档兼容)**:`LossMaskPolicy` 的字段保留可扩展形态,默认仍是静态规则;未来若引入
  per-trajectory scorer callable,应在 `_response_loss_mask_for_span`(L378)读取 policy 处预留一个
  `Optional[Callable[[RewardRecord], float | bool]]` 字段(默认 `None` 时退化为静态 bool),避免
  本期把字段固化成纯 bool 而阻塞后续演进。**本期不实现 callable,仅保证字段形状不排斥其接入。**

### 5.2 校验(turn 级,worker init 前)

- tool_calls turn 缺匹配 tool 结果 → `ValueError`。
- tool 结果无前置 tool_call → `ValueError`。
- **不**把 empty `response_tokens` 当非法:
  `_run_chat_request`(L516)空 fallback 是合法路径。

### 5.3 配置 + CLI(AGENTS.md "ask first")

- `TrainerConfig` 增 `trainable_turns: str = "all_assistant"`、
  `mask_tool_call_args: bool = False`;`__post_init__` 校验字面量集合。
- `policy_only._loss_mask_policy()` 把 config 映射到 policy,
  复用现有 `RolloutSession(loss_mask_policy=...)` 路径,不开新 trainer 表面。
- `areno/cli/train.py` 加 `--trainable-turns` / `--mask-tool-call-args`,
  纳入 Rollout section 与 config summary;非法模式 `click.UsageError`(早失败)。

### 5.4 可观测输出

- metrics 路径每 batch 输出:
  - `trainable_tokens = sum(loss_mask)`
  - `masked_response_tokens = sum(response_mask) - sum(loss_mask)`
- rollout 日志打印生效模式与 `mask_tool_call_args`。

### 5.5 演进路径与接口预留(A/B/C 档定位)

| 档位 | 范围 | 本计划归属 | 接入点 / 约束 | 业界/文献先例 |
|---|---|---|---|---|
| **A 静态配置** | 三模式 + `mask_tool_call_args`,纯静态规则 | ✅ 本期交付 | `policy_only._loss_mask_policy()`(L179)映射 config→policy,不碰 trainer | veRL delta-tokenization |
| **B per-trajectory 判断** | A + optional scorer callable(每条 trajectory 一个打分→mask/权重) | ⏸ 接口形状预留,不实现 | 最窄接入点 `policy_only.py:245-246`(rewards 已算、`_train_rows_from_samples` 未调),改写 `loss_mask_override`(agentic L162/373);reward 已是 per-trajectory scalar,不碰奖励契约 | STM(arXiv:2501.14315)/ RFT |
| **C per-turn credit assignment** | per-turn reward/verifier,turn 级 advantage 或丢弃 | ❌ 独立后续 issue | 需扩 `reward_fn` 单 scalar 契约(rewards.py:63)→ 撞"不替换 trainer";且 `_append_sample_response`(L627-666)已把 turn 边界折叠进单条 `loss_mask_override`(L665),**turn offset 丢失**,per-turn 判断须先补 offset 保留 | MT-GRPO(arXiv:2505.11821)/ PRM(arXiv:2305.20050)|

**设计取舍**:C 档是研究空白(§2),非本期;A 档贴 issue 原文且可窄改,故为本期范围。B 档以
"接口形状预留"折中——`LossMaskPolicy` 字段不固化成纯 bool,留出 callable 注入位,使后续 issue
无须破坏性返工即可演进到 per-trajectory 判断。**本计划不写 callable 实现代码**,仅保证数据契约与
字段形状不排斥其接入。

## 6. Task Breakdown(任务拆解 · 分阶段)

### Phase 0 — 方案对齐(ask-first)
- [T0.1] 把英文 plan 评论发到 #199,等待 maintainer 认可模式名/字段名/`mask_tool_call_args` 是否独立。
- [T0.2] 按反馈同步字段命名;若 maintainer 改名,本 spec 全文替换。
- verify: maintainer 明确 ack 或指派。

### Phase 1 — 核心逻辑(`areno/api/agentic.py`,CPU 可验)
- [T1.1] 定义 `LossSelectionMode` + 扩 `LossMaskPolicy` 字段(默认向后兼容)。
- [T1.2] `_response_loss_mask_for_span` / `_append_sample_response` 接入三种模式。
- [T1.3] 实现 tool-call 参数 span 屏蔽(独立于 `_tool_call_loss_mask`)。
- [T1.4] 实现 call/result 配对校验函数,在 sample 组装前调用。
- verify: `pytest tests/test_agentic_cpu.py -k cpu` 全绿(含新增用例)。

### Phase 2 — 配置 + CLI
- [T2.1] `TrainerConfig` 加字段 + `__post_init__` 校验。
- [T2.2] `policy_only._loss_mask_policy()` 映射 config→policy。
- [T2.3] `cli/train.py` 加两个选项 + Rollout section + config summary + `click.UsageError`。
- verify: CLI `--help` 可见;非法值早失败;默认值等价现状。

### Phase 3 — 可观测
- [T3.1] metrics 输出 `trainable_tokens` / `masked_response_tokens`。
- [T3.2] rollout 日志打印生效模式。
- verify: CPU 测试断言 metrics 字段值。

### Phase 4 — 测试(`tests/test_agentic_cpu.py`)
- [T4.1] 固定多工具 transcript:`assistant_text → tool_call → tool_result → assistant_text`。
- [T4.2] 三模式 + `mask_tool_call_args` 的**逐 token** `loss_mask` 断言。
- [T4.3] 非法输入(缺 tool 结果)→ turn 级错误;边界(空 final answer / 全 tool-call);默认关闭断言现状不变。
- [T4.4] 断言 `trainable_tokens` / `masked_response_tokens` 数值。
- verify: `pytest tests/ -k cpu` 全绿。

### Phase 5 — 文档 + 示例
- [T5.1] CLI/skills 页加 `--trainable-turns final_answer` 可复制示例 + 契约/默认/输出/限制说明。
- [T5.2] `examples/` 加一个无网络、无 sandbox 的最小确定性 fixture(含一例非法输入)。
- verify: 示例可离线运行;docs 构建无报错。

### Phase 6 — 研究化扩展(G8)
- [T6.1] `examples/` 或脚本:对同一批 trajectory 统计三模式 trainable_token 数(CPU 可跑)。
- [T6.2](有算力时)小模型 + 少步 ablation:对比三模式收敛趋势/最终奖励。
- [T6.3] 短 blog / 附录,挂简历。
- verify: 产出可链接 artifact(blog URL 或 PDF);缺算力时止于 T6.1 并如实标注范围。

## 7. Verification Criteria(验收标准 · 对齐 issue)

- [V1] 固定多工具 transcript,逐 token 断言四种配置的 mask;
  非法 call/result 对以 turn 级错误拒绝。
- [V2] 复用 AReno 现有契约,无外部 DB / 强制 sandbox。
- [V3] 默认行为向后兼容。
- [V4] 自动测试覆盖成功 / 非法 / 边界 / 失败路径。
- [V5] 文档含最小可运行示例 + 可观测输出说明。
- [V6](研究化)三模式 trainable-token 统计脚本产出可量化数据。

## 8. Documentation Requirements(文档要求)

更新 CLI 指南与 skills/troubleshooting 页:用户可见选项、输入契约、默认值、
输出字段(`trainable_tokens` / `masked_response_tokens`)、限制、一个可复制示例。

## 9. Risks & Mitigations(风险与缓解)

| 风险 | 影响 | 缓解 |
|---|---|---|
| `final_assistant_text` 仅 dead flag,改动需确认无外部依赖 | 低 | grep 全仓仅 L53,安全;PR 说明弃用 |
| tool-call 参数 span 定位需 tokenizer 一致性 | 中 | 用 chat-template 渲染边界 + CPU 逐 token 测试钉死 |
| GPU 缺失导致收敛 ablation 不可全跑 | 中(仅影响 G8) | 止于 T6.1 mask 统计;文书如实标注范围;借云算力补 T6.2 |
| `TrainerConfig`/CLI 属 ask-first 区,未授权直接改 | 高 | 严守 Phase 0 拿认可后再动 Phase 2 |
| maintainer 改字段名导致返工 | 低 | Phase 0 锁命名;spec 全文替换 |
| 误把空 `response_tokens` 当非法 | 中 | 显式排除;对应测试用例覆盖 |
| `_tool_call_loss_mask` markers 两元素相同(`<|tool_response>`,`<|tool_response>`),疑似 bug | 中 | 不在本期修复范围(避免 scope creep);`mask_tool_call_args` 实现时**不复用**该定位,按 tool-call span 独立实现;附记 PR 供 maintainer 决断是否单开 issue |
| C 档 per-turn 判断受 turn offset 丢失阻塞(`_append_sample_response` L627-666 折叠边界) | 低(属后续 issue) | 本期仅交付 A 档 + 预留 B 档接口;C 档独立 issue 须先补 turn offset 保留机制,计入其方案 |
| `reward_fn` 契约为单 scalar(rewards.py:63),per-turn reward 无 hook | 低(属后续 issue) | 同上,本期不扩契约;C 档 issue 须重定义奖励契约 |

## 10. Acceptance & Timeline(里程碑)

- M1(Phase 0): 评论已发,maintainer ack。— 阻塞后续。
- M2(Phase 1+4): 核心逻辑 + CPU 测试通过,PR draft 可开。
- M3(Phase 2+3+5): 配置/CLI/metrics/docs 完整,PR 转 review。
- M4(Phase 6): ablation artifact 完成,可挂简历。

> 每个里程碑以 `pytest tests/ -k cpu` 全绿为硬门槛;GPU 集成另注。

## 11. Deliverables(交付物)

- 开源:`areno/api/agentic.py`、`areno/api/trainer_config.py`、`areno/api/trainers/policy_only.py`、
  `areno/cli/train.py`、`tests/test_agentic_cpu.py`、docs/example 的窄改 PR(closes #199)。
- 申请素材:本 spec、三模式 mask 统计脚本/数据、(可选)小 ablation 结果、短 blog/附录。
- GitHub issue/PR thread 本身作为工程协作与思考过程的可佐证记录。

---

*Spec 模式:Problem → Goals/Non-goals → Design → Tasks(分阶段,每步带 verify)→ Verification → Risks → Milestones → Deliverables。*
