# PR #210 自我审查报告与运行记录

> Issue: [#210 — Normalize conversation roles and pair tool messages](https://github.com/inclusionAI/AReno/issues/210)
> 分支: `feature/normalize-conversation-roles-210`
> 作者: TTTLe

---

## 一、对 PR 任务的理解

### 1.1 当前代码存在什么问题，或缺少什么能力

AReno 训练 Agent 模型需要大量多轮对话数据，这些数据来自不同框架（ShareGPT、OpenAI、Anthropic、HuggingFace），存在三类格式问题：

1. **角色名不统一**：同一个角色在不同数据集里叫法不同（human/user、bot/assistant、function/tool），AReno 只认四种标准角色（user/assistant/tool/system），其他名字进来就出错。
2. **工具调用配对断裂**：assistant 发起 tool_call 后没有对应的 tool_response，或者凭空出现孤立的 tool_response，这种数据喂给模型会把它教坏。
3. **角色顺序违法**：连续两个 user 消息、system 消息出现在对话中间、tool 消息出现在不该出现的位置等。

当前 AReno 没有统一处理这些问题的模块，用户各写各的一次性代码，导致数据质量不一致，训练结果难以复现。

### 1.2 本 PR 的目标是什么

实现一个对话数据规范化器，在训练开始之前：

1. 将常见角色别名映射为 AReno 标准角色（human→user、bot→assistant、function→tool 等）
2. 验证每个 tool_call 都有对应的 tool_response，且 id 匹配、顺序正确
3. 验证角色交替顺序合法（不能连续 user、system 只能在开头等）
4. 错误精确报出到"第几条样本的第几轮"，不暴露对话内容
5. 提供人类可读和 JSON 结构化双格式输出
6. 通过 CLI 命令 `areno normalize-conversation` 暴露给用户

### 1.3 本 PR 明确不处理哪些内容

- **不修改 tool_call 的 arguments 内容**：只解析 JSON 字符串为 dict，不做 schema 验证
- **不截断或拆分超长对话**：长度处理是其他 Issue 的范围
- **不自动修复异常数据**：遇到问题报错，不静默修复（Issue 要求 "Never guess"）
- **不新增 trainer 配置项**：opt-in 设计，用户显式调用才生效
- **不读写文件**：纯内存处理，数据加载由调用方负责
- **不替换现有 trainer、rollout engine、dashboard 或 SDK 架构**

### 1.4 修改会影响哪些模块、接口或使用场景

| 模块 | 影响 | 说明 |
|---|---|---|
| `areno/engine/data/conversation_normalizer.py` | 新增 | 核心逻辑，不影响现有模块 |
| `areno/engine/data/__init__.py` | 修改 | 新增 normalizer 导出，保留原有 batch.py 导出 |
| `areno/api/data.py` | 修改 | 末尾追加 re-export，不修改现有 PromptItem/PromptBatch |
| `areno/cli/normalize.py` | 新增 | CLI 命令，独立文件 |
| `areno/cli/main.py` | 修改 | 新增一行注册命令，不影响其他命令 |
| `tests/test_conversation_normalizer_cpu.py` | 新增 | 46 个 CPU 测试 |
| `docs/troubleshooting/conversation-normalization.rst` | 新增 | 用户文档 |

**影响的使用场景**：用户在训练前调用 `normalize_conversation()` 或 `areno normalize-conversation` 清洗数据。不调用时所有现有行为不变。

### 1.5 完成任务的验收标准是什么

| 验收标准 | 状态 |
|---|---|
| 提供多种常见格式的 fixture（ShareGPT、OpenAI、并行、嵌套） | ✅ |
| 包含嵌套/并行和畸形的 tool call | ✅ |
| 错误指向 sample 和 turn | ✅ |
| 规范化后通过现有 agentic 契约 | ✅ |
| 使用现有 AReno 契约，不引入外部数据库/沙箱 | ✅ |
| 默认行为向后兼容 | ✅ |
| 测试覆盖成功/无效/边界 | ✅ |
| 用户文档含可运行示例 | ✅ |
| CLI 暴露时提供人类可读 + JSON 双输出 | ✅ |

---

## 二、实现思路

### 2.1 修改涉及的主要文件和模块

| 文件 | 类型 | 职责 |
|---|---|---|
| `areno/engine/data/conversation_normalizer.py` | 新增（~400行） | 核心逻辑：角色映射、tool 配对、角色交替、批量报告 |
| `areno/cli/normalize.py` | 新增（~60行） | CLI 命令：读取 JSON 文件，调用 normalizer，输出报告 |
| `tests/test_conversation_normalizer_cpu.py` | 新增（~600行） | 46 个 CPU 测试 |
| `docs/troubleshooting/conversation-normalization.rst` | 新增（~150行） | 用户文档 |
| `areno/engine/data/__init__.py` | 修改 | 新增 normalizer 公共 API 导出 |
| `areno/api/data.py` | 修改 | 末尾追加 re-export，统一调用路径 |
| `areno/cli/main.py` | 修改 | 注册 `normalize-conversation` 命令 |

### 2.2 核心流程或数据流

```
输入：各种格式的多轮 Agent 对话数据
  ↓
阶段1：角色映射（human→user, bot→assistant, function→tool）
  → 遍历每条消息，查表替换 role，未知角色报错
  → 同时归一化 tool_calls 格式（arguments 从 JSON 字符串解析为 dict）
  → content=None 替换为空字符串
  ↓
阶段2：Tool 消息配对验证
  → 维护 pending_tool_call_ids 列表
  → assistant 带 tool_call → 加入 pending
  → tool 消息 → 检查 id 在 pending 里则移除，不在则报"孤立 response"
  → user/assistant(无tool_call) 出现且 pending 非空 → 报"pending 未回答"
  → 对话结束且 pending 非空 → 报"末尾缺少 response"
  ↓
阶段3：角色交替规则验证
  → system 只能在开头
  → 不能连续两个 user
  → 不能连续两个 assistant（无 tool_call）
  → tool 必须跟在 assistant 或另一个 tool 后面（支持并行 response）
  ↓
阶段4：报告生成
  → 人类可读：Total/Passed/Failed + 每条错误的简述
  → JSON 结构化：sample/turn/type/detail
  ↓
输出：标准化的对话数据或精确的错误报告
```

### 2.3 关键数据结构、接口或算法

**数据结构**：

- `ROLE_ALIASES: dict[str, str]` — 角色别名映射表，18 种别名 → 4 种标准角色
- `pending_tool_call_ids: list[str]` — 待匹配的 tool_call id 列表，追踪配对状态
- `NormalizeResult` — 单条对话的规范化结果（messages + errors）
- `BatchNormalizeReport` — 批量处理报告（total/passed/failed/errors/normalized）

**接口**：

- `normalize_role(raw_role) → str` — 单个角色映射，未知角色抛 `UnknownRoleError`
- `normalize_conversation(messages, ...) → NormalizeResult` — 单条对话规范化
- `normalize_dataset(samples, ...) → BatchNormalizeReport` — 批量处理
- `normalize_dataset_iter(samples, ...) → Iterator` — 流式处理（省内存）

**算法**：

- 角色映射：查表，O(1) per message
- Tool 配对：线性遍历 + pending 列表，O(n) per conversation
- 角色交替：线性遍历 + 前一个角色记录，O(n) per conversation
- 总体复杂度：O(N×M)，N 条样本每条 M 轮对话

### 2.4 重要设计选择及理由

| 设计选择 | 理由 |
|---|---|
| **opt-in，不调用不影响现有行为** | Issue 要求 "safe default that preserves current behavior" |
| **未知角色直接报错不猜测** | Issue 要求 "Never guess when conversion is ambiguous"；错误的角色映射比报错更危险，会静默污染训练数据 |
| **不引入新依赖，只用 stdlib** | Issue 要求 "Use only existing AReno dependencies" |
| **放在 `areno/engine/data/` 下** | Issue 要求 "Start with areno/api/data.py, areno/engine/data/"，复用现有数据层架构 |
| **错误只报 sample #N, turn #N** | Issue 要求 "without exposing full training samples"，避免泄露敏感数据 |
| **同时提供 raise 和 collect 两种模式** | raise 适合 fail-fast（数据加载阶段），collect 适合批量看全部错误 |
| **支持并行 tool response 乱序** | 真实数据中并行 tool call 的 response 到达顺序不固定 |
| **`@dataclass(slots=True)`** | 和项目现有代码风格一致（`areno/api/data.py:15` 同样用法） |
| **CLI 懒加载 normalizer** | 不在 import 时拉 torch，和 `areno/cli/diagnostics.py` 的做法一致 |
| **测试用 importlib 直接加载** | 绕过 `__init__.py → batch.py → torch` 依赖链，和 `tests/test_agentic_cpu.py` 做法一致 |

### 2.5 是否考虑过其他方案，以及没有采用的原因

**方案A：把角色映射逻辑加到现有的 `areno/api/openai_chat.py:normalize_messages` 里**

没有采用：`normalize_messages` 的职责是处理 `content=null` 和 tool_calls 格式归一化，在 tokenizer 之前调用。角色映射和 tool 配对验证是独立的逻辑，混进去会让 `normalize_messages` 职责不清，也违反 Issue 的 "keep the change narrow" 要求。

**方案B：做成独立的 Python 包/服务**

没有采用：Issue 明确反对 "Build a separate service for the feature"，会增加部署复杂度。这个功能可以操作在 AReno 的现有本地 artifact 上，不需要独立服务。

**方案C：自动修复异常数据（比如缺失 tool_response 时自动补一条占位消息）**

没有采用：Issue 要求 "Never guess when conversion is ambiguous"。自动补的消息可能不符合真实数据语义，反而引入更隐蔽的错误。报错让用户自己决定怎么处理更安全。

### 2.6 兼容性、性能、异常处理等方面的考虑

**兼容性**：
- opt-in 设计，不调用 = 行为完全不变
- `areno/api/data.py` 只追加不修改现有代码
- `areno/engine/data/__init__.py` 保留原有导出
- `areno/cli/main.py` 只加一行注册命令

**性能**：
- 单条对话处理是 O(n)，n 是消息轮数
- 批量处理逐条独立，一条出错不影响其他条
- 提供 `normalize_dataset_iter` 流式接口，避免大数据集一次性加载到内存
- 不涉及 IO、不涉及模型推理，纯内存计算

**异常处理**：
- 未知角色 → `UnknownRoleError`（不猜测）
- 缺 tool_response → `missing_tool_response`（定位到 turn）
- 孤立 tool_response → `orphan_tool_response`（定位到 turn）
- 缺 tool_call_id → `missing_tool_call_id`
- user 打断 pending → `interrupted_tool_call`
- 角色连续 → `consecutive_user` / `consecutive_assistant`
- system 位置错误 → `misplaced_system`
- tool 位置错误 → `invalid_tool_position`
- 所有错误都带 sample_index + turn_index，不暴露对话内容

---

## 三、对自己代码的 Review

### 3.1 正确性：正常输入和边界输入是否符合预期

**正常输入**：
- ShareGPT 格式（human/bot/function）→ 正确映射为 user/assistant/tool ✅
- OpenAI 格式（标准角色）→ 保持不变 ✅
- 并行 tool call（一次 3 个 call + 3 个 response）→ 全部配对 ✅
- 嵌套 tool call（response 后触发新 call）→ 链式验证通过 ✅
- 乱序 response（先 c2 后 c1）→ 只要 id 匹配就通过 ✅

**边界输入**：
- 空对话 `[]` → 通过，返回空列表 ✅
- 单轮对话（user + assistant）→ 通过 ✅
- `content=None` → 替换为空字符串 ✅
- `tool_calls` 的 arguments 是 JSON 字符串 → 解析为 dict ✅
- `tool_calls` 是 flat 格式（name/arguments）→ 包装成 function 子字典 ✅

**异常输入**：
- 未知角色 → 报 `unknown_role` ✅
- 缺 tool_response → 报 `missing_tool_response`，定位到 turn ✅
- 孤立 tool_response → 报 `orphan_tool_response` ✅
- 缺 tool_call_id → 报 `missing_tool_call_id` ✅
- 连续 user → 报 `consecutive_user` ✅
- 连续 assistant（无 tool_call）→ 报 `consecutive_assistant` ✅
- system 在中间 → 报 `misplaced_system` ✅
- tool 不跟在 assistant 后面 → 报 `invalid_tool_position` ✅
- 非 dict 消息 → 报 `invalid_message` ✅
- 非 list 输入 → 报 `invalid_input` ✅

### 3.2 可读性：命名、注释、函数职责是否清晰

**命名**：
- `normalize_role` / `normalize_conversation` / `normalize_dataset` — 函数名清楚表达职责 ✅
- `pending_tool_call_ids` — 变量名自解释 ✅
- `ConversationValidationError` / `UnknownRoleError` — 错误类型名清晰 ✅
- `NormalizeResult` / `BatchNormalizeReport` — 结果类名明确 ✅
- `error_type` 字段使用 snake_case 命名（`missing_tool_response`、`orphan_tool_response`），语义清晰 ✅

**注释**：
- 每个函数都有 docstring，说明输入输出和设计意图 ✅
- 关键位置有中文设计注释（`normalize_role` 为什么不猜、Phase 1/2/3 各做什么） ✅
- `ConversationValidationError` 的 docstring 说明了为什么不暴露对话内容 ✅
- `NormalizeResult` 的 docstring 说明了两种模式的适用场景 ✅
- `BatchNormalizeReport` 的 docstring 说明了双格式输出的原因 ✅

**函数职责**：
- `normalize_role` — 只管角色映射，不管配对 ✅
- `_extract_tool_call_ids` — 只提取 id，不做验证 ✅
- `_normalize_tool_call` — 只管格式归一化，不管配对 ✅
- `normalize_conversation` — 主入口，协调三个阶段 ✅
- `_validate_role_sequence` — 只管角色交替，不管 tool 配对 ✅
- `normalize_dataset` — 只管批量调度和报告汇总 ✅
- 职责分离清晰，单个函数不超过 60 行 ✅

### 3.3 复用性：是否存在不必要的重复代码

- `_extract_tool_call_ids` 被 Phase 2 和 Phase 3 共用，没有重复实现 ✅
- `normalize_messages`（openai_chat.py 现有函数）的 `content=None → ""` 逻辑和本模块一致，没有重新发明 ✅
- `ConversationValidationError` 统一所有错误类型，`UnknownRoleError` 继承它，没有平行错误体系 ✅
- CLI 命令复用 `normalize_dataset` 函数，没有重新实现批量逻辑 ✅
- 未发现不必要的重复代码 ✅

### 3.4 兼容性：是否改变已有默认行为或公开接口

- `areno/api/data.py`：只在文件末尾追加 re-export，`PromptItem` 和 `PromptBatch` 完全没动 ✅
- `areno/engine/data/__init__.py`：新增 normalizer 导出，原有 `batch.py` 导出保留 ✅
- `areno/cli/main.py`：只加一行注册命令，其他命令不受影响 ✅
- `areno/api/openai_chat.py:normalize_messages`：没有修改 ✅
- 默认不调用 normalizer 时，所有现有流程不变 ✅
- `DefaultBehaviorTest` 7 个测试验证了向后兼容 ✅

### 3.5 异常处理：错误是否被正确发现并提供清晰信息

- 所有错误都包含 `error_type`（如 `missing_tool_response`）、`sample_index`、`turn_index`、`detail` ✅
- 错误信息不包含对话内容，只引用 tool_call id（如 `"c1"`），避免泄露 ✅
- `raise_on_error=True` 时第一个问题就抛异常（fail-fast） ✅
- `raise_on_error=False` 时收集所有问题（批量看全部错误） ✅
- CLI 失败时 `sys.exit(1)`，错误信息输出到 stderr ✅
- `to_human_string()` 和 `to_json()` 两种格式都包含完整错误信息 ✅

### 3.6 测试：新增逻辑是否有对应测试，原有测试是否通过

- 46 个测试覆盖 7 个测试类 ✅
- `RoleMappingTest`（7个）：别名映射、大小写、未知角色 ✅
- `ToolPairingTest`（11个）：单个/并行/嵌套/缺失/孤立/乱序 ✅
- `RoleSequenceTest`（7个）：交替/连续/system位置/空/单轮 ✅
- `BatchNormalizeTest`（9个）：批量/混合/错误定位/双格式/iter ✅
- `DefaultBehaviorTest`（7个）：re-export/content=None/flat/非dict ✅
- `AgenticContractTest`（3个）：标准角色/function子字典/兼容性 ✅
- `CliOutputTest`（3个）：人类可读/JSON字段/JSON有效性 ✅
- 测试 assert 具体的 error_type、sample、turn，不只是 exit code ✅

### 3.7 性能：是否引入明显的额外计算、内存或 I/O 开销

- 角色映射：查表 O(1) ✅
- Tool 配对：线性遍历 O(n)，n 是消息轮数 ✅
- 角色交替：线性遍历 O(n) ✅
- 批量处理：逐条独立，O(N×M) ✅
- 无 IO 操作，纯内存计算 ✅
- `normalize_dataset_iter` 提供流式接口，大数据集不会一次性加载到内存 ✅
- 46 个测试在 0.08 秒内完成 ✅
- 未引入明显的额外计算、内存或 I/O 开销 ✅

### 3.8 提交范围：是否混入与任务无关的格式化或文件修改

- 7 个文件全部与 #210 直接相关 ✅
- 没有修改无关文件的格式 ✅
- 没有删除现有注释或 docstring ✅
- `test_manual.py`（本地调试脚本）没有提交 ✅
- 提交范围干净，只包含 Issue #210 相关的改动 ✅

### 3.9 Review 后实际发现并处理的问题

1. **并行 tool response 误报 `invalid_tool_position`**：最初 `_validate_role_sequence` 中 tool 消息只允许跟在 assistant 后面，但并行场景下 `tool → tool` 是合法的（同一个 assistant 发起的多个 tool call 的 response）。修复为 `prev_role not in (ROLE_ASSISTANT, ROLE_TOOL)`。
2. **缺少 CLI 入口**：初版只有 Python API，Issue 要求 "both human-readable and structured output when the feature is exposed through the CLI"。补充了 `areno/cli/normalize.py` 并注册到 `main.py`。
3. **commit message 不符合项目规范**：项目使用 Conventional Commits 格式（`feat(data): ...`），初版标题不符合。已修正为 `feat(data): normalize conversation roles and pair tool messages (#210)`。

---

## 四、遇到的问题、挑战与解决方法

### 问题1：并行 tool response 触发 `invalid_tool_position` 误报

**现象**：测试并行 tool call 场景时，`test_parallel_tool_calls_paired` 和 `test_tool_response_id_order_does_not_matter` 报错 `invalid_tool_position: tool message at turn 3 must follow an assistant message (previous role: tool)`。

**定位过程**：查看 `_validate_role_sequence` 函数，发现 tool 消息的检查条件是 `prev_role != ROLE_ASSISTANT`。在并行场景中，连续两个 tool 消息（如 `tool(c1) → tool(c2)`）的 `prev_role` 是 `tool` 而不是 `assistant`，被误判为非法。

**根因**：最初只考虑了"tool 必须跟在 assistant 后面"这个单线程场景，没有考虑"一个 assistant 同时发起多个 tool call，后面跟多个 tool response"的并行场景。

**解决方法**：将条件从 `prev_role != ROLE_ASSISTANT` 改为 `prev_role not in (ROLE_ASSISTANT, ROLE_TOOL)`，允许 tool 后面跟 tool。

**验证方式**：重新运行 `test_parallel_tool_calls_paired` 和 `test_tool_response_id_order_does_not_matter`，全部通过。

**经验总结**：写验证规则时要考虑所有合法的消息排列组合，不能只想到最常见的单线程场景。并行场景在 agentic 数据中很常见，应该在设计验证规则时就考虑进去。

### 问题2：`pip install -e .` 在 Kaggle 上编译失败

**现象**：在 Kaggle 上运行 `pip install -e .` 安装 AReno 时报错 `Building editable for areno (pyproject.toml) did not run successfully`。

**定位过程**：查看错误日志，发现是 pip 尝试重新编译 CUDA 扩展，但 Kaggle 环境的编译工具链和 AReno 的 build isolation 不兼容。

**根因**：`pip install -e .` 默认使用 build isolation，会创建一个隔离环境重新安装所有依赖（包括 PyTorch），但 Kaggle 已经装了 PyTorch 2.10，重新装会导致版本冲突。

**解决方法**：按照 AReno 官方安装教程，使用 `pip install -e . --no-build-isolation`，让 pip 使用 Kaggle 已有的 PyTorch 环境。

**验证方式**：安装成功后运行 `areno check`，输出 `AReno check: ready`，确认安装正确。

**经验总结**：遇到安装报错时先查项目的官方文档/教程，很可能已经有对应环境的解决方案。不要盲目搜报错信息，要先理解报错的根本原因。

### 问题3：本地 Python 3.9 不支持 `@dataclass(slots=True)`

**现象**：本地运行测试时报错 `TypeError: dataclass() got an unexpected keyword argument 'slots'`。

**定位过程**：查看 Python 版本 `python3 --version` 输出 `3.9.6`，确认 `slots=True` 是 Python 3.10+ 才支持的参数。

**根因**：AReno 项目要求 Python 3.10+（README 和 CI 矩阵都是 3.10/3.11/3.12），但本地系统自带的是 Python 3.9。

**解决方法**：在本地测试脚本中用 monkey-patch 去掉 `slots` 参数绕过，正式代码保持 `slots=True` 不改（符合项目规范）。最终在 Kaggle 的 Python 3.12 环境下验证通过。

**验证方式**：Kaggle 上 `python -m pytest tests/test_conversation_normalizer_cpu.py -v` 输出 `46 passed in 0.08s`，确认 `slots=True` 在项目要求的环境下无问题。

**经验总结**：写代码前先确认目标环境的版本要求。本地环境不满足时，应该在符合要求的环境中做最终验证，不能只依赖本地绕过手段。

---

## 五、分步骤运行结果证明

### 步骤1：本地环境测试（Python 3.9，绕过依赖）

**步骤目的**：快速验证核心逻辑是否正确，不需要完整安装 AReno。

**完整命令**：
```bash
cd /Users/heiheiheilelele/AReno
python3 -c "
import dataclasses
_orig = dataclasses.dataclass
def _patched(*args, **kwargs):
    kwargs.pop('slots', None)
    if args and callable(args[0]):
        return _orig(**kwargs)(args[0])
    return _orig(**kwargs)
dataclasses.dataclass = _patched
import unittest
loader = unittest.TestLoader()
suite = loader.loadTestsFromName('tests.test_conversation_normalizer_cpu')
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
"
```

**关键输出**：
```
Ran 46 tests in 0.002s
OK
```

**对输出的解释**：46 个测试全部通过，说明核心逻辑正确。但本地环境是 Python 3.9（不满足项目 3.10+ 要求）、没有安装 torch，所以不能证明在 AReno 真实环境下能跑，需要到 Kaggle 上做完整验证。

### 步骤2：Kaggle 完整安装 AReno

**步骤目的**：在符合项目要求的环境下完整安装 AReno，为后续验证做准备。

**完整命令**：
```bash
git clone https://github.com/TTTLe/AReno.git
cd AReno
git checkout feature/normalize-conversation-roles-210
pip install psutil -q
pip install flash-linear-attention -q
pip install -e . --no-build-isolation
```

**关键输出**：
```
Successfully installed addict-2.4.0 areno-0.0.6 datasets-4.0.0 ...
```

**对输出的解释**：`areno-0.0.6` 安装成功，说明代码和 AReno 现有依赖兼容，`--no-build-isolation` 参数让 pip 使用 Kaggle 已有的 PyTorch 环境，避免了 CUDA 扩展编译失败。

### 步骤3：验证 AReno 安装完整性

**步骤目的**：确认 AReno 在 Kaggle 环境下安装正确，所有核心依赖可用。

**完整命令**：
```bash
areno check
```

**关键输出**：
```
AReno check: ready
OK  Python >= 3.10        found 3.12.13
OK  PyTorch >= 2.6        2.10.0+cu128
OK  torch.cuda.is_available()   visible_gpus=2
OK  areno_accel import    imported
```

**对输出的解释**：所有检查项通过，Python 3.12、PyTorch 2.10、CUDA、GPU 都正常。说明 AReno 完整安装成功，可以开始验证我的代码。

### 步骤4：验证 CLI 命令注册

**步骤目的**：确认 `normalize-conversation` 命令已正确注册到 AReno CLI。

**完整命令**：
```bash
areno --help
```

**关键输出**：
```
Commands:
  normalize-conversation  Normalize conversation roles and validate tool-
                          message pairing.
```

**对输出的解释**：`normalize-conversation` 出现在命令列表中，说明 `areno/cli/main.py` 的命令注册成功，`areno/cli/normalize.py` 被 CLI 正确加载。

### 步骤5：运行 46 个 CPU 测试

**步骤目的**：在完整安装环境下验证所有测试通过。

**完整命令**：
```bash
python -m pytest tests/test_conversation_normalizer_cpu.py -v
```

**关键输出**：
```
collected 46 items
tests/test_conversation_normalizer_cpu.py::RoleMappingTest::test_sharegpt_aliases_mapped PASSED
...（逐个输出 PASSED）
46 passed in 0.08s
```

**对输出的解释**：46 个测试在 Python 3.12 + PyTorch 2.10 环境下全部通过，0.08 秒完成。证明代码在项目要求的真实环境下功能正确，包括 `slots=True` 语法兼容、import 链完整、核心逻辑无误。

### 步骤6：验证 Python API 完整 import 链

**步骤目的**：确认 `from areno.api.data import normalize_conversation` 这条完整 import 链在真实环境下能跑通。

**完整命令**：
```python
from areno.api.data import normalize_conversation, normalize_dataset

messages = [
    {"role": "human", "content": "北京天气怎么样"},
    {"role": "bot", "content": None, "tool_calls": [
        {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}
    ]},
    {"role": "function", "content": "25°C 晴", "tool_call_id": "c1"},
    {"role": "bot", "content": "北京今天 25°C，晴天"},
]

result = normalize_conversation(messages)
print(f"通过: {result.ok}")
for msg in result.messages:
    print(f"  role={msg['role']}, content={msg['content'][:30]}")
```

**关键输出**：
```
通过: True
  role=user, content=北京天气怎么样
  role=assistant, content=
  role=tool, content=25°C 晴
  role=assistant, content=北京今天 25°C，晴天
```

**对输出的解释**：完整 import 链 `areno.api.data → areno.engine.data → conversation_normalizer → torch` 成功执行。ShareGPT 格式的 `human/bot/function` 被正确映射为 `user/assistant/tool`，`content=None` 被替换为空字符串，tool 配对验证通过。

### 步骤7：验证 CLI 双格式输出

**步骤目的**：确认 CLI 命令的人类可读和 JSON 输出都正确。

**完整命令**：
```bash
# 人类可读
areno normalize-conversation /tmp/test_cli.json

# JSON 结构化
areno normalize-conversation /tmp/test_cli.json --json
```

**关键输出（人类可读）**：
```
Total: 2  Passed: 1  Failed: 1  Skipped: 0

Errors:
  [missing_tool_response] sample #1, turn #2: assistant at turn 2 has no tool calls but pending tool calls ['x1'] are unanswered
```

**关键输出（JSON）**：
```json
{
  "total": 2, "passed": 1, "failed": 1, "skipped": 0,
  "errors": [
    {"sample": 1, "turn": 2, "type": "missing_tool_response", "detail": "..."}
  ]
}
```

**对输出的解释**：两种输出格式都正确。人类可读格式包含统计摘要和每条错误的简述；JSON 格式包含结构化的 sample/turn/type/detail 字段，可以被程序直接解析。错误精确定位到 `sample #1, turn #2`，没有暴露对话内容。CLI 失败时退出码为 1，方便脚本集成。

### 步骤8：验证复杂场景

**步骤目的**：确认嵌套、并行、未知角色等复杂场景在真实环境下正确处理。

**完整命令**：
```python
# 嵌套 tool call
nested = [
    {"role": "user", "content": "搜索然后总结"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "s1", "function": {"name": "search", "arguments": {"q": "AI"}}}]},
    {"role": "tool", "content": "找到3条结果", "tool_call_id": "s1"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "s2", "function": {"name": "summarize", "arguments": {"ids": [1,2,3]}}}]},
    {"role": "tool", "content": "AI正在快速发展", "tool_call_id": "s2"},
    {"role": "assistant", "content": "总结：AI正在快速发展"},
]
result = normalize_conversation(nested)
print(f"嵌套: {result.ok}")

# 未知角色
unknown = [{"role": "agent", "content": "未知"}]
result = normalize_conversation(unknown, raise_on_error=False)
print(f"未知角色: {result.ok}")
for err in result.errors:
    print(f"  [{err.error_type}] {err.detail}")
```

**关键输出**：
```
嵌套: True
未知角色: False
  [unknown_role] unknown role 'agent'; cannot normalize automatically
```

**对输出的解释**：嵌套 tool call（response 后触发新 call）验证通过，说明 pending 列表在链式调用中正确追踪。未知角色 `agent` 被正确拒绝并报错，没有静默跳过或猜测映射。

---

## 六、个人思考与反思

### 6.1 关于角色别名表的设计

我觉得角色别名表是硬编码的，如果以后有新的数据源用了新的角色名，就得改代码。更好的做法可能是支持用户在配置文件里自定义角色映射，但 Issue 说了"keep the change narrow"，所以我没有扩展这个功能。

### 6.2 关于 __init__.py 的 import 设计

我不太理解为什么 `areno/engine/data/__init__.py` 要在 import 时就拉 `batch.py`（依赖 torch）。如果做成懒加载，CPU 测试就不用用 importlib 绕了。但这是现有架构的设计，我不确定改了会不会影响其他模块。