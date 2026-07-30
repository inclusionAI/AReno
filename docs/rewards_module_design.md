# `rewards.py` 模块设计文档 — AReno 奖励计算子系统

> 本文档解析 `areno/api/rewards.py` 中各组件在 RL 后训练模型中的角色、调用链路、数据流和作用。

---

## 一、模块定位

`rewards.py` 是 AReno 框架中**奖励计算子系统的核心枢纽**。它定义了：

1. **数据契约**：`RewardEvent` 和 `RewardRecord` 两个数据类，统一了简单 prompt/completion 场景与复杂多轮 agentic 场景的奖励输入格式。
2. **动态加载机制**：`load_reward_fn` 允许用户在运行时传入任意 Python 文件，其中定义 `reward_fn(record) -> float`，实现奖励逻辑与训练框架的解耦。
3. **组优势计算**：`compute_group_advantages` 实现了 GRPO/GSPO 算法的核心——在同一 prompt 的多个采样之间做组内标准化，生成 advantage 信号。
4. **工厂方法**：`make_reward_record` 为 trainer 提供便捷的 RewardRecord 构造入口。

---

## 二、数据类详解

### 2.1 `RewardEvent`

```python
class RewardEvent(BaseModel):
    type: Literal["request", "assistant_text", "assistant_tool_call", "tool_result", "finish", "error"]
    text: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | str | None = None
    content: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**作用**：描述一次 agentic（多轮工具调用）rollout 过程中一个事件节点。

| 字段 | 类型 | 含义 |
|------|------|------|
| `type` | Literal | 事件类型，标记该节点是请求、文本回复、工具调用、工具结果、结束还是错误 |
| `text` | str | 文本内容（如 assistant 回复的文本、工具返回的文本） |
| `name` | str | 工具调用的函数名 |
| `arguments` | dict | 工具调用的参数 |
| `content` | str | 工具执行返回的内容 |
| `messages` | list | 完整消息列表（用于 request 类型） |
| `metadata` | dict | 额外元信息（如 finish_reason） |

**构造位置**：`areno/api/agentic.py` 第 583、591、593、595 行

**消费位置**：这些 RewardEvent 被组装成 `trace` 列表后，存入 `RewardRecord.trace` 字段，传递给用户定义的 `reward_fn`，让用户根据完整交互历史做出评分决策。

### 2.2 `RewardRecord`

```python
class RewardRecord(BaseModel):
    prompt: str
    completion: str
    rendered_completion: str | None = None
    final_answer: str | None = None
    answer: Any | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[RewardEvent] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    loss_mask: list[bool] = Field(default_factory=list)
    source_record: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

**作用**：奖励函数的**统一输入接口**。无论是简单问答还是多轮 agent 交互，都通过这个单一数据结构传递给用户定义的 `reward_fn`。

| 字段 | 含义 | 使用场景 |
|------|------|----------|
| `prompt` | 原始提示文本 | 所有场景，上下文参考 |
| `completion` | 模型生成的回复文本（decode 后） | 所有场景，评分主要依据 |
| `rendered_completion` | 渲染后的 completion | 非 agentic 场景通常等于 completion |
| `final_answer` | 最终答案 | 默认为 completion，agentic 场景为最后一轮文本 |
| `answer` | 标准答案（来自数据集） | 正确性验证场景（如数学题） |
| `messages` | 完整消息历史 | agentic 场景 |
| `trace` | 多轮交互轨迹（RewardEvent 列表） | agentic 场景 |
| `tool_calls` | 工具调用列表 | agentic 场景 |
| `tool_results` | 工具返回结果列表 | agentic 场景 |
| `tokens` | 完整 token 序列（prompt + response） | 框架内部传递 |
| `logprobs` | 每个位置的 log 概率 | 框架内部传递 |
| `loss_mask` | 哪些位置参与 loss 计算（标记 response 部分） | 框架内部传递 |
| `source_record` | 数据集的原始完整记录 | 需要访问数据集额外字段时 |
| `metadata` | 额外元信息（prompt_index, sample_index 等） | 分组标识 |

---

## 三、函数详解

### 3.1 `make_reward_record(...)` — 奖励记录工厂

```python
def make_reward_record(
    *, prompt: str, completion: str, source_record: dict[str, Any],
    answer: Any | None = None, tokens: list[int] | None = None,
    logprobs: list[float] | None = None, loss_mask: list[bool] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RewardRecord
```

| 项目 | 内容 |
|------|------|
| **输入** | prompt 文本、completion 文本、source_record 原始记录、answer 标准答案、tokens token 序列、logprobs log 概率、loss_mask 损失掩码、metadata 元信息 |
| **输出** | `RewardRecord` 实例 |
| **调用者** | `PolicyOnlyTrainer._materialize_train_batch`（`policy_only.py:533`）、`PPOTrainer`（`ppo.py:116`） |
| **调用时机** | 每次 rollout batch 之后，模型生成 completion 后立即调用 |

### 3.2 `load_reward_fn(path)` — 动态加载奖励函数

```python
def load_reward_fn(path: str) -> Callable[[RewardRecord], float]
```

| 项目 | 内容 |
|------|------|
| **输入** | Python 文件路径（如 `examples/math/math_verify_reward.py`） |
| **输出** | 用户定义的 `reward_fn(record: RewardRecord) -> float` 可调用对象 |
| **调用者** | `areno/cli/train.py:809` |
| **调用时机** | 训练启动时，解析 CLI 参数后调用一次 |
| **加载方式** | 使用 `importlib.util.spec_from_file_location` 动态导入，不污染 `sys.modules` |
| **契约约束** | 目标文件必须定义名为 `reward_fn` 的可调用对象，且签名匹配 `(record) -> float` |

### 3.3 `compute_group_advantages(rewards, eps)` — 组优势计算

```python
def compute_group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]
```

| 项目 | 内容 |
|------|------|
| **输入** | 同一 prompt 下所有采样样本的 reward 列表（如 `[0.5, 0.8, 0.2]`） |
| **输出** | 每个样本的优势值列表，标准化到均值为 0、标准差为 1 |
| **数学公式** | `A_i = (r_i - mean(r)) / (std(r) + eps)` |
| **调用者** | `PolicyOnlyTrainer._materialize_agentic_train_batch`（`policy_only.py:417`）、`PolicyOnlyTrainer._materialize_train_batch`（`policy_only.py:550`） |
| **调用时机** | 每次 rollout batch 后，对每个 prompt 组的 reward 列表调用一次 |

---

## 四、完整数据流与调用链路

### 4.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       训练启动阶段 (CLI)                                  │
│                                                                         │
│  areno train --reward-fn-path examples/math/math_verify_reward.py       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  areno/cli/train.py                                 │                │
│  │    ├─ 解析 --reward-fn-path → reward_fn_path         │                │
│  │    ├─ load_reward_fn(reward_fn_path) → reward_fn     │                │
│  │    └─ build_trainer(..., reward_fn=reward_fn)        │                │
│  └─────────────────────────────────────────────────────┘                │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  areno/api/trainer_factory.py                        │                │
│  │    build_trainer(config, reward_fn=fn)               │                │
│  │    → trainer_cls(config, reward_fn=reward_fn)        │                │
│  └─────────────────────────────────────────────────────┘                │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────┐                │
│  │  Trainer 初始化                                      │                │
│  │    self.reward_fn = reward_fn  (存储为实例变量)        │                │
│  │                                                        │                │
│  │  ─ SFT/DPO Trainer: del reward_fn  (丢弃, 不使用)     │                │
│  │  ─ PolicyOnlyTrainer: 存储并调用                      │                │
│  │  ─ PPOTrainer: 继承 PolicyOnlyTrainer, 支持双路径     │                │
│  └─────────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       训练循环 (Training Loop)                           │
│                                                                         │
│  每个 step:                                                             │
│    1. Rollout: 模型对 prompt 采样生成多个 completion                      │
│    2. 评分: reward_fn(record) → float scalar                            │
│    3. 优势计算: compute_group_advantages → list[float]                   │
│    4. 组装训练序列: TrainSequence(tokens, logprobs, advantages, reward)  │
│    5. 损失计算: policy gradient 使用 advantage 作为梯度缩放因子            │
│    6. 参数更新: 优化器 step                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 按算法分类的详细调用链路

#### 场景一：GRPO/GSPO（非 agentic，标准问答）

```
PolicyOnlyTrainer._materialize_train_batch()
│
├─ tokenizer.decode(seq.resp_tokens) → completion text          ← 解码模型输出
│
├─ make_reward_record(                                         ← 构造 RewardRecord
│     prompt=item.prompt,
│     completion=completion,
│     source_record=item.record,
│     answer=item.solutions,
│     tokens=item.input_tokens + seq.resp_tokens,
│     logprobs=[0.0]*prefix_len + seq.resp_logprobs,
│     loss_mask=[False]*prefix_len + [True]*len(seq.resp_tokens),
│     metadata={"prompt_index": item_idx, "sample_index": sample_idx}
│   )
│
├─ self.reward_fn(record) → float reward                       ← 用户自定义评分
│
├─ compute_group_advantages(rewards) → list[float] advantage   ← 组内标准化
│
└─ TrainSequence(                                               ← 组装训练序列
       tokens=...,
       logprobs=...,
       advantages=[0.0]*prefix_len + [advantage]*resp_len,
       reward=reward,
       ...
   )
```

**关键特征**：
- 每个 prompt 生成 `n_samples` 个 completion
- 对这 `n` 个 reward 做组内标准化得到 advantage
- advantage 在 response 的每个 token 位置重复（prompt 位置为 0）
- 这是 GRPO/GSPO 的标准实现，无需 critic 网络

#### 场景二：GRPO/GSPO（agentic，多轮工具调用）

```
PolicyOnlyTrainer._materialize_agentic_train_batch()
│
├─ AgentRolloutContext.reward_record(sample) → RewardRecord     ← agentic 专用构造
│    包含完整 trace（多轮对话历史、工具调用、工具结果）
│
├─ self.reward_fn(record) → float reward                       ← 用户根据轨迹评分
│
├─ 按 metadata.prompt_index 分组 reward                         ← 同 prompt 分一组
│  └─ compute_group_advantages(group_rewards) → advantage       ← 组内标准化
│
└─ TrainSequence(                                               ← 组装训练序列
       advantages=...,           ← advantage 复制到 loss_mask 为 True 的位置
       ...
   )
```

**关键特征**：
- RewardRecord 包含完整的 `trace`（RewardEvent 列表），记录 agent 的每一步操作
- 用户奖励函数可以基于完整的交互历史评分（如：是否成功调用工具、最终答案是否正确）
- 分组逻辑基于 `metadata.prompt_index`，确保同一 prompt 的采样之间做优势对比

#### 场景三：PPO

```
PPOTrainer.train()
│
├─ make_reward_record(...) → RewardRecord                       ← 构造 RewardRecord
│
├─ 路径 A: self.reward_fn is not None                            ← 用户自定义评分
│     self.reward_fn(record) → float reward
│
├─ 路径 B: self.reward_fn is None                                ← 后端 reward model 评分
│     self._score_rewards(token_rows) → float reward
│
├─ 然后使用 critic 网络估计 value
│  └─ advantage = reward - value  (GAE 方式)
│
└─ TrainSequence(                                               ← 组装训练序列
       advantages=...,           ← PPO 的 GAE advantage
       ...
   )
```

**关键特征**：
- PPO 支持双路径：Python `reward_fn` 或后端 reward model
- advantage 不是通过组内标准化得到，而是通过 critic 网络的 value 估计
- 这是 PPO 与 GRPO/GSPO 的核心区别

### 4.3 各方法的输入输出矩阵

| 方法 | 输入来源 | 输入数据 | 输出 | 输出去向 | 调用频次 |
|------|----------|----------|------|----------|----------|
| `make_reward_record` | Trainer rollout 结果 | prompt 文本、completion 文本、数据集记录、token 序列等 | `RewardRecord` | `reward_fn` | 每样本 1 次 |
| `reward_fn` (用户定义) | `make_reward_record` 的输出 | `RewardRecord` | `float` 标量 reward | `compute_group_advantages` 或直接写入 `TrainSequence` | 每样本 1 次 |
| `compute_group_advantages` | 同 prompt 的多个 reward | `list[float]` | `list[float]` advantage | `TrainSequence.advantages` | 每 prompt 组 1 次 |
| `load_reward_fn` | CLI 参数 `--reward-fn-path` | Python 文件路径 | `Callable[[RewardRecord], float]` | `build_trainer` → Trainer 实例 | 训练全程 1 次 |

### 4.4 reward 与 advantage 在训练中的作用

| 信号 | 含义 | 在训练中的作用 |
|------|------|----------------|
| **reward** | 绝对评分（如 0 或 1） | 衡量模型输出质量的**绝对标准**，反映"这个回答有多好" |
| **advantage** | 组内相对评分 | policy gradient 损失函数中**梯度的缩放因子**，决定"这个回答比平均水平好/差多少" |

在损失函数中，advantage 的作用为：
- **正 advantage** → 增加对应 token 的生成概率
- **负 advantage** → 降低对应 token 的生成概率
- **advantage 绝对值越大** → 梯度更新幅度越大

数学上，policy gradient 损失为：
```
L = -E[ advantage * log π(y|x) ]
```

其中 `π(y|x)` 是模型生成 token y 的概率。

---

## 五、配置与注入链路

```
CLI 参数: --reward-fn-path <path>
     │
     ▼
TrainerConfig.reward_fn_path: str | None   (areno/api/trainer_config.py:145)
     │
     ▼
areno/cli/train.py:809: load_reward_fn(path) → reward_fn callable
     │
     ▼
areno/api/trainer_factory.py:8: build_trainer(config, reward_fn=fn)
     │
     ▼
Trainer 构造函数: self.reward_fn = reward_fn
     │
     ├── PolicyOnlyTrainer (GSPO/GRPO):   调用 self.reward_fn(record)
     ├── PPOTrainer:                      调用 self.reward_fn(record) 或 self._score_rewards()
     ├── SFTTrainer:                      del reward_fn (不使用)
     └── DPOTrainer:                      del reward_fn (不使用)
```

---

## 六、用户自定义奖励函数示例

### 6.1 数学验证器 (`examples/math/math_verify_reward.py`)

该函数使用 `math_verify` 库对模型生成的数学答案进行符号级比较：

```
输入：RewardRecord
    ├─ record.answer        → 标准答案（来自数据集）
    └─ record.completion    → 模型生成的回答文本
处理流程：
    1. parse(ground_truth)  → 解析标准答案
    2. parse(record.completion) → 解析模型答案
    3. verify(gt, pred)     → 符号级比较
输出：
    └─ 1.0 (正确) 或 0.0 (错误)
```

### 6.2 一般模式

用户自定义奖励函数遵循统一契约：

```python
def reward_fn(record: RewardRecord) -> float:
    # 非 agentic 场景：基于 prompt + completion + answer 评分
    # agentic 场景：基于 trace/messages/tool_calls/tool_results 评分
    return score  # 任意浮点数
```

---

## 七、设计原则总结

1. **统一接口**：`RewardRecord` 是唯一的奖励输入数据类，同时服务简单问答和多轮 agent 场景
2. **可插拔**：通过 `load_reward_fn` 实现用户自定义奖励逻辑与训练框架的完全解耦
3. **算法无关**：reward 评分独立于算法实现，GRPO/GSPO/PPO 共享相同的评分接口
4. **组内标准化**：`compute_group_advantages` 是 GRPO/GSPO 的核心差异化特性，无需 critic 网络即可获得 advantage 信号
5. **零侵入**：SFT/DPO 等无需奖励函数的算法通过 `del reward_fn` 自然忽略该参数，不增加额外开销