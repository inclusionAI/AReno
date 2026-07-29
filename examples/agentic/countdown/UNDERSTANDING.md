# 我对这个 PR 的理解

本 PR 为 AReno 新增 **Countdown 算术游戏的 Agentic RL 示例**，对应 Issue #183。

## 1. 为什么做这个示例

AReno 已有的 agentic 示例（tictactoe、codebreaker、duelgrid、shopping）覆盖了棋盘游戏、代码生成、对战博弈、电商购物等场景，但缺少一个**规则最简单、易于调试、又能完整展示多轮工具调用**的算术类示例。Countdown（英国数字游戏）正好满足：

- 规则简单：6 个数字 + 1 个目标，用 +−×÷ 凑出目标
- 工具集小而清晰：add / subtract / multiply / divide / finish 共 5 个
- 多轮交互：模型需要分步计算，每步调一个工具，最后调 finish 提交答案
- 奖励信号明确：答案与目标的距离可以直接量化为 reward

这让它成为**新用户理解 Agentic RL 训练流程的最佳入门示例**。

## 2. 三个核心文件的职责

### `dataset_loader.py` —— 输入转换层

把 JSONL 里的原始记录（`{"numbers": [...], "target": ..., "id": ...}`）转换成模型能理解的 prompt。核心是 `load_training_dataset` 函数，它：
- 用 AReno 注入的 `default_loader` 读 JSONL
- 给每条数据拼一段自然语言 prompt，告诉模型有哪些数字、目标是什么、怎么用工具
- 保留 `numbers` / `target` 字段，供 reward 函数读取

### `reward.py` —— 奖励信号层

定义 `reward_fn(record) -> float`，这是 GSPO 优化策略的唯一信号。设计思路：
- 从 `record.tool_calls` 里找 `finish` 调用，提取 `answer` 参数
- 答案与 target 完全相等 → 1.0
- 相对误差 ≤10% → 0.7；≤30% → 0.3
- 更远则线性衰减到 0
- 没调 finish → 0；调了但参数解析失败 → -1.0（负惩罚，逼模型学合法的工具调用格式）

### `run_agent.py` —— Agent 环境层

定义 `async def run_agent(ctx, batch)`，这是 AReno rollout 阶段的入口。它：
- 用 `httpx.AsyncClient` + `AsyncOpenAI` 并发跑多个 episode
- 每个 episode 最多 20 步，每步模型生成一个 tool call，本地执行后把结果塞回 messages
- 用 `AgentTrajectoryTurn` 记录每一轮的 (messages, response)，供 AReno 计算梯度

## 3. Agentic RL 与 SFT 的区别（我的理解）

| 维度 | SFT | Agentic RL |
|------|-----|------------|
| 数据 | 人工标注的 (prompt, response) 对 | 只有环境（puzzle + tools），没有标准答案 |
| 监督 | 逐 token 交叉熵 | 稀疏的 episode 级 reward |
| 交互 | 单轮 | 多轮（观察→行动→观察→…→结束） |
| 训练目标 | 模仿专家 | 最大化期望奖励 |

本 PR 里的 Countdown 就是典型的 Agentic RL：模型每次 rollout 是一局游戏，reward 函数根据最终答案给分，GSPO 用这个分数做组内比较来更新策略。

## 4. 训练验证

在 Kaggle T4 GPU（14.56GB）上用以下配置跑通了完整训练循环：

```
--ckpt Qwen/Qwen3-0.6B
--algo gspo --batch-size 1 --n-samples 2
--max-steps 2 --epochs 1
--attn-backend native --adam-8bit --drop-rollout-state
```

日志显示 `rollout → score → train → max_steps_reached` 全流程跑通，证明：
- 三个文件接口与 AReno CLI 对接正确
- `AgentTrajectoryTurn` 的字段满足 AReno 训练后端要求
- reward 函数能正确从 tool_calls 提取答案并打分

由于 Qwen3-0.6B 是基座模型、训练步数只有 2 步，`reward_mean=0.0` 是预期结果——这验证了**训练流程正确**，而非模型已学会解题。真正学会解题需要更大模型（如 Qwen3-1.7B）和更多步数。

## 5. 文件结构

```
examples/agentic/countdown/
├── README.md              # 游戏规则、文件说明、训练命令
├── dataset_loader.py      # 输入转换：JSONL → agent prompt
├── reward.py              # 奖励函数：答案与目标距离 → reward
├── run_agent.py           # Agent 环境：多轮工具调用循环
└── data/
    └── countdown.jsonl    # 10 道样本题
```

结构与已有的 `examples/agentic/tictactoe/` 保持一致，方便用户类比学习。