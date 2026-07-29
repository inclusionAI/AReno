# Issue #187: 部分可观测迷宫 Agentic RL Demo — PR 理解与 Review 记录

## 一、对 Issue 的理解

Issue #187 要求构建一个**部分可观测迷宫（POMDP）**的 Agentic RL demo。核心特征是：

1. **部分可观测**：agent 只能看到自身周围有限范围内的格子，完整地图永远不可见
2. **多元素交互**：迷宫包含 walls、keys、doors、goal，agent 需要先找到钥匙才能通过门
3. **多轮决策**：agent 每次只能移动一步，需要多轮 tool call 才能到达终点
4. **可量化评估**：需要 shortest-path oracle 作为基准，衡量 agent 的路径效率

与现有的 tictactoe（单步决策）和 shopping（固定 4 轮）不同，迷宫 demo 的轮次数是不确定的——最短路径可能 8 步，但 agent 可能走 49 步（max_steps）都没到终点。这使得它在 AReno 的 agentic demo 矩阵中填补了"变长多轮 POMDP"这个空白。

## 二、对实现方案的思考

### 为什么选择参照 shopping 而非 tictactoe

tictactoe 是单步决策：给棋盘 → 选格子 → 结束。迷宫不可能这样，因为 agent 需要在一个持续变化的环境中做序列决策。

shopping 是固定 4 轮：search → inspect → check → submit。迷宫的轮次不确定，但"每轮调用一个 tool、环境返回新观察、agent 继续决策"这个循环模式是相同的。所以 `run_agent.py` 的多轮循环结构参照了 shopping，但去掉了固定轮次，改为 while 循环直到 terminal。

### POMDP 的核心设计：local_view

部分可观测的实现非常简洁：`local_view(state)` 函数只渲染 agent 周围 `(2r+1)²` 个格子，视野外的格子用 `?` 表示。这个函数是 agent 观察世界的唯一窗口——`run_agent` 把 `local_view` 的输出作为 tool result 返回给模型，模型永远看不到完整的 `maze` 二维数组。

这个设计意味着 agent 必须"记住"自己走过的路径和看到过的地形，才能规划出通往钥匙和终点的路线。对于一个 7×7 迷宫、3×3 视野的 agent 来说，它最多只能看到 9 个格子，而整个迷宫有 49 个格子——信息覆盖率不到 20%。这就是 POMDP 的难度所在。

### Key-gated BFS：一个被低估的设计难点

最初实现 `solve_shortest_path` 时，我假设 `has_key=True` 意味着 agent 从一开始就有钥匙，BFS 直接找一条能穿过门的最短路径。但实际 replay 时发现 agent 卡在门前——因为它**没有钥匙**。

真正的最短路径应该是两阶段的：先走到钥匙位置（不能穿门），拾取钥匙后再穿门到终点。修复方法是把 BFS 的搜索状态从 `Position` 扩展为 `(Position, has_key)`，在搜索过程中拾取钥匙并解锁门的通行权限。

这个 bug 的深层教训是：**在 POMDP 中，"状态"不只是位置，还包括 agent 携带的物品**。如果只追踪位置，就会忽略钥匙拾取对可达性的影响。

### 奖励设计的权衡

奖励函数 `score_episode` 的设计面临稀疏奖励问题：

- **到达终点**：`1.0 - 0.05 × excess_steps`，最优路径得 1.0，多走 14 步降到 0.3
- **未到达终点**：恒定 `-0.5`，无论走了多远
- **无效移动**：`-0.1` per move，惩罚撞墙和撞门

这个设计的弱点是：未到达终点时没有距离梯度。agent 如果走到离终点 1 格和离终点 10 格，得到的都是 -0.5，无法区分"差一点"和"差很远"。在 100 步训练中 reward_mean 稳定在 -0.5 也印证了这一点——agent 从未到达终点，奖励信号完全没有梯度方向。

改进方向是加入距离奖励（基于曼哈顿距离或 BFS 距离的负相关），但这会增加奖励工程的复杂度，作为第一版 demo 保持简单是合理的。

### 数据集生成器的健壮性

7×7 迷宫的唯一变体数量是有限的——DFS wall carving 在 25 个内部格子上生成的 spanning tree 数量虽然不少，但加上固定的 key/door 放置规则后，去重空间会快速收缩。最初用 `count × 50` 次尝试生成 2048 个唯一迷宫时直接 RuntimeError。

修复方案分两步：1) 随机变化 key/door 数量（1-2）和迷宫尺寸（7→9→11），扩大搜索空间；2) 当唯一变体耗尽后自动切换为允许重复模式，保证不报错。这是一个从"严格唯一"到"尽量唯一"的实用妥协。

## 三、AI 生成代码的 Self-Review

### 做得好的地方

1. **模式一致性**：所有 5 个核心文件的结构与 tictactoe/shopping 完全对齐——`sys.path.insert` + `import game`、`# noqa: E402`、相同的函数签名（`load_training_dataset`、`reward_fn`、`run_agent`）。一个熟悉 AReno 的 reviewer 不需要学习新的模式。

2. **零侵入性**：没有修改 `areno/` 下任何一行代码，没有改 `pyproject.toml`，没有动 CLI。所有功能都是通过 AReno 现有的 hook 机制（`--dataset-loader-fn`、`--reward-fn-path`、`--agent-fn`）接入的。

3. **纯函数环境**：`game.py` 零 AReno 依赖，`MazeState` 是 frozen dataclass，所有状态转移返回新对象。这使得测试不需要 mock 任何 AReno 组件，也不需要 GPU。

4. **测试覆盖**：11 个 CPU 测试覆盖了生成器可复现性、游戏规则（墙/门/钥匙）、部分可观测不泄露、多尺寸、奖励梯度、tool schema、终局停止、步数耗尽、loader 契约、无效输入。在 Kaggle 上实测全部通过。

### 需要反思的地方

1. **run_agent 的状态管理**：maze 的 `MazeState` 完全在 `run_agent` 内部维护，AReno 基础设施对此一无所知。如果 AReno 未来需要支持 rollout 重放（issue #211），这种"隐藏状态"模式会成为障碍。但参照 shopping demo 的做法，这确实是当前 agentic 框架的标准模式。

2. **reward 的 replay 开销**：`reward_fn` 通过完整 replay 所有 move 来计算分数，而不是直接从 `tool_results` 中解析最终状态。这增加了计算开销，但保证了奖励的一致性——如果 `run_agent` 和 `reward_fn` 各自维护一份状态逻辑，很容易出现不一致。这里选择一致性而非效率是正确的。

3. **测试中 unused import**：最初的测试文件有一个未使用的 `import asyncio`，在最终检查时才发现并移除。AI 生成代码时容易引入"以防万一"的 import，需要人工 review 时仔细检查。

4. **Kaggle 文档的迭代过程**：文档经历了多次修复——PATH 问题、CUDA 编译问题、OOM 问题、数据集生成问题、ngrok 顺序问题。这些问题在本地 macOS 开发时无法预见，只有在 Kaggle 实际运行时才暴露。这说明 agentic demo 的"可运行性"验证必须覆盖目标部署环境，不能只依赖本地 CPU 测试。

## 四、分步骤运行记录

### Step 1: 本地 CPU 测试（macOS, Python 3.12）

```
pytest tests/test_agentic_maze_example_cpu.py -v
→ 11 passed in 0.09s
```

所有 11 个测试在本地 macOS 上首次通过，覆盖：迷宫生成可复现性、游戏规则（墙碰撞/门锁定/钥匙拾取）、部分可观测不泄露完整地图、多尺寸支持、奖励评分（最优路径/失败/无效移动）、tool schema 封闭性、终局停止、步数耗尽、loader 契约、无效方向拒绝。

### Step 2: 全量 agentic 回归测试（macOS, Python 3.12 + CPU PyTorch）

```
pytest tests/test_agentic_maze_example_cpu.py tests/test_agentic_tictactoe_example_cpu.py tests/test_agentic_shopping_example_cpu.py tests/test_agentic_cpu.py -v
→ 85 passed in 2.27s
```

maze 11/11 + tictactoe 3/3 + shopping 7/7 + agentic 框架 64/64 = 85 全部通过，确认没有破坏任何现有功能。

### Step 3: Kaggle CPU 测试（T4×2, Python 3.12）

![Kaggle CPU Tests](kaggle-cpu-tests.png)

```
pytest tests/test_agentic_maze_example_cpu.py -v
→ 11 passed in 0.14s
```

在 Kaggle 环境验证，11 个测试全部通过，耗时 0.14s。

### Step 4: Kaggle GPU 训练（T4×2, Qwen3-0.6B, GSPO）

训练配置：`--batch-size 2 --n-samples 4 --max-new-tokens 64 --tp-size 2 --world-size 2 --max-steps 100`

第一次尝试（单卡 tp-size=1）遇到 CUDA OOM，改为双卡张量并行后成功启动训练。

![Training Dashboard](kaggle-training-dashboard.png)

训练在 Kaggle T4×2 上运行，每步约 9.9s（rollout 5.6s + train 4.2s），100 步约 16 分钟完成。

### Step 5: 训练结果分析

`rollout/rewards_mean` 在 100 步内稳定在 -0.5125 到 -0.5 之间，说明 agent 在 100 步训练中未到达过终点。

这在预期之内：
- 每步仅 8 个 rollout（batch-size 2 × n-samples 4），100 步 = 800 次尝试
- POMDP 环境复杂度高：7×7 迷宫，3×3 视野，需要钥匙→开门→到终点
- 奖励稀疏：只有到达终点才有正奖励，未到终点恒定 -0.5，缺乏梯度方向

后续改进方向：增加训练步数到 500+、引入距离奖励、先用 5×5 小迷宫降低难度预训练。

## 五、总结

这个 PR 的核心价值不在于训练效果（100 步不足以学到有效策略），而在于：

1. **填补了 AReno agentic demo 矩阵中"POMDP + 变长多轮"的空白**
2. **验证了 AReno 现有 agentic 框架可以支撑多轮环境内的状态管理**
3. **提供了从代码到 Kaggle GPU 训练的完整可复现路径**
4. **11 个 CPU 测试保证了代码逻辑的正确性，可在无 GPU 环境下验证**

迷宫 demo 作为一个研究测试床，后续可以在此基础上研究：不同视野半径对策略学习的影响、奖励工程（稠密 vs 稀疏）、多轮轨迹的 loss mask 策略等课题。
