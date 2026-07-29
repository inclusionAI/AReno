# Issue #187 迷宫 Agentic RL Demo — PR 记录

## Issue 要求什么

建一个部分可观测迷宫的 agentic RL demo。agent 只能看到周围几个格子，要走到一个被墙围住的终点，路上还要先找钥匙开锁门。和 tictactoe（一步结束）或 shopping（固定四步）不一样，迷宫的步数完全不确定，可能 8 步到，也可能 49 步耗尽都没到。

## 实现过程中几个值得记的点

### 参照了 shopping 的多轮结构

tictactoe 太简单了——给它一个棋盘，它选一个格子就完事。迷宫不行，agent 要在变化的环境里连续决策。shopping 做的是固定四轮 tool call 循环，我把这个循环改成了 while + terminal 判断，框架是一样的：调模型 → 解析 tool call → 执行 → 把新观察塞回 messages → 继续。

### local_view 怎么做的

就一个函数，渲染 agent 周围 (2r+1)² 个格子，外面的全填 `?`。agent 看到的 3×3 视野里只有 9 个格子，整个 7×7 迷宫有 49 个，它能看到的不到五分之一。模型永远拿不到完整的 maze 数组，只能靠对话历史里积累的局部观察去推断。

### 一个实际踩的坑：BFS 漏了钥匙状态

写 `solve_shortest_path` 的时候，我传了 `has_key=True` 让 BFS 能穿门，以为这样就能算出最短路径。结果 replay 的时候 agent 卡在门口出不去——因为它一开始没有钥匙，BFS 算的路径假设你一开始就能穿门。

真正的解法是把搜索状态从"位置"变成"位置 + 有没有钥匙"的组合，搜索过程中走到钥匙格自动拾取，这样才能正确算出"先去拿钥匙再回来穿门"的两段路径。改完之后所有 8 条测试数据都跑通了，之前全是 -1.0。

### 奖励太稀疏了

到达终点给 1.0 减去多走步数的惩罚，没到就给 -0.5。问题在于没到终点时没有区分度——走到离终点 1 格和 10 格都是 -0.5。实际训练 100 步，reward_mean 一直在 -0.5 附近，说明 agent 从来没到过终点，也就没有任何正向梯度。后面如果要让 agent 真的学出来，大概得加距离奖励或者先从 5×5 的小迷宫开始。

### 7×7 迷宫生成 2048 个会爆

7×7 迷宫的变体没那么多，DFS carve 加上 key/door 放置后去重空间收缩得很快。一开始写了个 `count * 50` 的上限，2048 个直接 RuntimeError。后来加了尺寸变化（7/9/11）和 key/door 数量变化（1-2），还在唯一变体用完后自动放行重复，不会再崩。

## 自己 review 下来觉得还行和不太行的地方

**还行：**
- 五个文件的结构跟 tictactoe 和 shopping 完全一样，reviewer 不用学新东西
- 没碰 `areno/` 下面任何代码，全靠 `--dataset-loader-fn`、`--reward-fn-path`、`--agent-fn` 三个 hook 接进去
- `game.py` 不依赖 AReno，`MazeState` 是 frozen dataclass，测试不用 mock 也不需要 GPU
- 11 个测试在本地和 Kaggle 上都跑过了

**不太行：**
- `run_agent` 里的 `MazeState` 对 AReno 是黑箱，以后如果要做 rollout 重放（issue #211）会麻烦。不过 shopping 也是这么做的，目前框架就这样
- `reward_fn` 完整 replay 所有 move 来算分，开销比直接解析 tool_results 大。但两套逻辑分开写更容易不一致，所以还是 replay 了
- 测试文件最初多了一个没用到的 `import asyncio`，最后检查才发现删掉
- Kaggle 文档来来回回改了好多次：PATH 找不到 areno、没编译 CUDA 扩展、显存不够、数据集生成崩、ngrok 顺序反了。这些在本地 Mac 上根本遇不到，得实际上 Kaggle 跑一遍才知道

## 运行记录

**本地 CPU 测试**（macOS, Python 3.12）：

```
pytest tests/test_agentic_maze_example_cpu.py -v
→ 11 passed in 0.09s
```

**全量回归**（含现有 agentic 测试）：

```
pytest tests/test_agentic_maze_example_cpu.py tests/test_agentic_tictactoe_example_cpu.py tests/test_agentic_shopping_example_cpu.py tests/test_agentic_cpu.py
→ 85 passed in 2.27s
```

没破坏任何现有测试。

**Kaggle CPU 测试**（T4×2, Python 3.12）：

```
pytest tests/test_agentic_maze_example_cpu.py -v
→ 11 passed in 0.14s
```

**Kaggle GPU 训练**（T4×2, Qwen3-0.6B, GSPO）：

配置：`--batch-size 2 --n-samples 4 --tp-size 2 --world-size 2 --max-steps 100`

![Kaggle CPU Tests](kaggle-cpu-tests.png)

单卡 tp-size=1 直接 OOM，换成双卡 tp-size=2 后跑通。每步大约 9.9s，rollout 5.6s + train 4.2s，100 步 16 分钟跑完。

![Training Dashboard](kaggle-training-dashboard.png)

**训练结果**：reward_mean 在 -0.5125 到 -0.5 之间，agent 100 步内没到过终点。每步只有 8 个 rollout，总共 800 次尝试，对 7×7 POMDP 来说太少了。后续打算跑 500 步 + 5×5 小迷宫试试。