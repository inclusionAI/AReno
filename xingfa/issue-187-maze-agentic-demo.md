# Issue #187: Build a partially observable maze agentic RL demo

> 原始 issue: https://github.com/inclusionAI/AReno/issues/187
> 创建者: xsuler (Collaborator) | 标签: `kind/feature` `area/agentic` `priority/backlog`

---

## 一、Issue 要做什么

### 1.1 核心目标

在 `examples/agentic/` 下新增一个 **部分可观测迷宫（partially observable maze）** 的 agentic RL demo，让用户能快速体验和测试 AReno 在多步决策、探索性场景下的 RL 训练能力。

### 1.2 迷宫功能规格

迷宫包含以下元素：

| 元素 | 符号 | 说明 |
|------|------|------|
| 墙壁 | `#` | 不可通行 |
| 空地 | `.` | 可通行 |
| Agent | `A` | 玩家起始位置 |
| 钥匙 | `K` | 可拾取，用于打开对应门 |
| 门 | `D` | 需要持有对应钥匙才能通过 |
| 目标 | `G` | 到达即成功 |

关键约束：

- **部分可观测**：agent 只能看到周围局部视野（如 3×3 或 5×5 窗口），不能看到完整地图
- **单步移动**：每次只能向上下左右移动一格
- **种子化生成**：迷宫通过 seed 确定性生成，保证可复现
- **保证可解**：生成的迷宫必须存在从起点到目标的合法路径

### 1.3 需要创建的文件

按照 AReno 现有 agentic 示例的**统一文件结构契约**（参考 `examples/agentic/tictactoe/` 和 `examples/agentic/duelgrid/`）：

```
examples/agentic/maze/
├── game.py              # 迷宫生成、状态管理、局部视野渲染、移动/拾取/开门逻辑
├── run_agent.py         # async run_agent(ctx, batch) — 调用模型选择动作
├── reward.py            # reward_fn(record) — 奖励函数
├── dataset_loader.py    # 加载预生成的迷宫 JSONL
├── dataset_generator.py # 种子化生成可解迷宫（含钥匙-门配对）
└── README.md            # 使用说明
```

### 1.4 各文件职责详解

#### `game.py` — 迷宫核心逻辑

- **迷宫生成**：使用种子化算法（如 Prim、DFS backtracking）生成保证可解的迷宫
- **钥匙-门配对**：在迷宫中放置钥匙和对应颜色的门，确保钥匙在门之前可达
- **局部视野渲染**：`render_local_view(state, radius=N)` 只返回 agent 周围 N 格范围内的地图
- **动作系统**：`move(direction)` 移动、`pickup()` 拾取钥匙、`use_key()` 开门
- **状态管理**：`State` dataclass 包含 grid、agent 位置、持有钥匙、步数等
- **合法性校验**：撞墙检测、无钥匙开门检测、动作耗尽检测

#### `run_agent.py` — Agent 入口

- 定义 `async run_agent(ctx, batch)` 函数（AReno agentic 的标准入口签名）
- 通过 `ctx.get_base_url()` 获取本地 OpenAI-compatible 端点
- 定义 tool schema（如 `move`、`pickup`、`use_key`、`look` 等工具）
- 每轮将局部视野作为 prompt 发给模型，模型通过 tool call 返回动作
- 返回 `AgentTrajectory`（包含多轮 `AgentTrajectoryTurn`）

#### `reward.py` — 奖励函数

- `reward_fn(record)` 根据轨迹计算奖励
- 稀疏奖励为主：到达目标 +1.0，撞墙/无效动作 -0.1
- 可选步数惩罚：鼓励最短路径

#### `dataset_generator.py` — 数据生成

- 种子化生成迷宫 JSONL 文件
- 每条记录包含：迷宫 grid、agent 起始位置、目标位置、钥匙-门配对信息
- 参考 `examples/agentic/duelgrid/dataset_generator.py` 的模式

#### `dataset_loader.py` — 数据加载

- `load_training_dataset(dataset_path, ...)` 加载 JSONL 并转为 AReno prompt 格式
- 直接复用 duelgrid 的加载模式

### 1.5 与现有代码的复用关系

迷宫示例**不需要修改任何 AReno 核心代码**，只需使用 `areno/api/__init__.py` 已导出的公共类型：

```python
from areno.api.agentic import (
    AgentBatch, AgentItem, AgentTrainBatch,
    AgentTrajectory, AgentTrajectoryTurn,
    LossMaskPolicy, RolloutSession,
)
```

CLI 入口 `areno/cli/train.py` 已支持 `--agent-fn` 参数（`areno/cli/train.py:1298`），用户只需：

```bash
areno train --ckpt Qwen/Qwen3-0.6B \
  --dataset-path examples/agentic/maze/maze_states.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo --tp-size 1
```

### 1.6 验收标准（来自 issue）

- [ ] 覆盖多种迷宫尺寸和布局、不可行走路径、需要钥匙的门、动作耗尽、确定性重放
- [ ] 指标覆盖：成功率、路径长度、无效移动次数、超出最短路径的步数
- [ ] 使用现有 AReno 契约，不引入外部数据库或强制沙箱
- [ ] 默认行为向后兼容（不影响现有功能）
- [ ] 聚焦的自动化 CPU 测试覆盖成功路径、无效输入、边界/失败路径
- [ ] 用户文档包含最小可运行示例

---

## 二、实现难度评估

### 2.1 总体评估

| 维度 | 评估 |
|------|------|
| 整体难度 | **中等** |
| 预估代码量 | ~800-1200 行（不含测试） |
| 核心挑战 | 迷宫生成算法、局部视野渲染、钥匙-门可达性保证 |
| 风险点 | 迷宫生成算法的可解性验证、多轮 tool-call 轨迹的正确 tokenization |

### 2.2 各模块难度分解

| 模块 | 难度 | 预估行数 | 说明 |
|------|------|----------|------|
| `game.py` | **中高** | 300-400 行 | 核心难点：迷宫生成算法需保证可解性；钥匙-门配对需保证钥匙在门之前可达；局部视野渲染需正确处理边界 |
| `run_agent.py` | **低** | 80-120 行 | 直接复用 tictactoe/duelgrid 的模式，只需定义不同的 tool schema 和 system prompt |
| `reward.py` | **低** | 40-60 行 | 稀疏奖励为主，逻辑简单 |
| `dataset_generator.py` | **中** | 150-200 行 | 种子化迷宫生成，参考 duelgrid 的 `dataset_generator.py` 模式 |
| `dataset_loader.py` | **低** | 30-50 行 | 直接复用 duelgrid 的加载模式 |
| `README.md` | **低** | 50-80 行 | 使用说明和可运行示例 |
| CPU 测试 | **中** | 200-300 行 | 需覆盖迷宫逻辑的各种边界情况 |

### 2.3 各模块技术要点

#### `game.py` — 难度中高

**迷宫生成算法**（核心挑战）：

- 推荐使用 **Prim 算法** 或 **递归回溯（DFS backtracking）** 生成保证连通的迷宫
- 需要保证从起点到目标存在路径
- 钥匙-门配对需要额外验证：钥匙必须在门之前可达（即不经过门就能到达钥匙）

**局部视野渲染**：

- `render_local_view(state, radius=2)` 返回 agent 周围 (2*radius+1)×(2*radius+1) 的地图
- 边界外的格子标记为未知（如 `?`）
- 已探索但当前不可见的区域可以保留记忆（可选，增加复杂度）

**动作合法性校验**：

- 移动：目标格不能是墙壁 `#` 或未打开的门 `D`
- 拾取：当前格必须是钥匙 `K`
- 开门：必须持有对应钥匙，且目标格是门 `D`

#### `run_agent.py` — 难度低

- 与 tictactoe/duelgrid 的 `run_agent.py` 结构几乎一致
- 差异仅在 tool schema 定义和 system prompt
- 多轮交互：agent 需要多轮 tool-call 才能完成迷宫（移动→拾取钥匙→开门→到达目标）

#### `dataset_generator.py` — 难度中

- 参考 `examples/agentic/duelgrid/dataset_generator.py` 的模式
- 迷宫生成需要保证多样性（不同尺寸、不同钥匙-门数量）
- 输出 JSONL 格式，每条记录包含完整的迷宫状态

#### CPU 测试 — 难度中

- 参考 `tests/test_agentic_cpu.py` 的测试模式
- 需要覆盖：
  - 迷宫生成的可解性验证
  - 局部视野渲染的正确性（边界处理、未知区域）
  - 移动/拾取/开门的合法性校验
  - 钥匙-门配对逻辑
  - 确定性重放（相同 seed → 相同迷宫）
  - 无效输入处理（撞墙、无钥匙开门、动作耗尽）
  - 指标计算（成功/路径长度/无效移动次数）

### 2.4 与现有示例的对比

| 特性 | tictactoe | duelgrid | maze（新增） |
|------|-----------|----------|-------------|
| 可观测性 | 完全信息 | 完全信息 | **部分可观测** |
| 交互轮次 | 单轮 | 多轮 | 多轮 |
| 工具数量 | 1 个 | 1 个 | 3-4 个 |
| 状态空间 | 小（3×3） | 中（11×11） | 可配置 |
| 核心挑战 | 对抗推理 | 战术决策 | **探索+规划** |
| 实现复杂度 | 低 | 中 | 中 |

### 2.5 实现顺序建议

1. **`game.py`** — 先实现迷宫核心逻辑，这是所有其他模块的基础
2. **`dataset_generator.py`** — 能生成数据后才能验证 game.py 的正确性
3. **CPU 测试（game 部分）** — 在写 agent 之前确保迷宫逻辑正确
4. **`run_agent.py`** — agent 入口，依赖 game.py 的局部视野渲染
5. **`reward.py`** — 奖励函数，依赖 game.py 的状态判定
6. **`dataset_loader.py`** — 数据加载，最简单
7. **CPU 测试（集成部分）** — 端到端验证
8. **`README.md`** — 最后写文档

---

## 三、非目标（明确排除）

- 不替换 AReno 的 trainer、rollout engine、dashboard 存储或公共 SDK 架构
- 不引入外部数据库、托管控制面板或重量级依赖
- 不自动修改用户配置、删除产物或终止无关进程
- 不解决可以拆分为独立 issue 的相邻功能
- 不需要实现图形化界面（可选的 web_ui 是后续工作）