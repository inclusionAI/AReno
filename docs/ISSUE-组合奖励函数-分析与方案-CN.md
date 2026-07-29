# Issue 分析:组合多个带权重的奖励函数

> 本文档翻译 GitHub Issue 原文,并给出**已对照源码核实**的解决思路。
> 已核实的代码:`areno/api/rewards.py`、`areno/api/data.py`、`areno/engine/data/batch.py`、`areno/api/trainers/policy_only.py`、`areno/api/trainers/ppo.py`、`areno/cli/train.py`、`areno/api/metrics.py`、`areno/api/trainer_factory.py`。

---

## 一、Issue 原文翻译

### 动机(Motivation)
AReno 用户需要把**多个带权重的奖励函数组合**起来,作为一个聚焦、可独立评审的能力。当前流程要么缺这个行为,要么得靠一次性的用户代码,这让后训练运行更难运维、对比和复现。

### 提议功能(Proposed feature)
允许一次训练注册多个**具名**奖励函数、配置权重,并**输出每个分量以及加权和**。定义当某个分量抛异常或返回非法值时的处理:是**快速失败(fail-fast)**还是**标记该样本无效**。

实现要复用 AReno 现有的公共契约和本地产物(artifact)格式。任何新增公共选项必须有**安全的默认值**(保持当前行为)、清晰的校验报错,且经 CLI 暴露时既有人类可读输出也有结构化输出。

### 预期用户流程(Expected user flow)
1. 用户用显式输入启用/调用该功能。
2. AReno 在昂贵的模型/worker 初始化**之前**校验输入。
3. 功能通过现有日志、指标、产物、CLI 输出或 dashboard 产生**可观测结果**。
4. 失败时指出受影响阶段和输入,**不暴露完整训练样本、也不掩盖原始错误**。

### 可能的实现位置(Likely implementation areas)
从 `areno/api/data.py`、`areno/engine/data/`、相关 trainer、CPU 测试入手。改动要**窄**;复用现有的数据、指标、生命周期、registry 契约,而不是引入一套并行的子系统。

### 最小示例(Minimal example)
给出一个**小型确定性示例或 fixture**,可脱离外部数据库运行。Agentic 示例也必须避开网络服务和沙箱。示例须演示成功路径以及至少一个非法/边界输入。

### 非目标(Non-goals)
- 不替换 AReno 的 trainer、rollout 引擎、本地 dashboard 存储或公共 SDK 架构。
- 不引入外部数据库、托管控制面、强制重型依赖。
- 不自动改用户配置、删产物、终止无关进程。
- 不在同一 issue 里捎带可单独评审的相邻特性。

### 参考(Reference)
只用 AReno 现有依赖,除非某一小的新依赖被单独论证。仓库当前实现和契约是首要参考;不需要论文复现。

### 已考虑的替代方案(Alternatives considered)
- 留在项目特定 loader / reward hook / shell 脚本里——没有统一契约、诊断或可复用测试。
- 单独建服务——给一个本可在现有本地产物上运行的能力增加部署和存储复杂度。
- 并入大范围 runtime/dashboard 重写——聚焦的小改动更好评审、测试和采用,不动无关行为。

### 测试要求(Testing requirements)
- 核心逻辑、畸形输入、边界值、禁用/默认行为、确定性输出都要有聚焦的 CPU 测试。
- 跨模块处加一个用小型本地 fixture 的集成式测试。
- 分布式或仅 GPU 的行为,用 fake 隔离编排逻辑,并记录剩余的最小 GPU 验证。
- 断言产出的**指标/产物字段和报错信息**,不仅是退出码。
- 功能未启用时验证现有行为不变。

### 文档要求(Documentation requirements)
记录用户选项/命令、输入契约、默认值、输出字段、限制、一个可复制示例;若改变了运维流程,更新相关 skill 或 troubleshooting 页。

### 验收标准(Acceptance criteria)
- 用手算 fixture 校验加权和;拒绝重名和不兼容的输出长度;让分量指标可被 CLI/dashboard 消费;保持现有单奖励行为。
- 用现有 AReno 契约实现,不引入外部数据库或强制沙箱。
- 默认行为向后兼容。
- 聚焦自动化测试覆盖成功、非法输入、一个边界/失败路径。
- 用户文档含最小可运行示例并说明可观测输出。

---

## 二、现状契约(已核实)

下面是读过代码后确认的事实,不是凭印象:

| 事实 | 位置 |
| --- | --- |
| 真实契约是 `reward_fn(record: RewardRecord) -> float`(单 record → 单标量) | `areno/api/rewards.py:63` `load_reward_fn` |
| 两个 rollout trainer 都这么调:`[float(self.reward_fn(record)) for record in reward_records]` | `policy_only.py:245`、`ppo.py:133` |
| **结构化指标通道**:`TrainStats.metrics: dict[str,float]\|None` —— worker 训练步把标量指标装这里,经 engine 聚合(`merge_train_stats`)→ backend 汇总 → 回到 SDK `train()` 的返回 dict | `engine/data/batch.py:18` |
| **rollout 指标通道**:`RolloutOutput.metrics: dict[str,float]\|None` —— rollout 结果自带可选 metrics dict | `engine/data/batch.py:42` |
| CLI 已有前置校验范式:`_preflight_task_hooks` → `_validate_python_callable`,在模型初始化前校验 `--reward-fn-path` 含 `reward_fn(record)` | `cli/train.py:536` |
| 所有 trainer 经 `build_trainer(config, instance=, dataset=, reward_fn=, loss_fn=)` 构造,统一收 `reward_fn`(SFT/DPO 里 `del reward_fn`) | `trainer_factory.py:8` |
| 指标已有出口:`MetricsRecorder.record_train_step` / `record_rollout_sample`;已有 `rollout/rewards_mean` 等标量写 TensorBoard | `metrics.py:193` |
| `make_reward_record(...)` 构造统一的 `RewardRecord`(prompt/completion/trace/tool_*/tokens/logprobs/loss_mask/…),prompt 与 agentic 共用同一契约 | `rewards.py:89` |
| README 里 `reward_fn(row, completions) -> list[float]` 是**旧文档写法**,代码已改成单 record 单标量 | `README.md:129` 与代码不符 |

> **关于验收里"incompatible output lengths"**:在 `RewardRecord → float` 契约下,单个 `reward_fn` 只返回一个标量,不存在 list 长度不匹配。这条理解为 **① 组合时各分量数量与注册数量对齐**,以及 **② 批量评分时每条 record 的输出长度与 `reward_records` 长度对齐**。会显式覆盖并断言,而不是造一个不存在的"返回 list"分支。

---

## 三、解决思路(落在 issue 指定的实现位置,全在现有合同内)

> 设计基线:分量奖励**沿着已有的结构化 metrics 通道**(`TrainStats.metrics` / `RolloutOutput.metrics`)流回 CLI/dashboard,**不新建并行子系统**。

### 步骤 1 — `areno/api/rewards.py`:组合奖励的纯逻辑(无 GPU、无副作用)

新增 `CompositeReward` + `CompositeScore`(纯 dataclass,放 `rewards.py` 与 `RewardRecord` 同处):

- `CompositeScore`:`total: float`、`components: dict[str, float]`、`invalid: list[str]`。
- `CompositeReward` 初始化收 `components: list[tuple[str, Callable[[RewardRecord], float], float]]`(名/函数/权重)、`on_error: Literal["raise","mark_invalid"]`、`invalid_value`。
- **构造时全部校验**(fail-fast 的第一道):重名抛 `ValueError`;权重非有限数抛错;空 components / 总权重为 0 的边界明确处理(直接报错,写清理由)。
- `score(record) -> CompositeScore`:逐分量调用,聚合加权和;非法值/异常按 `on_error` 处理(见步骤 4)。
- 纯函数层,核心逻辑(成功/重名/权重非法/边界)直接 CPU 测,不依赖任何 trainer。

### 步骤 2 — `areno/api/data.py` 与 `areno/engine/data/`:沿现有容器携带分量

issue 明确点名这两个目录,在此按"复用现有 data 容器"的最小方式接入:

- 在 `api/data.py` 的 `PromptBatch`(或随 trainer 批量结果一起的数据结构)不新增大对象,而是让评分结果把分量装进**已有 `dict` 形态**——trainer 的 `AgentTrainBatch.rewards`(`agentic.py:118` 的 `reward_records`)和批量 train batch 处用 `CompositeScore.components` 累加成一个 `dict[str, list[float]]`(每分量每行的值)。
- 该 dict 写入 **`RolloutOutput.metrics` / `TrainStats.metrics`**,即 `engine/data/batch.py` 已定义的那两个 `dict[str,float]|None` 字段。以 `reward/<name>_mean`、`reward/<name>_invalid_count` 等键名写入,**字段即契约**;CLI 和 dashboard 已经消费这些 metrics dict,无需新增上报路径。
- 关键:不新增并行数据结构,只是往已有 `metrics` dict 里按命名约定填键——满足"reuse existing data contracts rather than introducing a parallel subsystem"。

### 步骤 3 — 相关 trainer(`policy_only.py` / `ppo.py`):瘦适配器,几乎不动

trainer 只认"callable → float",调用形如 `[float(self.reward_fn(record)) for record in reward_records]`:

- 让 `CompositeReward.__call__(record) -> float` 直接返回 `.score(record).total` —— 两个 rollout trainer **一行都不用改**,完全向后兼容。
- 在**已有的批量评分循环**里顺带收集分量:`policy_only.py` 已在算 `tool_call_count` 一类诊断,同一循环把 `CompositeScore.components` 累加进步骤 2 的 dict。这是唯一对 trainer 的最小改动,且只在 `reward_fn` 是 `CompositeReward` 时触发(`isinstance` 分支),普通单函数路径不变。
- PPO 走 `make_reward_record` 后同样调 `self.reward_fn(record)`,适配器同样适用。

### 步骤 4 — 错误的两种模式(在 `CompositeReward.score` 内)

- **`raise`**(默认,保持当前行为):任一分量抛异常或返回非有限值 → 立即重抛,消息注明分量名(`ValueError: reward component 'format' returned non-finite value`),用 `raise ... from exc` 保留原始 traceback,**不掩盖**。
- **`mark_invalid`**:分量失败 → 该分量记 `invalid_value`、加入 `CompositeScore.invalid`;`total` 用其余分量的归一化权重算出;整步 invalid 比例可 warning。两种模式都有 CPU 测试。
- 失败信息**只含分量名 + 阶段 + 非法值的概要**(如 "non-finite"),不打印整条 prompt/completion,满足"identify the affected stage and input without exposing full training samples"。

### 步骤 5 — `areno/cli/train.py`:复用 `--reward-fn-path` 多值 + 前置校验(模型初始化前 fail-fast)

**CLI 形态(采用 issue 起草的命令)**:不新增 `--reward-components`,而是把已有
`--reward-fn-path` 从单值升级为**可重复**(Click `multiple=True`),每个值用
`reward_function_path:weight` 格式注册一个分量;权重省略时默认 `1.0`。最终训练 reward
按配置加权组合,**同时保留每个 reward component 的独立指标输出**。

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/accuracy_reward.py:0.7 \
  --reward-fn-path examples/math/format_reward.py:0.3 \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1
```

输入契约:
- `--reward-fn-path` 可出现 **0 或多次**。0 次 = 不启用 rollout 算法所需的奖励(沿用现有
  `--reward-ckpt` 或离线算法路径);1 次 = 现有单 reward 行为(向后兼容)。
- 每次 `--reward-fn-path` 值为 `path[:weight]`:
  - `path` 必须是定义 `reward_fn(record) -> float` 的 Python 文件
  - `weight` 可选,缺省 `1.0`,可为小数;非有限数在校验阶段 `click.UsageError`
  - 解析后分量名取**文件名的 stem**(如 `accuracy_reward`、`format_reward`),**重名报错**
    (验收:reject duplicate names)
- 可选 `--reward-on-error {raise,mark_invalid}`(默认 `raise`,保持当前 fail-fast 行为)。
- **单次 `--reward-fn-path`(且不带 `:weight`)→ 完全等价当前单函数路径,行为零变化**
  (验收:preserve existing single-reward behavior)。

前置校验(在 `_preflight_task_hooks` 内,模型初始化前):
- 复用 `_validate_python_callable` 校验每个 `path` 含 `reward_fn(record)`。
- 解析 `weight`、去重名、检查 `on_error` 取值 —— 任一失败 `click.UsageError`,消息指明是
  哪个分量(path / name)。
- `trainer_config.reward_fn_path`(单值字段)的流转保持不变;多分量解析结果在 CLI 侧组装成
  `CompositeReward`,经 `build_trainer(..., reward_fn=...)` 注入。配置摘要
  (`_format_training_config_summary`)打印分量登记表(名/路径/权重);分量 metrics 经
  `TrainStats.metrics` 自然出现在 `areno train` 的 `train_stats=...` 行与 TensorBoard
  ——"human-readable and structured output"。

> 注意:`--reward-fn-path` 从单值改 `multiple=True` 是**向后兼容的**:Click 允许该选项出现
> 0/1/N 次,单次行为不变;`TrainerConfig.reward_fn_path: str | None` 仍保留供 SDK/单值使用,
> 多分量解析只在 CLI 层进行,不强制改公共 config dataclass(遵循 AGENTS.md「改 config dataclass
> 先问」——此处不改字段,仅 CLI 侧组装)。

### 步骤 6 — CUDA/worker 边界:编排逻辑用 fake 隔离

- 真正的 GPU/分布式只在 trainer→backend→engine 那段;**组合奖励的聚合是纯 CPU 逻辑**,在 trainer 的评分循环里完成,不进 worker 进程、不经 IPC。
- 集成测试用 fake `instance`(不进 backend)跑 `build_trainer(..., reward_fn=CompositeReward(...))`,断言产出的 `AgentTrainBatch`/batch 与 `metrics` dict 含分量字段。
- 文档记录:GPU 上唯一需验证的是"分量 metrics 经 `TrainStats.metrics` 正常穿越 worker→engine 聚合",这一步可被 `merge_train_stats` 的现有单测覆盖,无需新增 GPU 用例。

### 步骤 7 — 最小确定性示例 + CPU 测试(无 GPU/网络/沙箱)

- `examples/math/` 下放与命令呼应的两个纯本地分量:`accuracy_reward.py`(判定 `\boxed{}` 内答案正确性)与 `format_reward.py`(判定格式合规,如是否包含 `\boxed{}`),数据用内联常量,不连任何数据库。命令即步骤 5 的那段。
- 新增 `tests/test_composite_reward_cpu.py`:
  - **手算加权和**(命中"verify totals against hand-calculated fixtures")
  - **重名拒绝**、**权重非法**、**`raise` 与 `mark_invalid` 两路径**、**边界**(总权重 0、单分量、全 invalid)
  - **默认/未启用时行为不变**(传入普通 `reward_fn`,断言 metrics dict 不含 `reward/<name>` 键)
  - **批量长度对齐断言**:分量输出数 == `reward_records` 数(命中"incompatible output lengths")
  - 确定性输出
- 集成式测试(fake trainer)断言 `TrainStats.metrics` 的分量字段与报错信息文本,**不止断言退出码**。

### 步骤 8 — 文档

- `docs/cli/training.rst`(新选项段)、`docs/reference/reward-function-api.rst`(输入契约/默认值/输出字段/限制 + 一个可复制示例)
- 若改动运维流程,在 `docs/troubleshooting/reward-function.rst` 补"分量失败时如何读诊断"
- 更新对应 skill(`.agents/skills/areno-validate-correctness` 或 reward 相关)如触及运维流程

---

## 四、逐条对照验收标准

| 验收标准 | 落点 |
| --- | --- |
| 用手算 fixture 校验加权和 | 步骤 1 纯逻辑 + 步骤 7 CPU 测试 |
| 拒绝重名 / 不兼容输出长度 | 步骤 1 构造校验(重名)+ 步骤 7 批量长度断言 |
| 分量指标可被 CLI/dashboard 消费 | 步骤 2 写入 `TrainStats.metrics`/`RolloutOutput.metrics`,CLI/dashboard 已消费 |
| 保持单奖励行为 | 步骤 5 默认 None + 步骤 3 适配器 `__call__`→float,trainer 不改 |
| 用现有 AReno 契约,无外部数据库/沙箱 | 全在 `areno` 内 + 现有 `pydantic`/`numpy`;复用 `metrics` dict |
| 默认向后兼容 | 步骤 5 默认值 + 步骤 3 `isinstance` 分支跳过普通函数 |
| 测试覆盖成功/非法/边界失败 | 步骤 7 三个路径均有用例 |
| 文档含最小示例与可观测输出 | 步骤 8 + 步骤 7 的 examples 示例 |
| fail-fast before init | 步骤 5 `_preflight_task_hooks`,与现有 `--reward-fn-path` 校验同站 |
| 失败定位且不泄露样本 | 步骤 4 错误只含分量名+阶段+值概要;`record_rollout_sample` 只存名→值 |

---

## 五、改动文件清单(精确到 issue 指定位置)

**新增**
- `tests/test_composite_reward_cpu.py` —— 核心逻辑 + 集成(fake trainer)
- `examples/math/accuracy_reward.py`、`examples/math/format_reward.py` —— 纯本地最小示例分量(含一个非法/边界输入)

**修改(落在 issue 点名的实现区)**
- `areno/api/rewards.py` —— `CompositeReward` / `CompositeScore`(核心纯逻辑)
- `areno/api/data.py` —— 批量评分结果的分量容器约定(最小,复用已有 dict 形态)
- `areno/engine/data/batch.py` —— **不改结构**,仅在文档/注释里明确 `TrainStats.metrics`/`RolloutOutput.metrics` 承载 `reward/<name>` 键的约定(若需类型注解微调,保持向后兼容)
- `areno/api/trainers/policy_only.py` / `ppo.py` —— 在已有评分循环 `isinstance(reward_fn, CompositeReward)` 分支里累加分量(普通函数路径不变)
- `areno/cli/train.py` —— `--reward-fn-path` 升级为 `multiple=True` + `path:weight` 解析 + `_preflight_task_hooks` 校验 + 配置摘要
- `areno/api/metrics.py` —— 确保 `reward/<name>` 标量写入 TensorBoard(复用现有 `record_training_stats` 循环)
- `docs/cli/training.rst`、`docs/reference/reward-function-api.rst`、`docs/troubleshooting/reward-function.rst`

**不动**:backend、engine worker/rollout runtime、accel、SDK 公共类(`Trainer` 及其方法签名);`TrainerConfig.reward_fn_path` 字段不改(多分量解析只在 CLI 层)。