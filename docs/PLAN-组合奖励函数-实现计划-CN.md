# 实现计划:组合多个带权重的奖励函数

> 依据:`docs/ISSUE-组合奖励函数-分析与方案-CN.md`(已对照源码核实)。
> 本计划精确到文件 / 函数 / 测试用例,并显式标注 **代码风格** 与 **注释规范**——这两点是基于真实代码归纳的,见末尾「代码风格与注释规范基线」。

---

## 0. 总目标(可验证)

- `--reward-fn-path` 可重复出现,格式 `path[:weight]`,注册多个具名 reward 分量,按权重加权求和为最终 reward。
- 每个分量的均值/非法计数作为独立指标输出到 `TrainStats.metrics` → TensorBoard / CLI / dashboard。
- 单次 `--reward-fn-path`(不带 `:weight`)= 现有单 reward 行为,零变化。
- 纯 CPU 逻辑可独立测试;GPU/worker 边界用 fake 隔离。
- **不**新增并行子系统、不新增依赖、不改 backend/engine/accel/SDK 公共签名、不改 `TrainerConfig.reward_fn_path` 公共字段。

---

## 1. 代码风格与注释规范(贯穿全部改动)

本仓库风格从 `areno/api/rewards.py`、`areno/api/metrics.py`、`areno/api/trainers/policy_only.py`、`tests/test_losses_rewards_cpu.py` 归纳,**所有新增代码必须遵循**:

### 风格
- `from __future__ import annotations` 置顶(`rewards.py:1`)。
- 显式类型注解,用 `list[...]` / `dict[...]` / `X | None` 现代语法,不用 `typing.List`。
- 函数签名用 keyword-only(裸 `*`)收关键参数,例:`def make_reward_record(*, prompt, ...) -> RewardRecord`。
- dataclass 用 `@dataclass(slots=True)`(`agentic.py`、`data.py` 现有惯例);纯数据容器优先 dataclass 而非 pydantic(rewards.py 用 pydantic 是因为 `RewardRecord` 要校验输入,新容器无此需求 → 用 dataclass)。
- 无 wildcard 导入;重导入放函数内(`policy_only.py:521` 的 `import areno.api` 范式)。
- 行宽 120,但 ruff 忽略 E501;Do not手动断行到 120 之外。

### 注释(本仓库重视注释,等级很高)
- **每个模块顶部 docstring**:说清"这个文件在管线里扮演什么角色 + 为什么这么设计",模仿 `rewards.py:1-7`、`metrics.py` 顶部、`policy_only.py:_materialize_train_batch` 的多步说明。
- **每个公共类/函数 docstring**:第一句说做什么,第二句起说**为什么**与边界,模仿 `compute_group_advantages`(`rewards.py:51` 解释 eps 的除零动机)。
- **行内 `#` 注释解释非显然决策**:模仿 `rewards.py:71`:"spec_from_file_location lets us import a module whose path is supplied at runtime without polluting sys.modules"、`policy_only.py:548`:"Group-relative advantage: A_i = (r_i - mean(r))/std(r)"、`policy_only.py:556`:"Prompt positions are masked (1=prompt, 0=response)"。
- **错误信息带 Why + Next steps**:模仿 `setup.py` 的报错风格("Why: ... Next steps: ...")与 `_validate_python_callable` 的 `click.UsageError(f"{option_name} ...; expected callable {expected}")`。
- **测试**:每条用例配单行 docstring 说"应满足的性质",模仿 `test_losses_rewards_cpu.py`(每方法都有 docstring,断言紧跟其后)。
- **不写废话注释**(不解释语法、不重复代码已表达的);只解释**为什么**和**非显然的约束**。

---

## 2. 改动清单(按依赖顺序)

### 阶段 A — 纯逻辑:`areno/api/rewards.py`(核心,无任何依赖)

新增两个 dataclass + 一个类,放在 `compute_group_advantages` 之后、`load_reward_fn` 之前(逻辑上属 reward 聚合)。

```python
@dataclass(slots=True)
class CompositeScore:
    """One record's weighted reward plus per-component breakdown."""

    total: float
    components: dict[str, float]      # name -> component value (invalid_value if failed+mark_invalid)
    invalid: list[str]                # names of components that raised / returned non-finite
```

```python
class CompositeReward:
    """Weighted sum of named reward callables sharing the RewardRecord contract.

    Behaves as a plain `reward_fn(record) -> float` (returns the weighted
    total) so existing rollout trainers need no change; per-component values
    are also exposed via `score()` for metrics collection.
    """

    def __init__(
        self,
        components: list[tuple[str, Callable[[RewardRecord], float], float]],
        *,
        on_error: Literal["raise", "mark_invalid"] = "raise",
        invalid_value: float = 0.0,
    ) -> None:
        # Validate up front so misconfiguration fails at construction (in the
        # CLI preflight), not mid-training on the Nth rollout.
        ...  # 见下"校验"

    def score(self, record: RewardRecord) -> CompositeScore: ...

    def __call__(self, record: RewardRecord) -> float:
        # Trainer calls [float(self.reward_fn(record)) for ...]; returning the
        # weighted total keeps the rollout path backward compatible.
        return self.score(record).total
```

**校验逻辑(构造时 fail-fast,对应验收「reject duplicate names」「boundary」)**:
- `components` 非空;每个 name 非空字符串;**重名** `ValueError("duplicate reward component name 'x'")`。
- 每个权重 `math.isfinite(weight)` 且 `weight >= 0`;否则 `ValueError(f"reward component 'x' weight must be a finite non-negative number, got {weight}")`。
- 总权重 `<= 0` → `ValueError`(全 0 权重无意义;半负半正凑 0 也拒)。
- `on_error` 必须是两个字面量之一。
- 保留**归一化权重的决定**:总 reward = Σ(wᵢ·vᵢ)/Σwᵢ(归一化)而非 Σ(wᵢ·vᵢ),理由写进 docstring:"归一化使权重表示相对比例而非绝对量级,与用户 `0.7/0.3` 的心智一致;且 mark_invalid 时可干净地用其余分量重新归一"。**这是关键决策,必须在注释里写清。**

**`score` 逻辑**:
- 对每个分量 `try: v = float(fn(record))`;`except Exception as exc` 或 `not math.isfinite(v)` → 按 `on_error`:
  - `"raise"`:`raise ValueError(f"reward component '{name}' {stage}") from exc`(`stage` 为 "raised" 或 "returned non-finite value"),**保留原始 traceback**,消息**不含 prompt/completion 全文**。
  - `"mark_invalid"`:`v = invalid_value`,`invalid.append(name)`,继续其余分量。
- 组装 `CompositeScore(total=归一化加权和, components={name: v, ...}, invalid=[...])`。

**公共导出**:在 `areno/api/__init__.py` 加 `CompositeReward, CompositeScore` 到 import 与 `__all__`(可选——若只在 CLI/trainers 内部用,可不暴露到顶层 SDK;决策:**暴露**,因用户可能从 SDK 直接构造,符合"现有 `register_algorithm`/`Trainer` 公共面"惯例)。

**verify check**:新增单元测试(见阶段 E)全绿;`ruff check areno/api/rewards.py` 通过。

---

### 阶段 B — metrics 通道约定:`areno/engine/data/batch.py`(仅注释,不改结构)

`TrainStats.metrics: dict[str, float] | None` 与 `RolloutOutput.metrics` 已是 `dict[str,float]|None`。**不改类型**;只在 `TrainStats` docstring 追加一段命名约定:

```python
@dataclass(slots=True)
class TrainStats:
    """Metrics returned by one worker train step.

    Composite-reward components, when enabled, are written here by the trainer
    using the keys `reward/<name>_mean` and `reward/<name>_invalid_count`, so
    CLI/dashboard consumers read them through the existing metrics dict without
    a parallel reporting subsystem.
    """
    loss: float
    stepped: bool = True
    metrics: dict[str, float] | None = None
```

**verify check**:`pytest tests/ -k cpu` 既有用例不受影响(纯注释改动)。

---

### 阶段 C — Trainer 接入:`areno/api/trainers/policy_only.py` 与 `ppo.py`(最小侵入)

**核心:不改 `self.reward_fn(record)` 的调用形**,只在评分循环里用 `isinstance` 分支顺带收集分量。

`policy_only.py:_materialize_train_batch`(527-547 行那段循环)改造:
```python
# 在循环开始前,准备分量累加器(仅当启用组合奖励时)
composite = self.reward_fn if isinstance(self.reward_fn, CompositeReward) else None
component_values: dict[str, list[float]] = {} if composite else None
component_invalid: dict[str, int] = {} if composite else None

for item_idx, (item, result) in enumerate(...):
    ...
    # 关键:保持 float(self.reward_fn(...)) 不变;但若是 CompositeReward,
    # 用 .score() 获取分量明细,再用 .total 喂给 rewards(group advantage 需要 float 标量)。
    if composite is not None:
        scores = [composite.score(make_reward_record(...)) for ...]
        rewards = [s.total for s in scores]
        for s in scores:
            for name, v in s.components.items():
                component_values.setdefault(name, []).append(v)
            for name in s.invalid:
                component_invalid[name] = component_invalid.get(name, 0) + 1
    else:
        rewards = [float(self.reward_fn(make_reward_record(...))) for ...]  # 现有路径,字节不变
```

- 在 `_materialize_train_batch` 的返回值 / 或上层 `fit` 循环里,把 `component_values`/`component_invalid` 聚合成 `{"reward/<name>_mean": mean(...), "reward/<name>_invalid_count": int}` 并注入该步的 metrics(经 `self.areno.train(...)` 返回的 dict 或 `record_training_stats` 路径——见阶段 D)。
- `ppo.py:133` 同样用 `isinstance` 分支;若 PPO 在此 issue 范围内不必完整落地分量指标,至少保证 `CompositeReward.__call__` 让它**行为正确**(total 正确),分量指标可标记为 PPO 的后续 enhancement 并在注释/TODO 说明。

**注释要点**:`isinstance` 分支处写注释解释"为什么用 isinstance 而非新接口"——答:保持普通单 reward 路径字节不变,符合向后兼容硬约束。

**verify check**:`test_trainer_api_cpu.py`、`test_algorithms_cpu.py` 既有用例不变(单 reward 路径走 else 分支);新增集成测试(阶段 E)验证分量字段。

---

### 阶段 D — metrics 落盘:`areno/api/metrics.py`

在 `record_training_stats`(已读 179-216 行)中,现有 `train_res` 循环 `for key, value in train_res.items(): writer.add_scalar(f"train/{key}", value, step)` 已经会把 `reward/<name>_mean` 写成 `train/reward/<name>_mean`。**因此只要阶段 C 把键塞进 train_res/metrics,这一步自动完成**。

- 决策:分量指标键名用 `reward/<name>_mean`,经现有 `train/` 前缀循环写出 → TensorBoard 标量 `train/reward/<name>_mean`。**不新写循环**,验证 `metrics.py:215-216` 足矣。
- 唯一可能需要的小改:若希望分量在 `rollout/` 命名空间而非 `train/`,则在 `record_training_stats` 新增一个明确循环(类似 192-199 的 rewards 块)。**决策:放 `rollout/`**——因为分量是 rollout 阶段产生的,与现有 `rollout/rewards_mean` 同空间,便于用户归并。新增循环:

```python
# Composite-reward per-component means (set by the trainer when multiple
# reward components are registered via --reward-fn-path).
for key in ("reward_components_mean",):  # 或直接遍历 stats 里带前缀的键
    ...
```

实际实现:在 `collect_train_batch_stats`(metrics.py:126)或上层把 `component_values` 放进 `stats["reward_components"]={name: [..]}`,然后在 `record_training_stats` 里:
```python
for name, vals in stats.get("reward_components", {}).items():
    if vals:
        writer.add_scalar(f"rollout/reward/{name}_mean", float(np.mean(vals)), step)
```
并对应 `invalid_count`。**注释解释为何放 rollout/ 命名空间。**

**verify check**:新增测试断言 TensorBoard 事件含 `rollout/reward/accuracy_reward_mean` 标量(用 `SummaryWriter` 的 fake / 内存 event 读取)。

---

### 阶段 E — CLI:`areno/cli/train.py`

**E1. 选项升级**(已读 1180 行):
```python
# multiple=True 让 --reward-fn-path 可重复注册分量;weight 用 path:weight 后缀,
# 省略时默认 1.0。单次出现且无 :weight 时行为与旧行为完全一致(向后兼容)。
@click.option("--reward-fn-path", "reward_fn_paths", multiple=True,
              help="Python file defining reward_fn(record). Repeatable; append :weight to set "
                   "a component weight (default 1.0), e.g. accuracy_reward.py:0.7.")
@click.option("--reward-on-error", type=click.Choice(["raise", "mark_invalid"]),
              default="raise", help="Behavior when a reward component raises or returns "
                                    "non-finite (default raise, preserving current fail-fast).")
```
- `multiple=True` 使 `args.reward_fn_paths` 成为 `tuple[str,...]`;现有单值读取点(536、690、737、802、987、1180、1355 等)**逐一改为处理 tuple**。
- **关键兼容处理**:若 `len == 1` 且不含 `:`,退化为旧行为(直接 `load_reward_fn` → 单 callable,即现有路径,字节级不变);若 `>= 1` 且至少一个含 `:` 或 `len > 1`,组装成 `CompositeReward`。
- 把 `--reward-fn-path` 与 `--reward-on-error` 注册进 `TRAIN_OPTION_GROUPS` 的 **Rollout** 段(已读 48-96 行,该段已含 `reward_fn_path`、`reward_ckpt`)。

**E2. 解析函数**(新增私有 helper):
```python
def _parse_reward_fn_paths(paths: tuple[str, ...]) -> list[tuple[str, str, float]] | None:
    """Parse repeatable --reward-fn-path values into (name, abs_path, weight).

    Returns None for the single-legacy case (one path, no :weight) so the caller
    keeps the historical single-reward path byte-for-byte; otherwise returns a
    list of (name, path, weight) for CompositeReward construction.
    """
    # name = Path(path).stem;重名在 _build_reward_fn 用 CompositeReward 构造时自然报错。
    # weight 解析失败 -> ValueError("could not parse weight ..."),由 preflight 转 UsageError。
```
**注释**:解释为何保留"单值旧路径"——向后兼容硬约束 + AGENTS.md「改 CLI 选项面先问」,此处不改选项**名**、不改公共 config 字段。

**E3. 前置校验**(已读 `_preflight_task_hooks` 521-543、`_validate_python_callable` 554-576):
- 在现有 `if algorithm.requires_rollout and args.reward_fn_path is not None:` 块里,改为遍历 `reward_fn_paths`,对每个解析出的 `path` 调 `_validate_python_callable(path, "reward_fn", option_name="--reward-fn-path", expected="reward_fn(record)", positional_args=1)`。
- 复用 `_validate_python_callable`,**不新写文件解析逻辑**;weight 解析与去重在 `_parse_reward_fn_paths` 内,失败→`click.UsageError`,消息指明是哪条 `--reward-fn-path`、不含样本正文。

**E4. 组装与注入**(`run` / `fit` 附近,已读 802-825):
```python
components = _parse_reward_fn_paths(args.reward_fn_paths)
if components is None and args.reward_fn_paths:
    reward_fn = load_reward_fn(resolve(args.reward_fn_paths[0]))   # 旧单值路径
elif components:
    reward_fn = CompositeReward(
        [(name, load_reward_fn(path), weight) for name, path, weight in components],
        on_error=args.reward_on_error,
    )
else:
    reward_fn = None
# build_trainer(..., reward_fn=reward_fn, ...) 不变
```

**E5. 配置摘要**:在 `_format_training_config_summary`(已读 376 行附近)里,当启用组合时打印分量登记表(name/path/weight/on_error),人类可读。

**verify check**:`test_train_cli_config_cpu.py`(已读 tests 列表)既有用例不变;新增 CLI 测试(阶段 G)。

---

### 阶段 F — 示例:`examples/math/accuracy_reward.py` + `format_reward.py`

- 两个纯本地文件,各定义 `reward_fn(record: RewardRecord) -> float`,**无网络/无沙箱/无数据库**:
  - `accuracy_reward.py`:从 `record.answer` / `record.final_answer` 校验 `\boxed{}` 内答案正确性,正确 1.0 否则 0.0。
  - `format_reward.py`:校验 `record.completion` 是否包含 `\boxed{}` 格式标记,含 1.0 否则 0.0。
- 与 issue 命令完全呼应:`--reward-fn-path examples/math/accuracy_reward.py:0.7 --reward-fn-path examples/math/format_reward.py:0.3`。
- **边界/非法示例**:在文件内或 `examples/math/` 旁放一个 `README` snippet,演示一条**故意抛异常**的坏 reward(`raise RuntimeError`)在 `mark_invalid` 下被记 0、在 `raise` 下 fail-fast。

**注释**:每个 `reward_fn` 顶部 docstring 说明"这是某分量的奖励语义 + 边界行为"。

**verify check**:`pytest tests/test_sft_example_cpu.py` 风格参考;新增 `test_composite_reward_example_cpu.py` 加载这两个文件断言返回值。

---

### 阶段 G — 测试:`tests/test_composite_reward_cpu.py`

遵循 `test_losses_rewards_cpu.py` 风格:每方法单行 docstring + `unittest.TestCase`。

| 用例 | 断言(命中验收) |
| --- | --- |
| `test_composite_total_matches_hand_calculation` | 给定 accuracy=1.0/0.0 + format=1.0/1.0,权 0.7/0.3,断言 total == 手算归一化加权和(**hand-calculated fixtures**) |
| `test_composite_rejects_duplicate_names` | 构造两个同名 → `ValueError`(**reject duplicate names**) |
| `test_composite_rejects_bad_weight` | 权重 `nan`/负数/总权 0 → 各自 `ValueError`(**boundary**) |
| `test_composite_raise_mode_propagates` | 某分量抛异常,`on_error="raise"` → `ValueError` 且 `__cause__` 保留原始(**fail-fast + 不掩盖错误**);消息含分量名、不含全文样本 |
| `test_composite_mark_invalid_records_and_continues` | `on_error="mark_invalid"` → total 由其余分量算出,`CompositeScore.invalid` 含失败名,invalid_count 指标 +1(**mark-sample-invalid**) |
| `test_composite_call_returns_total_for_trainer_compat` | `__call__(record)` == `score(record).total`,float 类型(**保持现有 trainer 调用契约**) |
| `test_composite_length_alignment` | 批量评分 N 条 record → 分量输出长度 == N(**incompatible output lengths**) |
| `test_single_reward_fn_path_backward_compatible` | 模拟 CLI `_parse_reward_fn_paths` 单值无 `:` → 返回 None → 走旧单 reward 路径(**verify existing behavior unchanged**) |
| `test_metrics_emit_component_keys` | 集成:fake trainer,断言 metrics dict 含 `reward/accuracy_reward_mean` 等键(**assert emitted metric fields**) |

集成用 fake:`build_trainer` 的 `instance` 传一个只记录 metrics 的桩,不进 backend(已读 `trainer_factory.py` 只做 `trainer_cls(config, instance=, dataset=, reward_fn=, loss_fn=)`)。

**verify check**:`pytest tests/test_composite_reward_cpu.py -v` 全绿;`pytest tests/ -k cpu` 全套不回归。

---

## 3. GPU / 分布式验证边界(documented minimal)

- 组合 reward 聚合是**纯 CPU**,在 trainer 评分循环完成,**不进 worker 进程、不经 IPC**——故无 GPU 用例需求。
- 唯一跨进程的是分量 metrics 经 `TrainStats.metrics` 穿越 worker→`merge_train_stats`(`engine/api.py:489`)→ backend → SDK。该聚合逻辑已由 `tests/test_*.py` 覆盖;新增 metrics 键不改变 dict 序列化,故**无需新增 GPU 用例**,在 PR 描述里「记录剩余的最小 GPU 验证」为:"确认 `rollout/reward/<name>_mean` 在真实 N 卡 `areno train` 下出现在 TensorBoard(手动/CI 单卡即可证,多卡仅复制路径)"。

---

## 4. 文档更新

- `docs/cli/training.rst`:新增多 reward 用法段,贴 issue 命令;说明 `path:weight`、默认 1.0、`--reward-on-error`。
- `docs/reference/reward-function-api.rst`:输入契约(`RewardRecord→float`)、组合语义(归一化加权)、输出字段(`rollout/reward/<name>_mean`、`reward/<name>_invalid_count`)、限制(命名取 stem、重名报错)。
- `docs/troubleshooting/reward-function.rst`:补"某分量失败如何读诊断"(看 `rollout/reward/<name>_invalid_count` 与异常消息里的分量名)。
- 一个可复制示例直接引用阶段 F 的两个文件 + issue 命令。

---

## 5. 向后兼容与风险核查

| 风险 | 处置 |
| --- | --- |
| `--reward-fn-path` 单值老脚本 | 单值无 `:` → 旧路径字节不变(阶段 E1/E2 显式退化) |
| `TrainerConfig.reward_fn_path` 公共字段 | **不改**;多分量解析仅在 CLI 层组装 `CompositeReward` |
| 改 CLI 选项面(AGENTS.md 要先问) | 不改选项**名**,仅 `multiple=True` + 新增 `--reward-on-error`;需在 PR 描述点明这一边界 |
| 普通单 reward 路径性能 | `isinstance` 分支跳过,零额外开销 |
| PPO 分量指标 | 至少 `__call__` 保证 total 正确;完整分量指标可后置并加 `TODO(agent)` |

---

## 6. 执行顺序与 verify 节点

1. **阶段 A**(纯逻辑)→ `pytest tests/test_composite_reward_cpu.py::阶A用例` 绿 + `ruff check`。
2. **阶段 B**(注释)→ `pytest tests/ -k cpu` 不回归。
3. **阶段 C-D**(trainer + metrics)→ 集成测试绿;`test_trainer_api_cpu.py`/`test_algorithms_cpu.py` 不回归。
4. **阶段 E**(CLI)→ `test_train_cli_config_cpu.py` 不回归 + 新 CLI 测试绿。
5. **阶段 F**(示例)→ 示例测试绿。
6. **阶段 G**(完整测试)→ `pytest tests/ -k cpu` 全绿;`pre-commit run -a` 通过;`pyright` 无新增错误。

每个阶段独立可评审、可回滚,符合 issue「focused, independently reviewable」。
