# PR Review 文档

## 一、对 Issue 的理解

这个 issue 要做的是在 Dashboard 里加一个对比两次训练运行的功能。用户跑了两次训练（比如 SFT 和 GSPO，或者同样算法不同超参数），想直接在页面上看两个 run 的差异——哪些配置变了、loss 降了多少、训练速度差多少。

Issue 特别强调要用 AReno 现有的本地数据，不引入外部数据库。这意味着所有数据都来自 `job.metrics`、`job.launch_config`、`job.timeperf` 这些已有的字段，不能搞一套新的存储。

## 二、实现思路

后端在 `server.py` 加了 `compare_jobs()` 方法和 `/api/compare` 端点。前端加了 `CompareRunsPanel` 组件，包含头部信息卡片、关键指标卡片、超参数对比表、指标曲线对比图、差异摘要、完整指标表和 Timing 表。CLI 加了 `areno compare` 命令。

核心数据流：前端选定两个 job → 调用 `/api/compare` → 后端从 `DashboardState` 获取两个 Job 对象 → 提取 config、metric_summaries、metric 时间序列、timeperf 计时 → 返回结构化 JSON 给前端渲染。

关键设计选择：
- 状态提升到 App 组件：compare 状态存在父组件而非子组件，避免页面切换时卸载丢失 state。
- 单次遍历 metric 分组：最初 29 个指标 x 2 个 job = 58 次全量遍历，改成 `defaultdict` 单次分组后只需 2 次，复杂度从 O(58N) 降到 O(2N)。
- 归一化模式默认开启：两个 run 步数差距大时绝对步数模式下短线看不见，归一化按训练进度拉伸到全宽，用 `(step - minStep) / (maxStep - minStep)` 确保两条线起点对齐。
- RL-only 字段注释：`n_samples` 等字段对 SFT 无意义时自动添加解释 note。
- 指标卡片自动隐藏：SFT 没有 reward 指标，对应卡片不显示。

## 三、Code Review

### 后端
`compare_jobs()` 方法的验证逻辑比较清晰，先检查 job 存不存在再做对比。RL-only 字段（如 n_samples）在 SFT 场景下有 note 解释为什么是空的。timing 对比在步数差异大于 3 时会提示"可能不够可靠"。

但有几个问题：

1. same job 提前返回时没带 `metric_charts`、`diff_summary` 等新字段。虽然前端不会崩溃（有可选链保护），但不够严谨。
2. 命令行启动的 job 没有 `launch_config`，导致超参数对比始终是 0 changed。这是 AReno 现有架构的限制，不是这个 PR 引入的，但确实影响了实际使用体验。
3. `diff_summary` 里算 ratio 时，`float(val_a) != 0` 的检查在 try 块里，如果 val_a 是非数值类型会先抛 TypeError。虽然异常会被捕获不影响运行，但代码不够干净。
4. throughput 计算最初放在了 `timing_a` 定义之前，导致 `UnboundLocalError`。后来移到了后面修复。这个 bug 的原因是 Python 函数内只要有一处对变量的赋值，整个函数里这个变量就被视为局部变量，在赋值前引用就会报错。

### 前端
状态提升到 App 组件解决了页面切换丢数据的根本问题。指标卡片会自动隐藏不存在的指标（SFT 没有 reward 就不显示 reward 卡片）。数字格式化处理了大数、小数和科学计数法。但问题也不少：

1. Compare 按钮的 onClick 一开始写成 `onClick={fetchComparison}`，React 会把 Event 对象当第一个参数传入，导致 API 请求变成 `job_a=[object Object]`。改成 `onClick={() => fetchComparison()}` 才修复。
2. 关键指标卡片的渲染逻辑用了 IIFE（立即执行函数），放在 `(function() { ... })()` 里，可读性很差。应该拆成子组件或者用 useMemo。
3. 曲线图的归一化逻辑改了好几版。最初用绝对步数，两条线步数差距大时短的线几乎看不见。改成归一化后起点又对不齐（因为用 step/maxStep，起点不一定是 0）。最后用 `(step - minStep) / (maxStep - minStep)` 才让两条线都从最左边开始。
4. 不归一化时两条线画在同一个坐标系里会重叠，后来改成上下两个独立子图。

### 测试
27 个 CPU 测试覆盖了成功路径、无效输入、边界情况、CLI 命令和新功能。

但缺了几个：没有集成测试（Issue 明确要求了），没有测 CLI 的 `--dashboard-url` 超时场景，没有测 `metric_charts` 的 200 点截取逻辑。

### 其他检查项
- 复用性：复用了 `metric_summaries()`、`metric_series()`、`COMPARE_CONFIG_KEYS` 等现有 contract，没有不必要的重复代码。
- 兼容性：不改变任何现有方法的行为，新增端点 `/api/compare` 不修改现有端点，`test_existing_behavior_unchanged` 验证。
- 异常处理：缺 job 报 ValueError 返回 HTTP 400，same job 返回 comparable=false，非数值 metric 的 diff 返回 null 不崩溃。
- 性能：单次遍历分组替代 58 次全量遍历，指标曲线数据限 200 个点避免 JSON 过大。
- 提交范围：没有混入无关的格式化或文件修改，所有文件都与 compare 功能直接相关。

## 四、遇到的 Bug 记录

1. `job [object Object] not found` — onClick 直接传函数名，React 传了 Event 对象进去。改成箭头函数修复。
2. `UnboundLocalError: timing_a` — throughput 计算放在 timing_a 定义之前。调整顺序修复。
3. 页面切换丢数据 — state 在子组件里，切页面时组件卸载。状态提升到 App 修复。
4. 曲线图短线看不见 — 绝对步数模式下步数少的线很短。加归一化模式修复。
5. 归一化起点不对齐 — 用 step/maxStep 算坐标，起点不一定是 0。改成 (step-minStep)/(maxStep-minStep) 修复。
6. 不归一化时线重叠 — 两条线画在同一坐标系会重叠。改成上下两个独立子图。
7. CLI 输出 Nones — None 格式化时多了 s。加 helper 函数修复。
8. Compare 响应慢 — 29 个指标每个都调 metric_series 全量遍历。改成单次遍历 defaultdict 分组。

## 五、运行记录

### Fork 和分支
```plain
git clone https://github.com/anna495/AReno.git
git checkout -b feat/dashboard-compare-runs
```

### 构建

<!-- 贴 npm run build 的终端截图 -->

```plain
npm install --prefix dashboard
npm run build --prefix dashboard
# built in 4.15s
```

### 测试

<!-- 贴 pytest 27 passed 的终端截图 -->

```plain
pytest tests/test_dashboard_compare_cpu.py -v
# 27 passed in 0.14s
```

### Kaggle 里跑训练
```python
!areno train --ckpt Qwen/Qwen3.5-0.8B --algo sft \
  --dataset-loader-fn /kaggle/working/AReno/examples/sft/alpaca/dataset_loader.py \
  --batch-size 2 --mini-bs 2 --max-context-len 2048 \
  --metrics-log-dir /tmp/areno/tfevent_sft
```

跑了两次 SFT 训练，一个到 step 268，一个到 step 85。

### CLI 输出

<!-- 贴 CLI 输出的终端截图 -->

```plain
Job A: train sft Qwen/Qwen3.5-0.8B (id=5a3660acfbe3, status=exited, step=268)
Job B: train sft Qwen/Qwen3.5-0.8B (id=770ad2cffcab, status=exited, step=85)

Config: 0 changed, 0 identical

Metrics (29):
    rollout/accuracy: A=0.0  B=0.0  diff=+0.0
    rollout/advantages_mean: A=0.0  B=0.0  diff=+0.0
    rollout/advantages_std: A=0.0  B=0.0  diff=+0.0
    rollout/logprobs_mean: A=0.0  B=0.0  diff=+0.0
    rollout/num_sequences: A=2.0  B=2.0  diff=+0.0
    rollout/prompt_len_mean: A=31.0  B=45.0  diff=-14.0
    rollout/response_len_mean: A=74.5  B=160.5  diff=-86.0
    rollout/rewards_max: A=0.0  B=0.0  diff=+0.0
    rollout/rewards_mean: A=0.0  B=0.0  diff=+0.0
    rollout/rewards_min: A=0.0  B=0.0  diff=+0.0
    rollout/rewards_std: A=0.0  B=0.0  diff=+0.0
    rollout/seq_len_mean: A=105.5  B=205.5  diff=-100.0
    rollout/skipped_long: A=0.0  B=0.0  diff=+0.0
    rollout/total_skipped_long: A=0.0  B=0.0  diff=+0.0
    time/train: A=0.8329417705535889  B=1.1695845127105713  diff=-0.336643
    train/grad_nonzero_count: A=752492224.0  B=752582272.0  diff=-90048.0
    train/grad_nonzero_ratio: A=0.7475043535232544  B=0.7475937604904175  diff=-8.9e-05
    train/grad_norm: A=35.44312286376953  B=30.288209915161133  diff=+5.154913
    train/grad_total_count: A=1006672704.0  B=1006672704.0  diff=+0.0
    train/grad_zero_count: A=254180496.0  B=254090448.0  diff=+90048.0
    train/grad_zero_ratio: A=0.252495676279068  B=0.25240620970726013  diff=+8.9e-05
    train/loss: A=1.2566077709197998  B=1.3997747898101807  diff=-0.143167
    train/lr: A=8.4970531588624e-07  B=9.840508710112772e-07  diff=-0.0
    train/sft_logprob_mean: A=-1.2566077709197998  B=-1.3997747898101807  diff=+0.143167
    train/sft_loss: A=1.2566077709197998  B=1.3997747898101807  diff=-0.143167
    train/sft_target_tokens: A=149.0  B=321.0  diff=-172.0
    train/step_e2e_time_s: A=0.8329443335533142  B=1.1695916652679443  diff=-0.336647
    train/step_rollout_time_s: A=0.0  B=0.0  diff=+0.0
    train/step_train_time_s: A=0.8329417705535889  B=1.1695845127105713  diff=-0.336643

Timing:
    Steps: A=268  B=85
    Avg total/step: A=1.07s  B=3.1s
    Avg rollout/step: A=-  B=-
    Avg train/step: A=1.07s  B=3.1s
    Step time diff: -2.03s
    Note: job A ran 268 steps, job B ran 85 steps; timing comparison may be less reliable
```

### Dashboard 验证
通过 ngrok 在浏览器打开，选两个 job 点 Compare：

<!-- 贴 Dashboard Compare 页面截图（5张） -->

- 头部卡片、指标卡片、曲线图（29 个指标 tab 可切换）、指标表、Timing 表都正常
- 从 Compare 切到 Jobs 再切回来，数据还在，不用重新点 Compare

## 六、总结与反思

一开始这个需求，做的时候只列出了表格那样的29个指标进行对比，后面觉得还是得有指标曲线那些，但是没理解这个side by side 是让要两个实验的曲线并排还是叠加，最后两个都做了，如果选择归一化按钮的话，就是两个实验的指标曲线叠加在一起进行对比，以便于不同step的实验直观比较指标。缺点是界面的按钮可能不是很美观，做前端果然还是要有审美还要比较有耐心去调试。