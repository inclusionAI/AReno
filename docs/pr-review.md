# PR Review 文档

## 一、对 Issue 的理解

这个 issue 要做的是在 Dashboard 里加一个对比两次训练运行的功能。用户跑了两次训练（比如 SFT 和 GSPO，或者同样算法不同超参数），想直接在页面上看两个 run 的差异——哪些配置变了、loss 降了多少、训练速度差多少。

Issue 特别强调要用 AReno 现有的本地数据，不引入外部数据库。这意味着所有数据都来自 `job.metrics`、`job.launch_config`、`job.timeperf` 这些已有的字段，不能搞一套新的存储。

"side by side" 我理解就是左右并排展示，让用户一眼能看出两个 run 的区别，而不是来回切换页面看。

## 二、实现了什么

后端在 `server.py` 加了 `compare_jobs()` 方法和 `/api/compare` 端点。前端加了 `CompareRunsPanel` 组件，包含头部信息卡片、关键指标卡片、超参数对比表、指标曲线对比图、差异摘要、完整指标表和 Timing 表。CLI 加了 `areno compare` 命令。

## 三、代码 Review

### 后端

`compare_jobs()` 方法的验证逻辑比较清晰，先检查 job 存不存在再做对比。RL-only 字段（如 n_samples）在 SFT 场景下有 note 解释为什么是空的。timing 对比在步数差异大于 3 时会提示"可能不够可靠"。

但有几个问题：

1. same job 提前返回时没带 `metric_charts`、`diff_summary` 等新字段。虽然前端不会崩溃（有可选链保护），但不够严谨。

2. 命令行启动的 job 没有 `launch_config`，导致超参数对比始终是 0 changed。这是 AReno 现有架构的限制，不是这个 PR 引入的，但确实影响了实际使用体验。

3. `diff_summary` 里算 ratio 时，`float(val_a) != 0` 的检查在 try 块里，如果 val_a 是非数值类型会先抛 TypeError。虽然异常会被捕获不影响运行，但代码不够干净。

4. throughput 计算最初放在了 `timing_a` 定义之前，导致 `UnboundLocalError`。后来移到了后面修复。这个 bug 的原因是 Python 函数内只要有一处对变量的赋值，整个函数里这个变量就被视为局部变量，在赋值前引用就会报错。

### 前端

状态提升到 App 组件解决了页面切换丢数据的根本问题。指标卡片会自动隐藏不存在的指标（SFT 没有 reward 就不显示 reward 卡片）。数字格式化处理了大数、小数和科学计数法。

但问题也不少：

5. Compare 按钮的 onClick 一开始写成 `onClick={fetchComparison}`，React 会把 Event 对象当第一个参数传入，导致 API 请求变成 `job_a=[object Object]`。改成 `onClick={() => fetchComparison()}` 才修复。

6. 关键指标卡片的渲染逻辑用了 IIFE（立即执行函数），200 多行塞在 `(function() { ... })()` 里，可读性很差。应该拆成子组件或者用 useMemo。

7. 前端没有任何测试。项目本身没配 Vitest/Jest，所以这部分代码没有自动化保障。

8. 曲线图的归一化逻辑改了好几版。最初用绝对步数，两条线步数差距大时短的线几乎看不见。改成归一化后起点又对不齐（因为用 step/maxStep，起点不一定是 0）。最后用 `(step - minStep) / (maxStep - minStep)` 才让两条线都从最左边开始。

9. 不归一化时两条线画在同一个坐标系里会重叠，后来改成上下两个独立子图。

### 测试

27 个 CPU 测试覆盖了成功路径、无效输入、边界情况、CLI 命令和新功能。

但缺了几个：没有集成测试（Issue 明确要求了），没有测 CLI 的 `--dashboard-url` 超时场景，没有测 `metric_charts` 的 200 点截取逻辑。

## 四、运行记录

### Fork 和分支

```
git clone https://github.com/anna495/AReno.git
git checkout -b feat/dashboard-compare-runs
```

### 构建

```
npm install --prefix dashboard
npm run build --prefix dashboard
# built in 4.15s
```

### 测试

```
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

```
Job A: train sft Qwen/Qwen3.5-0.8B (id=5a3660acfbe3, status=exited, step=268)
Job B: train sft Qwen/Qwen3.5-0.8B (id=770ad2cffcab, status=exited, step=85)

Config: 0 changed, 0 identical

Metrics (29):
    rollout/accuracy: A=0.0  B=0.0  diff=+0.0
    rollout/prompt_len_mean: A=31.0  B=45.0  diff=-14.0
    rollout/response_len_mean: A=74.5  B=160.5  diff=-86.0
    train/loss: A=1.2566  B=1.3998  diff=-0.143167
    train/grad_norm: A=35.443  B=30.288  diff=+5.154913
    ...

Timing:
    Steps: A=268  B=85
    Avg total/step: A=1.07s  B=3.1s
    Avg rollout/step: A=-  B=-
    Step time diff: -2.03s
    Note: job A ran 268 steps, job B ran 85 steps; timing comparison may be less reliable
```

### Dashboard 验证

通过 ngrok 在浏览器打开，选两个 job 点 Compare：
- 头部卡片、指标卡片、曲线图（29 个指标 tab 可切换）、指标表、Timing 表都正常
- 从 Compare 切到 Jobs 再切回来，数据还在，不用重新点 Compare

## 五、遇到的 Bug 记录

1. `job [object Object] not found` — onClick 直接传函数名，React 传了 Event 对象进去。改成箭头函数修复。
2. `UnboundLocalError: timing_a` — throughput 计算放在 timing_a 定义之前。调整顺序修复。
3. 页面切换丢数据 — state 在子组件里，切页面时组件卸载。状态提升到 App 修复。
4. 曲线图短线看不见 — 绝对步数模式下步数少的线很短。加归一化模式修复。
5. 归一化起点不对齐 — 用 step/maxStep 算坐标，起点不一定是 0。改成 (step-minStep)/(maxStep-minStep) 修复。
6. 不归一化时线重叠 — 两条线画在同一坐标系会重叠。改成上下两个独立子图。
7. CLI 输出 Nones — None 格式化时多了 s。加 helper 函数修复。
8. Compare 响应慢 — 29 个指标每个都调 metric_series 全量遍历。改成单次遍历 defaultdict 分组。

## 六、总结

<!-- 在这里写你自己的反思 -->