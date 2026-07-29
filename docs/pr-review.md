# PR 理解与代码 Review 文档

## 一、对 Issue #264 的理解

Issue 要求实现一个 Dashboard 中的"对比两次训练"功能，核心诉求是：
- 选择两个 run（活跃或已完成），并排展示算法/模型、变更的关键配置、最新指标、吞吐量、时长和各阶段耗时
- 折叠相同的配置项，解释无法对比的字段
- 使用 AReno 现有的本地数据契约，不引入外部数据库
- 提供 CLI 命令，支持 human 和 JSON 两种输出
- 默认行为向后兼容
- CPU 测试覆盖成功路径、无效输入、边界情况

### 我的理解

<!-- 在这里写你自己的理解，比如：
1. 这个 issue 的本质是什么？为什么需要这个功能？
2. "side by side" 意味着什么？用户看这个面板想得到什么信息？
3. 为什么强调"使用现有数据契约"？这说明了什么设计约束？
-->

## 二、实现方案概述

### 后端 (`areno/dashboard/server.py`)
- `compare_jobs()` 方法：接收两个 job ID，返回结构化对比数据
  - 配置对比：遍历 `COMPARE_CONFIG_KEYS`，分为 identical / different
  - 指标对比：按 name 聚合 `metric_summaries`，计算 diff
  - 指标曲线：单次遍历 `job.metrics` 按 name 分组，返回时间序列
  - 差异摘要：自动生成配置差异的文字摘要（2x 以上倍率标注）
  - 吞吐量：steps / duration
- `/api/compare` 端点：GET 请求，参数 `job_a` 和 `job_b`

### 前端 (`dashboard/src/main.jsx`, `dashboard/src/styles.css`)
- `CompareRunsPanel` 组件：7 个区域
  1. Run 头部卡片（名称、状态徽章、算法、步数）
  2. 关键指标卡片（Loss/Reward/Accuracy/LR/Steps/Duration/Throughput，A/B 彩色标签）
  3. 超参数对比表（差异高亮，可折叠相同项）
  4. 指标曲线对比图（tab 切换，归一化叠加 / 不归一化上下子图）
  5. 差异摘要
  6. 完整指标对比表
  7. Timing 对比表
- 状态提升到 `App` 组件，页面切换不丢数据

### CLI (`areno/cli/compare.py`)
- `areno compare --job-a --job-b` 命令
- `--format human|json` 两种输出
- Dashboard 不可用时 fallback 到本地 artifact

## 三、代码 Self-Review

### 3.1 后端代码 Review

#### `compare_jobs()` 方法

**正面：**
- 验证逻辑清晰：先检查 job_a/job_b 是否存在，再进行对比
- RL-only 字段有 note 解释（如 SFT 没有 n_samples）
- timing 对比有步数差异警告（steps_diff > 3 时提示）
- 使用 `defaultdict` 单次遍历分组，避免重复扫描 `job.metrics`

**发现的瑕疵：**

1. **`same job` 提前返回时缺少新字段**
   - 问题描述：当 `job_a_id == job_b_id` 时，直接返回 `comparable: false`，但没有返回 `metric_charts`、`diff_summary`、`throughput_a/b` 字段
   - 影响：前端如果依赖这些字段可能报错（实际上前端有 `?.` 可选链保护，不会崩溃，但不够严谨）
   - 我的思考：这个问题影响不大，因为 same job 是异常路径，前端会显示 "not comparable" 不渲染后续区域。但严格来说应该补上空值

2. **CLI 启动的 job 没有 `launch_config`**
   - 问题描述：命令行启动的 SFT 训练不会保存 `launch_config`，导致 `config_a` 和 `config_b` 都为空 dict，超参数对比显示 0 changed
   - 影响：用户在 Dashboard 里用命令行启动的两个 job 对比时，超参数对比区域是空的
   - 我的思考：这是 AReno 现有架构的限制，不是本 PR 引入的 bug。`launch_config` 是 Dashboard Launcher 启动时才保存的。文档的 Limitations 中已说明

3. **`diff_summary` 的 ratio 计算可能除零**
   - 问题描述：`float(val_a) != 0` 的检查在 try 块里，如果 `val_a` 是非数值类型会先抛 TypeError 被 catch
   - 影响：不影响正确性，异常会被捕获，但代码不够优雅

#### 前端代码 Review

**正面：**
- 状态提升到 App 组件解决了页面切换丢数据的核心问题
- 指标卡片自动隐藏不存在的指标（SFT 无 Reward）
- 数字格式化函数处理了大数、小数、科学计数法
- 曲线图归一化模式用各自 minStep 做起点，两条线对齐

**发现的瑕疵：**

4. **`onClick={() => fetchComparison()}` 曾经写成 `onClick={fetchComparison}`**
   - 问题描述：React 的 onClick 会把 Event 对象作为第一个参数传入，`fetchComparison(event)` 导致 `aId` 变成 Event 对象，API 请求变成 `job_a=[object Object]`
   - 修复：改为 `onClick={() => fetchComparison()}`
   - 教训：React 事件处理函数如果不接收参数，必须用箭头函数包装

5. **指标卡片内联函数 `(function() { ... })()` 过长**
   - 问题描述：关键指标卡片的渲染逻辑用了 IIFE（立即执行函数表达式），200 多行塞在一个函数里
   - 影响：可读性差，难维护
   - 我的思考：应该提取成独立的 `useMemo` 或子组件。当时为了快速实现用了 IIFE，不是最佳实践

6. **`timing_a` 变量未定义引用（已修复）**
   - 问题描述：`throughput` 计算放在了 `_timing_stats` 定义之前，引用了还未赋值的 `timing_a`
   - 修复：把 `result["throughput_a"] = _throughput(job_a, timing_a)` 移到 `timing_a = _timing_stats(job_a)` 之后
   - 教训：Python 函数作用域中，变量在函数内任何位置赋值都会被视为局部变量。如果在赋值前引用，会报 `UnboundLocalError`

7. **前端没有单元测试**
   - 问题描述：项目原本没有 Jest/Vitest 配置，所以前端代码没有测试覆盖
   - 影响：前端逻辑（如归一化坐标计算、指标卡片显示/隐藏逻辑）没有自动化测试保障
   - 我的思考：和项目现状一致，但如果要更严谨应该加 Vitest 测试

### 3.2 测试 Review

**覆盖的场景（27 个测试）：**
- 成功路径：正常对比、GSPO vs SFT、相同配置
- 无效输入：缺 job_a、缺 job_b、不存在的 job
- 边界：same job、无指标、不等步数、无 timeperf、空 config
- 活跃/兼容：运行中 job、duration 计算、原有方法不变
- CLI：human 格式、JSON 格式、无效 job
- 新功能：metric_charts、diff_summary、throughput、非数值 metric

**缺失的测试：**

8. **没有测试 CLI `--dashboard-url` 参数**
   - `_try_dashboard_api` 的超时和连接失败场景没有测试

9. **没有测试 `metric_charts` 的数据点限制**
   - 后端 `limit=200` 截取逻辑没有专门的测试

10. **没有集成测试**
    - Issue 要求 "Add an integration-style test using tiny local fixtures where the feature crosses modules"
    - 目前只有单元测试，没有跨模块的集成测试

## 四、分步骤运行记录

### 步骤 1：Fork 仓库并创建分支

```
git clone https://github.com/anna495/AReno.git
cd AReno
git checkout -b feat/dashboard-compare-runs
```

### 步骤 2：后端实现

实现 `compare_jobs()` 方法和 `/api/compare` 端点。

验证后端语法：
```
python3 -c "import py_compile; py_compile.compile('areno/dashboard/server.py', doraise=True)"
# 输出: (无错误)
```

### 步骤 3：前端实现

实现 `CompareRunsPanel` 组件和样式。

构建前端：
```
npm install --prefix dashboard
npm run build --prefix dashboard
# 输出:
# ../areno/dashboard/dist/index.html                   0.40 kB
# ../areno/dashboard/dist/assets/index-9lyAEhxf.css   36.04 kB
# ../areno/dashboard/dist/assets/index-Wzu9Xf6x.js   415.53 kB
# ✓ built in 4.15s
```

### 步骤 4：测试

```
pytest tests/test_dashboard_compare_cpu.py -v

# 输出:
# 27 passed in 0.14s
```

### 步骤 5：Kaggle 环境验证

启动 Dashboard 和训练：

```python
# 启动 SFT 训练（两次，不同步数）
!areno train --ckpt Qwen/Qwen3.5-0.8B --algo sft --dataset-loader-fn /kaggle/working/AReno/examples/sft/alpaca/dataset_loader.py ...

# 启动 Dashboard
subprocess.Popen(["python3", "-m", "areno.dashboard.server"], cwd="/kaggle/working/AReno")

# 通过 ngrok 暴露
public_url = ngrok.connect(addr="127.0.0.1:8765")
```

### 步骤 6：CLI 验证

```
$ python3 -m areno.cli.main compare --job-a 5a3660acfbe3 --job-b 770ad2cffcab

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

### 步骤 7：Dashboard 验证

在浏览器中打开 ngrok URL，点击 Compare 标签页：
- 选择 Job A 和 Job B，点击 Compare
- 确认 7 个区域都正常渲染：
  - ✅ Run 头部卡片：显示名称、状态、算法、步数
  - ✅ 关键指标卡片：Loss/LR/Steps/Duration/Throughput 有值，Reward/Accuracy 自动隐藏
  - ✅ 超参数对比：0 changed（CLI 启动的 job 无 config）
  - ✅ 指标曲线：29 个 tab 可切换，归一化模式两条线从同一起点叠加
  - ✅ 差异摘要：空（配置无差异）
  - ✅ 完整指标表：29 个指标对比
  - ✅ Timing 对比：步数、平均耗时、总时长

### 步骤 8：页面切换验证

从 Compare 切到 Jobs 再切回来：
- ✅ 选中的 Job A/B 保持不变
- ✅ 对比结果保持不变
- ✅ 不需要重新点击 Compare

## 五、遇到的 Bug 与修复记录

| # | Bug | 原因 | 修复 |
|---|-----|------|------|
| 1 | `job [object Object] not found` | `onClick={fetchComparison}` 传入 Event 对象 | 改为 `onClick={() => fetchComparison()}` |
| 2 | `UnboundLocalError: timing_a` | throughput 计算在 timing_a 定义之前 | 移到 timing_a 赋值之后 |
| 3 | 页面切换后 Compare 数据丢失 | state 在子组件内，切换时卸载 | 状态提升到 App 组件 |
| 4 | 下拉框遮挡 label | `align-items: flex-end` + padding 过大 | 改为 `flex-end` + 缩小字体 |
| 5 | 指标卡片 better 标签与内容重叠 | `position: absolute` 浮在内容上 | 改为 inline，和标题同一行 |
| 6 | 曲线图 B 只占左边一小段 | 绝对步数模式，短 run 的线很短 | 归一化模式默认开启 |
| 7 | 不归一化时两条线中间重叠 | 共享 X 轴坐标 | 改为上下两个独立子图 |
| 8 | 归一化时两条线起点不同 | 用绝对 step 值除以 maxStep | 用 `(step - minStep) / (maxStep - minStep)` |
| 9 | CLI 输出 `Nones` | f-string 格式化 None 时多了 s | 用 `_fmt_t()` helper 函数 |
| 10 | Compare 响应慢 | 29 个指标 x 2 个 job = 58 次全量遍历 | 单次遍历 defaultdict 分组 |

## 六、总结与反思

<!-- 在这里写你自己的反思，比如：
1. 这个 PR 你学到了什么？
2. 如果重做你会怎么做？
3. 你觉得代码质量怎么样？哪些地方需要改进？
4. AI 生成的代码有哪些常见问题你在 review 中发现了？
-->