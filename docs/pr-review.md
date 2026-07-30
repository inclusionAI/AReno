# PR Review 文档

## 一、对 PR 任务的理解

当前 AReno Dashboard 只能查看单个训练 run 的指标和配置，没有对比两次训练的能力。用户跑了两次训练（比如 SFT 和 GSPO，或者同样算法不同超参数），想直接在页面上看两个 run 的差异——哪些配置变了、loss 降了多少、训练速度差多少——只能手动开两个浏览器窗口来回切换，或者自己写脚本读 TensorBoard 数据。

本 PR 的目标是在 Dashboard 中新增一个 Compare 页面，允许用户选择任意两个 run（活跃或已完成），并排展示配置差异、指标对比、训练曲线、耗时和吞吐量。同时提供 `areno compare` CLI 命令支持终端输出。

本 PR 明确不处理以下内容：不引入外部数据库或云端服务，不修改 AReno 的训练器、rollout 引擎或现有 dashboard 存储，不自动修改用户配置或终止运行中的进程，不解决与此无关的其他 dashboard 改进需求。

修改影响的模块：`areno/dashboard/server.py` 新增 `compare_jobs()` 方法和 `/api/compare` 端点，`dashboard/src/main.jsx` 新增 `CompareRunsPanel` 前端组件，`dashboard/src/styles.css` 新增对比面板样式，`areno/cli/compare.py` 新增 CLI 命令，`areno/cli/main.py` 注册新命令。

验收标准：只使用现有本地 run 和 metric 数据；测试不等步数、缺失指标、活跃写入、页面导航和空/错误状态；不引入外部数据库；默认行为向后兼容；测试覆盖成功、无效输入和边界路径；文档包含最小可运行示例。

## 二、实现思路

主要文件和模块：后端在 `areno/dashboard/server.py` 的 `DashboardState` 类上新增 `compare_jobs()` 方法，前端在 `dashboard/src/main.jsx` 新增 `CompareRunsPanel` 组件，CLI 在 `areno/cli/compare.py` 新增 `areno compare` 命令。

核心数据流：前端选定 Job A 和 Job B → 调用 `GET /api/compare?job_a=xxx&job_b=yyy` → 后端 `compare_jobs()` 从 `DashboardState` 获取两个 Job 对象 → 分别提取 config、metric_summaries、metric 时间序列、timeperf 计时 → 配置对比遍历 `COMPARE_CONFIG_KEYS` 分为 identical/different → 指标对比按 name 聚合 summary 计算 diff → 指标曲线单次遍历 `job.metrics` 用 `defaultdict` 按 name 分组返回时间序列 → 返回结构化 JSON 给前端渲染。

关键设计选择及理由：
- 状态提升到 App 组件：AReno 的页面切换是条件渲染，子组件切走时会被卸载丢失 state。把 compareJobAId/compareJobBId/compareResult 存在父组件 App 里避免这个问题。
- 单次遍历 metric 分组：最初对每个指标调用 `metric_series()`，29 个指标 x 2 个 job = 58 次全量遍历。改成 `defaultdict` 单次分组后只需 2 次，复杂度从 O(58N) 降到 O(2N)。
- 归一化模式默认开启：两个 run 步数差距大时（如 267 vs 84）绝对步数模式下短线几乎看不见。归一化按训练进度拉伸到全宽，用 `(step - minStep) / (maxStep - minStep)` 确保两条线起点对齐。
- RL-only 字段注释：`n_samples`、`reward_fn_path` 等字段对 SFT 无意义，当一方是 SFT 时自动添加解释 note。
- 指标卡片自动隐藏：SFT 没有 reward 指标，对应卡片不显示，避免用户看到一堆无意义的空值。

考虑过的其他方案：最初用 URL 参数保存 compare 状态，但在 SPA 条件渲染下不可靠（页面切换不改变 URL），最终改为状态提升。不归一化时最初把两条线叠加在同一坐标系，步数差距大时短线看不见且中间部分重叠，改为上下两个独立子图。

兼容性：现有 API 和 dashboard 页面不变，只新增不修改。性能：单次遍历分组已优化，指标曲线数据限 200 个点避免 JSON 过大。异常处理：缺失 job 报 ValueError 返回 HTTP 400，same job 返回 comparable=false，非数值 metric 的 diff 返回 null 不崩溃。

## 三、对自己代码的 Review

正确性：正常输入两个不同 job 对比，输出配置/指标/计时/曲线，`test_compare_two_normal_jobs` 验证。边界输入 same job 返回 not comparable、空 config 显示 0 changed、无 metrics 返回空列表、非数值 metric 的 diff 返回 null，均有测试覆盖。review 后发现 throughput 计算引用了未定义的 timing_a 变量，已移到定义之后修复。

可读性：`compare_jobs()` 方法用 `# -- section --` 注释分隔了验证、配置对比、指标对比、曲线、计时等段落。不足：前端指标卡片渲染用了 IIFE（`(function() { ... })()`），200 多行塞在一个函数里，可读性差，应该拆成子组件或用 useMemo。

复用性：复用了 `metric_summaries()`、`metric_series()`、`COMPARE_CONFIG_KEYS` 等现有 contract。CLI 的 `_format_comparison_human()` 和 dashboard 的 `compare_jobs()` 共用后端逻辑。没有不必要的重复代码。

兼容性：不改变任何现有方法的行为，`test_existing_behavior_unchanged` 验证。新增端点 `/api/compare` 不修改现有端点。

异常处理：缺 job_a/job_b 报 `ValueError("job_a and job_b are required")`；job 不存在报 `ValueError(f"job {job_id} not found")`；same job 返回 `comparable: false, reason: "same job"`；非数值 metric diff 用 try/except 返回 null。测试覆盖了以上所有场景。

测试：27 个 CPU 测试覆盖成功路径（3）、无效输入（3）、边界（5）、活跃/兼容（3）、CLI（3）、新功能（10），全部通过。缺失：没有集成测试（Issue 要求了），没有测 CLI `--dashboard-url` 超时场景，没有测 `metric_charts` 的 200 点截取逻辑。

性能：单次遍历分组替代 58 次全量遍历已优化。指标曲线数据限 200 个点避免 JSON 过大。

提交范围：没有混入无关的格式化或文件修改，所有文件都与 compare 功能直接相关。

## 四、遇到的问题、挑战与解决方法

1. `job [object Object] not found`
   - 现象：点击 Compare 按钮后后端报错 `job [object Object] not found`
   - 定位过程：检查前端网络请求，发现 URL 是 `/api/compare?job_a=[object Object]`
   - 根因：`onClick={fetchComparison}` 会把 React Event 对象作为第一个参数传入，`fetchComparison(event)` 里 `aId` 变成 Event 对象
   - 解决方法：改为 `onClick={() => fetchComparison()}`
   - 验证方式：重新点击 Compare，请求 URL 正确变成 `job_a=5a3660acfbe3`
   - 经验总结：React 事件处理函数如果不接收参数，必须用箭头函数包装

2. `UnboundLocalError: timing_a`
   - 现象：后端报错 `cannot access local variable 'timing_a'`
   - 定位过程：检查 `compare_jobs()` 代码，发现 `throughput_a = _throughput(job_a, timing_a)` 写在了 `timing_a = _timing_stats(job_a)` 之前
   - 根因：Python 函数内只要有一处对变量的赋值，整个函数里这个变量就被视为局部变量，在赋值前引用会报 UnboundLocalError
   - 解决方法：把 throughput 计算移到 timing_a 赋值之后
   - 验证方式：重新调用 compare API，不再报错
   - 经验总结：Python 变量作用域和 JS 不同，赋值顺序很重要

3. 页面切换后 Compare 数据丢失
   - 现象：从 Compare 页面切到 Jobs 再切回来，之前选的 job 和对比结果都没了
   - 定位过程：检查 React 组件生命周期，AReno 的页面切换是条件渲染，切走时 CompareRunsPanel 被卸载，useState 丢失
   - 根因：compare 状态在子组件内部，卸载即丢失
   - 解决方法：把 compareJobAId/compareJobBId/compareResult 提升到 App 组件
   - 验证方式：切到 Jobs 再切回来，选中的 job 和对比结果都在
   - 经验总结：React 状态管理需要考虑组件生命周期，跨页面共享的状态应该提升到父组件

4. 曲线图短线看不见
   - 现象：归一化前，Job A 有 267 步但 Job B 只有 84 步，B 的线只占左边一小段
   - 定位过程：检查 X 坐标计算，`x = (step / maxStep) * width`，B 的 84 步只占 31% 宽度
   - 根因：绝对步数模式下，步数少的线被压缩
   - 解决方法：加归一化模式，各自按进度拉伸到 0-100%
   - 验证方式：归一化后两条线都铺满全宽
   - 经验总结：对比不同长度的时间序列需要考虑坐标轴归一化

5. 归一化后两条线起点不对齐
   - 现象：归一化后蓝色线起点在左边，橙色线起点偏右
   - 定位过程：检查归一化坐标计算 `x = (step / maxStep) * width`，第一个点的 step 不一定是 0
   - 根因：数据点的起始 step 可能不是 0，除以 maxStep 后起点偏右
   - 解决方法：改为 `(step - minStep) / (maxStep - minStep)`，每条线用自己的 minStep 做起点
   - 验证方式：两条线都从最左边开始
   - 经验总结：归一化要减去各自的起始值，否则起点不对齐

6. CLI 输出 `Nones`
   - 现象：Timing 输出 `Avg rollout/step: A=Nones` 而不是 `A=None`
   - 定位过程：检查 f-string，`f"A={ta.get('avg_rollout_s', '?')}s"` 当值为 None 时变成 `Nones`
   - 根因：None + "s" = "Nones"
   - 解决方法：加 `_fmt_t()` helper 函数，None 返回 "-"
   - 验证方式：重新运行 CLI，输出 `A=-`
   - 经验总结：f-string 拼接 None 时要注意类型转换

## 五、分步骤运行结果证明

### 步骤 1：测试通过

命令：`pytest tests/test_dashboard_compare_cpu.py -v`
输出：27 passed in 0.14s

<!-- 贴 pytest 27 passed 的终端截图 -->

### 步骤 2：CLI 输出

命令：`python3 -m areno.cli.main compare --job-a 5a3660acfbe3 --job-b 770ad2cffcab`

<!-- 贴 CLI 输出的终端截图 -->

### 步骤 3：Dashboard Compare 页面

通过 ngrok 在浏览器打开，选两个 SFT job 点 Compare。

<!-- 贴 Dashboard Compare 页面截图（5张） -->

## 六、总结与反思

一开始这个需求，做的时候只列出了表格那样的29个指标进行对比，后面觉得还是得有指标曲线那些，但是没理解这个side by side 是让要两个实验的曲线并排还是叠加，最后两个都做了，如果选择归一化按钮的话，就是两个实验的指标曲线叠加在一起进行对比，以便于不同step的实验直观比较指标。缺点是界面的按钮可能不是很美观，做前端果然还是要有审美还要比较有耐心去调试。