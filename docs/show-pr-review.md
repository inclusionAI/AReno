# Show Run Info PR Review 文档

## 一、对 PR 任务的理解

AReno 目前没有一个命令能直接在终端查看单个训练 run 的完整信息。用户想知道某个 run 用了什么模型、什么数据集、跑了多少步、最新指标是多少、有没有报错，只能启动 Dashboard 在浏览器里看，或者自己翻文件。对于 SSH 环境或快速排查来说不太方便。

本 PR 的目标是新增 `areno show <run_id>` CLI 命令，在终端直接展示一个 run 的 model、dataset、algorithm、关键配置、当前阶段、最新指标和最后的错误信息。支持 human/table/json 三种输出格式。

本 PR 明确不处理以下内容：不引入外部数据库，不修改 Dashboard，不修改训练器或 rollout 引擎，不打印完整训练样本或密钥。

修改影响的模块：`areno/cli/show.py` 新增 show 命令，`areno/cli/main.py` 注册新命令。

验收标准：测试模糊/缺失 ID、部分写入的 artifact、table 和 JSON 模式；确保不打印密钥或完整样本；使用现有 AReno 契约；默认行为向后兼容；测试覆盖成功、无效和边界路径；文档含最小可运行示例。

## 二、实现思路

主要文件：`areno/cli/show.py` 实现 show 命令，`areno/cli/main.py` 注册命令。

核心数据流：用户输入 run_id → 先尝试 Dashboard API `GET /api/jobs/{id}` 获取完整数据 → 如果 Dashboard 不可用则从本地 registry 和 state 文件解析 → Dashboard 返回的 config 含 sections 结构化数据需要用 `_normalize_dashboard_config()` 拍平 → 单独调 `GET /api/jobs/{id}/metrics` 获取指标摘要 → 格式化输出。

关键设计选择及理由：
- Dashboard API 优先，本地 fallback：Dashboard 运行时数据更全（有 metrics、timeperf、logs），不可用时 fallback 到 registry 文件，至少能显示基本信息。
- sections 配置拍平：Dashboard API 返回的 config 有一个 `sections` 键是结构化展示用的（按 Basic/Rollout/Train/Optimizer 等分组），CLI 需要的是 flat key-value，所以提取成扁平字典。
- 密钥自动隐藏：config 里可能包含 api_key、token 等敏感信息，用 `secret_patterns` 匹配 key 名自动跳过不打印。
- Partial ID 匹配：run ID 是 12 位 hex，用户不想输完整，支持前缀匹配。如果有多个匹配则报 ambiguous 错误。
- 移除重复代码：`_normalize_dashboard_config()` 和 `_build_details_from_dashboard()` 提取成共享函数，`show_command` 和 `_load_job_details` 都调用它们。

兼容性：只新增命令不修改原有行为。性能：API 调用有 5 秒超时。异常处理：run 不存在报 ClickException 退出码非 0，ambiguous ID 列出所有匹配。

## 三、对自己代码的 Review

正确性：正常输入显示完整信息，边界输入（缺失 ID、部分 artifact、空 metrics/config）都有测试覆盖。review 后发现 show_command 和 _load_job_details 里重复了 sections 解析逻辑，已提取成 `_normalize_dashboard_config()` 和 `_build_details_from_dashboard()` 消除重复。

可读性：用 `# -- section --` 注释分隔了 resolving、loading、formatting、command 段落。函数职责清晰：_resolve_job 负责 ID 匹配，_load_job_details 负责数据加载，_format_human 负责格式化，show_command 负责编排。

复用性：复用了 `GLOBAL_REGISTRY_FILE`、`DashboardState.get_job()`、`DashboardState.metric_summaries()` 等现有 contract。和项目其他 CLI 命令（diagnostics、dashboard）的 click 命令模式一致。

兼容性：不改变任何现有方法或端点。

异常处理：run 不存在 → `ClickException("Run 'xxx' not found")`；ambiguous partial ID → `ClickException` 列出匹配项；Dashboard 不可用 → fallback 到本地文件；API 超时 → 返回 None 走 fallback。测试覆盖了以上场景。

测试：20 个 CPU 测试覆盖 human 格式（7）、JSON（1）、table（1）、无效输入（6）、活跃 run（2）、job 解析（3），全部通过。

性能：API 调用 5 秒超时，不引入明显开销。

提交范围：只有 show.py、main.py（+1行）、test 文件和文档，都是 show 功能直接相关的。

## 四、遇到的问题、挑战与解决方法

1. Dashboard API 返回的 config 有 sections 结构化数据
   - 现象：`areno show` 输出的 Key settings 里出现了一个巨大的 `sections` 字段，把整个结构化配置原样打印出来了
   - 定位过程：检查 Dashboard API 的 `/api/jobs/{id}` 返回的 config，发现有个 `sections` 键包含分组展示数据
   - 根因：Dashboard 的 `to_json()` 返回的 config 带 sections 结构化字段，CLI 不应该原样打印
   - 解决方法：加 `_normalize_dashboard_config()` 把 sections 拍平成 flat key-value
   - 验证方式：重新运行 `areno show`，Key settings 正确显示 algo/ckpt/dataset_path 等扁平配置
   - 经验总结：不同消费者对数据格式的要求不同，Dashboard 前端需要结构化展示，CLI 需要 flat 输出

2. Dashboard API 不返回 metric summaries
   - 现象：`areno show` 输出里没有 Latest metrics 部分
   - 定位过程：检查 Dashboard API 的 `/api/jobs/{id}` 返回，发现 `to_json()` 只有 `metrics_count` 没有 metric summaries
   - 根因：`to_json()` 和 `to_summary_json()` 都不包含 metric summaries，那是 `metric_summaries()` 方法单独提供的
   - 解决方法：加 `_fetch_metric_summaries()` 单独调 `/api/jobs/{id}/metrics` 端点获取
   - 验证方式：重新运行 `areno show`，29 个指标都显示了
   - 经验总结：Dashboard API 的不同端点返回不同粒度的数据，需要组合调用

3. show_command 和 _load_job_details 里重复 sections 解析
   - 现象：review 时发现 sections 解析逻辑在两个地方都写了，代码重复
   - 定位过程：对比 show_command 的 348-390 行和 _load_job_details 的 82-127 行
   - 根因：最初实现时两处独立写了 sections 解析，没有提取共享函数
   - 解决方法：提取 `_normalize_dashboard_config()` 和 `_build_details_from_dashboard()`，两处都调用
   - 验证方式：重构后 20 个测试全部通过
   - 经验总结：写完后要检查重复代码，及时提取共享函数

## 五、分步骤运行结果证明

### 步骤 1：测试通过

命令：`pytest tests/test_cli_show_cpu.py -v`
输出：20 passed in 0.10s

这 20 个测试覆盖了：
- human 格式（7个）：基本输出、关键配置、指标、错误段落、计时、密钥隐藏
- JSON 格式（1个）：合法 JSON 且含预期字段
- table 格式（1个）：关键字段存在
- 无效输入（6个）：不存在的 ID、歧义 partial ID、partial 匹配、空 metrics/config、部分 artifact
- 活跃 run（2个）：running 状态、stage 显示、无 exit code
- job 解析（3个）：精确 ID 匹配、PID 匹配、无匹配

<!-- 贴 pytest 20 passed 的终端截图 -->

### 步骤 2：CLI 输出

命令：`python3 -m areno.cli.main show 841950e0f0ed`

<!-- 贴 CLI 输出的终端截图 -->

## 六、总结与反思

这个任务比 Compare 简单，主要是纯 CLI 不涉及前端。做的时候踩了一个坑：Dashboard API 返回的 config 里有 sections 结构化数据，直接打印出来一大坨，后来加了 normalize 函数拍平。还有 metrics 需要单独调 endpoint 获取，to_json() 里只有 count 没有 summary。这些问题说明拿 API 数据之前要先看清楚返回结构，不能想当然。