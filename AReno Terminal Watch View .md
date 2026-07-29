AReno Terminal Watch View 系统设计文档
1. 功能概述
1.1 需求背景
根据 GitHub Issue #252，AReno 用户需要一个非侵入性的终端 watch 视图来实时监控训练运行进度。该功能需要：
● 实时刷新显示：step、loss、reward、生成吞吐量、预计剩余时间
● Ctrl+C 只停止观察者，不停止训练任务
● 支持多种输出模式：TTY 刷新、非 TTY 行输出、JSON Lines
● 向后兼容：不启用时行为不变
1.2 实现功能
功能	说明
areno watch --latest	监控最近一次训练运行
areno watch --run-id <ID>	监控指定的运行
areno runs	列出所有可监控的运行
--json	JSON Lines 模式输出
--tail N	只显示最近 N 行
--fields f1,f2	只显示指定字段
--no-header	隐藏头部信息
--interval N	自定义刷新间隔
--timeout N	超时自动退出

2. 技术架构
2.1 技术选型
组件	技术选型	理由
CLI 框架	Click	AReno 现有项目使用 Click，风格统一
数据源	文件 Polling	实现简单、内存占用小、与训练进程解耦
状态文件	复用 dashboard_state.{pid}.json	复用现有 dashboard 基础设施
颜色输出	ANSI Escape Codes	标准终端颜色方案，无需额外依赖
信号处理	Python signal 模块	标准库，实现优雅退出
2.2 系统架构图
┌─────────────────────────────────────────────────────────────────┐
│                         User Terminal                           │
│                                                                  │
│   ┌─────────────────┐    ┌─────────────────────────────────────┐│
│   │  areno train    │    │         areno watch                 ││
│   │  (训练进程)     │    │  ┌─────────────┐  ┌───────────────┐ ││
│   │                 │    │  │watch_command│  │WatchObserver │ ││
│   │                 │    │  │ (CLI入口)   │  │ (核心类)     │ ││
│   └────────┬────────┘    │  └─────────────┘  └───────────────┘ ││
│            │              │         ▲              ▲            ││
│            │ 写入 metrics │         │              │            ││
│            ▼              │    ┌────┴──────────────┴──────────┐ ││
│   ┌───────────────────────┴────┤  TerminalRenderer            │ ││
│   │  ~/.areno/runs/{run_id}/   │  - render_tty() (颜色高亮)    │ ││
│   │  dashboard_state.{pid}.json│  - render_line() (fields过滤) │ ││
│   │                            │  - render_json() (JSON输出)   │ ││
│   └────────────────────────────┴──────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
2.3 模块设计
areno/cli/
├── main.py              # CLI 入口，注册 watch/runs 命令
├── watch.py             # 核心实现模块（新增）
│   ├── WatchConfig      # 配置数据类
│   ├── RunStatus        # 状态解析数据类
│   ├── Colors           # ANSI 颜色定义
│   ├── find_latest_run_id()     # 查找最近运行
│   ├── find_status_file()       # 定位状态文件
│   ├── read_status()            # 读取并解析状态
│   ├── calculate_eta()          # ETA 计算
│   ├── render_tty()             # TTY 渲染（含颜色）
│   ├── render_line()            # 行渲染（支持 fields/tail）
│   ├── render_json()            # JSON 渲染
│   ├── GracefulExit             # 优雅退出信号处理
│   ├── watch_command            # Click CLI 命令
│   └── runs_command             # Click CLI 命令（列出运行）
│
tests/
└── test_watch_cpu.py   # 33 个 CPU 测试（新增）

3. 接口设计
3.1 CLI 命令
3.1.1 areno watch
areno watch [OPTIONS]

# 监控最近一次运行
areno watch --latest

# 监控指定运行
areno watch --run-id 20240115_143022

# JSON Lines 模式（适合日志收集）
areno watch --latest --json

# 只显示最近 10 行
areno watch --latest --tail 10

# 只显示指定字段
areno watch --latest --fields step,loss,reward

# 刷新间隔 2 秒，超时 1 小时
areno watch --latest --interval 2 --timeout 3600
参数说明：
参数	类型	默认值	说明
--run-id	string	-	指定运行 ID
--latest	flag	false	监控最近运行
--interval	int	1	刷新间隔（秒）
--json	flag	false	JSON Lines 输出
--quiet	flag	false	静默模式
--no-header	flag	false	隐藏头部
--timeout	int	0	超时退出（0=无限）
--tail	int	-	只显示最近 N 行
--fields	string	-	逗号分隔的字段列表
3.1.2 areno runs
areno runs [OPTIONS]

# 列出所有运行
areno runs

# 详细信息
areno runs --verbose

# JSON 输出
areno runs --json
参数说明：
参数	类型	默认值	说明
-v, --verbose	flag	false	显示详细信息
--json	flag	false	JSON 格式输出
3.2 数据格式
3.2.1 状态文件格式（复用现有）
{
  "pid": 12345,
  "stage": "train",
  "status": "running",
  "updated_at": 1699999999.123,
  "step": 150,
  "epoch": 3,
  "role": "worker-0",
  "loss": 0.2345,
  "reward_mean": 0.8923,
  "throughput": 1200,
  "total_steps": 1000
}
3.2.2 JSON Lines 输出
{"step":150,"total_steps":1000,"loss":0.2345,"reward":0.8923,"throughput":1200,"eta_seconds":754,"stage":"train","status":"running"}

4. 核心逻辑
4.1 Polling 循环
def run_watch(config: WatchConfig) -> None:
    # 1. 解析 run_id
    run_id = config.run_id or find_latest_run_id()
    
    # 2. 查找状态文件
    status_file = find_status_file(run_id)
    
    # 3. 主循环
    while not exit_requested:
        status = read_status(status_file)
        eta = calculate_eta(status.step, status.total_steps, ...)
        
        # 4. 根据模式渲染输出
        if config.json_output:
            print(render_json(status, eta))
        elif is_tty:
            print(render_tty(status, eta, elapsed))
        else:
            print(render_line(status, eta, fields=config.fields, tail=config.tail))
        
        # 5. 检查是否完成
        if not check_training_active(status):
            break
        
        time.sleep(config.interval)
4.2 优雅退出
class GracefulExit:
    def __init__(self):
        self.exit_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, signum, frame):
        print("\n[Watch] Stopping observer (training continues)...")
        self.exit_requested = True
4.3 ETA 计算
def calculate_eta(current_step, total_steps, start_time, current_time):
    if current_step <= 0 or total_steps <= 0:
        return None
    
    elapsed = current_time - start_time
    rate = current_step / elapsed  # steps per second
    
    if rate <= 0:
        return None
    
    remaining = total_steps - current_step
    return int(remaining / rate)

5. 文件变更
5.1 修改的文件
文件	变更类型	说明
areno/cli/main.py	修改	注册 watch 和 runs 命令
修改内容：
_COMMANDS = {
    # ... 现有命令 ...
    "runs": ("areno.cli.watch", "runs_command", "List all training runs."),
    "watch": ("areno.cli.watch", "watch_command", "Watch training progress in terminal."),
}
5.2 新增的文件
文件	说明
areno/cli/watch.py	核心实现（~730 行）
tests/test_watch_cpu.py	CPU 测试（33 个测试用例）

6. 测试覆盖
6.1 单元测试（33 个）
测试类别	测试数量	覆盖内容
find_latest_run_id	3	空目录、单运行、多运行排序
find_status_file	2	找不到、找到
read_status	4	有效、缺失字段、无效 JSON、不存在
is_process_running	2	当前进程、不存在进程
check_training_active	2	运行中、已完成
calculate_eta	5	基础、零步、完成、无效参数
format_eta	5	None、零、秒、分钟、小时
render_line	2	完整数据、最小数据
render_json	1	JSON 格式
render_tty	1	TTY 渲染
GracefulExit	2	初始化、触发
WatchConfig	2	默认值、自定义值
其他	2	DEFAULT_INTERVAL、目录常量
6.2 测试结果
============================== 33 passed in 0.32s ==============================

7. 验收标准
验收项	状态
TTY 刷新模式	✅
非 TTY 行输出	✅
JSON Lines 模式	✅
计算 ETA	✅
优雅退出（Ctrl+C）	✅
状态文件轮询	✅
颜色高亮	✅
areno runs 命令	✅
tail 功能	✅
fields 过滤	✅
向后兼容	✅
CPU 测试覆盖	✅ 33 个测试全部通过

8. 使用示例
8.1 基本使用
# 查看有哪些运行
$ areno runs
Run ID                        Status      Step
------------------------------------------------------
run_20240115_143022           running     150
run_20240114_090000           completed   1000

# 监控最近一次运行
$ areno watch --latest
Watching run: run_20240115_143022
Status file: /home/user/.areno/runs/run_20240115_143022/dashboard_state.12345.json
Press Ctrl+C to stop watching (training will continue)...

╔═══════════════════════════════════════╗
║  AReno Watch - Run status             ║
╠═══════════════════════════════════════╣
║  Step: 150/1000  ████████░░░░░░  15.0% ║
║  Loss: 0.2345    Reward: 0.8923        ║
║  Throughput: 1200 tok/s                ║
║  Stage: train    ETA: 12m 34s          ║
╚═══════════════════════════════════════╝
8.2 进阶使用
# JSON Lines 模式（适合日志收集）
$ areno watch --latest --json >> training.log

# 只显示最近 10 行
$ watch --latest --tail 10

# 只显示特定字段
$ watch --latest --fields step,loss,reward

文档版本：v1.0
生成日期：2026-07-28
维护人：AReno Team