# 服务就绪状态

AReno 提供了详细的服务就绪状态追踪功能，帮助你监控和调试服务启动过程。

## 概述

当启动 AReno 服务时，系统会按顺序经过多个阶段，然后才能接受请求：

1. **model_loading** — 正在加载模型权重
2. **worker_ready** — 工作进程已初始化
3. **router_ready** — 请求路由器已就绪
4. **minimal_probe** — 基础健康检查通过
5. **ready** — 完全就绪，可以接受请求

如果任何阶段失败，系统会进入 **failed** 状态，并附带详细的错误信息。

## 启用就绪状态追踪

### CLI 选项

使用 `--enable-readiness` 参数启用详细的就绪状态追踪：

```bash
areno serve \
  --model-path Qwen/Qwen3-0.6B \
  --enable-readiness \
  --readiness-timeout 30 \
  --output text
```

可用选项：

- `--enable-readiness` — 启用就绪状态追踪（默认：关闭）
- `--readiness-timeout` — 每个阶段的超时时间（秒，默认：30，范围：1-3600）
- `--output` — 输出格式：`text` 或 `json`（默认：text）

### 输出格式

**文本格式**（默认）：
```
[AReno] Validating inputs... OK
[AReno] model_loading... (1/5)
[AReno] model_loading completed (1023ms) (1/5)
[AReno] worker_ready (2/5)
[AReno] router_ready (3/5)
[AReno] minimal_probe passed (4/5)
[AReno] ready - server listening (5/5)
```

**JSON 格式**（使用 `--output json`）：
```json
{
  "status": "ready",
  "stages": {
    "model_loading": {"state": "completed", "duration_ms": 1023},
    "worker_ready": {"state": "completed", "duration_ms": 45},
    "router_ready": {"state": "completed", "duration_ms": 12},
    "minimal_probe": {"state": "completed", "duration_ms": 5}
  },
  "last_completed_stage": "minimal_probe",
  "error": null
}
```

## HTTP 端点

启用就绪状态追踪后，提供以下额外端点：

### GET /health

基础存活探针。服务器运行时返回 `200 OK`：

```json
{"status": "ok"}
```

### GET /ready

就绪探针。返回当前的就绪状态：

```json
{"status": "ready", "stage": "ready"}
```

如果未就绪，返回 `503 Service Unavailable`：

```json
{"status": "not_ready", "stage": "model_loading"}
```

### GET /readiness/status

完整的就绪状态，包含所有阶段详情：

```json
{
  "status": "not_ready",
  "current_stage": "model_loading",
  "stages": {
    "model_loading": {"state": "in_progress", "duration_ms": 5234},
    "worker_ready": {"state": "pending"},
    "router_ready": {"state": "pending"},
    "minimal_probe": {"state": "pending"}
  },
  "last_completed_stage": null,
  "error": null
}
```

### GET /readiness/metrics

Prometheus 格式的指标：

```
# HELP areno_serve_readiness_state Current serve readiness state
# TYPE areno_serve_readiness_state gauge
areno_serve_readiness_state 4
# HELP areno_serve_readiness_stage_duration_ms Duration of each readiness stage in milliseconds
# TYPE areno_serve_readiness_stage_duration_ms gauge
areno_serve_readiness_stage_duration_ms{stage="model_loading"} 3421
areno_serve_readiness_stage_duration_ms{stage="worker_ready"} 512
# HELP areno_serve_probe_requests_total Total number of probe requests
# TYPE areno_serve_probe_requests_total counter
areno_serve_probe_requests_total 42
# HELP areno_serve_uptime_seconds Uptime since server start
# TYPE areno_serve_uptime_seconds gauge
areno_serve_uptime_seconds 12.345
```

## 错误处理

如果某个阶段失败，系统会进入 `failed` 状态并附带详细错误信息：

```
[AReno] Validating inputs... OK
[AReno] model_loading... (1/5)
[AReno] failed:model_loading - CUDA OOM: tried to allocate 12GB on device with 10GB free
```

JSON 格式：
```json
{
  "status": "failed",
  "current_stage": "failed",
  "stages": {
    "model_loading": {"state": "failed", "duration_ms": 5234, "error": "CUDA OOM: tried to allocate 12GB on device with 10GB free"},
    "worker_ready": {"state": "pending"},
    "router_ready": {"state": "pending"},
    "minimal_probe": {"state": "pending"}
  },
  "last_completed_stage": null,
  "error": "CUDA OOM: tried to allocate 12GB on device with 10GB free"
}
```

常见的失败场景：

- **超时** — 阶段执行时间超过了 `--readiness-timeout` 设置的值
- **CUDA 内存不足** — 模型加载时 GPU 内存不足
- **工作进程初始化失败** — 工作进程崩溃或启动失败
- **路由器初始化失败** — 路由器设置失败

## 输入验证

在昂贵的初始化操作之前会进行输入验证：

```bash
# 无效的超时值
areno serve --enable-readiness --readiness-timeout -1
# [AReno] Validation error: --readiness-timeout must be at least 1. Got: -1

# 非数字的超时值
areno serve --enable-readiness --readiness-timeout "not_a_number"
# [AReno] Validation error: --readiness-timeout must be a positive integer (invalid format). Got: 'not_a_number'
```

## 指标隔离

对 `/health`、`/ready`、`/readiness/status` 和 `/readiness/metrics` 的探针请求**不计入**业务请求指标。这确保了健康检查不会影响你的请求统计数据。

## 向后兼容

就绪状态追踪**默认关闭**。未启用时：

- 不会产生额外的日志输出
- 不会新增 HTTP 端点（`/health` 仍可用，仅返回基本状态）
- 指标不受影响
- 现有行为保持不变

## 配置文件示例

创建 YAML 配置文件以复用设置：

```yaml
# fixtures/readiness_minimal.yaml
serve:
  host: "127.0.0.1"
  port: 18080
  readiness:
    enabled: true
    timeout_per_stage_seconds: 30
    probe_interval_seconds: 2

model:
  backend: "mock"
  mock_config:
    load_delay_seconds: 1
    vocab_size: 1000
    hidden_size: 64

worker:
  count: 1
  mock: true

router:
  mock: true
```

使用方法：

```bash
areno serve --config fixtures/readiness_minimal.yaml --enable-readiness
```

## 故障排查

### 服务卡在 model_loading 阶段

检查：

1. 模型路径是否正确且可访问
2. GPU 内存是否充足
3. 如果从 Hugging Face 下载，网络连接是否正常

### 工作进程初始化失败

检查：

1. `--world-size` 和 `--tp-size` 是否兼容
2. 系统内存是否充足
3. 同一 GPU 上是否有冲突的进程

### 超时错误

为慢速系统增加超时时间：

```bash
areno serve --enable-readiness --readiness-timeout 120
```

### 使用完整状态进行调试

查询完整状态端点：

```bash
curl http://localhost:8000/readiness/status | jq
```

或查看指标：

```bash
curl http://localhost:8000/readiness/metrics
```

## API 参考

### ReadinessStateMachine

核心状态机类（供程序化使用）：

```python
from areno.engine.runtime.readiness import ReadinessStateMachine, ReadinessState

sm = ReadinessStateMachine(
    enabled=True,
    timeout_per_stage_seconds=30.0,
    on_state_change=lambda old, new, duration: print(f"{old} -> {new}: {duration}ms"),
)

# 按顺序经过各个阶段
sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
sm.mark_stage_complete(ReadinessState.WORKER_READY)

# 查看状态
status = sm.get_status()
print(status.to_dict())
```

### 输入验证

```python
from areno.engine.runtime.readiness_validation import validate_readiness_options, ValidationError

try:
    config = validate_readiness_options(
        enabled=True,
        timeout=30,
    )
    print(config)  # {'enabled': True, 'timeout_per_stage_seconds': 30, 'probe_interval_seconds': 2}
except ValidationError as e:
    print(e.to_dict())
```

### 指标收集

```python
from areno.engine.runtime.readiness import ReadinessStateMachine
from areno.engine.runtime.readiness_metrics import ReadinessMetricsCollector

sm = ReadinessStateMachine(enabled=True)
collector = ReadinessMetricsCollector(sm)

# 记录探针请求
collector.record_probe_request()

# 获取 Prometheus 格式的指标
metrics = collector.get_metrics()
print(metrics)
```