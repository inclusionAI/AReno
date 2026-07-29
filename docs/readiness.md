# 服务就绪状态追踪

## 概述

AReno 的服务就绪（readiness）状态追踪功能，让你可以精确监控服务启动的每个阶段，知道当前卡在哪里、每个阶段花了多长时间、以及出错时的具体原因。

## 使用方法

### 基本用法

在 `areno serve` 命令中加上 `--enable-readiness` 参数即可启用：

```bash
areno serve \
  --model-path Qwen/Qwen3-0.6B \
  --enable-readiness
```

### 完整参数

```bash
areno serve \
  --model-path Qwen/Qwen3-0.6B \
  --enable-readiness \
  --readiness-timeout 60 \
  --output json
```

| 参数 | 说明 | 默认值 | 限制 |
|------|------|--------|------|
| `--enable-readiness` | 启用就绪状态追踪 | 关闭 | 不带参数，加在命令中即可 |
| `--readiness-timeout` | 每个阶段的超时时间（秒） | `30` | 范围 1-3600，必须是正整数 |
| `--output` | 输出格式 | `text` | 可选值：`text`、`json` |

### 关闭功能

不加 `--enable-readiness` 就可以了，不产生任何额外输出。

## 输出内容

### 1. 日志 / CLI 终端输出

服务启动时，会实时打印每个阶段的状态：

```
[AReno] Validating inputs... OK
[AReno] model_loading... (1/5)
[AReno] model_loading completed (25969ms) (1/5)
[AReno] worker_ready (2/5)
[AReno] worker_ready completed (5630ms) (2/5)
[AReno] router_ready (3/5)
[AReno] router_ready completed (12378ms) (3/5)
[AReno] minimal_probe passed (4/5)
[AReno] ready - server listening (5/5)
```

每条日志的格式：`[阶段名称] [状态] ([第N阶段/共5阶段])`

出错时：

```
[AReno] failed:model_loading - CUDA OOM: tried to allocate 12GB on device with 10GB free
```

如果选择 JSON 格式（`--output json`），日志会输出 JSON 行。

### 2. HTTP 端点

启用就绪追踪后，以下端点可用：

| 端点 | 用途 | 未启用时的行为 |
|------|------|---------------|
| `GET /health` | 基础健康检查，服务在运行就返回 ok | 始终可用 |
| `GET /ready` | 就绪探针，服务完全就绪才返回 ready | 返回 404 |
| `GET /readiness/status` | 完整的阶段状态详情 | 返回 404 |
| `GET /readiness/metrics` | Prometheus 格式指标 | 返回 404 |

#### GET /ready

```bash
curl http://localhost:8000/ready
```

**就绪时：**
```json
{"status": "ready", "stage": "ready"}
```
HTTP 状态码：200

**未就绪时：**
```json
{"status": "not_ready", "stage": "model_loading"}
```
HTTP 状态码：503

#### GET /readiness/status

```bash
curl http://localhost:8000/readiness/status | python -m json.tool
```

```json
{
    "status": "ready",
    "current_stage": "ready",
    "stages": {
        "model_loading": {
            "state": "completed",
            "duration_ms": 5503,
            "error": null
        },
        "worker_ready": {
            "state": "completed",
            "duration_ms": 2042,
            "error": null
        },
        "router_ready": {
            "state": "completed",
            "duration_ms": 11868,
            "error": null
        },
        "minimal_probe": {
            "state": "completed",
            "duration_ms": 293,
            "error": null
        },
        "ready": {
            "state": "in_progress",
            "duration_ms": null,
            "error": null
        }
    },
    "last_completed_stage": "minimal_probe",
    "error": null
}
```

各字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"not_ready"`、`"ready"` 或 `"failed"` |
| `current_stage` | string | 当前正在执行的阶段名称 |
| `stages` | object | 包含 5 个阶段的详细状态 |
| `stages.*.state` | string | `"pending"`、`"in_progress"`、`"completed"`、`"failed"` |
| `stages.*.duration_ms` | int/null | 该阶段耗时（毫秒），未完成时为 null |
| `stages.*.error` | string/null | 错误信息，正常时为 null |
| `last_completed_stage` | string/null | 最后一个完成的阶段名，都未完成时为 null |
| `error` | string/null | 全局错误信息 |

#### GET /readiness/metrics

```bash
curl http://localhost:8000/readiness/metrics
```

输出 Prometheus 格式的指标，可用于被 Prometheus 服务抓取：

```
# HELP areno_serve_readiness_state Current serve readiness state
# TYPE areno_serve_readiness_state gauge
areno_serve_readiness_state 4
# HELP areno_serve_readiness_stage_duration_ms Duration of each readiness stage in milliseconds
# TYPE areno_serve_readiness_stage_duration_ms gauge
areno_serve_readiness_stage_duration_ms{stage="model_loading"} 5503
areno_serve_readiness_stage_duration_ms{stage="worker_ready"} 2042
areno_serve_readiness_stage_duration_ms{stage="router_ready"} 11868
areno_serve_readiness_stage_duration_ms{stage="minimal_probe"} 293
# HELP areno_serve_probe_requests_total Total number of probe requests
# TYPE areno_serve_probe_requests_total counter
areno_serve_probe_requests_total 3
# HELP areno_serve_uptime_seconds Uptime since server start
# TYPE areno_serve_uptime_seconds gauge
areno_serve_uptime_seconds 16.335
```

4 种指标：

| 指标名 | 类型 | label | 说明 |
|--------|------|-------|------|
| `areno_serve_readiness_state` | gauge | 无 | 当前状态数值（0=空, 1=pending, 2=model_loading, 3=worker_ready, 4=router_ready, 5=minimal_probe, 6=ready, -1=failed） |
| `areno_serve_readiness_stage_duration_ms` | gauge | `stage` | 每个阶段的耗时（毫秒） |
| `areno_serve_probe_requests_total` | counter | 无 | 探针请求总数（不含业务请求） |
| `areno_serve_uptime_seconds` | gauge | 无 | 服务启动后的运行时长（秒） |

## 实现原理

### 6 个就绪状态

```
model_loading → worker_ready → router_ready → minimal_probe → ready
                                                                    ↓
                                                                  failed（任何阶段出错）
```

| 状态 | 触发时机 | 说明 |
|------|---------|------|
| `model_loading` | 加载 tokenizer 和模型权重时 | 耗时取决于模型大小和硬件 |
| `worker_ready` | 工作进程初始化完成后 | 包括设备分配 |
| `router_ready` | 引擎（ArenoEngine）创建完成后 | 模型加载到 GPU 完成 |
| `minimal_probe` | 基础推理验证通过后 | 发送一个最小请求验证推理通路 |
| `ready` | 服务开始监听端口时 | 最终状态 |
| `failed` | 任何阶段出错时 | 附带错误信息，服务无法启动 |

### 状态流转规则

- 只能按顺序前进，不能跳阶段或回退
- 每个阶段有独立的超时计时
- 超时后自动进入 `failed` 状态
- 出错后停留在 `failed` 状态，不再变化

### 探针隔离

对 `/health`、`/ready`、`/readiness/status`、`/readiness/metrics` 的请求会被识别为探针请求，**不计入业务请求指标**，避免健康检查污染业务统计数据。

## 使用场景

### 场景 1：调试服务启动慢

查看各阶段耗时，定位瓶颈：

```
model_loading completed (25969ms)  ← 模型加载耗时过长
worker_ready completed (5630ms)
router_ready completed (12378ms)
```

### 场景 2：CI/CD 就绪判断

Kubernetes 等容器编排平台配置就绪探针：

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

### 场景 3：监控告警

Prometheus 配合 Grafana 监控：

```promql
# 服务启动失败告警（5分钟内状态为 failed）
areno_serve_readiness_state == -1
```

## 限制与注意事项

### 已知限制

| 限制 | 说明 |
|------|------|
| 单次启动 | readiness 状态机是服务级别的，一个进程只有一个状态机实例 |
| 不可逆 | 状态只能前进，不能回退。如果服务需要重启，必须重新启动进程 |
| 不适用于负载均衡 | readiness 只反映本进程的状态，不反映上游依赖的健康状况 |
| 超时粒度为阶段级别 | 每个阶段独立的超时计时，不能为某个阶段单独设置不同的超时值 |

### 注意事项

1. **必须显式启用**：`--enable-readiness` 默认不开启，不加这个参数不会有任何就绪追踪功能
2. **空响应**：在 `model_loading` 阶段，前几个 HTTP 端点可能还不可用
3. **超时值选择**：大模型（如 70B+）加载可能超过 30 秒，建议设大超时值或等模型完全加载完再启用
4. **端口冲突**：确保 `--port` 指定的端口未被占用，否则服务会启动失败
5. **无持久化**：状态信息只在进程内存中，进程重启后丢失

### 与其他系统的关系

| 系统 | 关系 |
|------|------|
| Prometheus | 可以通过 `/readiness/metrics` 端点抓取指标 |
| Kubernetes | `/ready` 端点可直接作为 `readinessProbe` 使用 |
| Grafana | 可以导入 Prometheus 指标做可视化 |
| AReno Dashboard | 当前版本暂未集成，可通过 `/readiness/status` API 自行集成 |

## 完整示例

### 启动并验证

```bash
# 终端 1：启动服务
areno serve --model-path Qwen/Qwen3-0.6B --enable-readiness --port 8000

# 终端 2：轮询等待就绪
while true; do
  RESP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ready)
  echo "Ready probe: $RESP"
  if [ "$RESP" = "200" ]; then
    echo "Server is ready!"
    break
  fi
  sleep 2
done

# 查看完整状态
curl -s http://localhost:8000/readiness/status | python -m json.tool
```

### 程序化使用（Python）

```python
import urllib.request, json, time

# 等待就绪
for i in range(30):
    try:
        resp = urllib.request.urlopen("http://localhost:8000/ready")
        print(f"Ready after {i*2}s")
        break
    except urllib.error.HTTPError as e:
        status = json.loads(e.read())
        print(f"Waiting... stage: {status['stage']}")
        time.sleep(2)
    except Exception:
        print(f"Not available yet... ({i*2}s)")
        time.sleep(2)
else:
    print("Timeout waiting for server")

# 获取详细状态
resp = urllib.request.urlopen("http://localhost:8000/readiness/status")
data = json.loads(resp.read())
print(json.dumps(data, indent=2, ensure_ascii=False))
```