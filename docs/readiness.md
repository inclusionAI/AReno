# Serve Readiness States

AReno provides detailed serve-readiness state tracking to help you monitor and debug the serve startup sequence.

## Overview

When starting an AReno serve instance, the system progresses through several stages before it's ready to accept requests:

1. **model_loading** - Model weights are being loaded
2. **worker_ready** - Worker processes are initialized
3. **router_ready** - Request router is ready
4. **minimal_probe** - Basic health check passed
5. **ready** - Fully ready to accept requests

If any stage fails, the system enters the **failed** state with detailed error information.

## Enabling Readiness Tracking

### CLI Options

Use the `--enable-readiness` flag to enable detailed readiness tracking:

```bash
areno serve \
  --model-path Qwen/Qwen3-0.6B \
  --enable-readiness \
  --readiness-timeout 30 \
  --output text
```

Available options:

- `--enable-readiness` - Enable readiness state tracking (default: disabled)
- `--readiness-timeout` - Timeout per stage in seconds (default: 30, range: 1-3600)
- `--output` - Output format: `text` or `json` (default: text)

### Output Formats

**Text format** (default):
```
[AReno] Validating inputs... OK
[AReno] model_loading... (1/5)
[AReno] model_loading completed (1023ms) (1/5)
[AReno] worker_ready (2/5)
[AReno] router_ready (3/5)
[AReno] minimal_probe passed (4/5)
[AReno] ready - server listening (5/5)
```

**JSON format** (with `--output json`):
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

## HTTP Endpoints

When readiness tracking is enabled, additional endpoints are available:

### GET /health

Basic liveness probe. Returns `200 OK` when server is running:

```json
{"status": "ok"}
```

### GET /ready

Readiness probe. Returns current readiness status:

```json
{"status": "ready", "stage": "ready"}
```

Returns `503 Service Unavailable` if not ready:

```json
{"status": "not_ready", "stage": "model_loading"}
```

### GET /readiness/status

Full readiness status including all stages:

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

Prometheus-format metrics:

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

## Error Handling

If a stage fails, the system enters the `failed` state with detailed error information:

```
[AReno] Validating inputs... OK
[AReno] model_loading... (1/5)
[AReno] failed:model_loading - CUDA OOM: tried to allocate 12GB on device with 10GB free
```

JSON format:
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

Common failure scenarios:

- **Timeout** - Stage took longer than `--readiness-timeout`
- **CUDA OOM** - GPU out of memory during model loading
- **Worker initialization failed** - Worker process crashed or failed to start
- **Router initialization failed** - Router setup failed

## Input Validation

Invalid inputs are caught before expensive initialization:

```bash
# Invalid timeout value
areno serve --enable-readiness --readiness-timeout -1
# [AReno] Validation error: --readiness-timeout must be at least 1. Got: -1

# Non-numeric timeout
areno serve --enable-readiness --readiness-timeout "not_a_number"
# [AReno] Validation error: --readiness-timeout must be a positive integer (invalid format). Got: 'not_a_number'
```

## Metrics Isolation

Probe requests to `/health`, `/ready`, `/readiness/status`, and `/readiness/metrics` are **not** counted in business request metrics. This ensures that health checks don't skew your request statistics.

## Backward Compatibility

Readiness tracking is **disabled by default**. When not enabled:

- No additional logging output
- No new HTTP endpoints (except `/health` which returns basic status)
- No metrics changes
- Existing behavior is unchanged

## Configuration File Example

Create a YAML configuration file for reusable settings:

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

Use with:

```bash
areno serve --config fixtures/readiness_minimal.yaml --enable-readiness
```

## Troubleshooting

### Server stuck in model_loading

Check:
1. Model path is correct and accessible
2. Sufficient GPU memory available
3. Network connectivity if downloading from Hugging Face

### Worker initialization fails

Check:
1. `--world-size` and `--tp-size` are compatible
2. Sufficient system memory
3. No conflicting processes on the same GPU

### Timeout errors

Increase timeout for slow systems:
```bash
areno serve --enable-readiness --readiness-timeout 120
```

### Debug with full status

Query the full status endpoint:
```bash
curl http://localhost:8000/readiness/status | jq
```

Or check metrics:
```bash
curl http://localhost:8000/readiness/metrics
```

## API Reference

### ReadinessStateMachine

The core state machine class (for programmatic use):

```python
from areno.engine.readiness import ReadinessStateMachine, ReadinessState

sm = ReadinessStateMachine(
    enabled=True,
    timeout_per_stage_seconds=30.0,
    on_state_change=lambda old, new, duration: print(f"{old} -> {new}: {duration}ms"),
)

# Progress through stages
sm.mark_stage_complete(ReadinessState.MODEL_LOADING)
sm.mark_stage_complete(ReadinessState.WORKER_READY)

# Check status
status = sm.get_status()
print(status.to_dict())
```

### Validation

```python
from areno.engine.readiness_validation import validate_readiness_options, ValidationError

try:
    config = validate_readiness_options(
        enabled=True,
        timeout=30,
    )
    print(config)  # {'enabled': True, 'timeout_per_stage_seconds': 30, 'probe_interval_seconds': 2}
except ValidationError as e:
    print(e.to_dict())
```

### Metrics Collection

```python
from areno.engine.readiness import ReadinessStateMachine
from areno.engine.readiness_metrics import ReadinessMetricsCollector

sm = ReadinessStateMachine(enabled=True)
collector = ReadinessMetricsCollector(sm)

# Record probe request
collector.record_probe_request()

# Get Prometheus-format metrics
metrics = collector.get_metrics()
print(metrics)
```
