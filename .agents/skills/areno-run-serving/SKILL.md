---
name: areno-run-serving
description: Start, validate, debug, and stop an AReno OpenAI-compatible serving endpoint. Use for areno serve commands, API probes, streaming, cancellation, cache or CUDA graph issues, and supported image or tool-call requests. Do not use for training jobs.
---

# Run AReno Serving

Read `AGENTS.md`, `CODEMAP.md`, and current `areno serve --help`. Serving uses `--model-path`, not training's `--ckpt`.

For a remote model reference, pass `--model-hub modelscope`. If ModelScope
cannot resolve it, report that failure or request a local model path instead of
silently changing hubs.

## Workflow

1. Record commit and inspect `areno env --json`, `areno check`, GPUs, model config, and adapter registration.
2. Validate TP/world-size constraints from current model code.
3. Start with requested cache/context, attention backend, and CUDA graph policy. Do not disable graphs merely to hide capture failures.
4. Probe model listing and chat:

```bash
python .agents/skills/areno-run-serving/scripts/probe_server.py \
  --base-url http://127.0.0.1:8000 [--model <name>]
```

5. Build image requests with [scripts/build_image_request.py](scripts/build_image_request.py), piping JSON to the HTTP client rather than shell argv.
6. When relevant, validate streaming, cancellation followed by a clean request, image input, and tool calls. Read [references/request-contracts.md](references/request-contracts.md).
7. Classify request, processor, model, cache, scheduler, or transport ownership before editing.

Report command, commit, endpoint, successful probes, and shutdown state. Startup alone is insufficient.
