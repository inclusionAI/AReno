---
name: areno-build-agentic-workflow
description: Create or debug an AReno multi-turn agentic dataset, run_agent implementation, tool schemas, tool execution, reward, loss masks, interactive TUI game with OpenAI-compatible LLM inference, or agentic training example. Use for text or multimodal agentic workflows, not ordinary single-turn rollout.
---

# Build an AReno Agentic Workflow

Inspect the nearest example under `examples/agentic/` and public types in `areno/api/agentic.py` before implementation.

Develop on a local branch and pull committed changes into remote validation
hosts. Use ModelScope for remote checkpoint and dataset references.

```bash
python .agents/skills/areno-build-agentic-workflow/scripts/validate_transcript.py transcript.json
python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
  --dataset-path data.jsonl --loader examples/agentic/<name>/dataset_loader.py --algo gspo
```

## Workflow

1. Define one deterministic environment/game domain and valid dataset records.
2. Keep dataset loaders processor/tokenizer independent.
3. Define strict JSON tool schemas and bounded execution. Read [references/message-contract.md](references/message-contract.md).
4. Build trajectories using AReno public agentic types. Preserve assistant tool-call and tool-result ordering.
5. Reward the intended outcome and process separately; test invalid, partial, and optimal paths.
6. Verify concurrency, timeout, cancellation, context truncation, multimodal content, and loss masks.
7. Treat a proxy `finish_reason == "length"` response as terminal for that item: append the turn, stop the item, and do not nudge the model to retry or execute a (half) tool call. Under `--agent-overlength-policy safe-stop` the proxy already drops half-finished tool calls / oversized tool results and emits `termination_reason` (`generation_limit` / `context_limit` / `oversized_tool_result`) plus `rollout/overlength_*` metrics; `run_agent` must honor the length signal or the loop spins to the turn limit. Default `off` preserves prior behavior.
8. Run a bounded agentic rollout, inspect one full transcript, then one real training step.

Do not fabricate missing tool calls or alter raw model text to make a trajectory appear valid. Surface parser/capability failures.
