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
7. Run a bounded agentic rollout, inspect one full transcript, then one real training step.

Do not fabricate missing tool calls or alter raw model text to make a trajectory appear valid. Surface parser/capability failures.
