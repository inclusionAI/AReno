# Agentic Logic Circuit Diagnosis

Diagnose a hidden stuck-at fault in a combinational logic circuit. A circuit
of AND/OR/NOT gates contains exactly one faulty gate that always outputs
`False` (stuck-at-0) or always outputs `True` (stuck-at-1). The agent sees the
circuit topology but **not** which gate is faulty. It must apply input vectors,
probe internal nodes, and identify the fault using as few probes as possible.

## Overview

- **Circuit**: Directed acyclic graph with 2–6 primary inputs, 3–8 *live*
  internal gates (AND/OR/NOT), and one output node. The training generator
  discards dead gates rather than treating its pre-pruning gate count as task
  difficulty.
- **Fault**: One injected gate is stuck-at-0 or stuck-at-1; generated records
  balance gate type and stuck value.
- **Tools**:
  - `set_input_vector(inputs)` — Free. Apply a boolean vector to the inputs
    and observe the output.
  - `inspect_node(node_id)` — Costs 1 probe. Read the actual value of one
    internal gate node.
  - `submit_diagnosis(node_id, fault_type)` — Submit final answer. Ends the
    episode.
- **Goal**: Identify the faulty gate and fault type with as few probes as
  possible.

## Generate data

```bash
python examples/agentic/logic_diagnosis/dataset_generator.py \
  --output /tmp/logic_diagnosis.jsonl --count 256 --seed 2026
```

Each JSONL record contains the full circuit topology, the injected fault
(ground truth), and a pre-rendered prompt. Every generated record is within
the eight-gate brute-force boundary and is checked for a unique distinguishing
I/O signature. The deterministic generator balances input counts, live gate
counts, and gate-type/stuck-value fault classes; it rejects exact duplicates
and faults that never alter the primary output. `n_gates` always reports the
post-pruning live-gate count.

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --dataset-path /tmp/logic_diagnosis.jsonl \
  --dataset-loader-fn examples/agentic/logic_diagnosis/dataset_loader.py \
  --reward-fn-path examples/agentic/logic_diagnosis/reward.py \
  --agent-fn examples/agentic/logic_diagnosis/run_agent.py \
  --algo gspo --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --max-new-tokens 128
```

## Reward

| Outcome | Reward |
|---|---|
| Correct diagnosis, 0 probes | 1.0 |
| Correct diagnosis, N probes | 0.5 + 0.5 × (1 − N/max) |
| Wrong diagnosis | 0.0 |
| No submission | −1.0 |
