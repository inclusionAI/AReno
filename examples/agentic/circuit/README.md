# Logic-Circuit Diagnosis Agentic RL Demo

A demo where an agent diagnoses a faulty logic circuit by probing wire outputs
and submitting the identified faulty gate.

## How it works

1. A small AND/OR/NOT circuit is generated with one injected stuck-at fault
   (a non-INPUT gate forced to 0 or 1).
2. The agent sees the circuit structure (gate types and connectivity) but
   not which gate is faulty.
3. The agent uses two tools:
   - **probe**: Set input values and inspect a wire's output.
   - **submit**: Submit the guessed faulty gate ID.
4. The reward is 1.0 for a correct diagnosis, 0.0 otherwise.

## Files

- `circuit.py` — Circuit generation, fault injection, simulation, diagnosis session, scoring, brute-force baseline.
- `reward.py` — Reward function extracting the `submit` tool call.
- `dataset_generator.py` — Generate JSONL circuit records.
- `dataset_loader.py` — Load JSONL records for training.
- `run_agent.py` — Agent function with `probe` and `submit` tools.

## Quick start

Generate a dataset:

```bash
python examples/agentic/circuit/dataset_generator.py \
    --output /tmp/circuits.jsonl \
    --count 256 \
    --seed 2026
```

Train with AReno:

```bash
areno train \
    --ckpt Qwen/Qwen3-0.6B \
    --dataset-path /tmp/circuits.jsonl \
    --dataset-loader-fn examples/agentic/circuit/dataset_loader.py \
    --reward-fn-path examples/agentic/circuit/reward.py \
    --agent-fn examples/agentic/circuit/run_agent.py \
    --algo gspo \
    --tp-size 1 \
    --world-size 1
```

## Metrics

- **Diagnosis accuracy**: Fraction of circuits correctly diagnosed.
- **Average probes**: Mean number of probes used per circuit (lower is better).
- **Brute-force baseline**: A reference baseline that systematically probes all gates.

## Limitations

- The demo uses single-turn tool calls (one probe + one submit per rollout).
- Multi-turn diagnosis (iterative probing) can be implemented by extending
  `run_agent.py` to loop on tool call responses.
- Circuits are small (default 3 inputs, 6 gates) for fast iteration.