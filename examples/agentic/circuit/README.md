# Logic-Circuit Diagnosis Agentic RL Demo

A demo where an agent diagnoses a faulty logic circuit by probing wire outputs
across multiple turns and submitting the identified faulty gate.

## How it works

1. A small AND/OR/NOT circuit is generated with one injected stuck-at fault
   (a non-INPUT gate forced to 0 or 1).
2. The agent sees the circuit structure (gate types and connectivity) but
   not which gate is faulty.
3. The agent interacts with the circuit through a **multi-turn** conversation
   (up to 10 turns):
   - **probe**: Set input values and inspect a wire's output from the faulty
     circuit. The result is returned as a tool message so the model can
     reason about expected vs observed values.
   - **submit**: Submit the guessed faulty gate ID. This ends the conversation.
4. The reward is 1.0 for a correct diagnosis, 0.0 otherwise.
5. Additional metrics (probes_used, submitted) are returned alongside the
   reward for observability.

## Files

- `circuit.py` — Circuit generation, fault injection, simulation, diagnosis
  session (with action limits for standalone use), scoring, brute-force baseline.
- `reward.py` — Reward function extracting the last `submit` tool call across
  all turns. Returns reward, probes_used, and submitted as a dict.
- `dataset_generator.py` — Generate JSONL circuit records (seeded, deterministic).
- `dataset_loader.py` — Load JSONL records for training.
- `run_agent.py` — Multi-turn agent function: loops up to 10 turns, executes
  `probe` on the faulty circuit, returns results as tool messages, ends on
  `submit`.

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

The reward function returns a dict with the following fields, which AReno
logs as training metrics:

- **reward**: 1.0 for correct diagnosis, 0.0 otherwise (the main training signal).
- **probes_used**: Number of probe calls made before submit (lower is more
  efficient; visible in training metrics as `probes_used`).
- **submitted**: Whether the agent made a submit call at all (0.0 or 1.0;
  useful for diagnosing cases where the model never submits).

The `circuit.py` module also includes a `brute_force_baseline()` function
that systematically probes all gates — use it as a reference baseline
outside of training to compare against the learned policy.

## Multi-turn flow

```
Turn 1: Model receives circuit description → calls probe(inputs, wire_id)
        → Server executes probe on faulty circuit → returns wire value
Turn 2: Model sees probe result → calls probe again with different inputs
        → Server executes → returns wire value
...
Turn N: Model calls submit(gate_id) → conversation ends
        → reward_fn checks if gate_id matches faulty_gate_id
```

If the model does not call any tool in a turn, a nudge message is appended
asking it to use probe or submit. If the model never calls submit within
10 turns, the reward is 0.0.

## Limitations

- Maximum 10 turns per rollout. The model may not always reach a submit
  call within this limit, resulting in 0.0 reward.
- The `DiagnosisSession` class in `circuit.py` provides standalone action
  limits (max_probes, max_submissions) for programmatic use, but the
  training loop uses MAX_TURNS (10) as the turn limit instead.
- Circuits are small (default 3 inputs, 6 gates) for fast iteration.
- The brute-force baseline is a standalone function, not integrated into
  the training loop.
