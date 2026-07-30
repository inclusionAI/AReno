# Logic-Circuit Diagnosis Agentic RL Demo

This example trains an agent to locate one stuck-at fault in a small Boolean
circuit. It is self-contained, deterministic, CPU-testable, and uses AReno's
existing `AgentTrajectory`, dataset-loader, and scalar-reward contracts.

## Task contract

Each record contains a topologically ordered AND/OR/NOT circuit and exactly one
hidden `stuck_at_0` or `stuck_at_1` fault on a non-input gate. The model sees
the gate types and connectivity, but never the reference outputs, faulty gate,
fault type, or a caller-provided prompt.

The agent has at most 10 turns and may call one tool per turn:

- `probe(inputs, wire_id)` sets the complete Boolean input vector and returns
  the selected faulty-circuit wire value.
- `submit(gate_id)` submits one non-input gate and ends the episode only after
  the argument passes validation.

Malformed JSON, input-width mismatches, invalid IDs, unknown tools, and
duplicate probes return the same stable envelope:

```json
{"ok": false, "error": {"code": "invalid_input_width", "message": "..."}}
```

Successful results use `{"ok": true, ...}`. An invalid action consumes a turn
but never crashes the rollout or leaks the correct answer. If a model emits
multiple tool calls in one response, only the first is executed. The raw calls
remain in the trajectory and every extra call receives a not-executed result.

## Dataset guarantees

`dataset_generator.py` provides these guarantees before training starts:

- identical arguments produce byte-equivalent records;
- the requested number of records is produced exactly, with duplicate
  circuit/fault tuples retried rather than silently reducing the dataset;
- every injected fault changes a circuit output for at least one input vector;
- every generated non-input gate is on a path to the output, so examples do
  not contain dead fault candidates;
- binary gates use distinct incoming wires and generation weights are
  AND 40%, OR 40%, NOT 20%;
- the default task is intentionally small: 3 inputs and 6 total gates.

`dataset_loader.py` validates required fields, contiguous gate IDs, arity,
backward-only edges, fault type/range, and output observability. Missing, empty,
or malformed data fails with a path and line number. The prompt is regenerated
from validated topology, so a JSONL `prompt` field cannot leak hidden labels.

## Quick start

```bash
python examples/agentic/circuit/dataset_generator.py \
    --output /tmp/circuits.jsonl \
    --count 256 \
    --seed 2026

areno train \
    --ckpt Qwen/Qwen3.5-0.8B \
    --dataset-path /tmp/circuits.jsonl \
    --dataset-loader-fn examples/agentic/circuit/dataset_loader.py \
    --reward-fn-path examples/agentic/circuit/reward.py \
    --agent-fn examples/agentic/circuit/run_agent.py \
    --algo gspo \
    --world-size 2 \
    --tp-size 2 \
    --n-samples 4 \
    --activation-checkpointing \
    --attn-backend native
```

The reward remains a scalar because AReno calls
`float(reward_fn(record))`: `1.0` for the correct gate and `0.0` otherwise.
AReno records this as `rollout/accuracy`. For evaluation batches,
`reward.summarize_diagnoses(records)` returns `diagnosis_accuracy`,
`average_probes`, and `submission_rate` without changing the trainer contract.

## Verification baseline

`brute_force_baseline()` enumerates every output-observable
`(gate, stuck-at value)` hypothesis, chooses the probe that best splits the
remaining hypotheses, observes that wire, and repeats until one gate remains.
It does not read `faulty_gate_id` when choosing a result and returns `-1` if the
probe budget cannot identify a unique gate. Dataset tests verify the baseline
across many independent seeds.

This exhaustive verification is practical for the default small circuits; its
cost grows exponentially with the number of inputs. Larger task distributions
should therefore define an explicit verification budget instead of silently
skipping ambiguity or observability checks.

## Files

- `circuit.py`: validated DAG model, simulation, observable fault injection,
  standalone diagnosis session, scoring, and hypothesis-elimination baseline.
- `dataset_generator.py`: exact-count deterministic JSONL generation.
- `dataset_loader.py`: strict preflight validation and prompt regeneration.
- `run_agent.py`: bounded concurrent multi-turn rollout using `probe` and
  `submit`.
- `reward.py`: scalar verifier plus offline evaluation metrics.
- `tests/test_circuit_diagnosis_cpu.py`: focused CPU and mocked-agent tests.

Run the focused suite with:

```bash
python -m pytest -q tests/test_circuit_diagnosis_cpu.py
```

## Scope

The example adds no service, database, sandbox, trainer fork, new public CLI
option, or mandatory dependency. Existing AReno behavior is unchanged unless
the user explicitly selects these example loader, reward, and agent files.
