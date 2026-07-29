# Composable Local Skill-Workflow Executor

A lightweight YAML-based workflow executor for AReno skill scripts. It lets
you chain multiple skill scripts into a reproducible pipeline with typed
inputs, step dependencies, structured output passing, and restart from a
failed step — all using the Python standard library, with no external
database or mandatory sandbox.

## Quick start

```bash
# Run the recipe -> tiny run -> summary example
python3 .agents/scripts/run_workflow.py examples/workflows/recipe_to_summary.yaml

# Run the serve -> probe -> summary example
python3 .agents/scripts/run_workflow.py examples/workflows/serve_to_summary.yaml

# Dry run (resolve and print commands without executing)
python3 .agents/scripts/run_workflow.py examples/workflows/recipe_to_summary.yaml --dry-run

# Restart from a specific step (skip all steps up to and including the given id)
python3 .agents/scripts/run_workflow.py examples/workflows/recipe_to_summary.yaml --start-from run
```

## Workflow YAML format

```yaml
name: my-workflow              # required, string
description: What it does      # optional, string

steps:                          # required, non-empty list
  - id: step1                   #   required, unique string
    script: scripts/foo.py      #   required, path relative to the YAML file
    inputs:                     #   optional, mapping of CLI flag -> value
      --arg1: value1
      --flag: true              #   boolean true -> --flag (store_true)
    depends_on: []              #   optional, list of step ids this step needs

  - id: step2
    script: scripts/bar.py
    inputs:
      --input: ${steps.step1.output_key}  # reference step1's JSON output
    depends_on: [step1]
```

### Input contract

| Type in YAML      | Behaviour                                                |
|-------------------|----------------------------------------------------------|
| String            | Passed as `--flag value`                                 |
| Integer / Float   | Stringified and passed as `--flag value`                 |
| `true`            | Passed as `--flag` only (matches argparse `store_true`)  |
| `false` / `null`  | Flag is omitted entirely                                 |
| `${steps.X.Y}`    | Resolved to the JSON output `Y` from step `X`. If the    |
|                   | placeholder is the entire value, the typed object is     |
|                   | returned; if embedded in a string, it is stringified.   |

### Output fields

The executor prints a single JSON object to stdout:

```json
{
  "ok": true,
  "workflow": "path/to/workflow.yaml",
  "name": "my-workflow",
  "description": "...",
  "steps_total": 2,
  "steps_executed": 2,
  "results": [
    {
      "step_id": "step1",
      "ok": true,
      "output": { ... },        // parsed JSON from the script's stdout
      "error": "",
      "returncode": 0
    }
  ],
  "outputs": {                   // accumulated JSON outputs by step id
    "step1": { ... },
    "step2": { ... }
  }
}
```

On failure, `ok` is `false` and an `error` field is added at the top level.
Subsequent steps after a failure are marked `skipped (prior failure)`.

## Script contract

Each step's `script` must:

1. Be a Python script invoked as `python3 script.py --flags ...`.
2. Print a single JSON object to stdout on success.
3. Exit with a non-zero code on failure.

This matches the convention already used by all scripts under
`.agents/skills/*/scripts/` (e.g. `probe_server.py`, `read_metrics.py`).

## Limitations

- The YAML parser supports a deliberate subset (mappings, sequences,
  scalars, comments). It does not support anchors, aliases, multi-line
  block scalars, or inline mappings `{...}`.
- Step outputs are kept in memory; there is no persistent checkpoint
  file. Restart (`--start-from`) re-executes the named step and all steps
  it transitively depends on, so their outputs are available for
  placeholder resolution. Steps not needed by the resume point are
  skipped. There is no file-based checkpoint — if the process exits you
  must re-run from the beginning or use `--start-from` which re-executes
  dependencies as needed.
- Scripts are executed with a 300-second timeout per step.

## Examples

Two deterministic, self-contained examples ship under `examples/workflows/`:

1. **`recipe_to_summary.yaml`** — builds a tiny training recipe, simulates
   a training run, and summarises the result.
2. **`serve_to_summary.yaml`** — mocks a serving endpoint, probes it, and
   summarises the probe result.

Neither requires a GPU, network, or external database.

## Tests

```bash
pytest tests/test_workflow_executor_cpu.py -v
```

Covers success paths, invalid input, dependency cycles, missing output
keys, restart-from-failed-step, dry-run, and boundary conditions.