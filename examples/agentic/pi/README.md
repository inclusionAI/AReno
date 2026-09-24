# Train with the pi coding agent

Run the real [pi coding agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
as AReno's rollout harness. All integration code lives in this example.

```text
pi CLI → example secondary proxy → existing AReno rollout proxy → current policy
                ↓
       exact per-turn trajectories → existing AReno trainer
```

The secondary proxy adapts pi's streaming Chat Completions requests to AReno's
non-streaming endpoint. It saves the upstream `areno` metadata, including exact
input tokens, generated tokens, logprobs and routing replay IDs. It then emits
buffered SSE chunks for pi. This is protocol compatibility, not incremental token
streaming. Pi's own read, bash, edit and write tools run unchanged. The adapter was
smoke-tested with pi 0.83.0 against the existing AReno proxy and a CPU policy
double, including a real pi tool call and verification.

## Software tasks without task-image builds

For a lightweight start, use the [ModelScope BigCodeBench local demo](LOCAL_DEMO.md).
It scans all 1,140 upstream tasks, selecting standard-library tasks or a larger
set with dependencies installed once in the existing environment. It includes
reference-check filtering and selection reports; the original ten tasks remain
available with `--profile smoke`. Pi runs in temporary workspaces without a
Docker daemon or per-task image.

## Repository tasks with Docker-in-Docker

For real software-engineering issues, use the [ModelScope SWE-bench generator and
DinD runner](dind/README.md). It runs pi in per-attempt nested containers and grades
patches in separate clean SWE-bench containers. The two local tasks below remain
small integration fixtures; they are not the recommended training dataset.

## Run the local fixture

Install pi separately; it is not an AReno dependency:

```bash
npm install -g @mariozechner/pi-coding-agent
```

Use a local, tool-capable checkpoint and an existing AReno training environment:

```bash
areno train \
  --ckpt /path/to/tool-capable-checkpoint \
  --dataset-path examples/agentic/pi/dataset.jsonl \
  --dataset-loader-fn examples/agentic/pi/dataset_loader.py \
  --agent-fn examples/agentic/pi/run_agent.py \
  --reward-fn-path examples/agentic/pi/reward.py \
  --algo grpo \
  --world-size 1 --tp-size 1 \
  --batch-size 1 --n-samples 4 \
  --max-running-prompts 4 \
  --max-new-tokens 2048 --max-context-len 32768
```

`--algo gspo` uses the same example. Set `ARENO_PI_EXECUTABLE` to a different pi
executable or wrapper if needed. No separate `areno serve` process is required:
the training session owns the upstream policy and its lifecycle. This example
uses synchronous rollout batches, so all attempts finish before weights change.
Pi's advertised output limit is removed by the secondary proxy; AReno's
`--max-new-tokens` controls generation length.

Each attempt gets a fresh workspace, pi config directory, loopback proxy and
bearer key. Pi runs in print mode without saved sessions, extensions, skills or
prompt templates. Automatic compaction and retries are disabled to keep
auxiliary model calls out of the training trace. The upstream context limit
still applies to every actual model call. Concurrency follows
`--max-running-prompts`.

## Dataset and reward

Each JSONL row contains:

- `prompt` or `problem_statement`: the task given to pi.
- `files`: relative file paths mapped to initial text contents.
- `verify`: trusted Python assertions run on the resulting workspace after pi exits.
- Optional `max_turns` (32), `timeout` (300 seconds), `verify_timeout` (30 seconds).

The loader always normalizes the task to `prompt`. Verification code is kept out
of the agent's initial files and executed by the example, rather than accepting
the agent's claim that its tests passed. Replace it with a task-specific verifier
for larger datasets.

Reward is 1 when pi finishes successfully within budget and verification passes;
otherwise it is 0. Failed or timed-out attempts with usable traces remain
training examples. Transport/metadata failures and attempts with no model calls
are marked invalid and excluded, with a warning. Every attempt's model turns
share its task reward through AReno's existing trajectory grouping. Actual
per-turn contexts are preserved; tool results are prompt tokens, not newly
generated actions. The sample-local `pi_result` also retains process status and
the tail of pi/verifier logs for reward inspection.

Workspaces and configuration are isolated between attempts, but a temporary
directory is **not an OS security sandbox**. Pi executes shell commands and the
verifier executes dataset code. Run this example with trusted tasks in a
disposable container or VM; do not give an untrusted policy access to host secrets.
Timeouts terminate the process group, including tool children. Proxy shutdown
drains outstanding requests before returning trajectories; an in-flight model
request can take up to the upstream HTTP timeout after pi is stopped.

## CPU checks

```bash
python -m pytest -q examples/agentic/pi/test_pi_cpu.py
```

These checks use a fake policy and a fixture process, not a GPU or downloaded
model. They cover the secondary HTTP/SSE path, exact training tokens and masks,
sample isolation, verifier rewards, request limits and process cleanup. A real
pi/model training run remains a separate hardware integration check.
