# Agentic Coding Example

This example trains a policy on small SWE-bench-style coding tasks. Each sample
creates an isolated temporary repository from the task record, then runs a
multi-turn coding-agent loop:

1. inspect the repository tree
2. read files or search with `rg`
3. apply a unified diff
4. run a bounded shell/test command
5. submit `solved` or `blocked`

The dataset uses familiar SWE-bench fields such as `instance_id`, `repo`,
`base_commit`, `problem_statement`, `FAIL_TO_PASS`, and `PASS_TO_PASS`. The
example also includes `files` and `test_commands` so it can run locally without
network access or external repository checkout. `dataset.jsonl` contains 100
self-contained records that range from easy one-file fixes to harder multi-file
tasks, with the majority marked hard. The tasks use only local Python files and
pytest commands; they do not require sandbox services, package installation, or
network access.

## Tools

The shared agent loop exposes constrained Codex-style tools:

- `inspect_tree`: compact directory tree under a relative path
- `list_files`: flat source file listing
- `read_file`: bounded line-range reads with line numbers
- `rg`: regex search over workspace files
- `apply_patch`: unified diff patch application inside the workspace
- `replace_text`: exact text replacement for simple file edits
- `write_file`: create, overwrite, or append one workspace file
- `run_command`: bounded shell/test command with timeout and output limits; pass `background=true` for long-running commands
- `read_background_output`: read a character range from a background command log and check whether it is still running
- `submit`: final status and summary

During training, paths are resolved inside the temporary workspace created from
each task record. In the interactive CLI, paths are resolved inside the
directory passed with `--repo`. Commands run inside the workspace with short
timeouts and truncated outputs; destructive `rm` commands are blocked. For
long-running commands, start `run_command` with `background=true`, use a short
`sleep` command to wait, then poll a range of the log with
`read_background_output`.

## Interactive Coding CLI

Start an AReno-compatible serving endpoint, then run the same loop used by
training against the current repository:

```bash
python examples/agentic/coding/code_cli.py \
  --repo . \
  --base-url http://127.0.0.1:8000/v1 \
  --model policy
```

The CLI prompts for the coding task and operates in the directory passed with
`--repo` (default: `.`), so run it from a disposable checkout or review the
generated patch before keeping the result. Pass `--test-command` when you want
the prompt and reward function to know the expected validation command.
After each model/tool loop, the CLI prompts for another instruction and keeps
the historical conversation in context. Long histories are compacted
automatically; tune this with `--compact-chars` and `--compact-keep-messages`.

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path examples/agentic/coding/dataset.jsonl \
  --dataset-loader-fn examples/agentic/coding/dataset_loader.py \
  --reward-fn-path examples/agentic/coding/reward.py \
  --agent-fn examples/agentic/coding/run_agent.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 2 \
  --max-prompt-tokens 4096 \
  --max-new-tokens 256
```

The reward is `1.0` when the trajectory applies a patch, runs every required
test command successfully, and submits `solved`. It gives partial credit for
passing tests without a final solved submission and negative reward for failed
or unsupported trajectories.

## AReno Repo Targets

`areno_agentic_targets.jsonl` contains four higher-level AReno repository
tasks. Each record asks the coding agent to prepare one agentic example for
training from `/home/admin/Qwen3.5-4B`: DuelGrid, shopping, Tic-Tac-Toe, and
the coding agent itself.

Use `run_areno_agent.py` for these records. The runner copies `/home/admin/AReno` into a
temporary workspace before the model sees the task, then reuses the same coding
tools to inspect, patch, test, and submit. The target tasks ask the agent to
use each example's `dataset_generator.py` to create the training dataset before
documenting the final train command. It runs samples under a process-wide lock
so only one AReno agent task is active at a time. Tool-run subprocesses are
pinned to `CUDA_VISIBLE_DEVICES=4,5,6,7` by the runner, so the model should call
plain commands rather than adding GPU environment prefixes itself. The prompt
also tells the model to run AReno training commands in the background, wait with
`sleep`, and inspect the background log before submitting.

```bash
areno train \
  --ckpt /home/admin/Qwen3.5-4B/ \
  --dataset-path ./examples/agentic/coding/dataset.jsonl \
  --dataset-loader-fn examples/agentic/coding/dataset_loader.py \
  --reward-fn-path examples/agentic/coding/reward.py \
  --algo gspo \
  --tp-size 4 \
  --world-size 4 \
  --mini-bs 1 \
  --agent-fn examples/agentic/coding/run_agent.py \
  --save-path /pcache-mnt/rw/checkpoint/54229/462250/316390214/260624105554/coding \
  --drop-rollout-state \
  --max-running-prompts 4 \
  --batch-size 1 \
  --n-samples 8 \
  --max-new-tokens 2048 \
  --max-context-len 32768
```
