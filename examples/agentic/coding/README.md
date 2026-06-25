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
- `run_command`: bounded shell/test command with timeout and output limits; destructive `rm` commands are blocked
- `submit`: final status and summary

During training, paths are resolved inside the temporary workspace created from
each task record. In the interactive CLI, paths are resolved inside the
directory passed with `--repo`. Commands run inside the workspace with short
timeouts and truncated outputs; destructive `rm` commands are blocked.

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
