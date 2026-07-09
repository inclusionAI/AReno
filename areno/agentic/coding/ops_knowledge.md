# AReno Train/Serve Operations Knowledge

You are running inside an existing AReno checkout and should use the local files
and commands available in the current environment. Do not clone another AReno
repo. Your job is to start a train or serve task successfully, inspect failures,
and retry with adjusted parameters when the failure is likely recoverable.

## Basic workflow

1. Inspect the repository and examples before choosing commands:
   - `pwd`
   - `ls`
   - `areno --help`
   - `areno train --help`
   - `areno serve --help`
   - `areno check`
   - `areno env`
2. Inspect GPUs and memory before launching:
   - `nvidia-smi`
   - `nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv`
   - `ps aux | grep -E "areno|python" | grep -v grep`
   - `df -h . /tmp`
3. Smoke checks are available diagnostics, not a mandatory first step. Use them
   when the user asks for validation, when a real run would be expensive, or
   when a failure suggests missing dependencies, unsupported model adapters,
   tensor-parallel divisibility errors, checkpoint shape mismatches, CUDA graph
   capture failures, or basic CUDA startup failures. Useful checks:
   - `areno check`
   - `areno env`
   - `areno train ... --smoke-infer`
   - `areno train ... --smoke-train`
4. For rollout/RL, keep `--n-samples 8` unless the user requests another value.
   Keep rollout demand and concurrency consistent: normally
   `batch_size * n_samples <= max_running_prompts`. If you raise
   `--max-running-prompts` to improve utilization, also raise `--batch-size`
   when the dataset and training memory allow it; otherwise the run may not
   produce enough requests to use the configured concurrency.
5. If a command fails with a recoverable capacity error, adjust the relevant
   dimension first:
   - rollout/KV OOM: reduce `--max-running-prompts`.
   - train/backward OOM: reduce `--mini-bs`.
   - full step OOM: reduce the dimension named by the failing phase first, then
     reduce `--batch-size` if needed.
   - do not tune `--max-new-tokens` to make smoke or train fit. Treat it as part
     of the task quality target unless the user explicitly changes it.
   - for agentic train or serve tasks, if the user did not provide
     `--max-new-tokens` or `--max-context-len`, ask for those values before
     running commands. Do not silently assume defaults for these two agentic
     limits.
   - divisibility or unsupported-model errors are not capacity search problems;
     fix the invalid setting or report the blocker.
6. Read the error message, adjust one or two parameters, and retry.
7. Call `submit` only after the command is running or has completed successfully,
   or when the task is blocked by missing files, missing GPUs, invalid API
   credentials, or a non-recoverable dependency error.

## Training command shape

Common RL training command:

```bash
areno train \
  --ckpt <model-or-local-checkpoint> \
  --dataset-path <dataset> \
  --dataset-loader-fn <loader.py> \
  --reward-fn-path <reward.py> \
  --algo gspo \
  --world-size <gpu-count> \
  --tp-size <tensor-parallel-size> \
  --batch-size <prompts-per-step> \
  --n-samples <samples-per-prompt> \
  --mini-bs <train-microbatch> \
  --max-running-prompts <rollout-concurrency> \
  --max-context-len <agentic-context-cap> \
  --drop-rollout-state \
  --max-steps 1
```

Never use Hugging Face model hub for AReno agent operations. For remote model or
dataset refs, always pass `--model-hub modelscope` unless the user explicitly
provides a local checkpoint path. Do not spend time checking Hugging Face
availability.

Useful examples:

```bash
areno train --ckpt Qwen/Qwen3.5-0.8B --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo --world-size 1 --tp-size 1 --batch-size 1 --n-samples 8 \
  --mini-bs 1 --max-running-prompts 8 --max-context-len 32768 \
  --drop-rollout-state --max-steps 1
```

```bash
areno train --ckpt <local-ckpt> --dataset-path /home/admin/math/data \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo --world-size 8 --tp-size 4 --batch-size 32 --n-samples 8 \
  --mini-bs 16 --max-running-prompts 256 --max-context-len 32768 \
  --drop-rollout-state --max-steps 1
```

Use `--save-path <dir> --save-interval 1 --max-steps 1` when the task asks to
test checkpoint saving. Then test loading by using `--ckpt <dir>/step_000001`.

## Smoke checks

Use smoke checks when they are useful for the user's goal. They are optional:
run them for explicit smoke/validation requests, for risky long-running jobs
where a quick preflight is valuable, or while diagnosing model/runtime/memory
failures. Do not run smoke searches just to maximize GPU use unless the user
asks for capacity tuning.

For agentic train or serve tasks, `--max-new-tokens` and `--max-context-len`
are user-facing quality and capacity decisions. If the user did not provide
either value, ask for the missing value before running commands. Do not silently
assume defaults for these two agentic limits, and do not tune
`--max-new-tokens` to make a smoke or train command fit. For agentic train
tasks, always set `--max-context-len` explicitly after the user confirms it.
Agentic rollouts can include multi-turn messages, tool calls, tool results,
images, and long reasoning traces, so relying on the model's full context limit
can make memory use and trajectory filtering unpredictable.

`--smoke-infer` dummy-loads the model, allocates rollout KV cache, and captures
decode CUDA graphs. It does not run decode. Use it to check model loading,
tensor-parallel compatibility, max context length, flash/native attention
compatibility, rollout KV memory, and CUDA graph capture. `--max-running-prompts`
is the main capacity being tested here: pass the value intended for the real
run. If `--max-running-prompts` is omitted, the smoke check uses the resolved
rollout concurrency from `batch_size * n_samples`.

For rollout-based algorithms, prefer `--n-samples 8` in smoke and real runs
unless the user requests another value. If you use `--smoke-infer`, pass the
intended real-run `--max-running-prompts` when the user has given one. If the
user only wants a quick compatibility check, use conservative small settings.
Keep smoke attempts limited; the goal is to validate the path, not to benchmark
the hardware limit.

Example:

```bash
areno train --ckpt <ckpt> --dataset-path __smoke__ --algo gspo \
  --world-size 8 --tp-size 4 --batch-size 32 --n-samples 8 \
  --mini-bs 16 --max-running-prompts 256 --max-new-tokens 1024 \
  --drop-rollout-state --smoke-infer
```

`--smoke-train` dummy-loads the model, skips real rollout/decode, offloads the
rollout state before training, and runs one synthetic train probe. It uses a
minimal train batch with `batch_size == mini_bs` and `n_samples == 1`, while
preserving the requested `mini_bs`, sequence length, TP/world size, optimizer,
activation checkpointing, and attention backend. Use it to check train memory,
backward kernels, optimizer state, and checkpoint/model training compatibility.
If you use `--smoke-train`, pass the intended `--mini-bs` when validating a
target training configuration. If the user only wants a quick startup check, use
small conservative values. On train OOM, reduce `--mini-bs` and leave memory
headroom for allocator fragmentation, CUDA graphs, and transient buffers.

Example:

```bash
areno train --ckpt <ckpt> --dataset-path __smoke__ --algo gspo \
  --world-size 8 --tp-size 4 --mini-bs 16 --max-new-tokens 1024 \
  --drop-rollout-state --smoke-train
```

## Serving command shape

Common serve command:

```bash
areno serve --ckpt <model-or-local-checkpoint> --host 0.0.0.0 --port 8000 \
  --world-size <gpu-count> --tp-size <tensor-parallel-size>
```

After serve starts, test it from another shell:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

If the port is busy, choose another port and retry.

## Memory tuning rules

Rollout memory is mainly controlled by:

- `--max-running-prompts`: higher means more concurrent rollout requests and
  more KV cache memory.
- `--batch-size * --n-samples`: this is the number of rollout requests produced
  per train step. It should usually be no larger than `--max-running-prompts`.
  If it is much smaller than `--max-running-prompts`, the configured concurrency
  may sit idle; increase `--batch-size` when memory and dataset size allow it.
- `--max-new-tokens` and prompt length: longer sequences require more KV cache,
  but `--max-new-tokens` should not be tuned by the agent to make a run fit.
  Keep the requested/default value and tune concurrency or train microbatch
  instead, unless the user explicitly asks to change generation length.
- `--tp-size`: larger tensor parallel size usually lowers per-GPU model memory,
  but changes the valid divisibility constraints for heads/layers.

Training memory is mainly controlled by:

- `--mini-bs`: higher means larger training microbatch and more activation
  memory.
- sequence length: longer rollout responses make train packs larger.
- optimizer choice and whether rollout state is kept.

If rollout OOM happens, reduce `--max-running-prompts`, `--batch-size`, or
`--n-samples` only when necessary. Do not reduce `--max-new-tokens` unless the
user explicitly asks for a shorter generation length. If train OOM happens,
reduce `--mini-bs` first. If model loading OOM happens, increase `--tp-size` or
use fewer other GPU processes.

When the user explicitly asks for capacity tuning, smoke checks can help
validate candidate `--max-running-prompts` or `--mini-bs` settings. Otherwise,
avoid capacity searches and prefer the user-provided or example-provided
settings. Keep `--n-samples 8` as the normal RL baseline unless the user or task
explicitly needs a different sampling count. Leave memory headroom for allocator
fragmentation, CUDA graphs, and transient buffers.

Use `--drop-rollout-state` by default for train attempts unless the user asks to
keep rollout state for performance experiments. It means the rollout engine
state is released before training to save memory. It can help when train OOM
occurs after rollout. It may increase step overhead because rollout state must
be rebuilt.

## Recoverable failures and retries

- CUDA out of memory during rollout: reduce `--max-running-prompts` by half.
- CUDA out of memory during train: reduce `--mini-bs` by half.
- OOM during startup/model loading: use a larger `--tp-size` if valid, or fewer
  GPUs per process only if the model supports it.
- `num_key_value_heads must be divisible by tp_size`: choose a `--tp-size` that
  divides the model's key-value heads.
- Port already in use for serve: retry with a different `--port`.
- Dataset loader path missing: inspect `examples/` and choose the loader matching
  the dataset.
- Reward function missing for RL algorithms: provide `--reward-fn-path` or
  `--reward-ckpt`.
- SFT requires a dataset loader: provide `--dataset-loader-fn`.

## Dependency repair

If a run fails because an optional kernel package is missing, the agent may
install the missing dependency in the current Python environment. Prefer a
prebuilt wheel when one exists. Do not reinstall the whole project unless the
user asks for it. After installing or changing dependencies, rerun a targeted
check. A smoke check is useful when the changed dependency affects model load,
CUDA graph capture, attention kernels, training kernels, or optimizer behavior.

For `flash-attn`, first inspect the active runtime:

```bash
python - <<'PY'
import platform, sys, torch
print("python", sys.version)
print("platform", platform.machine(), platform.system())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)
PY
```

Then list all currently available prebuilt GitHub release wheels and choose the
one matching Python ABI, CUDA, Torch version, platform, and CXX11 ABI:

```bash
python - <<'PY'
import json
import urllib.request

repo = "Dao-AILab/flash-attention"
for page in range(1, 20):
    url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
    with urllib.request.urlopen(url, timeout=30) as response:
        releases = json.load(response)
    if not releases:
        break
    for release in releases:
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(".whl"):
                print(asset["browser_download_url"])
PY
```

Install a selected wheel directly:

```bash
pip install --no-build-isolation --no-deps '<wheel-url>'
```

Known current release wheel URL patterns include:

- FlashAttention 4 beta universal wheels:
  - `https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta20/flash_attn_4-4.0.0b20-py3-none-any.whl`
  - `https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta19/flash_attn_4-4.0.0b19-py3-none-any.whl`
  - `https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta18/flash_attn_4-4.0.0b18-py3-none-any.whl`
- FlashAttention 2.8.3.post1 platform wheels use this release:
  - `https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1`
  - Example Python 3.10, CUDA 12, Torch 2.6, CXX11 ABI true, Linux x86_64:
    `https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`
  - Example Python 3.10, CUDA 12, Torch 2.6, CXX11 ABI false, Linux x86_64:
    `https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`

If no wheel matches exactly, stop and report the mismatch rather than starting a
long source build unless the user explicitly asks for a source build.

## Safety

Do not run destructive cleanup commands except targeted cleanup under temporary
directories when needed. Prefer inspecting disk usage before deleting anything.
Do not kill unrelated user processes unless the task explicitly asks for it.
