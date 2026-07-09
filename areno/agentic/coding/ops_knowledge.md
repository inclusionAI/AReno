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
3. Before a real train/serve run, first try smoke checks for the target
   checkpoint. The goal is to catch missing dependencies, unsupported model
   adapters, tensor-parallel divisibility errors, checkpoint shape mismatches,
   CUDA graph capture failures, and basic CUDA startup failures before spending
   time on a rollout or training step. Useful checks:
   - `areno check`
   - `areno env`
   - `areno train ... --smoke-infer`
   - `areno train ... --smoke-train`
4. Do not start smoke tuning from the smallest possible settings. Estimate the
   largest plausible target from available GPU memory, GPU count, model size,
   user-provided parameters, and nearby examples, then smoke-test that target
   first. For rollout/RL, keep `--n-samples 8` unless the user requests another
   value.
   Keep rollout demand and concurrency consistent: normally
   `batch_size * n_samples <= max_running_prompts`. If you raise
   `--max-running-prompts` to improve utilization, also raise `--batch-size`
   when the dataset and training memory allow it; otherwise the run may not
   produce enough requests to use the configured concurrency.
5. If the large smoke target fails with a recoverable capacity error, binary
   search the failing dimension instead of walking one step at a time:
   - rollout/KV OOM: binary search `--max-running-prompts`.
   - train/backward OOM: binary search `--mini-bs`.
   - full step OOM: reduce the dimension named by the failing phase first, then
     reduce `--batch-size` if needed.
   - do not tune `--max-new-tokens` to make smoke or train fit. Treat it as part
     of the task quality target unless the user explicitly changes it.
   - divisibility or unsupported-model errors are not capacity search problems;
     fix the invalid setting or report the blocker.
6. If the large smoke target succeeds with substantial free memory, try a larger
   upper bound and continue binary search until you have a largest stable value
   or a clear practical cap. Low GPU memory use usually means poor throughput.
   Prefer finding the largest stable settings under the available memory instead
   of keeping tiny smoke parameters.
   Keep this search short so the user does not wait on excessive smoke runs:
   usually one large attempt plus at most two or three capacity retries is
   enough before choosing a practical setting and moving on to the real command.
   The smoke target must leave headroom: keep peak GPU memory at or below about
   90% of total memory (`mem_frac <= 0.9`). If a smoke run exceeds that target
   or leaves too little free memory, reduce the searched capacity parameter.
7. Read the error message, adjust one or two parameters, and retry.
8. Call `submit` only after the command is running or has completed successfully,
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

Use smoke checks before long train/serve jobs.

For agentic train tasks, always set `--max-context-len` explicitly. Agentic
rollouts can include multi-turn messages, tool calls, tool results, images, and
long reasoning traces, so relying on the model's full context limit can make
memory use and trajectory filtering unpredictable. Use the user-provided
context cap when available; otherwise start with a practical value such as
`--max-context-len 32768` for coding/agentic RL and keep `--max-new-tokens`
unchanged.

`--smoke-infer` dummy-loads the model, allocates rollout KV cache, and captures
decode CUDA graphs. It does not run decode. Use it to check model loading,
tensor-parallel compatibility, max context length, flash/native attention
compatibility, rollout KV memory, and CUDA graph capture. `--max-running-prompts`
is the main capacity being tested here: pass the value intended for the real
run. If `--max-running-prompts` is omitted, the smoke check uses the resolved
rollout concurrency from `batch_size * n_samples`.

For rollout-based algorithms, prefer `--n-samples 8` in smoke and real runs
unless the user requests another value. Start smoke-infer from the largest
plausible real-run `--max-running-prompts` you can infer, not from a tiny value.
If it OOMs, halve the interval and binary search for the largest value that
passes. If it passes with lots of free memory, double or otherwise raise the
upper bound and then binary search down after the first failure. Do not spend a
long time chasing the absolute maximum: cap smoke-infer search to a few attempts
and prefer a good-enough stable setting over keeping the user waiting. Treat
about 90% GPU memory usage as the upper bound; do not choose settings that rely
on using nearly all memory.

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
Start from the largest plausible `--mini-bs`; on train OOM, binary search down to
the largest stable microbatch. If it passes with substantial free memory, raise
the upper bound and continue searching. Keep smoke-train search short; after a
large attempt and a few binary-search retries, use the best stable setting found.
Do not accept a smoke-train setting that pushes peak GPU memory above about 90%
of total memory.

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

For smoke tuning, search from the largest plausible setting first. If GPU memory
remains far below capacity after smoke succeeds, increase `--max-running-prompts`
first for rollout utilization, then increase `--batch-size` if the
dataset/algorithm supports it. For training utilization, increase `--mini-bs`
until train memory is close to the safe target. Use binary search after the first
capacity failure. Keep `--n-samples 8` as the normal RL baseline unless the user
or task explicitly needs a different sampling count.
Do not run too many smoke attempts: one high initial attempt and two or three
follow-up retries is normally enough. The goal is to avoid obviously bad
settings, not to benchmark the exact hardware limit. Keep the chosen smoke and
real settings within `mem_frac <= 0.9`; leave memory headroom for allocator
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
user asks for it. After installing or changing dependencies, first rerun
`--smoke-infer` or `--smoke-train` before retrying the original long command.

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
