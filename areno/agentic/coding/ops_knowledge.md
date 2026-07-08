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
3. Before a real train/serve run, first try a dummy load or minimal load check
   for the target checkpoint. The goal is to catch missing dependencies,
   unsupported model adapters, tensor-parallel divisibility errors, checkpoint
   shape mismatches, and basic CUDA startup failures before spending time on a
   rollout or training step. Useful checks:
   - `areno check`
   - `areno env`
   - a minimal `areno serve` on an unused port followed by `/v1/models`, then
     stop it if the user asked for training
   - a one-step `areno train --max-steps 1` with small batch settings if serve
     is not appropriate
4. Prefer a small smoke test first:
   - one epoch or `--max-steps 1`
   - small `--batch-size`
   - small `--n-samples`
   - conservative `--mini-bs`
   - a valid `--save-path` only when checkpoint save must be tested
5. Read the error message, adjust one or two parameters, and retry.
6. Call `submit` only after the command is running or has completed successfully,
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
  --max-steps 1
```

Useful examples:

```bash
areno train --ckpt Qwen/Qwen3.5-0.8B --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo --world-size 1 --tp-size 1 --batch-size 1 --n-samples 2 \
  --mini-bs 1 --max-running-prompts 2 --max-steps 1
```

```bash
areno train --ckpt <local-ckpt> --dataset-path /home/admin/math/data \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo --world-size 8 --tp-size 4 --batch-size 32 --n-samples 8 \
  --mini-bs 16 --max-running-prompts 256 --max-steps 1
```

Use `--save-path <dir> --save-interval 1 --max-steps 1` when the task asks to
test checkpoint saving. Then test loading by using `--ckpt <dir>/step_000001`.

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
- `--max-new-tokens` and prompt length: longer sequences require more KV cache.
- `--tp-size`: larger tensor parallel size usually lowers per-GPU model memory,
  but changes the valid divisibility constraints for heads/layers.

Training memory is mainly controlled by:

- `--mini-bs`: higher means larger training microbatch and more activation
  memory.
- sequence length: longer rollout responses make train packs larger.
- optimizer choice and whether rollout state is kept.

If rollout OOM happens, reduce `--max-running-prompts`, `--batch-size`,
`--n-samples`, or max sequence length. If train OOM happens, reduce `--mini-bs`
first. If model loading OOM happens, increase `--tp-size` or use fewer other GPU
processes.

`--drop-rollout-state` means the rollout engine state is released before
training to save memory. It can help when train OOM occurs after rollout. It may
increase step overhead because rollout state must be rebuilt.

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
user asks for it. After installing or changing dependencies, first rerun the
dummy load/minimal load check before retrying the original long command.

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
