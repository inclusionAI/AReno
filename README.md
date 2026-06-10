👋 Hi, everyone! AReno is a fast, effortless, and self-contained toolkit that
scales RL post-training up locally, initiated by the inclusionAI ASystem team
and maintained by the AReno community.

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.6+" src="https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg"></a>
</p>

<!-- TODO: replace with the project logo when ready -->
<p align="center">
  <img alt="inclusionAI" src="https://raw.githubusercontent.com/inclusionAI/.github/main/assets/banner.png">
</p>

## AReno: ASystem Reinforcement Learning Nano

AReno is a reinforcement learning (RL) toolkit that lets you train and run models locally. It was originally developed by engineers from the ASystem Team at Ant Group.

Built on a **self-contained, full-stack design**, AReno is optimized to extract maximum performance from a single node, making it well-suited for **fast, local RL post-training** with no external training or inference backend in the loop.

AReno's mission is to make LLM RL **accessible** for a broad community of researchers and developers — so you can go from a base checkpoint to a trained, served model on a single node, without standing up a cluster or wiring together a training framework, an inference server, and a kernel library.

> Small but complete, like its name — nano in footprint, full-stack in
> capability. We hope AReno makes scaling up your ideas locally both fast and
> delightful. Enjoy!

## Highlights

- ✨ **Plug-and-play**: various fine-tuning methods are easily accessible via the `--algo` flag or the same `Trainer` class from Python, no cluster or launcher to set up.
- 🪶 **Lightweight**: single self-contained package, no external training/inference backend, just PyTorch, FlashAttention, and a handful of other libraries.
- 🧩 **Extensible**: easily register new algorithms, model adapters, reward functions, and hardware backends without changing the core.

## Installation

**Requirements:**

- Linux with an NVIDIA GPU (CUDA compute capability 8.0+)
- CUDA toolkit, with `CUDA_HOME` set (so `nvcc` is on the build path)
- PyTorch >= 2.6, matching your installed CUDA version

> **Other platforms:** Apple Silicon (M-series) and AMD GPUs are not supported —
> the engine requires NVIDIA CUDA. On Windows, install under
> [WSL2](https://learn.microsoft.com/windows/wsl/) and follow the Linux
> instructions. DGX Spark and other Grace/Blackwell systems work, but install an
> `aarch64` PyTorch build first.

**To install:**

```bash
pip install "git+https://github.com/inclusionAI/asystem-areno.git" --no-build-isolation
pip install flash-attn flash-linear-attention
```

`--no-build-isolation` is required so that pip uses your existing CUDA-enabled
PyTorch instead of installing a CPU-only torch in an isolated build environment.

> Prebuilt PyPI wheels (`pip install areno`) are coming. Until then, install
> from the repository as shown above.

**From source** (recommended if you want the examples or plan to contribute):

```bash
git clone https://github.com/inclusionAI/asystem-areno.git
cd asystem-areno
pip install -e . --no-build-isolation
pip install flash-attn flash-linear-attention
```

**Tips:**

- Install `ninja` (`pip install ninja`) before building. With ninja, the CUDA
  kernels compile in parallel (a few minutes); without it, compilation falls
  back to a single core and is much slower.
- Compiling the CUDA kernels takes a few minutes. If your machine has many CPU
  cores but limited RAM, cap the parallel build jobs with `MAX_JOBS`:
  ```bash
  MAX_JOBS=4 pip install -e . --no-build-isolation
  ```
- To install the Python package without building the CUDA extension (for
  docs/metadata or a dry run), set `ARENO_BUILD_EXT=0`. The engine will not run
  without the extension, but the install will succeed.

## Quick Start

With the SDK, RL loop is a short cycle of `Trainer` calls. Each step below maps a
concept to the SDK call that performs it:

```mermaid
flowchart LR
    A["Trainer<br/>init()"] -->
    B["rollout_batch<br/>on-policy samples"] -->
    C["reward fn<br/>score"] -->
    D["train<br/>optimizer step"] -->|repeat| B
```

1. **Create the trainer** — construct a `Trainer` on the AReno backend and
   `init()` it to load the tokenizer and start workers.
2. **Roll out** — `rollout_batch(...)` generates on-policy completions for each
   prompt.
3. **Score** — reward each completion and turn rewards into advantages (your
   reward function, not AReno's).
4. **Train** — pack the rollout into `TrainSequence` objects and call
   `train(batch, loss_fn)` to run one optimizer step.
5. **Repeat** — new weights produce new rollouts; loop until done, then
   `close()`.

```python
from functools import partial

from datasets import load_dataset

from areno.api import (
    Areno,
    ArenoConfig,
    SamplingParams,
    Trainer,
    TrainSequence,
    gspo_loss_fn,
)
from examples.math.math_verify_reward import reward_fn


def to_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards), 1)
    std = max(var**0.5, 1e-6)
    return [(r - mean) / std for r in rewards]


# 1. Create the trainer
trainer = Trainer(
    world_size=1,
    model_path="Qwen/Qwen3-0.6B",
    backend_type=Areno,
    custom_config=ArenoConfig(tp_size=1),
)
trainer.init()

# 2. Roll out on-policy completions for one GSM8K prompt
row = load_dataset("gsm8k", "main", split="train[0:1]")[0]
prompt = (
    "Solve the problem and put the final answer in \\boxed{}.\n\n"
    f"Problem: {row['question']}\nSolution:"
)
prompt_tokens = trainer.get_tokenizer().encode(prompt)
rollout = trainer.rollout_batch(
    [prompt],
    n_samples=8,
    sampling_params=SamplingParams(max_new_tokens=512, temperature=1.0),
)[0]

# 3. Score with the same reward function the CLI uses, then form advantages
completions = [trainer.get_tokenizer().decode(seq.resp_tokens) for seq in rollout.sequences]
rewards = reward_fn(row, completions)
advantages = to_advantages(rewards)

batch = []
for seq, reward, advantage in zip(rollout.sequences, rewards, advantages, strict=True):
    response_len = len(seq.resp_tokens)
    batch.append(
        TrainSequence(
            prompt_mask=[True] * len(prompt_tokens) + [False] * response_len,
            tokens=prompt_tokens + seq.resp_tokens,
            logprobs=[0.0] * len(prompt_tokens) + seq.resp_logprobs,
            advantages=[0.0] * len(prompt_tokens) + [advantage] * response_len,
            reward=reward,
            eos_token_id=trainer.get_tokenizer().eos_token_id,
        )
    )

# 4. Train one step
stats = trainer.train(batch, partial(gspo_loss_fn, clip_eps=3.0e-4), mini_bs=4)

# 5. Repeat the loop over more prompts, then close
trainer.close()
```

See the documentation for the full `Trainer` API.

## Command Line Interface (CLI)

You can use the AReno Command Line Interface (CLI) to quickly get started with post-training without writing any Python. Run GSPO on a GSM8K-style dataset with a reward function:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path gsm8k:main \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo \
  --tp-size 4
```

`--ckpt` and `--dataset-path` accept either local paths or Hugging Face repo
IDs. Switch algorithms by changing `--algo` (e.g. `--algo grpo`, `--algo sft`).

Serve a trained checkpoint behind an OpenAI-compatible endpoint:

```bash
areno serve \
  --model-path /path/to/model \
  --tp-size 1 \
  --world-size 1 \
  --port 8000
```

This starts a server with continuous batching; point any OpenAI client at
`http://localhost:8000/v1/chat/completions`. Run `areno train --help` or
`areno serve --help` for the full option surface.

## Development

If you want to contribute to AReno or customize it for your own needs, read
the contribution guide and make a development install:

```bash
git clone https://github.com/inclusionAI/asystem-areno.git
cd asystem-areno
pip install -e . --no-build-isolation
```

New algorithms, model adapters, kernels, reward functions, and hardware
backends all have first-class extension points, so most contributions land
without forking the core.

## Citation

```bibtex
@misc{areno2026,
  title        = {AReno: A Self-Contained, Full-Stack Toolkit for Single-Node LLM RL Post-Training},
  author       = {Zibo He and Le Su and Zongyu Li},
  year         = {2026},
  url          = {https://github.com/inclusionAI/asystem-areno},
  license      = {Apache-2.0}
}
```

## License

This repository's source code is available under the [Apache 2.0 License](LICENSE).
