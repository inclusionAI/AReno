# Phi-4 Multimodal image/text validation

AReno loads the official Phi-4 Multimodal checkpoint's language weights,
SigLIP NaViT encoder, HD projector, and vision LoRA. The checkpoint processor
performs image resizing, crop padding, and image-token expansion. Both dataset
rows and OpenAI-style image messages are converted to Phi's numbered image
markers before the string-only chat template is rendered.

The native model replaces the expanded image-token embeddings and enables
vision LoRA for the entire image-bearing sequence, including later text-only
prefill chunks and cached decode. Text rows in a mixed batch keep LoRA disabled.
Audio inference is outside this example's scope.

## Requirements

Use a local, complete `microsoft/Phi-4-multimodal-instruct` snapshot obtained
from ModelScope. Set `PHI4MM_MODEL_PATH` to that directory. The validation reads
local assets only and executes the official checkpoint's Python code for the
reference and processor. The AReno model uses native AReno modules.

The recorded reference environment uses Python 3.10, PyTorch 2.9.1+cu128,
Transformers 4.48.2, and PEFT 0.13.2. These are the versions used to execute
this snapshot's reference code, not changes to AReno's package dependencies.
Compatibility with other Transformers versions must be checked separately.

Run the commands below from the repository root in an environment where AReno
is importable. Full model inference requires CUDA. `--stream-weights` optionally
stages weights one module at a time when a GPU cannot hold the entire model;
this is a validation facility, not a serving implementation or speed benchmark.

## Processor, intermediate tensors, and generation

```bash
python examples/multimodal/phi4mm/validate_vision.py \
  --model-path "$PHI4MM_MODEL_PATH" \
  --output-dir validation/phi4mm \
  --device cuda:0 --steps 16
```

The deterministic fixtures cover a square red image, a portrait blue image,
two differently shaped images in one prompt, and a text-only prompt. The script:

1. Checks exact token, image tensor, size, and mask equality between the
   official processor, dataset encoding, and serving request encoding.
2. Loads the official model with eager attention explicitly selected in its
   config. This snapshot stores `_attn_implementation` in `config.json`, so
   relying on a `from_pretrained` keyword alone can leave FlashAttention active.
3. Records the selected vision layer, projector, replaced embeddings, decoder
   layers 0/16/31, final norm, and cached generation logits.
4. Runs AReno full prefill and 256-token chunked prefill followed by paged decode.
   Generation stops at the checkpoint's EOS IDs or `--steps`.
5. Requires exact greedy-token equality, finite tensors, intermediate cosine
   similarity at least 0.99, and every compared logits cosine at least 0.999.

`reference.pt` and `areno.pt` contain CPU tensors. `report.json` contains error
metrics, generated tokens/text, and reference-source SHA-256 hashes.
`checkpoint_audit.json` accounts for all loaded and deliberately skipped keys.
These fixtures test implementation equivalence; they are not an image-understanding
benchmark and do not establish arbitrary-prompt bitwise equivalence.

Use `--phase reference`, `--phase areno`, and `--phase compare` to run these
steps separately in the same output directory. Keep the checkpoint, fixtures,
`--steps`, and `--chunk-size` consistent between phases.

## Tensor parallelism

First generate the reference above. Then run:

```bash
torchrun --standalone --nproc-per-node=2 \
  examples/multimodal/phi4mm/validate_vision.py \
  --model-path "$PHI4MM_MODEL_PATH" --output-dir validation/phi4mm \
  --phase areno --steps 16

python examples/multimodal/phi4mm/validate_vision.py \
  --model-path "$PHI4MM_MODEL_PATH" --output-dir validation/phi4mm \
  --phase compare --tp-size 2 --steps 16
```

This produces separate `areno-tp2.pt` and `report-tp2.json` artifacts. The vision
tower is replicated; decoder and vision-LoRA weights use their TP layouts.

## Real worker inference

```bash
python examples/multimodal/phi4mm/validate_engine.py \
  --model-path "$PHI4MM_MODEL_PATH" \
  --reference validation/phi4mm/reference.pt \
  --output validation/phi4mm/engine.json --device 0
```

This uses `ArenoEngine.from_pretrained(role="rollout")` with native attention,
CUDA-graph decode enabled, and two running slots. It submits all four fixtures,
then reverses their order in a second call on the same engine. Both rounds must
match the official tokens, covering mixed image/text rows and slot/cache reuse.
Pass `--eager-decode` to exercise the eager runtime instead.

## Save and reload

```bash
python examples/multimodal/phi4mm/validate_vision.py \
  --model-path "$PHI4MM_MODEL_PATH" --output-dir validation/before \
  --phase areno --cases single multi --steps 16 \
  --save-dir validation/saved

python examples/multimodal/phi4mm/validate_vision.py \
  --model-path validation/saved --output-dir validation/after \
  --phase areno --cases single multi --steps 16
```

Compare `logits`, `full_logits`, and `tokens` in the two `areno.pt` files. For
an unchanged checkpoint they must be identical. Saving requires the original
snapshot so audio/speech tensors and processor assets can be preserved even
though the current runtime does not execute audio inference. Use a new or empty
save directory.

## Regression tests

```bash
python -m pytest tests/test_phi4mm_vision_cpu.py \
  tests/test_phi4mm_adapter_cpu.py tests/test_phi4mm_checkpoint_cpu.py \
  tests/test_serve_cli_cpu.py tests/test_inference_scheduler_cpu.py -q
python -m pytest tests/test_native_attention_gpu.py -q
```

CUDA-specific tests skip without a GPU. They cover cached-prefix attention
against SDPA for native and FlashAttention backends, CUDA-graph modality-slot
updates, and image prefill split across a reduced LongRoPE boundary. CPU tests
also verify two synthetic optimizer steps with gradients reaching the encoder,
projector, and vision LoRA; this is not a full-checkpoint training validation.

See [VALIDATION.md](VALIDATION.md) for measured results and limitations.
