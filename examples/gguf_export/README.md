# Direct GGUF export

The CUDA SDK can export a live dense Qwen3 policy without a converter or an
intermediate Safetensors checkpoint. Call after training and before closing:

```python
trainer.export_gguf(output_dir, file_type="f16")
```

For `areno train`, append `--export-gguf ./exports/qwen3` to the training
command. `--gguf-file-type f16` is the default; `q8_0` is also available.
Export runs once after successful training, including a normal stop at
`--max-steps`, and before releasing the live weights. It also works without
`--save-path`. Training failures skip export; export failures fail the command
and still close the trainer. Smoke/tuning probes do not export dummy weights.

The output is a new directory containing `model.gguf`, `Modelfile`, and
`export.json`. Existing directories are rejected. `gguf` is a Python package
dependency; no separate llama.cpp installation or executable is needed.
`file_type="q8_0"` quantizes eligible matrices in Python; vectors remain F32.
Standard native LoRA is merged into temporary export buffers. Packed QLoRA,
other architectures, MLX, and multimodal exports are not implemented yet.

`Modelfile` provides ordinary text chat with thinking disabled. Tools are not
advertised. Import the package with `ollama create NAME -f /path/to/Modelfile`,
then use `ollama run NAME`.

`run.py` performs one real SFT update on two inspectable arithmetic examples,
generates before and after, exports directly, and checks that selected output
weights differ from the original base model. This verifies the pipeline, not
training quality. Pass `--model-path`, `--output-dir`, and `--report` as local
paths. `--lora-rank 8` tests native LoRA merging instead of full-weight SFT.

With Ollama running, import the exported package, then verify raw generation
and its chat template against the same prompt:

```bash
ollama create areno-qwen3 -f /path/to/export/Modelfile
python examples/gguf_export/verify_ollama.py \
  --model areno-qwen3 --native-report /path/to/native-report.json \
  --report /path/to/ollama-report.json
```

The verifier records both outputs and whether the raw text matches. Successful
generation demonstrates loading and execution; it does not establish numerical
parity between AReno's BF16 execution and the exported F16 or Q8_0 model.
