# Phi-4 Multimodal audio

This example exercises the same native Phi-4 audio processor and Conformer
encoder in serving, rollout, log-probability scoring, and training.

## Serve and send audio

Start the server with a local Phi-4 Multimodal checkpoint:

```bash
areno serve \
  --model-path /models/Phi-4-multimodal-instruct \
  --tp-size 1 \
  --world-size 1 \
  --max-running-prompts 1 \
  --port 8000
```

Send a local WAV or MP3 file. The client embeds it as base64, so the server
does not need access to the client's filesystem:

```bash
python examples/multimodal/phi4mm_audio/request.py ./sample.wav \
  --prompt "Transcribe the speech, then summarize it."
```

## Training data

Create a JSONL manifest. Relative audio paths are resolved from the manifest
directory:

```json
{"audio_path":"audio/0001.wav","prompt":"What is being said?","response":"A short reference answer."}
```

The included loader turns each row into the same structured audio message used
by serving. A minimal GRPO command is:

```bash
areno train \
  --ckpt /models/Phi-4-multimodal-instruct \
  --dataset-path /data/audio/train.jsonl \
  --dataset-loader-fn examples/multimodal/phi4mm_audio/dataset_loader.py \
  --reward-fn-path /path/to/reward.py \
  --algo grpo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --mini-bs 1 \
  --n-samples 4 \
  --max-running-prompts 1 \
  --max-prompt-tokens 4096 \
  --max-context-len 8192
```

Media towers and projectors are frozen by default. To train the audio tower or
projector, add `--unfreeze-mm-tower` and/or `--unfreeze-mm-projector`; begin
with a small independent learning rate such as `--mm-projector-lr 1e-5`.

See `docs/models/phi4mm.rst` for checkpoint requirements, supported modes, and
memory guidance.
