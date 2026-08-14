# Gemma 4 AVE Temporal Grounding

This agentic RL example trains Gemma 4 to localize one named audiovisual event
inside a 10-second AVE clip. Each prompt includes one video, its extracted audio
track, and one target event class. The policy returns exactly one
`report_event_range` tool call with `start_seconds` and `end_seconds`.

The source is the original AVE dataset mirrored on
[Hugging Face](https://huggingface.co/datasets/UnFaZeD07/AVE-Dataset). Its
`Annotations.txt` stores `Category&VideoID&Quality&StartTime&EndTime`, and the
official train, validation, and test files use the same row format. One video
can therefore produce multiple records when it contains multiple annotated
events. Degenerate zero-duration annotation rows are skipped because they do
not define a learnable temporal range.

## Generate the dataset

The first invocation downloads about 5.67 GB, safely extracts `videos.zip`, and
uses `ffmpeg` to create a 16 kHz mono WAV once per unique video. Hugging Face's
cache resumes interrupted downloads.

```bash
python examples/agentic/ave_temporal_grounding/dataset_generator.py \
  --dataset-root ~/datasets/AVE \
  --output ~/datasets/AVE/train.jsonl \
  --split train
```

For an existing download, pass `--skip-download`. Generate validation or test
manifests by selecting `--split val` or `--split test`.

Each JSONL row is one event question:

```json
{
  "video_id": "MH3m4AwEcRY",
  "event_class": "Church bell",
  "start_seconds": 6.0,
  "end_seconds": 8.0,
  "video_path": "videos/MH3m4AwEcRY.mp4",
  "audio_path": "audio/MH3m4AwEcRY.wav"
}
```

## Reward

The numerical reward is a strict but dense score:

```text
quality = 0.75 * temporal_IoU^2 + 0.25 * boundary_accuracy^2
reward = 2 * quality - 1
boundary_accuracy = max(0, 1 - (|start-start*| + |end-end*|) / 20)
```

A perfect range receives `1.0`. Squaring overlap and boundary accuracy makes
near misses noticeably less rewarding, while preserving a continuous signal
for valid predictions instead of collapsing them all to `-1.0`.
Malformed, non-finite, reversed, negative, or greater-than-10-second ranges
receive `-1.0`. Missing or repeated tool calls also receive `-1.0`.

## Train

```bash
ARENO_LOG_COMPLETIONS=1 areno train \
  --ckpt /path/to/gemma-4 \
  --dataset-path ~/datasets/AVE/train.jsonl \
  --dataset-loader-fn examples/agentic/ave_temporal_grounding/dataset_loader.py \
  --reward-fn-path examples/agentic/ave_temporal_grounding/reward.py \
  --agent-fn examples/agentic/ave_temporal_grounding/run_agent.py \
  --algo ppo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --mini-bs 1 \
  --max-running-prompts 1 \
  --max-new-tokens 64 \
  --max-prompt-tokens 8192
```

Use a larger tensor-parallel or world size for the selected Gemma checkpoint.
The same files can be used with GSPO or GRPO when grouped sampling is preferred.

## Citation

AVE was introduced in *Audio-Visual Event Localization in Unconstrained
Videos*, ECCV 2018, by Tian et al. Follow the dataset repository's terms and
cite the original paper when using the data.
