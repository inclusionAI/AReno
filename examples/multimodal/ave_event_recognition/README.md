# Gemma 4 AVE Event Recognition

This agentic RL example trains Gemma 4 to recognize every audiovisual event
inside a requested interval of a 10-second AVE clip. Each prompt includes the
video, its synchronized audio track, and a question such as:

```text
Which audiovisual events occur between 6 and 10 seconds in this clip?
```

The policy returns one `report_events` tool call with an `events` string array.
The prompt never contains the reference labels.

The source is the original AVE dataset mirrored on
[Hugging Face](https://huggingface.co/datasets/UnFaZeD07/AVE-Dataset). AVE has
28 event categories and annotations in this format:

```text
Category&VideoID&Quality&StartTime&EndTime
Church bell&fCZi6I6kPpU&good&6&10
```

Some videos contain multiple labels. The generator creates one query per
distinct annotated interval and puts every temporally overlapping label in the
reference `event_classes` list. This makes multi-event supervision explicit.

## Generate the dataset

Install `ffmpeg` before generating the manifest. A Conda build avoids conflicts
with incompatible system GLib libraries:

```bash
conda install -y -c conda-forge ffmpeg
export PATH="${CONDA_PREFIX:-/opt/conda}/bin:$PATH"
ffmpeg -version | head
```

Download and extract AVE automatically:

```bash
python examples/multimodal/ave_event_recognition/dataset_generator.py \
  --dataset-root ~/datasets/AVE \
  --output ~/datasets/AVE/train.jsonl \
  --split train
```

For an existing extracted `AVE.zip`, use `--skip-download`:

```bash
python examples/multimodal/ave_event_recognition/dataset_generator.py \
  --dataset-root /home/admin/tiny/AVE \
  --output /home/admin/tiny/AVE/train.jsonl \
  --split train \
  --skip-download
```

The generator extracts a 16 kHz mono WAV once per video and writes the new
event-list schema:

```json
{
  "video_id": "fCZi6I6kPpU",
  "start_seconds": 6.0,
  "end_seconds": 10.0,
  "event_classes": ["Church bell"],
  "video_path": "videos/videos/fCZi6I6kPpU.mp4",
  "audio_path": "audio/fCZi6I6kPpU.wav"
}
```

Old single-label manifests containing only `event_class` are not compatible;
regenerate them with the command above.

## Judge reward

The reward calls an OpenAI-compatible judge model to compare the predicted and
reference event sets semantically. It accepts synonyms such as `dog barking`
and `bark`, while penalizing missing or extra events. Configure it with:

```bash
export JUDGE_BASE_URL=http://judge-host:8000/v1
export JUDGE_MODEL=judge-model
export JUDGE_API_KEY=your-key
export JUDGE_MAX_WORKERS=16
```

Lowercase forms (`judge_base_url`, `judge_model`, and `judge_api_key`) are also
accepted. The judge returns semantic similarity from `0` for unrelated events
to `10` for an exact semantic set match, with partial matches receiving
intermediate scores. Training normalizes this to `reward = score / 10`, giving
a `0–1` reward range. Malformed tool calls receive `0`. Missing judge
configuration, request failures, and invalid judge responses fail loudly
instead of silently assigning bad rewards. Repeated label-pair decisions are
cached within each trainer process. Judge calls run in parallel while preserving
rollout order; `JUDGE_MAX_WORKERS` controls concurrency and defaults to 16.

## Train

```bash
JUDGE_BASE_URL=http://judge-host:8000/v1 \
JUDGE_MODEL=judge-model \
JUDGE_API_KEY=your-key \
ARENO_LOG_COMPLETIONS=1 \
python -m areno.cli.main train \
  --ckpt /home/admin/gemma-4-E2B-it \
  --model-hub hf \
  --dataset-path /home/admin/tiny/AVE/train.jsonl \
  --dataset-loader-fn examples/multimodal/ave_event_recognition/dataset_loader.py \
  --reward-fn-path examples/multimodal/ave_event_recognition/reward.py \
  --agent-fn examples/multimodal/ave_event_recognition/run_agent.py \
  --algo grpo \
  --tp-size 1 \
  --world-size 1 \
  --train-devices 0 \
  --batch-size 1 \
  --mini-bs 1 \
  --n-samples 8 \
  --max-running-prompts 1 \
  --temperature 1.3 \
  --top-p 0.95 \
  --max-new-tokens 128 \
  --max-prompt-tokens 8192 \
  --max-context-len 8192 \
  --drop-rollout-state
```

AVE was introduced in *Audio-Visual Event Localization in Unconstrained
Videos*, ECCV 2018, by Tian et al. Follow the dataset terms and cite the
original paper when using the data.
