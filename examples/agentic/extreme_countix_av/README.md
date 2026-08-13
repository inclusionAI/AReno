# Gemma4 Extreme Countix-AV Agentic RL

This example trains Gemma4 to recognize a repetitive physical activity from
paired video and audio and to count its completed repetitions. The policy
returns one structured `report_repetitions` tool call. The reward scores only
relative repetition-count accuracy; the reported action class is diagnostic.

## Download the dataset

Download **Extreme Countix-AV** from the authors' official
[RepetitionCounting repository](https://github.com/xiaobai1217/RepetitionCounting#datasets).
The repository links the official
[Extreme Countix-AV archive](https://drive.google.com/file/d/1eKYbN_fXetv6Dw_ks8eNeNkErGvrsDC6/view?usp=sharing).
The dataset accompanies the CVPR 2021 paper
[Repetitive Activity Counting by Sight and Sound](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Repetitive_Activity_Counting_by_Sight_and_Sound_CVPR_2021_paper.html).

After extraction, the directory must contain:

```text
ExtremeCountixAV/
├── ExtremeLabels.csv
├── Audio/<condition>/*.wav
└── Videos/<condition>/*.mp4
```

The official CSV includes an unlabelled VGGSound appendix. This example skips
those rows because they have no `action_class`. A labelled video appearing in
multiple challenge-condition directories remains a separate training sample.

## Files

- `dataset_generator.py` creates a shuffled JSONL manifest containing labels
  and media paths, not embedded media bytes.
- `dataset_loader.py` accepts either the extracted dataset directory or a
  generated manifest.
- `run_agent.py` sends paired `video_url` and `audio_url` inputs and requires a
  `report_repetitions` tool call.
- `reward.py` scores relative repetition-count accuracy only. Action-class
  correctness does not affect training reward.
- `web_ui.py` provides synchronized playback, filtering, inference, and reward
  inspection.

## Optional manifest

AReno can train directly from the extracted directory. A manifest is useful
when you want a deterministic shuffled order:

```bash
python examples/agentic/extreme_countix_av/dataset_generator.py \
  --dataset-root ~/Spaceship/dataset/ExtremeCountixAV \
  --output ~/Spaceship/dataset/ExtremeCountixAV/train.jsonl \
  --seed 42
```

## Agentic GSPO

Gemma4 video inputs are large, so begin with one running prompt and one sample
per batch. Increase these only after measuring GPU memory on your checkpoint.

```bash
ARENO_LOG_COMPLETIONS=1 areno train \
  --ckpt /home/admin/gemma-4-E2B-it \
  --dataset-path /path/to/ExtremeCountixAV \
  --dataset-loader-fn examples/agentic/extreme_countix_av/dataset_loader.py \
  --reward-fn-path examples/agentic/extreme_countix_av/reward.py \
  --agent-fn examples/agentic/extreme_countix_av/run_agent.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 2 \
  --mini-bs 1 \
  --epochs 1 \
  --max-running-prompts 1 \
  --max-new-tokens 128 \
  --max-prompt-tokens 8192 \
  --drop-rollout-state
```

Use `--algo grpo` with the same files if group-relative sequence-level updates
fit your experiment better. Keep `--n-samples` above one for either grouped
method so each prompt has alternative completions to compare.

## Serve and WebUI

Start the Gemma4 OpenAI-compatible server:

```bash
areno serve \
  --model-path /home/admin/gemma-4-E2B-it \
  --tp-size 1 \
  --world-size 1 \
  --max-running-prompts 1 \
  --port 8014
```

Then start the workbench:

```bash
python examples/agentic/extreme_countix_av/web_ui.py \
  --dataset-root ~/Spaceship/dataset/ExtremeCountixAV \
  --browser-video-root /path/to/h264-video-mirror \
  --base-url http://127.0.0.1:8014/v1 \
  --port 8770
```

The official videos use MPEG-4 Part 2, which many modern browsers cannot
decode even though the files have an `.mp4` extension. Point
`--browser-video-root` at a directory containing an H.264 mirror of `Videos`
with the same condition subdirectories and filenames. This affects playback
only; inference and training continue to use the original dataset files.

Create the browser-compatible mirror with FFmpeg:

```bash
SOURCE=/path/to/ExtremeCountixAV/Videos
TARGET=/path/to/ExtremeCountixAV-H264

find "$SOURCE" -type f -name '*.mp4' -print0 | while IFS= read -r -d '' video; do
  relative=${video#"$SOURCE"/}
  output="$TARGET/$relative"
  mkdir -p "$(dirname "$output")"
  ffmpeg -loglevel error -y -i "$video" -an \
    -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
    -movflags +faststart "$output"
done
```

`yuv420p` maximizes browser compatibility and `+faststart` moves MP4 metadata
to the beginning of the file so playback can begin before the full video is
downloaded. Audio remains available through the separate synchronized WAV
player in the workbench. The mirror may contain only a representative subset;
when `--browser-video-root` is set, the workbench lists only samples whose
browser-compatible video exists in that mirror. The original dataset remains
complete and available to training and inference.

## Live camera and microphone

The workbench can capture a new activity from the browser camera and
microphone. Choose **Use camera**, record the movement, stop, and then choose
**Analyze capture**. The browser records video and PCM WAV audio together; the
server stores them only in a temporary directory for one inference request and
does not add them to the training dataset.

Browser media permissions require a secure context. Access a remote Web UI
through HTTPS; plain HTTP camera and microphone access is normally allowed
only on `localhost`. The video control bar remains the master preview timeline.

For a camera-and-microphone-only interface that does not load a dataset, run:

```bash
python examples/agentic/extreme_countix_av/web_ui.py \
  --live-only \
  --segment-seconds 4 \
  --base-url http://127.0.0.1:8001/v1 \
  --host 0.0.0.0 \
  --port 8770
```

Live mode is segmented near-real-time inference rather than frame-by-frame
streaming. The browser records consecutive non-overlapping audiovisual clips,
records the next clip while the previous clip waits for inference, and sends
requests serially. Each result counts only cycles completed inside that clip.
The workbench accumulates a session total and separate totals for each
normalized action label, so switching activities does not merge their counts.
Shorter segments reduce detection latency but give the model less motion
context; four seconds is a practical starting point.

Open `http://127.0.0.1:8770`. Behind a reverse proxy, `--base-url /v1` is also
supported; the WebUI resolves it against the incoming request origin.
