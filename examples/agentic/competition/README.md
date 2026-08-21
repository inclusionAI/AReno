# Competition Agentic Example

Two LLM agents compete to generate the best sandwich feedback for a user's daily
diary. The core capability is "sandwich feedback": affirm effort, gently point
out one area for improvement with a specific suggestion, then affirm again.

## Quick Start

### 1. Configure your profile

Edit `user_profile.json` with your name, personality, and preferences.

### 2. Generate training data

```bash
python examples/agentic/competition/dataset_generator.py \
  --output /tmp/areno-competition.jsonl \
  --count 64 \
  --seed 2026
```

Generate a fixed evaluation set too:

```bash
python examples/agentic/competition/dataset_generator.py \
  --output /tmp/areno-competition-eval.jsonl \
  --count 16 \
  --seed 9001
```

### 3. Train

For a constrained free Kaggle GPU, start with this small smoke run:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --world-size 1 \
  --algo gspo \
  --tp-size 1 \
  --model-hub modelscope \
  --dataset-path /tmp/areno-competition.jsonl \
  --dataset-loader-fn examples/agentic/competition/dataset_loader.py \
  --reward-fn-path examples/agentic/competition/reward.py \
  --agent-fn examples/agentic/competition/run_agent.py \
  --batch-size 1 \
  --n-samples 2 \
  --mini-bs 1 \
  --score-micro-bs 1 \
  --max-running-prompts 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --max-context-len 3072 \
  --max-steps 5 \
  --save-interval 5 \
  --drop-rollout-state \
  --save-path /kaggle/working/areno-competition-smoke
```

If the smoke run is stable, a short 30-step run is enough to validate the full
training-serving-evaluation loop on a free Kaggle T4 session:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --world-size 1 \
  --algo gspo \
  --tp-size 1 \
  --model-hub modelscope \
  --dataset-path /tmp/areno-competition.jsonl \
  --dataset-loader-fn examples/agentic/competition/dataset_loader.py \
  --reward-fn-path examples/agentic/competition/reward.py \
  --agent-fn examples/agentic/competition/run_agent.py \
  --batch-size 1 \
  --n-samples 2 \
  --mini-bs 1 \
  --score-micro-bs 1 \
  --max-running-prompts 2 \
  --max-prompt-tokens 512 \
  --max-new-tokens 64 \
  --max-context-len 3072 \
  --max-steps 30 \
  --save-interval 5 \
  --drop-rollout-state \
  --save-path /kaggle/working/areno-competition-30step
```

For longer experiments, increase `--max-steps` after the 30-step run is saved
and evaluated.

### 4. Evaluate Before And After Training

Run the same evaluation diary set against the base model and the trained
checkpoint. Start `areno serve` first, then call the evaluator:

```bash
areno serve \
  --model-path Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --tp-size 1 \
  --world-size 1 \
  --max-running-prompts 2 \
  --default-max-tokens 128 \
  --disable-thinking \
  --port 8000
```

```bash
python examples/agentic/competition/eval_feedback.py \
  --dataset-path /tmp/areno-competition-eval.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model policy \
  --label before \
  --output-jsonl /kaggle/working/eval-before.jsonl \
  --report-md /kaggle/working/eval-before.md \
  --strip-reasoning \
  --no-think
```

After training, serve the saved checkpoint and repeat with `--label after`.
Compare the two Markdown reports and manually inspect several shared diary
records.

```bash
areno serve \
  --model-path /kaggle/working/areno-competition-checkpoints \
  --tp-size 1 \
  --world-size 1 \
  --max-running-prompts 2 \
  --default-max-tokens 128 \
  --disable-thinking \
  --port 8000
```

```bash
python examples/agentic/competition/eval_feedback.py \
  --dataset-path /tmp/areno-competition-eval.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model policy \
  --label after \
  --output-jsonl /kaggle/working/eval-after.jsonl \
  --report-md /kaggle/working/eval-after.md \
  --compare-jsonl /kaggle/working/eval-before.jsonl \
  --strip-reasoning \
  --no-think
```

For a fast Kaggle smoke check, evaluate only one or two records before running
the full set:

```bash
python examples/agentic/competition/eval_feedback.py \
  --dataset-path /tmp/areno-competition-eval.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model policy \
  --label after-smoke \
  --output-jsonl /kaggle/working/eval-after-smoke.jsonl \
  --report-md /kaggle/working/eval-after-smoke.md \
  --limit 2 \
  --candidates 1 \
  --max-tokens 96 \
  --request-timeout 180 \
  --strip-reasoning \
  --no-think
```

### 5. Serve a daily diary

After serving a trained checkpoint with `areno serve`, generate two competing
daily feedback candidates:

```bash
python examples/agentic/competition/serve_daily.py \
  --diary "今天跑通了GSPO训练，下午改简历改了三个小时。" \
  --mood "充实但有点累" \
  --output /tmp/areno-competition-daily.jsonl
```

## How It Works

Each training sample runs a 4-turn tool-call sequence:

1. `fetch_profile` - get user personality and preferences
2. `generate_content` - generate sandwich feedback for the diary entry
3. `self_score` - self-evaluate honestly (0-1)
4. `peer_score` - evaluate the opponent fairly (0-1)

The reward function combines:
- **User score (0.5)**: rule-based structure check + content relevance
- **Self score (0.2)**: agent's self-evaluation
- **Peer score (0.3)**: opponent's evaluation
- **Structure bonus (0.2)**: sandwich structure completeness
- **Calibration penalties**: self-eval deviation + peer lowballing

## Files

- `user_profile.json` - configurable user profile
- `game.py` - environment, tools, compute shares, simulated user scoring
- `dataset_generator.py` - synthetic diary generation
- `dataset_loader.py` - data loading with profile attachment
- `run_agent.py` - agent entrypoint with 4-turn tool-call sequence
- `reward.py` - three-dimensional scoring reward function
- `eval_feedback.py` - before/after evaluation helper
- `serve_daily.py` - deployment helper for daily diary feedback

## Kaggle Free GPU Notes

Use this example as a constrained validation run on free Kaggle GPUs. Prefer a
small model, one GPU, short generations, and a small number of saved checkpoints. If Kaggle
assigns a P100, run a real CUDA tensor operation before spending time on setup,
because some default images may expose CUDA but fail once PyTorch launches a
kernel for that GPU.

Reduce memory pressure in this order:

1. Lower `--max-new-tokens`.
2. Lower `--max-context-len`.
3. Keep `--batch-size 1` and `--n-samples 2`.
4. Keep `--mini-bs 1` and `--score-micro-bs 1`.
5. Use `--drop-rollout-state`.

For longer runs, increase `--save-interval` or remove older checkpoint
directories. Frequent saves can fill the free Kaggle working disk because each
Qwen3-0.6B checkpoint is large.

Training quality should be judged with a fixed before/after evaluation set, not
only with loss values. Useful signs include higher structure and relevance
scores, more concrete diary references, and suggestions that are specific enough
to act on the next day.

Some reasoning checkpoints, including Qwen3-style models, may emit
`<think>...</think>` traces instead of the final user-facing feedback. Use
`--strip-reasoning` and `--no-think` during evaluation so the report scores the
content intended for the user rather than private reasoning text.
