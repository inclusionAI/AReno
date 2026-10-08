# Kaggle 30-Step Findings

This note records a small free-GPU validation run for the competition agentic
example. It is intended as contributor evidence, not as a claim of final model
quality.

## Setup

- Platform: Kaggle notebook
- GPU: Tesla T4
- Model: `Qwen/Qwen3-0.6B`
- Training steps: 30
- Save interval: 5
- Checkpoint: `/kaggle/working/areno-competition-30step-v1/step_000030`
- Serving: `areno serve` on port 8000

## What Worked

- Training completed and produced checkpoints.
- The `step_000030` checkpoint loaded with `areno serve`.
- The OpenAI-compatible `/v1/chat/completions` endpoint returned HTTP 200.
- A smoke evaluation with two records completed without endpoint errors.
- A later longer run reached a complete `step_000060` checkpoint and that
  checkpoint also served successfully through the OpenAI-compatible API.

## Observations

The model often emitted Qwen-style reasoning traces such as
`<think>...</think>` instead of only the user-facing feedback. In the smoke
evaluation this reduced relevance and structure scores because the evaluator
scored reasoning text rather than final feedback content.

This is a useful diagnostic result: the training, serving, and evaluation loop
works, but the evaluation helper should make reasoning-model behavior easier to
handle on constrained Kaggle runs.

The longer run also exposed a Kaggle storage limitation. Saving checkpoints too
frequently can fill `/kaggle/working`; in one run, checkpoint saving failed at
`step_000070` with `No space left on device` while `step_000060` remained usable.
For free-GPU sessions, longer runs should use a wider `--save-interval` or clean
up older checkpoint directories after archiving the latest complete checkpoint.

## Follow-Up Contribution

The evaluator should support:

- tiny smoke runs with `--limit`
- shorter generations with `--max-tokens`
- bounded endpoint waits with `--request-timeout`
- reasoning trace cleanup with `--strip-reasoning`
- optional no-think prompting with `--no-think`

These options make the example easier to reproduce for contributors using free
Kaggle GPUs.
