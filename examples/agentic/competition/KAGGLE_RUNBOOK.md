# Kaggle Free GPU Runbook

This runbook is for a constrained free Kaggle session. The goal is to collect
before/after evidence, not to maximize training scale.

## 0. Environment Check

Run this before training:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    x = torch.ones((1024, 1024), device="cuda")
    print("cuda sum", x.sum().item())
PY
```

If the CUDA tensor operation fails, do not start training in that session.

## 1. Data

```bash
python examples/agentic/competition/dataset_generator.py \
  --output /tmp/areno-competition-train.jsonl \
  --count 64 \
  --seed 2026

python examples/agentic/competition/dataset_generator.py \
  --output /tmp/areno-competition-eval.jsonl \
  --count 16 \
  --seed 9001
```

## 2. Baseline Evaluation

Serve the base checkpoint, then run:

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

## 3. Smoke Training

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --world-size 1 \
  --algo gspo \
  --tp-size 1 \
  --model-hub modelscope \
  --dataset-path /tmp/areno-competition-train.jsonl \
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

## 4. Main Training

Only run this after the smoke run succeeds. A 30-step run is a practical first
target on a free T4 session because it validates training, checkpointing,
serving, and evaluation without requiring a long uninterrupted notebook run:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --world-size 1 \
  --algo gspo \
  --tp-size 1 \
  --model-hub modelscope \
  --dataset-path /tmp/areno-competition-train.jsonl \
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
  --save-path /kaggle/working/areno-competition-30step-v1
```

If memory is stable and the 30-step report looks healthy, increase
`--max-steps` in a later run, but increase `--save-interval` too. Qwen3-0.6B
checkpoints can be around 1.5 GB each, so frequent checkpointing may fill the
free Kaggle working disk before training finishes.

For a longer 100-step validation run, prefer a wider checkpoint interval:

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --world-size 1 \
  --algo gspo \
  --tp-size 1 \
  --model-hub modelscope \
  --dataset-path /tmp/areno-competition-train.jsonl \
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
  --max-steps 100 \
  --save-interval 50 \
  --drop-rollout-state \
  --save-path /kaggle/working/areno-competition-100step-v1
```

## 5. Save The Checkpoint

Kaggle sessions can lose `/kaggle/working` after a restart. Save the latest
checkpoint before leaving the session:

```bash
tar -czf /kaggle/working/areno-competition-30step-v1-step30.tar.gz \
  -C /kaggle/working/areno-competition-30step-v1 step_000030

ls -lh /kaggle/working/areno-competition-30step-v1-step30.tar.gz
```

Download the `.tar.gz` from the notebook output panel or create a notebook
version after the archive appears. `Save Version` preserves outputs for a
version, but it does not recreate a live runtime with the old `/kaggle/working`
state.

## 6. After Evaluation

Serve the trained checkpoint, then run:

```bash
areno serve \
  --model-path /kaggle/working/areno-competition-30step-v1/step_000030 \
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

If the full evaluation is slow on a free T4 session, first run a tiny smoke
evaluation:

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

Kaggle notebooks may reject shell background processes such as
`!nohup areno serve ... &`. Start `areno serve` from Python instead:

```python
import subprocess, time

subprocess.run("pkill -f 'areno serve' || true", shell=True)

log_path = "/kaggle/working/serve-after-30step.log"
log_file = open(log_path, "w")
cmd = [
    "areno", "serve",
    "--model-path", "/kaggle/working/areno-competition-30step-v1/step_000030",
    "--tp-size", "1",
    "--world-size", "1",
    "--max-running-prompts", "2",
    "--default-max-tokens", "64",
    "--disable-thinking",
    "--port", "8000",
]
proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
print("serve pid:", proc.pid)
time.sleep(30)
print(open(log_path).read()[-4000:])
```

## 7. What To Save

- exact training command
- GPU type
- environment check output
- smoke run result
- main training logs
- checkpoint path
- `eval-before.md`
- `eval-after.md`
- 5 before/after examples you can explain

## 8. Disk Cleanup

If a longer run stops while saving a checkpoint with `No space left on device`,
keep the latest complete checkpoint and remove partial or older checkpoint
directories. For example, if `step_000070` failed while saving and
`step_000060` is complete:

```bash
rm -rf /kaggle/working/areno-competition-100step-v1/step_000070
rm -rf /kaggle/working/areno-competition-100step-v1/step_000010
rm -rf /kaggle/working/areno-competition-100step-v1/step_000020
rm -rf /kaggle/working/areno-competition-100step-v1/step_000030
rm -rf /kaggle/working/areno-competition-100step-v1/step_000040
rm -rf /kaggle/working/areno-competition-100step-v1/step_000050
```

Archive the checkpoint you want to preserve before ending the session:

```bash
tar -czf /kaggle/working/areno-competition-100step-v1-step60.tar.gz \
  -C /kaggle/working/areno-competition-100step-v1 step_000060
```
