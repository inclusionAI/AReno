# Agentic Codebreaker TUI

Codebreaker is a deterministic multi-turn Bulls and Cows game. A hidden code
contains four distinct digits and may begin with zero. The policy calls
`guess_code` up to six times; each tool result reports `exact` digits in the
right position and `present` digits in the wrong position. The secret never
appears in the prompt or tool result.

This differs from the other agentic examples: it is a partial-observability
deduction game with state accumulated exclusively through tool results. Missing,
malformed, repeated, and invalid guesses remain failures rather than being
rewritten into successful calls.

## Play in the terminal

```bash
python examples/agentic/codebreaker/tui.py --seed 7
```

Point the same TUI at an OpenAI-compatible inference endpoint to let a model
play. It preserves the complete assistant tool-call and tool-result history:

```bash
python examples/agentic/codebreaker/tui.py --agent --seed 7 \
  --base-url http://127.0.0.1:8000/v1 --model policy --api-key token
```

## Generate data

```bash
python examples/agentic/codebreaker/dataset_generator.py \
  --output /tmp/codebreaker.jsonl --count 256 --seed 2026
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --dataset-path /tmp/codebreaker.jsonl \
  --dataset-loader-fn examples/agentic/codebreaker/dataset_loader.py \
  --reward-fn-path examples/agentic/codebreaker/reward.py \
  --agent-fn examples/agentic/codebreaker/run_agent.py \
  --algo gspo --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --max-new-tokens 64
```
