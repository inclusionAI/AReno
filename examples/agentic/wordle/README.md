# Agentic Wordle

Wordle is a deterministic multi-turn word-guessing game. A hidden 5-letter
English word must be guessed within 6 attempts. The policy calls `guess_word`
up to six times; each tool result reports per-position feedback:

- `exact`: correct letter in correct position
- `present`: letter exists in the word but at a different position
- `absent`: letter not in the word

The secret never appears in the prompt or tool result. Repeated letters are
handled according to standard Wordle counting rules (each secret letter can
only match once). The bundled word list is public domain.

This differs from the Codebreaker example: Wordle allows repeated letters in
the secret (e.g. "eerie", "llama"), requiring quota-based feedback instead of
simple set intersection. The feedback is a per-position list of `exact`,
`present`, and `absent` rather than integer counts.

## Generate data

```bash
python examples/agentic/wordle/dataset_generator.py \
  --output /tmp/wordle.jsonl --count 256 --seed 2026
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --dataset-path /tmp/wordle.jsonl \
  --dataset-loader-fn examples/agentic/wordle/dataset_loader.py \
  --reward-fn-path examples/agentic/wordle/reward.py \
  --agent-fn examples/agentic/wordle/run_agent.py \
  --algo gspo --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --max-new-tokens 64
```

## Observable outputs

After training, each trajectory contains:

- The full multi-turn message history (system + user prompts + assistant tool calls + tool results)
- Per-turn `response_tokens` and `response_logprobs` extracted from model outputs
- `parsed_tool_calls` for each turn where `guess_word` was invoked
- `score_episode` reward: `1.0` for first-guess solve, `0.8 + 0.2 * efficiency` for
  later solves, `0.1 * best_info / word_length` for partial progress, `-1.0` for
  invalid guesses, `-0.5` for repeated guesses

Use `evaluate_wordle` to report deterministic metrics:

```python
from game import evaluate_wordle

result = evaluate_wordle("eerie", ["eerie"])
# {"solved": True, "guesses_to_solve": 1, "word_length": 5}
```

