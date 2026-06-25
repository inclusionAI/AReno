# Agentic Tic-Tac-Toe Example

This example trains a policy to choose one Tic-Tac-Toe move for `X` from a
rendered board. It includes both an OpenAI tool-call variant and an XML no-tool
variant. The environment is deterministic and self-contained.

## Files

- `dataset_generator.py` generates reproducible Tic-Tac-Toe board JSONL.
- `dataset_loader.py`, `run_agent.py`, and `reward.py` define the tool-call
  variant.
- `dataset_loader_no_tool.py`, `run_agent_no_tool.py`, and
  `reward_no_tool.py` define the XML no-tool variant.
- `game.py` contains board validation, minimax best moves, and scoring.
- `web_ui.py` serves a cartoon browser game backed by an OpenAI-compatible
  tool-call model.

## Generate Boards

```bash
python examples/agentic/tictactoe/dataset_generator.py \
  --output /tmp/areno-tictactoe-boards.jsonl \
  --count 2048 \
  --seed 2026
```

## Run with Tool Calls

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-tictactoe-boards.jsonl \
  --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
  --reward-fn-path examples/agentic/tictactoe/reward.py \
  --agent-fn examples/agentic/tictactoe/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 32
```

## Play in the Web UI

Start a policy server, then point the UI at its OpenAI-compatible endpoint:

```bash
python examples/agentic/tictactoe/web_ui.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key token \
  --model policy
```

Open `http://127.0.0.1:8767`. The UI supports mouse clicks, keyboard squares
`1`-`9`, and switching whether the LLM or player moves first.

## Run without Tool Calls

The XML no-tool variant asks the model to answer with a move tag such as
`<move>5</move>`.

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-tictactoe-boards.jsonl \
  --dataset-loader-fn examples/agentic/tictactoe/dataset_loader_no_tool.py \
  --reward-fn-path examples/agentic/tictactoe/reward_no_tool.py \
  --agent-fn examples/agentic/tictactoe/run_agent_no_tool.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 64
```
