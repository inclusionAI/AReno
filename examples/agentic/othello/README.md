# Agentic 6x6 Othello Example

This example trains a policy to choose one Othello move from a rendered 6x6
board. The environment implements legal-move enumeration with 8-direction
flipping, pass handling, double-pass termination, and terminal disc scoring.
It is deterministic and self-contained with no external dependencies beyond
AReno's existing contracts.

## Files

- `game.py` contains board validation, 8-direction flip logic, legal-move
  enumeration, pass handling, terminal scoring, and move evaluation.
- `dataset_generator.py` generates reproducible reachable opening positions
  from the standard 6x6 Othello starting board.
- `dataset_loader.py`, `run_agent.py`, and `reward.py` define the tool-call
  agentic variant using AReno's file-callback contracts.

## Generate Boards

```bash
python examples/agentic/othello/dataset_generator.py \
  --output /tmp/areno-othello-boards.jsonl \
  --count 2048 \
  --seed 2026
```

## Train with Tool Calls

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-othello-boards.jsonl \
  --dataset-loader-fn examples/agentic/othello/dataset_loader.py \
  --reward-fn-path examples/agentic/othello/reward.py \
  --agent-fn examples/agentic/othello/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 64
```

## Observable Output

- **Reward**: -1.0 for illegal moves, 0.0 for legal non-terminal moves, 1.0
  for a move that ends the game with the player having more discs.
- **Win rate and illegal-move rate**: computed by playing full episodes
  against a seeded random opponent (see `game.play_episode` and the test
  suite).
- **Training logs and metrics**: surfaced through AReno's existing logging,
  TensorBoard, and dashboard infrastructure.

## Minimal Runnable Example

```bash
# Generate a small dataset
python examples/agentic/othello/dataset_generator.py \
  --output /tmp/othello-demo.jsonl --count 16 --seed 2026

# Verify the generator output
head -3 /tmp/othello-demo.jsonl

# Run CPU tests
python -m pytest tests/test_agentic_othello_example_cpu.py -v
```