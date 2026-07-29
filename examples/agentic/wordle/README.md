# Wordle Agentic RL Demo

This example demonstrates how to use AReno to train a local LLM to play Wordle through reinforcement learning.

## What is Wordle?

Wordle is a word guessing game where:
- You have 6 attempts to guess a 5-letter hidden word
- After each guess, you receive feedback for each letter:
  - **EXACT** (green): Correct letter in correct position
  - **PRESENT** (yellow): Correct letter in wrong position
  - **ABSENT** (gray): Letter not in the word
- Repeated letters are handled correctly per Wordle rules

## Project Structure

```
wordle/
├── game.py                 # Core Wordle game logic
├── dataset_generator.py    # Generates Wordle game datasets
├── dataset_loader.py       # Loads datasets for AReno training
├── run_agent.py            # Agent entrypoint for tool-call rollouts
├── reward.py               # Reward function for RL training
├── tests/
│   ├── __init__.py
│   └── test_game.py        # Unit tests for game logic
└── README.md               # This file
```

## Quick Start

### Prerequisites

```bash
pip install openai httpx pytest
```

### Running Tests

```bash
cd examples/agentic/wordle
python -m pytest tests/ -v
```

### Running the Demo

1. First, generate game data:

```bash
python -m dataset_generator --count 1024 --output /tmp/areno-wordle.jsonl
```

2. Train with AReno (requires GPU + AReno installed):

```bash
areno train \
--save-path /tmp/areno-wordle-ckpt \
--save-interval 10 \
--ckpt Qwen/Qwen3-1.7B \
--adam-8bit \
--model-hub hf \
--world-size 2 \
--tp-size 2 \
--algo gspo \
--batch-size 2 \
--n-samples 8 \
--drop-rollout-state \
--max-running-prompts 16 \
--max-new-tokens 256 \
--mini-bs 1 \
--max-steps 200 \
--dataset-path /tmp/areno-wordle.jsonl \
--reward-fn-path examples/agentic/wordle/reward.py \
--dataset-loader-fn examples/agentic/wordle/dataset_loader.py \
--agent-fn examples/agentic/wordle/run_agent.py \
--lr 1e-7 \
--grad-clip-norm 0.5
```

### Evaluation

Run a deterministic evaluation to report solve rate and average guesses:

```bash
python -m evaluate --dataset /tmp/areno-wordle.jsonl
```

Output example:
```
Wordle Statistics
========================================
Overall Solve Rate: 85.2%
Overall Avg Guesses (when solved): 4.12

By Word Length:
  5-letter words: 85.2% solved, avg 4.12 guesses (872/1024)
```

## Word List

The word list contains common 5-letter English words, split into:
- **Target words**: ~500 common words that can be the hidden word
- **Valid guess words**: ~1000 words that can be used as guesses

The word list is MIT licensed for redistribution.

## Reward Function

The reward function provides:
- **+1.0 to +1.5**: Correctly guessed the word (higher reward for fewer guesses)
- **+0.5 to +0.7**: Made progress (at least one letter in correct position)
- **+0.2 to +0.3**: Valid word but no correct letters
- **0.0**: Lost the game (exhausted all 6 guesses)
- **-1.0**: Invalid word (not in word list)

## AReno Integration

This example uses the AReno agentic RL framework:

1. **dataset_loader.py**: Loads Wordle games and converts them to AReno prompt records
2. **run_agent.py**: Implements the agent that calls the `guess_word` tool
3. **reward.py**: Scores each agent response based on game outcome

### Training with AReno

```yaml
# Example AReno config
datasets:
  - path: ./wordle/games.jsonl
    loader: wordle.dataset_loader.load_training_dataset

agent:
  entry: wordle.run_agent.run_agent
  reward_fn: wordle.reward.reward_fn

model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
```

## Key Features

- **No network access at runtime**: All data is bundled locally
- **Deterministic evaluation**: Same input always produces same output
- **CPU-friendly tests**: Core game logic runs without GPU
- **Proper repeated letter handling**: Correctly handles edge cases like double letters

## Documentation

For more information about AReno, see:
- [AReno Documentation](https://github.com/inclusionAI/AReno)
- [Agentic RL Examples](./tictactoe)

## License

This example is MIT licensed, same as the AReno project.