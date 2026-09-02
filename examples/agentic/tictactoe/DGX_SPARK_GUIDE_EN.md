# Training Ling 3.0 Tiny to Play Tic-Tac-Toe with AReno on DGX Spark

**English** | [中文](DGX_SPARK_GUIDE_CN.md)

This guide shows how to use AReno and `inclusionAI/Ling-3.0-tiny` on NVIDIA DGX Spark to train a model that plays Tic-Tac-Toe through tool calls with GSPO reinforcement learning.

## Prepare the Environment

We recommend running AReno in Docker:

```bash
git clone https://github.com/inclusionAI/AReno
cd AReno
docker build -t areno:latest .
```

## Start the Docker Container

```bash
docker run -d \
  --name areno \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  areno:latest \
  sleep infinity
```

Enter the container and switch to the AReno project directory:

```bash
docker exec -it areno bash
cd /workspace/AReno
```

The exact project path depends on the working directory configured by the Dockerfile.

## Generate the Tic-Tac-Toe Dataset

```bash
python examples/agentic/tictactoe/dataset_generator.py \
  --count 2048 \
  --output ./tictactoe_boards.jsonl
```

The generator creates data as follows:

1. Create an empty 3x3 board, using `"."` for empty squares.
2. Start with X and play between 0 and 6 random moves.
3. Each move must select a currently legal empty square.
4. Stop as soon as the game reaches a terminal state.
5. Keep only unique, non-terminal positions where X is the next player.
6. Use a fixed random seed so the same arguments reproduce the same dataset.
7. If the random sequence ends with O to move, add one random O move so that X moves next.

One JSONL record looks like this:

```json
{"id":"generated-00000","board":[["X",".","O"],[".",".","."],["X","O","."]]}
```

The raw JSONL stores only board positions. During training, `dataset_loader.py` converts each board into a prompt and computes its legal and minimax-optimal moves.

## Reward Design

The task uses a direct reward function:

- No valid `choose_square` tool call: `-1.0`
- Unparseable arguments or an occupied square: `-1.0`
- The move immediately wins for X: `1.0`
- The move is minimax-optimal but does not win immediately: `0.8`
- The move is legal but suboptimal: `0.0`

This reward trains three core behaviors:

- Produce actions using the correct tool-call format.
- Respect the board and never select occupied squares.
- Choose stronger moves after action validity has been learned.

This example demonstrates AReno's intended post-training workflow: start with a bounded task, define its environment and reward, run rollout and training, and then observe how model behavior changes.

## Run Training

First configure the checkpoint output directory:

```bash
export SAVE_PATH=/workspace/checkpoints/tictactoe
mkdir -p "$SAVE_PATH"
```

Start training:

```bash
areno train \
  --algo gspo \
  --mini-bs 1 \
  --ckpt inclusionAI/Ling-3.0-tiny \
  --dataset-path tictactoe_boards.jsonl \
  --dataset-loader-fn examples/agentic/tictactoe/dataset_loader.py \
  --reward-fn-path examples/agentic/tictactoe/reward.py \
  --agent-fn examples/agentic/tictactoe/run_agent.py \
  --save-interval 100 \
  --save-path "$SAVE_PATH" \
  --world-size 1 \
  --tp-size 1 \
  --drop-rollout-state \
  --batch-size 4 \
  --n-samples 4 \
  --max-running-prompts 16 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 1471 \
  --adam-8bit \
  --lr 0.00000001 \
  --min-lr 0.000000001
```

This command trains Ling 3.0 Tiny with GSPO on one GPU. The model receives a board and must call `choose_square` to select its next move.

### Important Parameters

- `--algo gspo`: Use GSPO to update the policy from rollout rewards.
- `--ckpt inclusionAI/Ling-3.0-tiny`: Select the initial model checkpoint.
- `--dataset-path`: Select the JSONL file containing initial board states.
- `--dataset-loader-fn`: Convert boards into prompts and compute legal and minimax-optimal actions.
- `--agent-fn`: Request one `choose_square` tool call for each sample.
- `--reward-fn-path`: Extract and score the selected square from the tool call.
- `--batch-size 4`: Read four distinct board prompts per training step.
- `--n-samples 4`: Sample four responses for each board, producing `4 x 4 = 16` rollouts per step.
- `--max-running-prompts 16`: Process up to 16 generation requests concurrently, covering the entire rollout batch.
- `--mini-bs 1`: Train on one rollout per microbatch. This lowers memory use but increases the number of training passes.
- `--max-prompt-tokens 1024`: Set the maximum prompt length. Tic-Tac-Toe prompts are normally much shorter.
- `--max-new-tokens 1471`: Set the maximum generated length. Since this task needs one tool call, reducing this to `64` to `256` can lower cost when the model behaves reliably.
- `--world-size 1 --tp-size 1`: Use one GPU without tensor parallelism.
- `--drop-rollout-state`: Release rollout state after every step, reducing persistent memory at the cost of rebuilding it next step.
- `--adam-8bit`: Store Adam optimizer states in 8-bit form to reduce memory use.
- `--lr 0.00000001`: Set the peak learning rate to `1e-8`.
- `--min-lr 0.000000001`: Set the minimum learning rate to `1e-9`.
- `--save-interval 100`: Save a checkpoint every 100 training steps.
- `--save-path "$SAVE_PATH"`: Select the checkpoint output directory.

In each step, four boards produce 16 candidate trajectories. AReno scores the selected squares and applies GSPO updates with `mini_bs=1`.

## Monitor Training with the Dashboard

```bash
areno dashboard --start --host 0.0.0.0 --port 8000
```

Open port `8000` on the DGX Spark in a browser:

![AReno dashboard](assets/dashboard.jpg)

## Evaluate the Model

Start an inference service using a saved checkpoint:

```bash
areno serve \
  --model-path /new_tiny/step_000400/ \
  --max-running-prompts 1 \
  --default-max-tokens 4096 \
  --port 8001
```

Replace `/new_tiny/step_000400/` with the actual saved checkpoint path.

Start the Tic-Tac-Toe Web UI:

```bash
python examples/agentic/tictactoe/web_ui.py \
  --host 0.0.0.0 \
  --port 8002 \
  --agent-mode llm \
  --base-url http://localhost:8001/v1 \
  --api-key xx \
  --model ling
```

Open port `8002` on the DGX Spark to play against the model:

![Tic-Tac-Toe Web UI](assets/tictactoe.jpg)
