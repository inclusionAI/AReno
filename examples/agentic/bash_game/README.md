# Bash Game (巴什博弈 / 取石子游戏)

A single-turn, trainable strategy-game demo for AReno RLVR: learn the optimal
first move of the classic **take-away stones game**.

## Game rules

There is **one pile of `n` stones**. Two players alternate; on each turn a
player must remove between **1 and `m` stones**; the player who removes the
**last stone** wins.

The winning strategy is a pure **modular** rule (not an XOR / GF(2) system):

* `k = n mod (m + 1)`.
* If `k == 0`, the position is **losing** (a P-position) — the correct reply is
  to **resign** (no winning move exists).
* Otherwise the position is **winning** and the **unique** optimal move is to
  take exactly `k` stones, leaving a multiple of `m + 1` to the opponent.

This rule is periodic and local (it depends only on `n` and `m`), which is what
makes it *learnable* by a small LM — in contrast to Lights Out, whose optimal
solution requires global XOR/GF(2) cancellation that the model could not learn.

## Single-turn contract

```
state: (n, m)                     one pile of n stones, max take m
action: submit_move(take=k)       winning move,   k = n mod (m+1) ≠ 0
        submit_move(resign=true)  losing position, k = 0
```

The model emits exactly one tool call per position. There is no environment
loop, no second request, and no multi-turn trajectory.

## Files

| File | Role |
|------|------|
| `game.py` | Pure oracle (`BashGame`), prompt formatting, tool schema, and the `submit_move` parser shared by training + reward + eval + WebUI. |
| `dataset_generator.py` | Deterministic JSONL generator (disjoint train/val/test by split seeds + global `(n,m)` fingerprint dedup; always self-checks generated rows). |
| `dataset_loader.py` | Trainer adapter: attaches the prompt, keeps the oracle hidden in the record. |
| `reward.py` | Deterministic verifier: `1.0` exact, dense `closeness` for legal-but-wrong takes (winning positions), `0.0` for illegal / resign-when-won / take-when-lost. |
| `run_agent.py` | Single-tool-call rollout for training (no agent loop). |
| `eval.py` | Held-out evaluation against a served policy (mean reward, solve rate, per-`m` breakdown). |
| `web_ui.py` | Playable human-vs-agent WebUI (LLM or perfect oracle). |

## Rules sources (中文)

* 巴什博弈（取石子）—— CSDN 小K算法
  `https://blog.csdn.net/fhqfjevfhp/article/details/116957284`
* 博弈论：巴什博弈 —— CSDN
  `https://blog.csdn.net/qq_40788630/article/details/89364690`

Both describe the identical rule: single pile, take 1..m per turn, last-taker
wins, and the winning formula `n % (m+1)`.

## Generate data

```bash
python examples/agentic/bash_game/dataset_generator.py \
  --output /new/bash_data --split-all 400 48 48 \
  --seed 42 --n-max 100 --max-takes 2 3 4 5 6
```

## Train (RLVR: GSPO + adam4bit, deterministic reward)

Single-GPU (NVIDIA GB10): pin world/tp to device 0.

```bash
areno train \
  --ckpt /new/modelscope_cache/models/inclusionAI--Ling-3.0-tiny/snapshots/master \
  --algo gspo --adam-4bit --disable-thinking \
  --world-size 1 --tp-size 1 --train-devices 0 \
  --dataset-path /new/bash_data/train.jsonl \
  --dataset-loader-fn examples/agentic/bash_game/dataset_loader.py \
  --agent-fn examples/agentic/bash_game/run_agent.py \
  --reward-fn-path examples/agentic/bash_game/reward.py \
  --save-path /new/bash_ckpt --max-steps 120 --save-interval 20 \
  --batch-size 8 --n-samples 8 --mini-bs 1 --max-running-prompts 8 \
  --temperature 1.0 --max-new-tokens 48 --max-prompt-tokens 2048 \
  --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine \
  --metrics-log-dir /new/bash_tfevent
```

### Results

Held-out `test.jsonl` (48 positions, deterministic `--greedy` / temperature 1.0,
seed 7), evaluated with the exact same `reward_fn` used in training:

| checkpoint | mean_reward | solved_frac | legal_frac |
|------------|------------|-------------|------------|
| base (Ling-3.0-tiny) | 0.6858 | 0.4375 | 1.0 |
| **trained (step 40)** | **0.8556** | **0.6667** | 1.0 |

Per-`m` solve rate for the trained policy: `m=2,3,4 → 1.0`, `m=5 → 0.3636`,
`m=6 → 0.1000` (larger modulus = harder).

### Checkpoint note

`areno train` writes checkpoints (`/new/bash_ckpt/step_000040`) in its internal
`distributed_tp_incremental` sharding. Those load fine for *training* but
`areno serve`'s spawned worker can fail to map their weights. To serve/eval a
trained checkpoint, re-pack it into standard HF sharded format first (the base
model uses `model-XXXX-of-0032.safetensors` + a plain `model.safetensors.index.json`
without the `areno_checkpoint_writer` marker). A one-off repack script (iterate
`weight_map`, `save_file` into 32 balanced shards, emit a clean index) is all
that is required.

## Evaluate / serve / play

```bash
# serve the trained checkpoint
areno serve ... --port 8000 --max-running-prompts 2 --disable-thinking

# held-out eval
python examples/agentic/bash_game/eval.py \
  --dataset /new/bash_data/test.jsonl \
  --base-url http://127.0.0.1:8000/v1 --api-key token --model policy \
  --out /new/eval_bash_test.json

# play in the browser (human vs agent)
python examples/agentic/bash_game/web_ui.py \
  --base-url http://127.0.0.1:8000/v1 --api-key token --port 8001
```
