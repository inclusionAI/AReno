# Retro Terminal Hacking — Agentic RL

An original retro-futurist word-deduction environment inspired by classic CRT terminal minigames. It reproduces the interaction grammar—memory dumps, positional word matches, limited attempts, and bracket bonuses—without shipping game logos, text, or visual assets.

## How it works

- Each puzzle uses 16–20 same-length candidate words, three bracket probes, and at least 18 rows of memory dump.
- A wrong guess consumes one of three buffered attempts and returns `likeness`, the number of letters matching the password in the same positions.
- RLVR is a single-step contextual-bandit task: one compact state goes in and one `submit_candidates` set comes out. No game action is executed inside the training rollout.
- WebUI keeps the playable loop: it uses the exact RLVR candidate-filter prompt, randomly guesses one returned candidate, and uses a free probe at the last ambiguous attempt.
- Hidden passwords remain in `source_record`; prompts expose only observable guesses and likeness values.
- Reward is dense and asymmetric: extra inconsistent candidates are heavily penalized, while omitting consistent candidates receives only a small proportional penalty.

## Generate 8,192 unique puzzles

```bash
python examples/agentic/terminal_hacking/dataset_generator.py \
  --output /tmp/terminal_hacking.jsonl \
  --count 8192 \
  --workers 20 \
  --seed 2026
```

The generator uses unique per-record seeds and verifies complete puzzle fingerprints, so duplicate records are regenerated rather than merely receiving different IDs.

## Train

```bash
areno train \
  --ckpt inclusionai/ling-3.0-tiny \
  --dataset-path /tmp/terminal_hacking.jsonl \
  --dataset-loader-fn examples/agentic/terminal_hacking/dataset_loader.py \
  --reward-fn-path examples/agentic/terminal_hacking/reward.py \
  --agent-fn examples/agentic/terminal_hacking/run_agent.py \
  --algo gspo \
  --batch-size 4 \
  --n-samples 8 \
  --mini-bs 1 \
  --max-running-prompts 32 \
  --max-context-len 8192 \
  --max-prompt-tokens 4096 \
  --max-new-tokens 1024
```

RLVR and the WebUI candidate-selection call use the same helper in `game.py`: candidate-filter system prompt, immutable memory-dump prompt, compact current state, and one dynamic `submit_candidates` schema. RLVR produces exactly one completion and never runs an episode. WebUI keeps candidates internal and converts them into a random visible guess; probe handling remains part of the WebUI game controller.

## WebUI

Manual mode:

```bash
python examples/agentic/terminal_hacking/web_ui.py --port 8771
```

Agent mode against multiple OpenAI-compatible endpoints:

```bash
python examples/agentic/terminal_hacking/web_ui.py \
  --port 8771 \
  --base-url http://127.0.0.1:8000/v1 https://api.example.com/v1 \
  --api-key EMPTY "$SECOND_API_KEY" \
  --model policy second-model
```

The three lists are zipped by position; one API key or model value may be broadcast to every base URL. Open `http://127.0.0.1:8771` and choose the inference target from the model selector. The interface also supports English/Chinese help, green/amber phosphor themes, and an `ALGO` mode that needs no endpoint.

## Files

- `game.py` — deterministic rules, tool schema, replay, prompt rendering
- `dataset_generator.py` — parallel 8,192-record generator
- `dataset_loader.py` — validation and observable prompt construction
- `run_agent.py` — concurrent single-turn state-to-candidates rollout
- `reward.py` — dense asymmetric candidate-set reward
- `web_ui.py` — self-contained local CRT WebUI
