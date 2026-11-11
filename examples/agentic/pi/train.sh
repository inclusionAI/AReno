#!/usr/bin/env bash
# Supply --ckpt, --dataset-path, --save-path and model-specific options as arguments.
# Docker: ARENO_HARBOR_ENVIRONMENT=docker ARENO_HARBOR_IMAGE=<preinstalled-pi-image>
# Local: ARENO_HARBOR_ENVIRONMENT=local (Pi, Node and Chromium on the host).
# Extra Harbor settings: ARENO_HARBOR_ENVIRONMENT_OPTIONS='{"task_env": {...}}'.
# The controller reads ARENO_VLM_BASE_URL, ARENO_VLM_MODEL and ARENO_VLM_API_KEY.
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$example_dir/../../.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export ARENO_HARBOR_ENVIRONMENT="${ARENO_HARBOR_ENVIRONMENT:-docker}"

exec python -m areno.cli.main train \
    --model-hub modelscope --algo gspo \
    --dataset-loader-fn "$example_dir/dataset_loader.py" \
    --agent-fn "$example_dir/run_agent.py" \
    --reward-fn-path "$example_dir/reward.py" \
    --batch-size 4 --n-samples 8 --max-running-prompts 32 \
    --max-context-len 16384 --max-prompt-tokens 12288 --max-new-tokens 4096 \
    --mini-bs 1 --score-micro-bs 1 --disable-thinking --drop-rollout-state \
    --max-steps 1000 --save-interval 20 --mem-frac 0.7 \
    "$@"
