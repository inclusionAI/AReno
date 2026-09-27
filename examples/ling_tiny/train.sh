#!/usr/bin/env bash
# Post-training recipes for inclusionAI/Ling-3.0-tiny (bailing_hybrid, MoE).
#
#   bash examples/ling_tiny/train.sh <recipe> [extra areno flags...]
#
# Recipes: lora-sft | full-sft | dpo | gspo | classify | serve
# Every recipe reads its inputs from environment variables (see README.md);
# extra arguments are appended to the underlying command and override defaults.
set -euo pipefail

RECIPE="${1:-}"
[[ $# -gt 0 ]] && shift

MODEL="${MODEL:-inclusionai/ling-3.0-tiny}"
MODEL_HUB="${MODEL_HUB:-modelscope}"
TP_SIZE="${TP_SIZE:-1}"
WORLD_SIZE="${WORLD_SIZE:-1}"
SAVE_PATH="${SAVE_PATH:-outputs/ling-tiny-${RECIPE}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

require() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "error: recipe '${RECIPE}' needs ${name}=..." >&2
      exit 2
    fi
  done
}

loader_args=()
if [[ -n "${LOADER:-}" ]]; then
  loader_args=(--dataset-loader-fn "${LOADER}")
fi

common=(
  --ckpt "${MODEL}" --model-hub "${MODEL_HUB}"
  --tp-size "${TP_SIZE}" --world-size "${WORLD_SIZE}"
  --save-path "${SAVE_PATH}" --save-interval "${SAVE_INTERVAL}"
  --disable-thinking --activation-checkpointing
)

case "${RECIPE}" in
  lora-sft)
    # Rows: {"prompt": ..., "response": ...}; the prompt gets Ling's chat template.
    require DATASET
    exec areno train --algo sft "${common[@]}" \
      --dataset-path "${DATASET}" ${loader_args[@]+"${loader_args[@]}"} \
      --batch-size "${BATCH_SIZE:-8}" --mini-bs "${MINI_BS:-4}" --epochs "${EPOCHS:-3}" \
      --max-prompt-tokens "${MAX_PROMPT_TOKENS:-1024}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
      --adam-4bit --lr "${LR:-1e-4}" --min-lr "${MIN_LR:-1e-5}" \
      --lora-rank "${LORA_RANK:-16}" \
      "$@"
    ;;
  full-sft)
    # Updates every weight, including all 128 routed experts; optimizer state is
    # the dominant memory cost, so keep 4-bit Adam and offload it to host memory.
    require DATASET
    exec areno train --algo sft "${common[@]}" \
      --dataset-path "${DATASET}" ${loader_args[@]+"${loader_args[@]}"} \
      --batch-size "${BATCH_SIZE:-8}" --mini-bs "${MINI_BS:-2}" --epochs "${EPOCHS:-2}" \
      --max-prompt-tokens "${MAX_PROMPT_TOKENS:-1024}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
      --adam-4bit --optimizer-state-offload "${OPTIMIZER_OFFLOAD:-cpu}" \
      --lr "${LR:-1e-5}" --min-lr "${MIN_LR:-1e-6}" \
      "$@"
    ;;
  dpo)
    # Rows: {"prompt": ..., "chosen": ..., "rejected": ...}. With LoRA the frozen
    # base doubles as the reference model, so no second copy is loaded.
    require DATASET
    exec areno train --algo dpo "${common[@]}" \
      --dataset-path "${DATASET}" ${loader_args[@]+"${loader_args[@]}"} \
      --batch-size "${BATCH_SIZE:-8}" --mini-bs "${MINI_BS:-2}" --epochs "${EPOCHS:-1}" \
      --max-prompt-tokens "${MAX_PROMPT_TOKENS:-1024}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
      --adam-4bit --lr "${LR:-5e-6}" --min-lr "${MIN_LR:-5e-7}" \
      --dpo-beta "${DPO_BETA:-0.1}" \
      --lora-rank "${LORA_RANK:-16}" --reference-mode reuse_actor_base \
      "$@"
    ;;
  gspo)
    # RLVR / agentic RL. AGENT is optional (single-turn tasks omit it).
    require DATASET REWARD
    agent_args=()
    if [[ -n "${AGENT:-}" ]]; then
      agent_args=(--agent-fn "${AGENT}")
    fi
    exec areno train --algo gspo "${common[@]}" \
      --dataset-path "${DATASET}" ${loader_args[@]+"${loader_args[@]}"} \
      --reward-fn-path "${REWARD}" ${agent_args[@]+"${agent_args[@]}"} \
      --batch-size "${BATCH_SIZE:-8}" --n-samples "${N_SAMPLES:-8}" --mini-bs "${MINI_BS:-1}" \
      --max-running-prompts "${MAX_RUNNING_PROMPTS:-64}" \
      --max-prompt-tokens "${MAX_PROMPT_TOKENS:-2048}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
      --adam-4bit --lr "${LR:-1e-6}" --min-lr "${MIN_LR:-1e-7}" \
      "$@"
    ;;
  classify)
    # Grouped-softmax scorer (JevForge records); full-parameter, no LoRA.
    require RECORDS
    exec python examples/classify/jev/train.py \
      --records "${RECORDS}" --ckpt "${MODEL}" --model-hub "${MODEL_HUB}" \
      --save-path "${SAVE_PATH}" --save-interval "${SAVE_INTERVAL}" \
      --world-size "${WORLD_SIZE}" --tp-size "${TP_SIZE}" \
      "$@"
    ;;
  serve)
    # ADAPTER: a LoRA step_* directory. For full-parameter runs point MODEL at
    # the checkpoint directory instead.
    adapter_args=()
    if [[ -n "${ADAPTER:-}" ]]; then
      adapter_args=(--lora-adapter-path "${ADAPTER}")
    fi
    exec areno serve --model-path "${MODEL}" --model-hub "${MODEL_HUB}" \
      ${adapter_args[@]+"${adapter_args[@]}"} --disable-thinking \
      --tp-size "${TP_SIZE}" --world-size "${WORLD_SIZE}" --port "${PORT:-8000}" \
      "$@"
    ;;
  *)
    echo "usage: $0 {lora-sft|full-sft|dpo|gspo|classify|serve} [extra flags...]" >&2
    exit 2
    ;;
esac
