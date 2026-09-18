"""Editable Hugging Face starting checkpoints for repository-discovered adapters.

This is a suggestion table, not a second model registry. Unknown new adapters
remain visible and require a checkpoint until a recommendation is added here.
"""

CHECKPOINTS = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "qwen3_5": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_vl": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_moe": "Qwen/Qwen3.5-35B-A3B",
    "qwen3_5_vl_moe": "Qwen/Qwen3.5-35B-A3B",
    "gemma4": "google/gemma-4-E2B-it",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "minicpmv46": "openbmb/MiniCPM-V-4.6",
    "olmo2": "allenai/OLMo-2-0425-1B-Instruct",
    "phi4mm": "microsoft/Phi-4-multimodal-instruct",
    "bailing_moe_v3": "inclusionAI/Ling-3.0-tiny",
}
