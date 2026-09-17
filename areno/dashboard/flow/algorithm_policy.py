"""Algorithm applicability derived from the CLI's actual config construction.

Only semantic exceptions (fields shared by config classes but unused by a loss)
are declared here. New CLI parameters remain discoverable automatically.
"""

from __future__ import annotations

import ast
from pathlib import Path

CONFIG_ALGORITHMS = {
    "TrainerConfig": {"sft"},
    "DPOTrainerConfig": {"dpo"},
    "PolicyTrainerConfig": {"gspo", "grpo"},
    "PPOTrainerConfig": {"ppo"},
}
OBJECTIVES = {
    "gspo_clip_eps": {"gspo"},
    "grpo_clip_eps": {"grpo"},
    "dpo_beta": {"dpo"},
    "ref_ckpt": {"dpo", "ppo"},
    "reference_mode": {"dpo", "gspo", "grpo", "ppo"},
}
ROLLOUT_ONLY = {"agent_fn", "train_tool_results", "drop_rollout_state", "eager_decode"}


def annotate_options(schema: list[dict], source: Path) -> list[dict]:
    tree = ast.parse(source.read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_trainer_config_from_args")
    consumed = {name: set() for name in CONFIG_ALGORITHMS}
    for node in ast.walk(builder):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in consumed:
            consumed[node.func.id].update(
                n.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args"
            )
    all_algorithms = set().union(*CONFIG_ALGORITHMS.values())
    mapped = set().union(*consumed.values())
    for option in schema:
        name = option["name"]
        supported = set().union(*(CONFIG_ALGORITHMS[c] for c, fields in consumed.items() if name in fields))
        # CLI-level controls, e.g. LoRA construction and tuning, are outside config calls.
        if name not in mapped:
            supported = all_algorithms.copy()
        if name in OBJECTIVES:
            supported &= OBJECTIVES[name]
        if name in ROLLOUT_ONLY:
            supported &= {"gspo", "grpo", "ppo"}
        option["algorithms"] = sorted(supported)
        if supported == {"ppo"}:
            option["group"] = "PPO objective & roles"
        elif name in ("gspo_clip_eps", "grpo_clip_eps", "dpo_beta"):
            option["group"] = f"{next(iter(supported)).upper()} objective"
        elif name in ("reward_fn_path", "reward_ckpt"):
            option["group"] = "Reward"
        elif option["group"] == "Rollout" and not supported <= {"gspo", "grpo", "ppo"}:
            option["group"] = "Data & token limits"
    return schema
