from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CODING_DIR = ROOT / "examples" / "agentic" / "coding"
if str(CODING_DIR) not in sys.path:
    sys.path.insert(0, str(CODING_DIR))

from areno_target_reward import reward_fn  # noqa: E402


REAL_LOG_WITH_REWARD_UP = """
loading checkpoint: 100%|██████████| 33/33 [00:03<00:00, 10.80stage/s]
2026-06-29 16:55:45 INFO areno.api.trainers.policy_only.PolicyOnlyTrainer policy_only.py:76 - epoch=0 step=0 role=policy stage=rollout_start
2026-06-29 16:55:45 INFO areno.api.trainers.policy_only.PolicyOnlyTrainer policy_only.py:183 - agentic rollout batch prompts=1 n_samples=8 expected_requests=8 max_running_prompts=4
2026-06-29 16:58:19 INFO areno.api.trainers.policy_only.PolicyOnlyTrainer policy_only.py:98 - epoch=0 step=0 metric=reward_mean value=0.125000
2026-06-29 16:58:19 INFO areno.api.trainers.policy_only.PolicyOnlyTrainer policy_only.py:119 - epoch=0 step=0 role=policy stage=train_start
2026-06-29 17:01:19 INFO areno.api.trainers.policy_only.PolicyOnlyTrainer policy_only.py:98 - epoch=0 step=1 metric=reward_mean value=0.375000
"""


def test_areno_target_reward_scores_real_train_log_with_reward_improvement():
    record = _record(
        tool_calls=[
            _call(
                "run_command",
                {
                    "command": (
                        "mkdir -p /tmp/areno-agentic-targets/areno__target && "
                        "areno train --ckpt /home/admin/Qwen3.5-4B/ "
                        "--dataset-path /tmp/areno-agentic-targets/areno__target/generated_dataset.jsonl"
                    ),
                    "background": True,
                },
            ),
            _call("submit", {"status": "solved"}),
        ],
        tool_results=[
            _result(
                {
                    "task_id": "bg-1",
                    "command": "areno train --ckpt /home/admin/Qwen3.5-4B/",
                    "running": True,
                    "output_path": "/tmp/areno-agentic-targets/areno__target/bg-1.log",
                }
            ),
            _result({"task_id": "bg-1", "output": REAL_LOG_WITH_REWARD_UP, "running": False}),
        ],
    )

    assert reward_fn(record) == 1.0


def test_areno_target_reward_requires_real_log_evidence():
    record = _record(
        tool_calls=[
            _call(
                "run_command",
                {
                    "command": (
                        "areno train --ckpt /home/admin/Qwen3.5-4B/ "
                        "--dataset-path /tmp/areno-agentic-targets/areno__target/generated_dataset.jsonl"
                    ),
                    "background": True,
                },
            )
        ],
        tool_results=[_result({"output_path": "/tmp/areno-agentic-targets/areno__target/bg-1.log"})],
    )

    assert reward_fn(record) < 0.5


def test_areno_target_reward_penalizes_save_path():
    record = _record(
        tool_calls=[
            _call(
                "run_command",
                {
                    "command": (
                        "areno train --ckpt /home/admin/Qwen3.5-4B/ "
                        "--dataset-path /tmp/areno-agentic-targets/areno__target/generated_dataset.jsonl "
                        "--save-path /tmp/areno-agentic-targets/areno__target/ckpt"
                    ),
                    "background": True,
                },
            )
        ],
        tool_results=[_result({"output_path": "/tmp/areno-agentic-targets/areno__target/bg-1.log", "output": REAL_LOG_WITH_REWARD_UP})],
    )

    assert reward_fn(record) < 0.8


def _record(tool_calls, tool_results):
    return SimpleNamespace(
        source_record={
            "instance_id": "areno__target",
            "output_dir": "/tmp/areno-agentic-targets/areno__target",
        },
        tool_calls=tool_calls,
        tool_results=tool_results,
    )


def _call(name: str, arguments: dict):
    return {"name": name, "arguments": json.dumps(arguments)}


def _result(content: dict):
    return {"content": json.dumps(content)}
