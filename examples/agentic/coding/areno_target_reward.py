"""Reward for AReno target-checkpoint coding tasks.

This reward is intentionally evidence-based. It does not inspect external
files; it only scores the command/log evidence that the agent exposed through
tool calls and tool results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_REWARD_RE = re.compile(r"metric=reward_mean\s+value=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
_STEP_RE = re.compile(r"epoch=\d+\s+step=\d+")
_CKPT_RE = re.compile(r"--ckpt(?:=|\s+)/home/admin/Qwen3\.5-4B/?(?:\s|$)")
_TRAIN_MARKERS = (
    "areno.api.trainers.policy_only.PolicyOnlyTrainer",
    "role=policy stage=rollout_start",
    "role=policy stage=train_start",
    "agentic rollout batch",
    "agentic train batch built",
    "loading checkpoint:",
)


def reward_fn(record) -> float:
    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)
    tool_results = [_decode_result(result.get("content")) for result in record.tool_results]

    if "files" in source:
        return _local_coding_reward(source, tool_calls, tool_results)
    return _areno_target_reward(source, tool_calls, tool_results)


def _areno_target_reward(
    source: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> float:
    output_dir = str(source.get("output_dir") or "")
    log_text = _combined_background_log(tool_results)
    score = 0.0

    if _started_areno_train(tool_calls, tool_results):
        score += 0.20
    if output_dir and _uses_output_dir(output_dir, tool_calls, tool_results):
        score += 0.20
    if _looks_like_real_areno_log(log_text):
        score += 0.25

    rewards = _parse_reward_values(log_text)
    if _reward_improved(rewards):
        score += 0.35
    elif rewards:
        score += 0.10

    if _uses_save_path(tool_calls):
        score -= 0.40
    if _submitted_solved(tool_calls) and score >= 0.80:
        score += 0.10
    return max(-1.0, min(1.0, score if score > 0 else -1.0))


def _local_coding_reward(
    source: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> float:
    required_commands = [str(command) for command in source.get("test_commands", [])]
    successful_commands = {
        str(result.get("command"))
        for result in tool_results
        if result.get("returncode") == 0 and isinstance(result.get("command"), str)
    }
    all_tests_passed = (
        all(command in successful_commands for command in required_commands) if required_commands else True
    )
    applied_patch = any(call.get("name") == "apply_patch" for call in tool_calls)
    if _submitted_solved(tool_calls) and all_tests_passed and applied_patch:
        return 1.0
    if all_tests_passed and required_commands:
        return 0.5
    if _submitted_solved(tool_calls):
        return -0.5
    return -1.0


def _started_areno_train(tool_calls: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> bool:
    for call in tool_calls:
        if call.get("name") != "run_command":
            continue
        args = _decode_args(call.get("arguments"))
        command = str(args.get("command", ""))
        if "areno train" in command and _CKPT_RE.search(command):
            return True
    return any("areno train" in str(result.get("command", "")) for result in tool_results)


def _uses_output_dir(
    output_dir: str,
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> bool:
    normalized = Path(output_dir).as_posix().rstrip("/")
    if not normalized:
        return False
    for call in tool_calls:
        args = _decode_args(call.get("arguments"))
        if normalized in str(args.get("command", "")):
            return True
    for result in tool_results:
        if normalized in str(result.get("output_path", "")) or normalized in str(result.get("command", "")):
            return True
    return False


def _combined_background_log(tool_results: list[dict[str, Any]]) -> str:
    chunks = []
    for result in tool_results:
        if isinstance(result.get("output"), str):
            chunks.append(str(result["output"]))
        if isinstance(result.get("stdout"), str):
            chunks.append(str(result["stdout"]))
        if isinstance(result.get("stderr"), str):
            chunks.append(str(result["stderr"]))
    return "\n".join(chunks)


def _looks_like_real_areno_log(text: str) -> bool:
    if not text:
        return False
    marker_count = sum(1 for marker in _TRAIN_MARKERS if marker in text)
    has_step = _STEP_RE.search(text) is not None
    has_reward = bool(_parse_reward_values(text))
    return marker_count >= 2 and has_step and has_reward


def _parse_reward_values(text: str) -> list[float]:
    values = []
    for match in _REWARD_RE.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _reward_improved(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    best_so_far = values[0]
    for value in values[1:]:
        if value > best_so_far + 1e-6:
            return True
        best_so_far = min(best_so_far, value)
    return False


def _uses_save_path(tool_calls: list[dict[str, Any]]) -> bool:
    for call in tool_calls:
        args = _decode_args(call.get("arguments"))
        if "--save-path" in str(args.get("command", "")):
            return True
    return False


def _submitted_solved(tool_calls: list[dict[str, Any]]) -> bool:
    for call in reversed(tool_calls):
        if call.get("name") != "submit":
            continue
        args = _decode_args(call.get("arguments"))
        return str(args.get("status")) == "solved"
    return False


def _decode_result(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _decode_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    if not isinstance(args, str):
        return {}
    try:
        value = json.loads(args)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
