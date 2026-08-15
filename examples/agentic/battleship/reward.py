"""Reward function for the Battleship tool-call example."""

from __future__ import annotations

# 导入标准库
import json
import sys
from pathlib import Path
from typing import Any

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


# 奖励常数定义
# Reward constants
WIN_BONUS = 1.0      # 获胜时的奖励
HIT_REWARD = 0.05    # 每次命中船只的奖励
SUNK_REWARD = 0.15   # 击沉每艘船的奖励
INVALID_SHOT_PENALTY = 0.02  # 每次无效射击的惩罚
EFFICIENCY_PENALTY = 0.05    # 每发一弹的效率惩罚：让"慢赢/穷举"接近 0 分，"高效赢"拿高分，拉开 RL 梯度


# RL 奖励函数：评估一次完整回合的得分
def reward_fn(record: Any) -> float:
    """Score one completion by extracting the fire tool calls and replaying the episode."""
    # 从记录中获取源数据（包含初始舰队布局）
    source = record.source_record

    # 解析工具调用，提取 fire 工具的调用列表
    # Parse tool calls
    fire_calls = _extract_fire_calls(record)
    # 如果没有有效的 fire 调用，返回零分
    if not fire_calls:
        return 0.0

    # 从源记录初始化船只布局：score_episode 只读取这里的 ships 与 seed，
    # 并在内部用一份全新的 test_state 重放所有 fire 调用，因此此处无需、
    # 也不应提前 fire（此前把整个 call dict 当坐标传给 game.fire 会把每发
    # 都记成 invalid 且从不真正开火，是无效且会误导维护者的死代码）。
    # Initialize state from source record; score_episode replays the fire
    # calls into a fresh state and only reads the fleet layout + seed here.
    state = game.init_state(source)

    # 对回合进行评分（重放在 score_episode 内部完成）
    # Score the episode (replay happens inside score_episode).
    score = game.score_episode(state, fire_calls)

    # 计算形状化奖励
    # Compute shaped reward
    reward = 0.0

    # 获胜奖励
    # Win bonus
    if score["win"]:
        reward += WIN_BONUS

    # 命中奖励：根据命中次数
    # Hit reward
    reward += score["hits"] * HIT_REWARD

    # 击沉奖励：根据击沉船只数
    # Sunk ship reward
    reward += score["sunk_ships"] * SUNK_REWARD

    # 无效射击惩罚
    # Invalid shot penalty
    reward -= score["invalid_shots"] * INVALID_SHOT_PENALTY

    # 效率惩罚：射击越多惩罚越重，激励用更少射击获胜
    # Efficiency penalty (slight penalty for taking more shots)
    reward -= score["shots_used"] * EFFICIENCY_PENALTY

    # 确保有效回合的奖励不低于零
    # Ensure non-negative minimum for valid episodes
    return max(0.0, reward)


# 从记录中提取 fire 工具调用的内部函数
def _extract_fire_calls(record: Any) -> list[dict]:
    """Extract fire tool calls from the record."""
    # 存储提取的 fire 调用
    calls = []
    # 遍历记录中的所有工具调用
    for call in record.tool_calls:
        # 获取调用名称
        name = call.get("name") if isinstance(call, dict) else None
        # 只处理 fire 工具调用
        if name != "fire":
            continue

        # 获取参数，可能是字典或 JSON 字符串
        arguments = call.get("arguments", {})
        # 如果是字符串，解析为字典
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        # 规范化调用格式
        # Normalize the call format
        calls.append({
            "name": "fire",
            "arguments": arguments,
        })

    return calls