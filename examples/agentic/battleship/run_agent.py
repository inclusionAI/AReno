"""Agent entrypoint for multi-turn Battleship tool-call rollouts."""

from __future__ import annotations

# 导入标准库
import asyncio
import json
import logging
import sys
from pathlib import Path

# 从 areno 导入 agentic 模块的类型
from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

# 获取模块日志记录器
logger = logging.getLogger(__name__)
# 将 httpx 日志级别设置为 WARNING，减少无关输出
logging.getLogger("httpx").setLevel(logging.WARNING)

# 系统提示词：告诉 agent 如何玩 Battleship 游戏
SYSTEM_PROMPT = (
    "You are playing Battleship. Use the fire(coordinate) tool to sink all ships.\n\n"
    "Rules:\n"
    "- The grid is 8x8, columns numbered 1-8, rows labeled A-H.\n"
    "- Coordinates are given as letter+number, e.g., A1 (top-left), H8 (bottom-right).\n"
    "- The fire tool returns: miss (no ship), hit (ship not sunk), sunk (ship destroyed).\n"
    "- Do not fire at the same coordinate twice.\n"
    "- Do not fire outside the A1-H8 range.\n"
    "- Win by sinking all ships with as few shots as possible.\n"
    "- After each shot, you will see an updated board showing your hits (X), misses (o), and unknown cells (.).\n"
    "- You also see 'Already fired: ...' listing every coordinate you have fired this game; do not fire any of them again."
)

# fire 工具的 JSON Schema 定义
FIRE_TOOL = {
    "type": "function",
    "function": {
        "name": "fire",
        "description": "Fire a shot at a coordinate on the Battleship board.",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinate": {
                    "type": "string",
                    "description": "The coordinate to fire at, e.g., 'A1', 'B7', 'H8'. Uses letter row (A-H) and number column (1-8).",
                    "pattern": "^[A-H][1-8]$",
                },
            },
            "required": ["coordinate"],
            "additionalProperties": False,
        },
    },
}

# 使用游戏模块的最大回合数
MAX_TURNS = game.MAX_TURNS
# 每回合模型生成的 token 上限。fire 工具调用本身约 30 token；强制 tool_choice=fire
# 时小模型仍可能输出冗长 content，不设上限会让多轮轨迹长度不可控（采样涨落时
# 单回合可达数百 token），易超过 max_context_len 被过滤、或在 train 阶段 OOM。
MAX_RESPONSE_TOKENS = 128


# Agent 入口函数：运行多轮 Battleship 工具调用 rollout
async def run_agent(ctx, batch):
    """Run multi-turn Battleship fire-until-win (or cap)."""

    # 尝试导入必需的依赖
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The Battleship agentic example requires `openai` and `httpx`. Install them with `pip install openai httpx`."
        ) from exc

    # 从 batch 中获取所有样本
    items = list(batch.iter_samples())
    logger.info("Battleship agent start tasks=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)

    # 创建 HTTP 客户端，用于与模型服务通信
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    # 创建 OpenAI 兼容的异步客户端
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    # 定义处理单个样本的内部函数
    async def run_one(item):
        # 存储每个回合的轨迹
        turns = []
        # 初始化消息列表，包含系统提示和用户 prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]

        # 从记录中初始化游戏状态
        # Initialize game state from the record
        record = item.record
        state = game.init_state(record)

        # 循环直到游戏结束或达到回合上限
        # Loop until win or turn cap
        while not game.is_terminal(state) and state.shots_used < MAX_TURNS:
            # 调用模型获取下一步动作
            assistant_message, turn = await _call_model(item, client, messages, state)
            turns.append(turn)

            # 执行工具调用
            # Execute the tool
            result = _run_tool(assistant_message, state)
            state = result.get("state", state)

            # 添加助手消息和工具结果到消息列表
            # Append assistant message and tool result
            messages.extend(_tool_messages(assistant_message, result))

            # 添加棋盘观察作为用户消息（让模型看到它的视角）
            # Add board observation as user message (so the model sees its view)
            # 追加已射击坐标明文列表，避免小模型需要解析 o/X 符号才能不重复射击
            # Append the explicit fired-coordinate list so small untrained models
            # do not have to parse o/X symbols to avoid repeats.
            fired = [game.format_coordinate(r, c) for r, c in state.shots_history]
            fired_line = f"Already fired: {', '.join(fired)}\n" if fired else ""
            board_msg = {
                "role": "user",
                "content": f"\nBoard after your shot:\n{game.board_text(state)}\n{fired_line}",
            }
            messages.append(board_msg)

            # 检查游戏是否结束
            # Check if game ended
            if game.is_terminal(state):
                break

        return turns

    try:
        # 并发运行所有样本
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        # 展平结果并返回 AgentTrajectory
        return AgentTrajectory(turns=[turn for turns in grouped for turn in turns])
    finally:
        # 关闭客户端连接
        await client.close()


# 调用模型获取单次 fire 动作的内部异步函数
async def _call_model(item, client, messages: list[dict], state):
    """Call the model for a single fire action."""
    # 复制消息列表，避免修改原始列表
    turn_messages = [*messages]
    # 将 fire 工具添加到可用工具列表
    tools = [FIRE_TOOL]
    # 强制使用 fire 工具
    tool_choice = {"type": "function", "function": {"name": "fire"}}

    # 调用 OpenAI 兼容的聊天完成接口
    response = await client.chat.completions.create(
        model="policy",
        messages=turn_messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=MAX_RESPONSE_TOKENS,
        stream=False,
    )

    # 获取模型返回的消息
    message = response.choices[0].message
    # 过滤出 fire 工具调用（只取第一个）
    tool_calls = [call for call in (message.tool_calls or []) if call.function.name == "fire"][:1]

    # 构建助手消息格式
    assistant_message = {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }

    # 如果没有 fire 工具调用，创建一个虚拟的空调用
    if not assistant_message["tool_calls"]:
        assistant_message["tool_calls"] = [
            {
                "id": "missing_fire",
                "type": "function",
                "function": {
                    "name": "fire",
                    "arguments": "{}",
                },
            }
        ]

    # 返回助手消息和轨迹回合对象
    return assistant_message, AgentTrajectoryTurn(
        item=item,
        messages=turn_messages,
        response=response,
        tools=tools,
        tool_choice=tool_choice,
    )


# 构建工具调用和工具结果消息的辅助函数
def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    """Build the messages for assistant tool call and tool result."""
    messages = [assistant_message]
    # `state` is internal bookkeeping for run_one (line 113) and is not
    # JSON-serializable (GameState/Ship). Strip it so the tool message only
    # carries serializable status metadata — never the hidden ship positions.
    content = {k: v for k, v in tool_result.items() if k != "state"}
    for call in assistant_message.get("tool_calls") or []:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": json.dumps(content, ensure_ascii=False),
            }
        )
    return messages


# 模型这回合未产出合法 fire 动作时，把这回合作废：不碰棋盘，只推进 shots_used，
# 否则 run_one 的 while 循环在未训练模型上永不终止（shots_used 恒为 0）。
def _wasted_turn(state, reason: str) -> dict:
    """Mark a turn as wasted (no valid shot) and advance the turn counter."""
    state.shots_used += 1
    return {
        "status": "invalid",
        "reason": reason,
        "shots_used": state.shots_used,
        "state": state,
    }


# 执行 fire 工具的辅助函数
def _run_tool(assistant_message: dict, state) -> dict:
    """Execute the fire tool and return the result with updated state."""
    # 获取工具调用列表
    calls = assistant_message.get("tool_calls") or []
    # 如果没有工具调用，返回错误
    if not calls:
        return _wasted_turn(state, "no tool call")

    # 只处理第一个工具调用
    call = calls[0]
    name = call["function"]["name"]
    # 验证是 fire 工具
    if name != "fire":
        return _wasted_turn(state, f"unexpected tool: {name}")

    # 解析工具参数
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return _wasted_turn(state, "invalid JSON arguments")

    # 获取坐标参数
    coord = args.get("coordinate")
    if not coord:
        return _wasted_turn(state, "missing coordinate")

    # 执行射击动作
    # Execute the fire action
    result = game.fire(state, coord)

    # 在结果中包含棋盘文本，让模型能够看到
    # Include board text in the result for the model to see
    result["board"] = game.board_text(state)

    return result


# 从工具调用消息中解析坐标的辅助函数
def _parse_coord(assistant_message: dict) -> str | None:
    """Parse the coordinate from a tool call message."""
    # 获取工具调用列表
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None

    # 尝试解析第一个工具调用的参数
    try:
        args = json.loads(calls[0]["function"]["arguments"])
        return args.get("coordinate")
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
