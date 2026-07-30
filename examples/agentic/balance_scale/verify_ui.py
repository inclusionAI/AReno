"""Gradio verification UI for the odd-ball balance-scale example.

Lets the user configure a puzzle (num_balls, odd_ball_index, direction —
or random), then runs the trained model through the multi-turn weigh /
submit_answer loop and displays the full reasoning trace, per-weighing
results, and the final verdict (correct/wrong, weighings used, reward).

Usage in Colab:
    python examples/agentic/balance_scale/verify_ui.py \\
        --base-url http://127.0.0.1:8000/v1 \\
        --api-key EMPTY \\
        --model policy

Usage without an LLM (random agent for demo):
    python examples/agentic/balance_scale/verify_ui.py --agent-mode random
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

# run_agent imports are deferred to avoid torch dependency at import time.


def run_puzzle_with_model(
    num_balls: int,
    odd_ball_index: int,
    direction: str,
    max_weighings: int,
    base_url: str,
    api_key: str,
    model: str,
):
    """Run one puzzle through the OpenAI-compatible model and return the trace."""

    import asyncio

    import httpx
    from openai import AsyncOpenAI

    ball_set = game.BallSet(
        num_balls=num_balls,
        odd_ball_index=odd_ball_index,
        direction=direction,
        max_weighings=max_weighings,
    )

    # Load tool definitions from run_agent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_agent as ra  # noqa: E402

    system_prompt = ra.SYSTEM_PROMPT
    user_prompt = game.format_prompt(ball_set)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    trace_lines = []
    weighings_used = 0
    turns_log = []

    async def _run():
        nonlocal weighings_used
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)

        try:
            for turn_idx in range(max_weighings + 1):
                if weighings_used >= max_weighings:
                    tools = [ra.SUBMIT_ANSWER_TOOL]
                    tool_choice = {"type": "function", "function": {"name": "submit_answer"}}
                    hint = f"You have used all {max_weighings} weighings. Submit your best guess now."
                    turn_messages = [*messages, {"role": "user", "content": hint}]
                elif weighings_used == 0 and turn_idx == 0:
                    # First turn: force weigh to bootstrap tool usage
                    tools = [ra.WEIGH_TOOL]
                    tool_choice = {"type": "function", "function": {"name": "weigh"}}
                    turn_messages = [*messages]
                else:
                    tools = ra.TOOLS
                    tool_choice = "auto"
                    turn_messages = [*messages]

                trace_lines.append(f"--- Turn {turn_idx + 1} ---")

                response = await client.chat.completions.create(
                    model=model,
                    messages=turn_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=64,
                    stream=False,
                )

                msg = response.choices[0].message
                tool_calls_raw = msg.tool_calls or []

                # Fallback: parse tool call from text if proxy didn't return one
                if not tool_calls_raw and msg.content:
                    parsed = ra._parse_tool_call_from_text(msg.content, tool_choice)
                    if parsed:
                        import uuid
                        trace_lines.append(f"  (parsed from text) {parsed['name']}")
                        name = parsed["name"]
                        args_str = parsed["arguments"]
                        args = json.loads(args_str) if args_str else {}
                    else:
                        trace_lines.append(f"  Model returned no tool call. Content: {msg.content}")
                        break
                else:
                    call = tool_calls_raw[0]
                    name = call.function.name
                    args_str = call.function.arguments or "{}"
                    args = json.loads(args_str) if args_str else {}

                turns_log.append({"turn": turn_idx + 1, "tool": name, "args": args})

                # Build a unified call_id for message construction
                call_id = call.id if tool_calls_raw else f"parsed_{turn_idx}"

                if name == "weigh":
                    left = args.get("left", [])
                    right = args.get("right", [])
                    trace_lines.append(f"  weigh(left={left}, right={right})")
                    result_dict, did_weigh = ra._run_weigh(args_str, ball_set, weighings_used)
                    if did_weigh:
                        trace_lines.append(f"  → {result_dict['result']}")
                        weighings_used += 1
                    else:
                        trace_lines.append(f"  → ERROR: {result_dict.get('error', 'unknown')}")

                    # Append to conversation
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [{
                            "id": call_id, "type": "function",
                            "function": {"name": name, "arguments": args_str},
                        }],
                    })
                    messages.append(ra._tool_result_message(
                        {"id": call_id, "function": {"name": name, "arguments": args_str}},
                        result_dict,
                    ))

                elif name == "submit_answer":
                    ball_idx = args.get("ball_index")
                    dir_val = args.get("direction")
                    trace_lines.append(f"  submit_answer(ball_index={ball_idx}, direction={dir_val})")
                    result = ra._run_submit_answer(args_str)
                    trace_lines.append(f"  → {result}")
                    break

            await http_client.aclose()
        finally:
            await http_client.aclose()

    asyncio.run(_run())

    return trace_lines, turns_log, weighings_used


def run_puzzle_random_agent(
    num_balls: int,
    odd_ball_index: int,
    direction: str,
    max_weighings: int,
):
    """Run a puzzle with a simple random agent (no LLM needed, for demo)."""

    ball_set = game.BallSet(
        num_balls=num_balls,
        odd_ball_index=odd_ball_index,
        direction=direction,
        max_weighings=max_weighings,
    )

    trace_lines = []
    turns_log = []
    weighings_used = 0
    rng = random.Random(42)

    candidates = list(range(num_balls))
    possible_directions = ["heavier", "lighter"]

    for turn_idx in range(max_weighings + 1):
        trace_lines.append(f"--- Turn {turn_idx + 1} ---")

        if weighings_used >= max_weighings or len(candidates) <= 1:
            # Submit answer
            guess_ball = candidates[0] if candidates else rng.randint(0, num_balls - 1)
            guess_dir = rng.choice(possible_directions)
            trace_lines.append(f"  submit_answer(ball_index={guess_ball}, direction={guess_dir})")
            turns_log.append({"turn": turn_idx + 1, "tool": "submit_answer",
                              "args": {"ball_index": guess_ball, "direction": guess_dir}})
            break

        # Random weigh
        half = len(candidates) // 2
        if half < 1:
            half = 1
        left = sorted(rng.sample(candidates, min(half, len(candidates))))
        right_candidates = [c for c in candidates if c not in left][:len(left)]
        if not right_candidates or len(right_candidates) < len(left):
            right = sorted(rng.sample([c for c in range(num_balls) if c not in left], len(left)))
        else:
            right = sorted(right_candidates[:len(left)])

        trace_lines.append(f"  weigh(left={left}, right={right})")
        result = game.weigh(ball_set, left, right, weighings_used=weighings_used)
        trace_lines.append(f"  → {result}")
        turns_log.append({"turn": turn_idx + 1, "tool": "weigh",
                          "args": {"left": left, "right": right}, "result": result})
        weighings_used += 1

        # Simple elimination
        if result == "balanced":
            candidates = [c for c in candidates if c not in left and c not in right]
        elif result == "left_heavy":
            candidates = [c for c in candidates if c in left or c in right]
        elif result == "right_heavy":
            candidates = [c for c in candidates if c in left or c in right]

    return trace_lines, turns_log, weighings_used


def build_gradio_app(base_url: str, api_key: str, model: str, agent_mode: str):
    """Build and return the Gradio Blocks interface."""

    import gradio as gr

    def run_and_display(num_balls, odd_ball_index, direction, randomize):
        if randomize:
            rng = random.Random()
            odd_ball_index = rng.randint(0, num_balls - 1)
            direction = rng.choice(["heavier", "lighter"])

        # Auto-compute max_weighings
        import math
        min_w = max(1, math.ceil(math.log(num_balls * 2, 3)))
        max_w = min_w * 2

        if agent_mode == "random":
            trace_lines, turns_log, weighings_used = run_puzzle_random_agent(
                num_balls, odd_ball_index, direction, max_w
            )
        else:
            trace_lines, turns_log, weighings_used = run_puzzle_with_model(
                num_balls, odd_ball_index, direction, max_w,
                base_url, api_key, model
            )

        # Build result summary
        ball_set = game.BallSet(num_balls, odd_ball_index, direction, max_w)
        last_submit = None
        for t in reversed(turns_log):
            if t["tool"] == "submit_answer":
                last_submit = t["args"]
                break

        if last_submit:
            result = game.check_answer(ball_set, last_submit["ball_index"], last_submit["direction"])
            if result["full_correct"]:
                verdict = "✅ 完全正确！球编号和方向都对。"
            elif result["ball_correct"]:
                verdict = "🟡 部分正确：球编号对了，但方向错了。"
            else:
                verdict = "❌ 完全错误。"
            answer_str = f"球 {last_submit['ball_index']} ({last_submit['direction']})"
        else:
            verdict = "❌ 模型未提交答案。"
            answer_str = "N/A"

        import math
        min_w = max(1, math.ceil(math.log(num_balls * 2, 3)))
        efficiency = f"{weighings_used}/{min_w} (实际/理论最优)"

        summary = f"""## 结果

| 项目 | 值 |
|---|---|
| 球数 | {num_balls} |
| 异常球 | #{odd_ball_index} |
| 真实方向 | {direction} |
| 模型答案 | {answer_str} |
| 判定 | {verdict} |
| 称量次数 | {weighings_used} |
| 理论最优 | {min_w} |
| 效率 | {efficiency} |
| 最大允许 | {max_w} |
"""

        trace_text = "\n".join(trace_lines)

        turns_json = json.dumps(turns_log, indent=2, ensure_ascii=False, default=str)

        return summary, trace_text, turns_json

    with gr.Blocks(title="Balance-Scale 验证") as app:
        gr.Markdown("# Odd-Ball Balance-Scale 验证")
        gr.Markdown("设定球参数，运行模型称量过程，查看结果。")

        with gr.Row():
            with gr.Column():
                num_balls = gr.Slider(2, 200, value=12, step=1, label="球数")
                odd_ball_index = gr.Number(value=5, label="异常球编号 (0-based)", precision=0)
                direction = gr.Radio(["heavier", "lighter"], value="heavier", label="异常球方向")
                randomize = gr.Checkbox(value=False, label="随机生成异常球")
                run_btn = gr.Button("运行", variant="primary")

        with gr.Row():
            summary = gr.Markdown()
            trace = gr.Textbox(label="称量过程", lines=20, max_lines=40)
            turns_json = gr.Code(label="结构化日志 (JSON)", language="json")

        run_btn.click(
            run_and_display,
            inputs=[num_balls, odd_ball_index, direction, randomize],
            outputs=[summary, trace, turns_json],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Balance-scale verification Gradio UI.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key.")
    parser.add_argument("--model", default="policy", help="Model name.")
    parser.add_argument("--agent-mode", choices=["model", "random"], default="model",
                        help="Agent mode: 'model' uses LLM, 'random' uses random agent (no LLM needed).")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port.")
    args = parser.parse_args()

    app = build_gradio_app(args.base_url, args.api_key, args.model, args.agent_mode)
    app.launch(server_port=args.port, share=True)


if __name__ == "__main__":
    main()
