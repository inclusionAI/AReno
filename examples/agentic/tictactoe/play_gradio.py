"""Gradio-based visual Tic-Tac-Toe board for playing against the model.

Uses a single HTML component for the board (rendered with inline CSS/JS)
and a hidden textbox as the click→Python bridge.

Usage (Kaggle / local):
    pip install gradio
    python play_gradio.py --model-path /path/to/Qwen3-0.6B --share
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import game  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import torch  # noqa: E402

EMPTY = game.EMPTY
HUMAN = "O"
MODEL = "X"

# Global board state (server-side, no Gradio State needed)
_board: game.Board = [[EMPTY] * 3 for _ in range(3)]
_finished = False
_message = "点击下方棋盘开始！"


def board_to_html(board: game.Board, msg: str = "") -> str:
    """Render the board as a self-contained HTML grid."""
    cells = ""
    for i in range(9):
        r, c = divmod(i, 3)
        cell = board[r][c]
        if cell == "X":
            display = "❌"
            clickable = ""
        elif cell == "O":
            display = "⭕"
            clickable = ""
        else:
            display = ""
            clickable = f"onclick=\"clickCell({i+1})\""
        cells += f'<div class="cell" {clickable}>{display}</div>'

    status = f'<div class="status">{msg}</div>' if msg else ""

    return f"""
    <div class="ttt-wrap">
      <div class="ttt-board">
        {cells}
      </div>
      {status}
    </div>
    <script>
    function clickCell(square) {{
        // Send the square number to Python via the hidden textbox
        const bridge = document.getElementById('click_bridge');
        if (bridge) {{
            bridge.value = square;
            bridge.dispatchEvent(new Event('input', {{bubbles: true}}));
        }}
    }}
    </script>
    <style>
    .ttt-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 12px; }}
    .ttt-board {{
        display: grid;
        grid-template-columns: repeat(3, 120px);
        grid-template-rows: repeat(3, 120px);
        gap: 6px;
        justify-content: center;
    }}
    .cell {{
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3em;
        font-weight: bold;
        background: #f0f0f0;
        border-radius: 10px;
        cursor: pointer;
        transition: background 0.15s;
        user-select: none;
    }}
    .cell:hover {{ background: #d0e8ff; }}
    .status {{ font-size: 1.3em; font-weight: bold; padding: 8px; }}
    </style>
    """


def model_move(board: game.Board) -> int | None:
    """Ask the model for a move."""
    prompt = game.format_xml_prompt(board)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    move = game.parse_xml_move(text)
    print(f"[model] raw output: {text.strip()!r}  parsed move: {move}")
    return move


def check_result(board: game.Board) -> str | None:
    if game.is_terminal(board):
        w = game.winner(board)
        if w == MODEL:
            return "😔 模型赢了！"
        elif w == HUMAN:
            return "🎉 你赢了！"
        else:
            return "🤝 平局！"
    return None


def on_click(square_str: str) -> tuple[str, str]:
    """Handle a cell click: human moves, then model responds."""
    global _board, _finished, _message

    if _finished:
        return board_to_html(_board, _message), ""

    if not square_str or square_str.strip() == "":
        return board_to_html(_board, _message), ""

    square = int(square_str.strip())

    # ── Human move ──
    if square not in game.legal_moves(_board):
        return board_to_html(_board, "⚠️ 该位置已被占用！"), ""

    _board = game.apply_move(_board, square, HUMAN)
    result = check_result(_board)
    if result:
        _finished = True
        _message = result
        return board_to_html(_board, _message), ""

    # ── Model move ──
    move = model_move(_board)
    if move and move in game.legal_moves(_board):
        _board = game.apply_move(_board, move, MODEL)
    result = check_result(_board)
    if result:
        _finished = True
        _message = result
        return board_to_html(_board, _message), ""

    _message = "你的回合，请落子 ⭕"
    return board_to_html(_board, _message), ""


def reset() -> tuple[str, str]:
    """Reset the game."""
    global _board, _finished, _message
    _board = [[EMPTY] * 3 for _ in range(3)]
    _finished = False
    _message = "新对局开始！你执 ⭕，模型执 ❌"
    return board_to_html(_board, _message), ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Tic-Tac-Toe vs Model", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🎮 井字棋对弈 — 你 vs 模型")
        gr.Markdown("你执 ⭕，模型执 ❌。点击空格落子，模型会自动应手。")

        board_html = gr.HTML(value=board_to_html(_board, _message))

        # Hidden textbox as JS→Python bridge
        click_bridge = gr.Textbox(
            value="",
            elem_id="click_bridge",
            visible=False,
        )

        with gr.Row():
            reset_btn = gr.Button("🔄 重新开始")

        # When the hidden textbox changes (via JS), trigger the game logic
        click_bridge.change(
            on_click,
            inputs=[click_bridge],
            outputs=[board_html, click_bridge],
        )

        reset_btn.click(
            reset,
            outputs=[board_html, click_bridge],
        )

    return ui


def main():
    parser = argparse.ArgumentParser(description="Gradio Tic-Tac-Toe vs Model")
    parser.add_argument("--model-path", required=True, help="Path or HF id of the model")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    global tokenizer, model
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    print("Model loaded. Starting Gradio UI ...")

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()