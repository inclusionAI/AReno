"""Gradio-based visual Tic-Tac-Toe board for playing against the model.

Uses 9 gr.Button cells + a global variable for state (no gr.State).
Each button reads its position from a hidden textbox with a fixed value.

Usage (Kaggle / local):
    pip install gradio
    python play_gradio.py --model-path /path/to/Qwen3-0.6B --share
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import threading
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import game  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import torch  # noqa: E402

EMPTY = game.EMPTY
HUMAN = "O"
MODEL = "X"

# ── Server-side state (no gr.State needed) ───────────────────────────────
_board: game.Board = [[EMPTY] * 3 for _ in range(3)]
_finished: bool = False
_message: str = "点击下方棋盘开始！"
_lock = threading.Lock()


def cell_display(cell: str) -> str:
    """Convert internal cell to display emoji."""
    if cell == "X":
        return "❌"
    if cell == "O":
        return "⭕"
    return ""


def get_all_displays() -> list[str]:
    """Return 9 display strings for the current board."""
    return [cell_display(_board[r][c]) for r in range(3) for c in range(3)]


def model_move(board: game.Board) -> int | None:
    """Ask the model for a move."""
    prompt = game.format_xml_prompt(board)
    messages = [
        {"role": "system", "content": "You are a Tic-Tac-Toe player. Output ONLY the XML tag <move>N</move> where N is the square number. No other text."},
        {"role": "user", "content": prompt},
    ]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    move = game.parse_xml_move(text)
    # Fallback: if no <move> tag found, try to find a bare digit
    if move is None:
        digits = re.findall(r'[1-9]', text)
        if digits:
            move = int(digits[-1])
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


def make_click_handler(square: int):
    """Return a handler for clicking cell `square` (1-9)."""

    def handler() -> tuple:
        """No inputs needed — reads global state directly."""
        global _board, _finished, _message

        with _lock:
            print(f"[click] square={square} board={_board} finished={_finished}")

            if _finished:
                return _message, *get_all_displays()

            # ── Human move ──
            if square not in game.legal_moves(_board):
                _message = "⚠️ 该位置已被占用！"
                return _message, *get_all_displays()

            _board = game.apply_move(_board, square, HUMAN)
            print(f"[human] placed O at {square}, board={_board}")
            result = check_result(_board)
            if result:
                _finished = True
                _message = result
                return _message, *get_all_displays()

            # ── Model move (retry up to 3 times, then fallback to random) ──
            model_placed = False
            for attempt in range(3):
                move = model_move(_board)
                if move and move in game.legal_moves(_board):
                    _board = game.apply_move(_board, move, MODEL)
                    print(f"[model] placed X at {move} (attempt {attempt+1}), board={_board}")
                    model_placed = True
                    break
                else:
                    print(f"[model] illegal move={move} (attempt {attempt+1}), legal={game.legal_moves(_board)}")

            if not model_placed:
                # Fallback: pick a random legal move
                legal = game.legal_moves(_board)
                if legal:
                    fallback = random.choice(legal)
                    _board = game.apply_move(_board, fallback, MODEL)
                    print(f"[model] fallback random X at {fallback}, board={_board}")
                    model_placed = True

            result = check_result(_board)
            if result:
                _finished = True
                _message = result
                return _message, *get_all_displays()

            _message = "你的回合，请落子 ⭕"
            return _message, *get_all_displays()

    return handler


def reset_game() -> tuple:
    """Reset to a fresh board."""
    global _board, _finished, _message
    _board = [[EMPTY] * 3 for _ in range(3)]
    _finished = False
    _message = "新对局开始！你执 ⭕，模型执 ❌"
    return _message, *get_all_displays()


def build_ui() -> gr.Blocks:
    css = """
    .ttt-cell {
        min-height: 100px !important;
        min-width: 100px !important;
        font-size: 2.5em !important;
        font-weight: bold !important;
        aspect-ratio: 1 / 1 !important;
    }
    """

    with gr.Blocks(title="Tic-Tac-Toe vs Model", theme=gr.themes.Soft(), css=css) as ui:
        gr.Markdown("# 🎮 井字棋对弈 — 你 vs 模型")
        gr.Markdown("你执 ⭕，模型执 ❌。点击空格落子，模型会自动应手。")

        status = gr.Markdown(_message)

        # 3x3 grid of buttons
        grid: list[list[gr.Button]] = []
        for row_idx in range(3):
            row_buttons = []
            with gr.Row():
                for col_idx in range(3):
                    btn = gr.Button(
                        "",
                        scale=1,
                        elem_classes="ttt-cell",
                        size="lg",
                    )
                    row_buttons.append(btn)
            grid.append(row_buttons)

        with gr.Row():
            reset_btn = gr.Button("🔄 重新开始")

        all_buttons = [grid[i][j] for i in range(3) for j in range(3)]
        all_outputs = [status, *all_buttons]

        # Each button click: no inputs (handler reads global state), outputs = status + 9 buttons
        for r in range(3):
            for c in range(3):
                square = r * 3 + c + 1
                grid[r][c].click(
                    make_click_handler(square),
                    inputs=None,
                    outputs=all_outputs,
                )

        reset_btn.click(
            reset_game,
            inputs=None,
            outputs=all_outputs,
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
    from os.path import isdir
    if isdir(args.model_path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded. Starting Gradio UI ...")

    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()