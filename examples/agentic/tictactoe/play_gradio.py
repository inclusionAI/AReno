"""Gradio-based visual Tic-Tac-Toe board for playing against the model.

Usage (Kaggle / local):
    pip install gradio
    python play_gradio.py --model-path /path/to/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

# Ensure the tictactoe game module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import game  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import torch  # noqa: E402

# ── Globals ──────────────────────────────────────────────────────────────
EMPTY = game.EMPTY
HUMAN = "O"
MODEL = "X"


def board_to_flat(board: game.Board) -> list[str]:
    """Convert internal board to 9 flat display strings (empty -> "")."""
    return [
        "" if cell == EMPTY else ("❌" if cell == "X" else "⭕")
        for row in board
        for cell in row
    ]


def new_state(board=None, finished=False, message="") -> dict:
    """Always create a fresh dict so Gradio state updates correctly."""
    if board is None:
        board = [[EMPTY] * 3 for _ in range(3)]
    return {"board": board, "finished": finished, "message": message}


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
    """Return status message if game ended, else None."""
    if game.is_terminal(board):
        w = game.winner(board)
        if w == MODEL:
            return "😔 模型赢了！"
        elif w == HUMAN:
            return "🎉 你赢了！"
        else:
            return "🤝 平局！"
    return None


def make_click_handler(row: int, col: int):
    """Return a click handler with row/col captured at build time."""

    def handler(state: dict) -> tuple:
        board = state["board"]
        finished = state["finished"]

        if finished:
            return new_state(board, True, state["message"]), state["message"], *board_to_flat(board)

        square = row * 3 + col + 1

        # ── Human move ──
        if square not in game.legal_moves(board):
            return new_state(board, False, "⚠️ 该位置已被占用！"), "⚠️ 该位置已被占用！", *board_to_flat(board)

        board = game.apply_move(board, square, HUMAN)
        result = check_result(board)
        if result:
            return new_state(board, True, result), result, *board_to_flat(board)

        # ── Model move ──
        move = model_move(board)
        if move and move in game.legal_moves(board):
            board = game.apply_move(board, move, MODEL)
        result = check_result(board)
        if result:
            return new_state(board, True, result), result, *board_to_flat(board)

        msg = "你的回合，请落子 ⭕"
        return new_state(board, False, msg), msg, *board_to_flat(board)

    return handler


def reset_game() -> tuple:
    """Reset to a fresh board."""
    state = new_state(message="新对局开始！你执 ⭕，模型执 ❌")
    return state, state["message"], *board_to_flat(state["board"])


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    css = """
    #cell-0-0 button, #cell-0-1 button, #cell-0-2 button,
    #cell-1-0 button, #cell-1-1 button, #cell-1-2 button,
    #cell-2-0 button, #cell-2-1 button, #cell-2-2 button {
        aspect-ratio: 1 / 1;
        min-width: 100px;
        min-height: 100px;
        font-size: 2.5em;
        font-weight: bold;
    }
    """

    with gr.Blocks(title="Tic-Tac-Toe vs Model", theme=gr.themes.Soft(), css=css) as ui:
        gr.Markdown("# 🎮 井字棋对弈 — 你 vs 模型")
        gr.Markdown("你执 ⭕，模型执 ❌。点击空格落子，模型会自动应手。")

        state = gr.State(new_state(message="点击下方棋盘开始！"))

        status = gr.Markdown("点击下方棋盘开始！")

        # 3x3 grid of buttons
        grid: list[list[gr.Button]] = []
        for row_idx in range(3):
            row_buttons = []
            with gr.Row():
                for col_idx in range(3):
                    btn = gr.Button(
                        "",
                        scale=1,
                        elem_id=f"cell-{row_idx}-{col_idx}",
                    )
                    row_buttons.append(btn)
            grid.append(row_buttons)

        with gr.Row():
            reset_btn = gr.Button("🔄 重新开始")

        all_buttons = [grid[i][j] for i in range(3) for j in range(3)]

        # Wire click events — closures capture row/col at build time
        for r in range(3):
            for c in range(3):
                grid[r][c].click(
                    make_click_handler(r, c),
                    inputs=[state],
                    outputs=[state, status, *all_buttons],
                )

        reset_btn.click(
            reset_game,
            outputs=[state, status, *all_buttons],
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