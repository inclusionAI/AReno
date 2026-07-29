"""Local browser demo for the 2048 agentic example.

Modes:
  * human  -- arrow keys play one move at a time (offline, no model).
  * random -- a uniform-random direction (all four, not legal-only) plays one
              move per request (offline, no model).
  * llm    -- a trained policy's ``choose_moves`` tool call is replayed as one
              episode against the same seeded engine used in training; the
              random baseline on the same board+seed is shown alongside for the
              trained-vs-baseline improvement. Requires ``--base-url``.

Run from the repository root:

    python examples/agentic/2048/web_ui.py --agent-mode random
    python examples/agentic/2048/web_ui.py --base-url http://127.0.0.1:8000/v1 \
        --api-key token --model policy --agent-mode llm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768
START_SPAWNS = 2

CHOOSE_MOVES_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_moves",
        "description": "Choose the 2048 direction sequence to play from the current board.",
        "parameters": {
            "type": "object",
            "properties": {
                "moves": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "description": "Legal directions to play, in order.",
                    "items": {"type": "string", "enum": ["up", "down", "left", "right"]},
                }
            },
            "required": ["moves"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are an expert 2048 player. "
    "Choose a sequence of legal directions by calling the choose_moves tool. "
    "Order moves to maximize merges and grow toward larger tiles. "
    "Stop once no direction changes the board; do not pad with no-op moves. "
    f"Emit at most {game.DEFAULT_EPISODE_CAP} directions; anything beyond that is dropped."
)

# DEV-LOG: remove before launch -- verbose dev logging for diagnosing the
# LLM-mode "board never changes" failure. Every `_dev_log` call (and this
# flag/helper) is temporary and should be deleted at launch.
DEV_VERBOSE = True


def _dev_log(message: str) -> None:
    # DEV-LOG: remove before launch
    if DEV_VERBOSE:
        sys.stderr.write(f"2048-web [DEV]: {message}\n")


class Game2048Server(ThreadingHTTPServer):
    """Small stateful HTTP server for one local 2048 game."""

    def __init__(self, server_address, request_handler, *, seed: int | None, cap: int, args):
        super().__init__(server_address, request_handler)
        self.args = args
        self.cap = cap
        self.seed = seed if seed is not None else random.randrange(0, 2**31)
        self.rng = random.Random(self.seed)
        self.agent_mode = args.agent_mode
        self.start_board: game.Board = _empty_board()
        self.board: game.Board = _empty_board()
        self.score = 0
        self.moves_played = 0
        self.invalid_moves = 0
        self.terminal = False
        self.events: list[str] = []
        self.openai_client = None
        # Last LLM episode's action source ("tool_call" | "text_fallback" | None),
        # surfaced in the per-request CLI summary line so it lands in the log file.
        self.last_policy_source: str | None = None
        # Last LLM episode metrics (score / improvement / reward) shown as
        # dedicated UI pills; None until the first LLM episode lands.
        self.last_episode: dict | None = None
        _reset(self)


class Game2048Handler(BaseHTTPRequestHandler):
    server: Game2048Server

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            payload = _payload(self.server)
            self._log_action("state", payload)
            self._send_json(payload)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route_path(self.path)
        handler = {
            "new": lambda body: _on_new(self.server, body),
            "move": lambda body: _on_move(self.server, body),
            "agent": lambda body: _agent_turn(self.server, str(body.get("mode", "random")) if isinstance(body, dict) else "random"),
        }.get(route)
        if handler is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        body = self._read_json()
        payload = self._handle_route(route, handler, body)
        self._send_json(payload)

    def _handle_route(self, route: str, handler, body: object) -> dict[str, Any]:
        """Run a POST handler; never let an exception reach the client.

        On error, append a friendly event (no traceback) and still return a 200
        JSON payload so the browser keeps a usable state.
        """
        try:
            payload = handler(body)
            self._log_action(route, payload, body)
            return payload
        except Exception as exc:  # noqa: BLE001 -- convert any failure to an event
            message = _friendly_error(exc)
            _append_event(self.server, message)
            payload = _payload(self.server)
            self._log_action(route, payload, body, error=message)
            return payload

    def log_message(self, fmt: str, *args: object) -> None:
        # Silence the default access-log lines; meaningful logs come from _log_action.
        return

    def _log_action(
        self, route: str, payload: dict[str, Any], body: object | None = None, error: str | None = None
    ) -> None:
        action = _describe_action(self.server, route, body, error)
        stats = (
            f"Score {payload['score']} | Max {payload['max_tile']} | "
            f"Invalid {payload['invalid_moves']}/{payload['moves_played']}"
            f"{' [game over]' if payload['terminal'] else ''}"
        )
        sys.stderr.write(f"2048-web: {action}  --  {stats}\n")
        sys.stderr.write(_board_ascii(payload["board"]) + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _route_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    if path.endswith("/api/state"):
        return "state"
    if path.endswith("/api/new"):
        return "new"
    if path.endswith("/api/move"):
        return "move"
    if path.endswith("/api/agent"):
        return "agent"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _empty_board() -> game.Board:
    return [[0 for _ in range(game.SIZE)] for _ in range(game.SIZE)]


def _on_new(server: Game2048Server, body: object) -> dict[str, Any]:
    agent_mode = body.get("agent_mode") if isinstance(body, dict) else None
    seed = body.get("seed") if isinstance(body, dict) else None
    _reset(server, agent_mode=agent_mode, seed=seed)
    return _payload(server)


def _on_move(server: Game2048Server, body: object) -> dict[str, Any]:
    direction = body.get("direction") if isinstance(body, dict) else None
    return _human_move(server, direction)


def _friendly_error(exc: Exception) -> str:
    """Turn a raised exception into a short, user-facing event line."""

    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    return f"Agent error: {text}. (Switch modes with the buttons, then New Game.)"


def _describe_action(
    server: Game2048Server, route: str, body: object | None, error: str | None
) -> str:
    """A one-line, human-readable description of what the user just did."""

    mode = server.agent_mode
    if route == "new":
        return f"new game (seed={server.seed})"
    if route == "move":
        direction = body.get("direction") if isinstance(body, dict) else "?"
        return f"human move: {direction}"
    if route == "agent":
        who = body.get("mode") if isinstance(body, dict) else mode
        last = server.events[-1] if server.events else ""
        # For LLM episodes, tag the summary with the policy action source so the
        # single per-request CLI line (which lands in the log file) states whether
        # the episode came from a real choose_moves tool_call or the text fallback.
        if who == "llm" and server.last_policy_source:
            return f"{who} step [{server.last_policy_source}]: {last}"
        return f"{who} step: {last}"
    if route == "state":
        return "state poll"
    return route


def _board_ascii(board: game.Board) -> str:
    """A compact 4-line board snapshot for the CLI log."""

    rows = [
        "  " + " ".join((str(c) if c != 0 else ".") for c in row)
        for row in board
    ]
    return "\n".join(rows)


def _reset(server: Game2048Server, *, agent_mode: Any = None, seed: Any = None) -> None:
    if agent_mode in {"human", "random", "llm"}:
        server.agent_mode = str(agent_mode)
    if seed is not None:
        try:
            server.seed = int(seed)
        except (TypeError, ValueError):
            pass  # ignore an unparseable seed and keep the current value
    server.rng = random.Random(server.seed)
    server.start_board = _empty_board()
    for _ in range(START_SPAWNS):
        server.start_board = game.spawn_tile(server.start_board, server.rng)
    server.board = [list(row) for row in server.start_board]
    server.score = 0
    server.moves_played = 0
    server.invalid_moves = 0
    server.terminal = game.is_terminal(server.board)
    server.events = [
        f"New game (seed={server.seed}).",
        f"Max tile so far: {game.max_tile(server.board)}.",
    ]
    server.last_episode = None


def _append_event(server: Game2048Server, message: str) -> None:
    """Append a history event; events are never truncated so the whole game stays
    scrollable. Oldest first, newest at the bottom."""

    server.events.append(message)


def _step(server: Game2048Server, direction: str, *, who: str = "human") -> bool:
    """Apply one direction to the live board; return whether it changed."""

    if server.terminal:
        _append_event(server, "Game over. Start a new board.")
        return False
    new_board, gained, changed = game.slide(server.board, direction)
    server.moves_played += 1
    if not changed:
        server.invalid_moves += 1
        _append_event(server, f"[{who}] invalid (no-op) move: {direction}.")
        return False
    server.board = game.spawn_tile(new_board, server.rng)
    server.score += gained
    won = game.max_tile(server.board) >= game.WIN_TILE
    stuck = game.is_terminal(server.board)
    server.terminal = won or stuck
    if won:
        _append_event(server, f"[{who}] played {direction} (+{gained}). Reached 2048 — you win!")
    elif stuck:
        _append_event(server, f"[{who}] played {direction} (+{gained}). Board full, no merge — game over.")
    else:
        _append_event(server, f"[{who}] played {direction} (+{gained}). Max tile: {game.max_tile(server.board)}.")
    return True


def _human_move(server: Game2048Server, direction: Any) -> dict[str, Any]:
    direction = str(direction).lower().strip() if direction else ""
    if direction not in game.ACTIONS:
        _append_event(server, f"Unknown direction: {direction!r}")
        return _payload(server)
    server.agent_mode = "human"
    _step(server, direction, who="human")
    return _payload(server)


def _agent_turn(server: Game2048Server, mode: str) -> dict[str, Any]:
    """Run one agent action: ``random`` (one uniform step) or ``llm`` (episode)."""

    if mode == "llm":
        return _llm_episode(server)
    return _random_step(server)


def _random_step(server: Game2048Server) -> dict[str, Any]:
    if server.terminal:
        _append_event(server, "Random: game over. Start a new board.")
        return _payload(server)
    server.agent_mode = "random"
    # Match game.random_episode: a uniform-random *direction* (all four), not a
    # legal-only pick. A no-op direction is counted as invalid by _step, so the
    # browser's Random mode reports the same invalid-rate semantics as the
    # training/eval baseline.
    direction = server.rng.choice(game.ACTIONS)
    _step(server, direction, who="random")
    return _payload(server)


def _llm_episode(server: Game2048Server) -> dict[str, Any]:
    if not server.args.base_url:
        raise ValueError("LLM mode requires --base-url")
    if server.openai_client is None:
        server.openai_client = _make_openai_client(server.args)

    server.agent_mode = "llm"
    start_board = [list(row) for row in server.board]
    moves, policy_source = _policy_moves(server, start_board)
    server.last_policy_source = policy_source
    # Surface whether the policy actually emitted a choose_moves tool_call, or
    # whether the plan came from the text fallback. The History panel shows it
    # directly so a no-tool-call degradation is visible without grepping logs.
    # --strict-tool-call rejects the fallback outright, mirroring reward.py's
    # no-tool-call penalty: a policy that cannot produce a tool call must fail
    # loudly instead of silently playing prose-parsed directions.
    if server.args.strict_tool_call and policy_source != "tool_call":
        raise ValueError(
            f"policy did not emit a choose_moves tool call (source={policy_source!r}); "
            "--strict-tool-call rejects the text fallback. Train longer or rerun without the flag."
        )
    moves_preview = ", ".join(moves) if moves else "-"
    if policy_source == "tool_call":
        _append_event(server, f"[llm] tool_call OK: choose_moves → [{moves_preview}] ({len(moves)} moves).")
    else:
        _append_event(
            server,
            f"[llm] NO tool_call: text fallback → [{moves_preview}] ({len(moves)} moves).",
        )
    # DEV-LOG: remove before launch
    _dev_log(f"episode start seed={server.seed} cap={server.cap} moves={moves}")
    # DEV-LOG: remove before launch
    _dev_log("episode start board:\n" + _board_ascii(start_board))
    result, frames = game.play_episode_frames(start_board, moves, seed=server.seed, cap=server.cap)
    # DEV-LOG: remove before launch
    _dev_log(
        f"episode result: score={result.score} max_tile={result.max_tile} "
        f"total_moves={result.total_moves} invalid={result.invalid_moves} "
        f"reached_2048={result.reached_2048} truncated={result.truncated} frames={len(frames)}"
    )
    for index, frame in enumerate(frames):
        # DEV-LOG: remove before launch -- one board snapshot per replayed move.
        # Prefix every frame line with the policy source so a `tail` of the
        # stderr log shows whether this episode came from a real choose_moves
        # tool_call or the text fallback, no matter which frames are in view
        # (the per-frame board dumps otherwise bury the source line above).
        _dev_log(
            f"  frame[{index}] src={policy_source} move={frame.get('move')!r} "
            f"changed={frame.get('changed')} gained={frame.get('gained')} score={frame.get('score')}\n"
            + _board_ascii(frame.get("board"))
        )
    server.board = result.board
    # Accumulate (don't overwrite): result.score is the merge score gained
    # *during this episode* from the start board, so adding it keeps Score as a
    # whole-game running total across human/random/llm modes instead of jumping
    # back to an episode-only value. moves/invalid follow the same rule so the
    # Invalid stat stays cumulative too.
    server.score += result.score
    server.moves_played += result.total_moves
    server.invalid_moves += result.invalid_moves
    # ``truncated`` only means the policy's plan hit the 32-step cap before the
    # board reached 2048 or a true dead end -- the game is NOT over and the user
    # may continue playing from the resulting board. Counting it as terminal
    # surfaced a false "Game over" the moment an LLM episode ran the full cap,
    # so terminal is gated on real end conditions only, mirroring ``_step``.
    server.terminal = result.reached_2048 or game.is_terminal(server.board)

    baseline = game.random_episode(start_board, seed=server.seed, cap=server.cap, trials=8)
    improvement = result.score - baseline["score"]
    reward = improvement - game.INVALID_PENALTY * result.invalid_moves
    # Last LLM episode metrics surfaced as dedicated UI pills: the episode's
    # own merge score, the trained-vs-random improvement, and the full RL
    # reward (improvement minus the no-op penalty) used by reward_fn. The UI
    # shows "-" for these while not in LLM mode.
    server.last_episode = {
        "score": result.score,
        "improvement": float(improvement),
        "reward": float(reward),
    }
    _append_event(
        server,
        f"[llm] episode: score={result.score} max_tile={result.max_tile} "
        f"invalid_rate={result.invalid_rate:.2f} moves={result.total_moves}.",
    )
    _append_event(
        server,
        f"[llm] random baseline (same board+seed): score={baseline['score']:.1f} "
        f"max_tile={baseline['max_tile']:.1f}.",
    )
    _append_event(server, f"[llm] trained-vs-baseline improvement: {improvement:+.1f}.")
    payload = _payload(server)
    payload["frames"] = frames
    return payload


def _policy_moves(server: Game2048Server, board: game.Board) -> tuple[list[str], str]:
    # DEV-LOG: remove before launch -- dump the board we send and the raw policy
    # response so we can tell whether the model emits a choose_moves tool_call
    # or plain-text directions that the serve parser does not recognise.
    _dev_log("policy request board:\n" + _board_ascii(board))
    response = server.openai_client.chat.completions.create(
        model=server.args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game.format_prompt(board)},
        ],
        tools=[CHOOSE_MOVES_TOOL],
        tool_choice={"type": "function", "function": {"name": "choose_moves"}},
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    # `tool_calls` may be ``None`` (model produced no tool call this turn) per the
    # OpenAI spec; fall back to ``[]`` so the iteration below does not crash.
    message = (choices[0].get("message", {}) or {}) if choices else {}
    finish_reason = choices[0].get("finish_reason") if choices else None
    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    # DEV-LOG: remove before launch
    _dev_log(f"policy response finish_reason={finish_reason!r} tool_calls={tool_calls!r}")
    # DEV-LOG: remove before launch -- full content, not truncated, for analysis
    _dev_log(f"policy response content={content!r}")
    for call in tool_calls:
        if call.get("function", {}).get("name") != "choose_moves":
            continue
        arguments = call.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # DEV-LOG: remove before launch
                _dev_log("choose_moves arguments not JSON; falling back to content text.")
                break
        moves = game.parse_moves(arguments)
        # DEV-LOG: remove before launch
        _dev_log(f"parsed moves from tool_call: {moves}")
        return moves, "tool_call"
    # No recognised tool_call: mirror reward.py's text fallback so the demo can
    # replay plain-text directions the policy may emit before it learns tool use.
    # The returned source label lets the UI (and --strict-tool-call) distinguish
    # a real tool call from this prose fallback, so a half-trained policy that
    # never emits tool_calls cannot silently coast on parsed text.
    moves = game.parse_moves(content) if isinstance(content, str) else []
    # DEV-LOG: remove before launch
    _dev_log(f"no choose_moves tool_call parsed; text-fallback moves: {moves}")
    return moves, "text_fallback"


def _payload(server: Game2048Server) -> dict[str, Any]:
    return {
        "board": server.board,
        "score": server.score,
        "seed": server.seed,
        "max_tile": game.max_tile(server.board),
        "moves_played": server.moves_played,
        "invalid_moves": server.invalid_moves,
        "invalid_rate": (server.invalid_moves / server.moves_played) if server.moves_played else 0.0,
        "terminal": server.terminal,
        "legal_moves": game.legal_moves(server.board),
        "agent_mode": server.agent_mode,
        "agent_name": _agent_name(server),
        "last_episode": server.last_episode,
        "events": server.events,
    }


def _agent_name(server: Game2048Server) -> str:
    return {"human": "Human", "random": "Random", "llm": "LLM"}.get(server.agent_mode, server.agent_mode)


def _make_openai_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("LLM mode requires `openai`. Install it with `pip install openai`.") from exc
    return OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2048</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#24313a;background:#faf8ef}
html,body{height:100%}
body{margin:0;height:100vh;background:linear-gradient(135deg,#faf8ef,#f6e7c8 60%,#e9c46a);display:flex;align-items:center;justify-content:center;overflow:hidden;padding:10px;box-sizing:border-box}
.app{width:min(1100px,98vw);height:calc(100vh - 20px);display:grid;grid-template-columns:minmax(min-content,440px) 230px minmax(260px,1fr);gap:12px;align-items:stretch}
.panel{background:#fffaf0;border:4px solid #27313a;border-radius:22px;box-shadow:8px 8px 0 #27313a;padding:12px 14px;min-height:0}
.board-col{display:flex;flex-direction:column;gap:8px;justify-content:center;overflow:hidden}
.head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:34px;line-height:1;margin:0;color:#e05d2f;text-shadow:2px 2px 0 #ffd977}
.subtitle{font-weight:900;color:#5d3d1d}
.board{background:#bbada0;border:6px solid #27313a;border-radius:16px;padding:7px;display:grid;grid-template-columns:repeat(4,84px);gap:7px;box-shadow:inset 0 6px 0 rgba(255,255,255,.25)}
.cell{width:84px;height:84px;background:#cdc1b4;border:3px solid #27313a;border-radius:12px;display:grid;place-items:center;font-weight:1000;transition:.12s transform}
.cell .tile{font-size:30px;line-height:.8}
.cell.drop{animation:pop .2s ease}@keyframes pop{0%{transform:scale(.7)}70%{transform:scale(1.06)}100%{transform:scale(1)}}
.t2{background:#eee4da}.t4{background:#ede0c8}.t8{background:#f2b179;color:#fff}.t16{background:#f59563;color:#fff}.t32{background:#f67c5f;color:#fff}.t64{background:#f65e3b;color:#fff}.t128{background:#edcf72;font-size:26px}.t256{background:#edcc61;font-size:26px}.t512{background:#edc850;font-size:26px}.t1024{background:#edc53a;font-size:22px;color:#fff}.t2048{background:#edc22e;font-size:22px;color:#fff}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 6px}.ep-stats{display:flex;gap:8px;flex-wrap:wrap;flex-basis:100%}.pill{background:#fff;border:3px solid #27313a;border-radius:999px;padding:6px 10px;font-weight:1000}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.mode-control,.actions,.seed-row{display:flex;gap:8px;align-items:center}
button{border:3px solid #27313a;border-radius:14px;background:#ffd166;box-shadow:4px 4px 0 #27313a;color:#27313a;font-weight:1000;padding:9px 12px;cursor:pointer}
button:hover{transform:translateY(-1px)}button:disabled{filter:grayscale(.75);opacity:.55;cursor:not-allowed}.choice.active,#llmAutoBtn.active{background:#8ec9ec}
.label{font-weight:1000;color:#5d3d1d}
input[type=number]{width:110px;border:3px solid #27313a;border-radius:12px;padding:7px 10px;font-weight:1000;background:#fff;color:#27313a;font-family:inherit}
.reserve{margin-top:6px}
.thinking{display:none;padding:2px 6px;border:3px solid #27313a;border-radius:10px;background:#dff6ff;font-weight:1000;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.thinking.on{display:block}
.dots::after{content:"";animation:dots 1s steps(4,end) infinite}@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}100%{content:""}}
.banner{display:none;padding:6px 10px;border:3px solid #27313a;border-radius:12px;font-weight:1000;text-align:center;white-space:nowrap}
.banner.on{display:block}.banner.win{background:#9be564;color:#1d3d12}.banner.lose{background:#f3d0d0;color:#5d1d1d}
.controls-col{display:flex;flex-direction:column;gap:10px;justify-content:flex-start;min-height:0;overflow:hidden}
.events-panel{display:flex;flex-direction:column;min-height:0;overflow:hidden}
.events-head{font-weight:1000;color:#14866f;margin:0 0 6px}
.events{display:flex;flex-direction:column;gap:6px;overflow-y:auto;flex:1;min-height:0;padding-right:4px}
.event{background:#fff;border:3px solid #27313a;border-radius:12px;padding:8px 10px;font-weight:800;box-sizing:border-box;overflow-wrap:break-word}
.rules{font-weight:800;line-height:1.4;color:#5d3d1d;margin:0 0 8px;font-size:14px}
@media(max-width:560px){h1{font-size:28px}.head{gap:8px}}
</style>
</head>
<body>
<main class="app">
  <section class="panel" style="display:flex;flex-direction:column;gap:8px">
    <div class="head">
      <h1>2048</h1>
      <div class="subtitle">seeded engine · agentic RL demo</div>
    </div>
    <div id="board" class="board" aria-label="2048 board"></div>
    <div class="stats">
      <div class="pill" id="score"></div>
      <div class="pill" id="maxTile"></div>
      <div class="pill" id="invalid"></div>
      <div class="ep-stats">
        <div class="pill" id="epScore"></div>
        <div class="pill" id="epImprove"></div>
        <div class="pill" id="epReward"></div>
      </div>
    </div>
    <div class="reserve">
      <div id="thinking" class="thinking">Agent is thinking<span class="dots"></span></div>
      <div id="banner" class="banner"></div>
    </div>
  </section>
  <section class="panel controls-col">
    <div class="label">Actions</div>
    <button id="randomBtn">Random Step</button>
    <button id="llmBtn">LLM Episode</button>
    <button id="llmAutoBtn">LLM Auto</button>
    <div class="seed-row" style="display:flex;flex-direction:column;gap:4px;margin-top:8px">
      <span class="label">Seed</span>
      <input type="number" id="seedInput" aria-label="Random seed">
      <button id="new">New Game</button>
    </div>
    <p class="rules" style="margin-top:auto">Human: arrows/WASD. Random Step: one uniformly-random direction (all four; no-ops counted as invalid). LLM Episode: a served policy plays a whole episode, animated step-by-step. LLM Auto: loop LLM episodes automatically (current board fed back each turn) until you stop or the game ends.</p>
  </section>
  <aside class="panel events-panel">
    <div class="events-head">History</div>
    <div id="events" class="events"></div>
  </aside>
</main>
<script>
const api = (path) => new URL(path, window.location.href).toString();
let state = null, lastBoard = "", agentBusy = false, lastMode = "", llmAuto = false, autoTimer = null;
const FRAME_MS = 220;
async function request(path, body){
  const opts = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const res = await fetch(api(path), opts);
  state = await res.json();
  render();
}
function drawBoard(board){
  const el = document.getElementById("board");
  const key = JSON.stringify(board);
  el.innerHTML = "";
  board.flat().forEach((cell) => {
    const div = document.createElement("div");
    div.className = `cell ${tileClass(cell)} ${key !== lastBoard ? "drop" : ""}`;
    div.innerHTML = cell ? `<span class="tile">${cell}</span>` : "";
    el.appendChild(div);
  });
  lastBoard = key;
}
function tileClass(v){return v ? "t"+v : "";}
function render(){
  drawBoard(state.board);
  document.getElementById("score").textContent = `Score ${state.score}`;
  document.getElementById("maxTile").textContent = `Max ${state.max_tile}`;
  document.getElementById("invalid").textContent = `Invalid ${state.invalid_moves}/${state.moves_played}`;
  // LLM-only pills: show the last LLM episode's metrics, or "-" while not in
  // LLM mode (human/random) or before the first episode lands.
  const ep = (state.agent_mode === "llm" && state.last_episode) ? state.last_episode : null;
  const epFmt = v => (v > 0 ? "+" : "") + v.toFixed(0);
  document.getElementById("epScore").textContent = ep ? `Episode ${ep.score}` : "Episode -";
  document.getElementById("epImprove").textContent = ep ? `vsR ${epFmt(ep.improvement)}` : "vsR -";
  document.getElementById("epReward").textContent = ep ? `Reward ${epFmt(ep.reward)}` : "Reward -";
  const seed = document.getElementById("seedInput");
  if (document.activeElement !== seed) seed.value = state.seed;  // don't clobber while typing
  const blocked = agentBusy || state.terminal;
  document.getElementById("randomBtn").disabled = blocked;
  document.getElementById("llmBtn").disabled = blocked || llmAuto;
  document.getElementById("llmAutoBtn").disabled = state.terminal;
  document.getElementById("llmAutoBtn").textContent = llmAuto ? "LLM Stop" : "LLM Auto";
  document.getElementById("llmAutoBtn").classList.toggle("active", llmAuto);
  if (state.terminal && llmAuto) stopAuto();
  document.getElementById("thinking").classList.toggle("on", agentBusy && lastMode === "llm");
  const banner = document.getElementById("banner");
  banner.classList.remove("on","win","lose");
  if (state.terminal) {
    const won = state.max_tile >= 2048;
    banner.classList.add("on", won ? "win" : "lose");
    banner.textContent = won ? "You win — reached 2048!" : "Game over — no moves left.";
  }
  // History: oldest on top, newest at the bottom; keep pinned to the newest entry
  // when the user isn't scrolled up reviewing older moves.
  const box = document.getElementById("events");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.innerHTML = state.events.map(e => `<div class="event">${escapeHtml(e)}</div>`).join("");
  if (nearBottom) box.scrollTop = box.scrollHeight;
}
function newGame(){
  const seed = Number(document.getElementById("seedInput").value);
  stopAuto();
  request("api/new", {seed: seed});
}
function sleep(ms){return new Promise(r => setTimeout(r, ms));}
async function fetchAgent(mode){
  const res = await fetch(api("api/agent"), {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode: mode})});
  return res.json();
}
async function agentAction(mode){
  if(agentBusy || state.terminal) return;
  lastMode = mode;
  agentBusy = true; render();
  try {
    const data = await fetchAgent(mode);
    if (mode === "llm" && Array.isArray(data.frames) && data.frames.length){
      // Animate the policy's plan one frame at a time so the board transitions
      // smoothly instead of jumping to the final state.
      for (let i = 0; i < data.frames.length; i++){
        const f = data.frames[i];
        state.board = f.board; state.score = f.score;
        render();
        await sleep(FRAME_MS);
      }
    }
    state = data;
  } finally { agentBusy = false; render(); }
}
function toggleAuto(){
  if (llmAuto){ stopAuto(); return; }
  if (state.terminal) return;
  llmAuto = true; render(); autoLoop();
}
function stopAuto(){
  llmAuto = false;
  if (autoTimer){ clearTimeout(autoTimer); autoTimer = null; }
  render();
}
async function autoLoop(){
  if (!llmAuto) return;
  if (state.terminal){ stopAuto(); return; }
  await agentAction("llm");
  if (!llmAuto || state.terminal){ stopAuto(); return; }
  autoTimer = setTimeout(autoLoop, 600);
}
function escapeHtml(t){return t.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));}
document.getElementById("randomBtn").onclick = () => agentAction("random");
document.getElementById("llmBtn").onclick = () => agentAction("llm");
document.getElementById("llmAutoBtn").onclick = toggleAuto;
document.getElementById("new").onclick = newGame;
document.getElementById("seedInput").addEventListener("keydown", (e) => { if(e.key === "Enter"){ e.preventDefault(); newGame(); } });
window.addEventListener("keydown", (e) => {
  if(document.activeElement && document.activeElement.id === "seedInput") return;
  if(!state || state.terminal) return;
  const map = {ArrowUp:"up",ArrowDown:"down",ArrowLeft:"left",ArrowRight:"right",w:"up",s:"down",a:"left",d:"right",W:"up",S:"down",A:"left",D:"right"};
  const dir = map[e.key];
  if(dir) { e.preventDefault(); stopAuto(); request("api/move", {direction:dir}); }
});
fetch(api("api/state")).then(r=>r.json()).then(s=>{state=s;render();});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2048 local web UI demo.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cap", type=int, default=game.DEFAULT_EPISODE_CAP)
    parser.add_argument("--agent-mode", choices=("human", "random", "llm"), default="human")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for LLM mode.")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    parser.add_argument(
        "--strict-tool-call",
        action="store_true",
        help=(
            "Reject LLM episodes where the policy did not emit a choose_moves tool call. "
            "Without this flag, plain-text directions are parsed as a fallback so the demo "
            "still plays; with it, a no-tool-call turn raises an error event so a "
            "half-trained policy cannot coast on prose."
        ),
    )
    args = parser.parse_args()

    server = Game2048Server((args.host, args.port), Game2048Handler, seed=args.seed, cap=args.cap, args=args)
    url = f"http://{args.host}:{args.port}"
    print(f"2048 web UI running at {url} (agent-mode={args.agent_mode})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()