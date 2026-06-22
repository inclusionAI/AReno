"""Web UI server for the DuelGrid agentic example.

Run from the repository root:

    python examples/agentic/duelgrid/web_ui.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class LLMActionParseError(RuntimeError):
    """Raised when an LLM response cannot be parsed into DuelGrid actions."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


SYSTEM_PROMPT = (
    "You are an expert DuelGrid player controlling A. "
    "Choose one or more legal actions by calling the choose_action tool. "
    "The tool name is always choose_action; never use MOVE, ATTACK, or another action as the tool name. "
    "Copy legal action objects from the prompt into the actions array until energy is spent."
)

CHOOSE_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_action",
        "description": "Choose the next legal DuelGrid action sequence for player A.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["MOVE", "ATTACK", "RANGED_ATTACK", "SHIELD", "PICKUP", "WAIT"],
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["UP", "DOWN", "LEFT", "RIGHT"],
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
    },
}


class DuelGridWebServer(ThreadingHTTPServer):
    """Small stateful HTTP server for one local DuelGrid game."""

    def __init__(
        self,
        server_address,
        request_handler,
        *,
        seed: int | None = None,
        llm_args: argparse.Namespace | None = None,
    ):
        super().__init__(server_address, request_handler)
        self.rng = random.Random(seed)
        self.state = _new_state(self.rng)
        self.pending_user_actions: list[dict[str, str]] = []
        self.user_reward = 0.0
        self.animations: list[dict[str, Any]] = []
        self.awaiting_agent = False
        self.llm_args = llm_args
        self.llm_client = _make_openai_client(llm_args) if llm_args else None
        self.events = ["New random map. You control U; the LLM controls A."]


class DuelGridHandler(BaseHTTPRequestHandler):
    server: DuelGridWebServer

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            self._send_json(_payload(self.server))
        elif route == "new":
            _reset_game(self.server)
            self._send_json(_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route_path(self.path)
        if route == "new":
            _reset_game(self.server)
            self._send_json(_payload(self.server))
        elif route == "action":
            body = self._read_json()
            action = body.get("action") if isinstance(body, dict) else None
            if not isinstance(action, dict):
                self._send_json({"error": "expected JSON body with action object"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(_apply_user_action(self.server, action))
        elif route == "agent":
            self._send_json(_apply_agent_turn(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("duelgrid-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data)

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
    if path.endswith("/api/action"):
        return "action"
    if path.endswith("/api/agent"):
        return "agent"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _new_state(rng: random.Random) -> game.State:
    record = dataset_generator.generate_records(1, seed=rng.randrange(1_000_000_000))[0]
    return dataset_generator.record_to_state(record)


def _reset_game(server: DuelGridWebServer) -> None:
    server.state = _new_state(server.rng)
    server.pending_user_actions = []
    server.user_reward = 0.0
    server.animations = []
    server.awaiting_agent = False
    server.events = ["Switched to a new random map."]


def _apply_user_action(server: DuelGridWebServer, action: dict[str, str]) -> dict[str, Any]:
    state = server.state
    server.animations = []
    if getattr(server, "awaiting_agent", False):
        server.events.insert(0, "Agent is still choosing an action.")
        return _payload(server)
    if game.is_terminal(state):
        server.events.insert(0, "Game is over. Start a new map.")
        return _payload(server)

    parsed = game.parse_action(action)
    if parsed is None:
        server.events.insert(0, f"Invalid action payload: {action}")
        return _payload(server)

    next_state, reward, done, info = game.step(state, parsed, game.USER, advance_turn=False)
    if info.get("illegal"):
        cost = min(game.ACTION_COSTS.get(parsed["action"], 0), state.user.energy)
        if cost > 0:
            server.state = game.spend_energy(state, game.USER, cost)
            server.pending_user_actions.append(parsed)
            server.user_reward -= 0.3
            server.events.insert(0, f"Illegal {parsed} spent {cost} energy.")
        else:
            server.events.insert(0, f"Illegal {parsed}.")
        if server.state.user.energy <= 0:
            _finish_user_turn(server)
        return _payload(server)

    server.state = next_state
    server.pending_user_actions.append(parsed)
    server.user_reward += reward
    animation = _animation_for_action(state, parsed, game.USER)
    if animation:
        server.animations.append(animation)
    server.events.insert(0, f"User {parsed} reward={reward:+.2f}")
    if done or parsed["action"] == "WAIT" or server.state.user.energy <= 0:
        _finish_user_turn(server)
    return _payload(server)


def _finish_user_turn(server: DuelGridWebServer) -> None:
    state = game.end_turn(server.state, game.USER)
    server.events.insert(0, f"User turn: {server.pending_user_actions} total={server.user_reward:+.2f}")
    server.pending_user_actions = []
    server.user_reward = 0.0
    server.state = state
    if game.is_terminal(state):
        return
    server.awaiting_agent = True
    server.events.insert(0, "Waiting for LLM agent action.")
    return


def _apply_agent_turn(server: DuelGridWebServer) -> dict[str, Any]:
    server.animations = []
    if game.is_terminal(server.state):
        server.awaiting_agent = False
        return _payload(server)
    if not getattr(server, "awaiting_agent", False):
        server.events.insert(0, "No pending agent turn.")
        return _payload(server)
    _run_agent_turn(server)
    return _payload(server)


def _run_agent_turn(server: DuelGridWebServer) -> None:
    state = server.state
    server.awaiting_agent = False
    try:
        agent_actions, raw_response = _agent_actions(server, state)
    except LLMActionParseError as exc:
        server.state = state
        server.events.insert(0, f"Agent LLM call failed: {exc}")
        if getattr(server.llm_args, "debug_llm", False) and exc.raw_response:
            server.events.insert(1, f"LLM raw output: {exc.raw_response}")
        return
    except Exception as exc:  # pragma: no cover - interactive/network error path
        server.state = state
        server.events.insert(0, f"Agent LLM call failed: {exc}")
        return
    for action in agent_actions:
        animation = _animation_for_action(state, action, game.AGENT)
        if animation:
            server.animations.append(animation)
        next_state, _reward, done, _info = game.step(state, action, game.AGENT, advance_turn=False)
        if done:
            break
        state = next_state
    state, reward, _done, info = game.step_turn(server.state, agent_actions, game.AGENT)
    server.state = state
    server.events.insert(0, f"Agent turn: {info['applied_actions']} reward={reward:+.2f}")
    if getattr(getattr(server, "llm_args", None), "debug_llm", False) and raw_response:
        server.events.insert(1, f"LLM raw output: {raw_response}")


def _agent_actions(server: DuelGridWebServer, state: game.State) -> tuple[list[dict[str, str]], str | None]:
    if getattr(server, "llm_client", None) is None or getattr(server, "llm_args", None) is None:
        raise RuntimeError("LLM mode is enabled but the OpenAI-compatible client is not configured")
    return _llm_agent_actions(state, server.llm_args, server.llm_client)


def _make_openai_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("LLM mode requires `openai`. Install it with `pip install openai`.") from exc
    kwargs = {"api_key": args.api_key}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    return OpenAI(**kwargs)


def _llm_agent_actions(
    state: game.State, args: argparse.Namespace, llm_client
) -> tuple[list[dict[str, str]], str | None]:
    response = llm_client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game.format_prompt(state)},
        ],
        tools=[CHOOSE_ACTION_TOOL],
        tool_choice={"type": "function", "function": {"name": "choose_action"}},
        temperature=args.temperature,
        stream=False,
    )
    raw_response = _format_llm_raw_response(response)
    actions = _actions_from_openai_response(response)
    if not actions:
        raise LLMActionParseError(
            "LLM response did not contain a parseable choose_action tool call or JSON action",
            raw_response,
        )
    return actions, raw_response


def _actions_from_openai_response(response) -> list[dict[str, str]]:
    """Extract DuelGrid actions from an OpenAI-format chat completion."""

    choices = getattr(response, "choices", None) or []
    if not choices and isinstance(response, dict):
        choices = response.get("choices") or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    if message is None and isinstance(choices[0], dict):
        message = choices[0].get("message")
    tool_calls = _message_get(message, "tool_calls") or []
    for call in tool_calls:
        function = _message_get(call, "function") or {}
        if _message_get(function, "name") != "choose_action":
            continue
        arguments = _message_get(function, "arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return []
        return game.parse_actions(arguments)
    return game.parse_actions(_message_get(message, "content"))


def _message_get(value, key: str):
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _format_llm_raw_response(response) -> str:
    """Return a stable debug string for OpenAI SDK or dict chat responses."""

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), indent=2, default=str)
    return json.dumps(response, indent=2, default=str) if isinstance(response, dict) else repr(response)


def _animation_for_action(state: game.State, action: dict[str, str], actor: str) -> dict[str, Any] | None:
    parsed = game.parse_action(action)
    if not parsed or parsed["action"] not in {"ATTACK", "RANGED_ATTACK"}:
        return None
    direction = parsed.get("direction")
    if direction not in game.DIRECTIONS:
        return None
    player = state.user if actor == game.USER else state.agent
    opponent = state.agent if actor == game.USER else state.user
    dr, dc = game.DIRECTIONS[direction]
    source = {"row": player.row, "col": player.col}
    if parsed["action"] == "ATTACK":
        target = {"row": player.row + dr, "col": player.col + dc}
    else:
        target = {"row": opponent.row, "col": opponent.col}
        if (dr and opponent.col != player.col) or (dc and opponent.row != player.row):
            target = {"row": player.row + dr * 3, "col": player.col + dc * 3}
    return {
        "actor": "user" if actor == game.USER else "agent",
        "kind": "melee" if parsed["action"] == "ATTACK" else "ranged",
        "direction": direction,
        "source": source,
        "target": target,
    }


def _payload(server: DuelGridWebServer) -> dict[str, Any]:
    state = server.state
    return {
        "state": _state_dict(state),
        "map": game.render_map(state).splitlines(),
        "legal_actions": game.legal_actions(state, game.USER),
        "events": server.events[:8],
        "pending_user_actions": server.pending_user_actions,
        "animations": getattr(server, "animations", []),
        "awaiting_agent": getattr(server, "awaiting_agent", False),
        "done": game.is_terminal(state),
        "terminal_score": game.terminal_score(state) if game.is_terminal(state) else None,
    }


def _state_dict(state: game.State) -> dict[str, Any]:
    return {
        "agent": _player_dict(state.agent),
        "user": _player_dict(state.user),
        "turn": state.turn,
        "max_turns": state.max_turns,
    }


def _player_dict(player: game.Player) -> dict[str, int]:
    return {
        "row": player.row,
        "col": player.col,
        "hp": player.hp,
        "energy": player.energy,
        "max_energy": player.max_energy,
        "shield": player.shield,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the DuelGrid browser UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generated maps.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="OpenAI API key.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "policy"), help="Model name for LLM mode.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for LLM mode.")
    parser.add_argument(
        "--debug-llm", action="store_true", help="Show raw LLM chat completion output in the event log."
    )
    args = parser.parse_args()

    server = DuelGridWebServer(
        (args.host, args.port),
        DuelGridHandler,
        seed=args.seed,
        llm_args=args,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"DuelGrid web UI serving at {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping DuelGrid web UI.")
    finally:
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DuelGrid</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #dff7ff;
      --panel: #fff6d8;
      --panel-2: #fffdf5;
      --panel-3: #ffe5a2;
      --ink: #27333a;
      --muted: #67777f;
      --line: #d3a85f;
      --cyan: #2fa8e7;
      --red: #ef6969;
      --green: #45bd68;
      --gold: #f3b832;
      --orange: #f18846;
      --floor: #9be377;
      --floor-2: #76c85c;
      --wall: #7c6042;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 14% 12%, rgba(255,255,255,.95) 0 5rem, rgba(255,255,255,0) 5.2rem),
        radial-gradient(circle at 82% 8%, rgba(255,255,255,.78) 0 4rem, rgba(255,255,255,0) 4.2rem),
        linear-gradient(180deg, #98ddff 0, var(--bg) 38%, #bdf0a3 100%);
      color: var(--ink);
      font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1060px, calc(100vw - 20px));
      margin: 0 auto;
      padding: 10px 0 18px;
      display: grid;
      grid-template-columns: minmax(390px, 1fr) 330px;
      gap: 12px;
    }
    header {
      grid-column: 1 / -1;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 8px 2px 6px;
    }
    h1 { margin: 0; font-size: 26px; letter-spacing: 0; }
    .subtitle { margin: 4px 0 0; color: var(--muted); }
    button {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, #ffe9a8);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      cursor: pointer;
      font: inherit;
      min-height: 38px;
    }
    button:hover:not(:disabled) { border-color: #bf8d35; background: linear-gradient(180deg, #fffdf4, #ffd875); }
    button:disabled { opacity: .46; cursor: not-allowed; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      box-shadow: 0 10px 0 #c8954e, 0 16px 26px rgba(88, 66, 32, .18);
    }
    .arena {
      display: grid;
      gap: 12px;
      background: #fff0bd;
      border-color: #cf9a49;
    }
    .board-wrap {
      width: min(100%, 560px, calc(100vh - 232px));
      min-width: 0;
      margin: 0 auto;
      padding: 8px;
      background: #e7b765;
      border: 1px solid #b67d35;
      border-radius: 8px;
    }
    .board {
      display: grid;
      grid-template-columns: repeat(11, 1fr);
      grid-auto-rows: 1fr;
      gap: 3px;
      width: 100%;
      aspect-ratio: 1;
      position: relative;
    }
    .tile {
      display: grid;
      place-items: center;
      min-width: 0;
      min-height: 0;
      aspect-ratio: 1;
      border-radius: 6px;
      border: 1px solid rgba(85, 122, 58, .28);
      background: linear-gradient(145deg, var(--floor), var(--floor-2));
      box-shadow: inset 0 -5px 0 rgba(60, 128, 52, .2), inset 0 2px 0 rgba(255,255,255,.45);
      font-weight: 900;
      font-size: clamp(15px, 2vw, 22px);
      position: relative;
      overflow: hidden;
    }
    .tile:after {
      content: "";
      position: absolute;
      left: 12%;
      right: 12%;
      bottom: 9%;
      height: 15%;
      border-radius: 999px;
      background: rgba(69, 104, 46, .18);
    }
    .wall {
      background:
        linear-gradient(135deg, #8a6745 25%, #715133 25%, #715133 50%, #8a6745 50%, #8a6745 75%, #715133 75%);
      background-size: 18px 18px;
      border-color: #5c422b;
    }
    .wall:after { display: none; }
    .health { color: var(--green); background: linear-gradient(145deg, #d7ffd7, #a9eda5); }
    .energy { color: var(--gold); background: linear-gradient(145deg, #fff3af, #ffd46f); }
    .trap { color: var(--orange); background: linear-gradient(145deg, #ffd1b2, #f5a06d); }
    .item { z-index: 1; filter: drop-shadow(0 2px 0 rgba(255,255,255,.7)); }
    .sprite {
      width: 58%;
      height: 70%;
      position: relative;
      z-index: 2;
      animation: walk .78s steps(2, end) infinite;
      filter: drop-shadow(0 5px 2px rgba(111,83,43,.35));
    }
    .sprite .head {
      position: absolute;
      width: 42%;
      height: 34%;
      left: 29%;
      top: 0;
      border-radius: 50%;
      background: #f2d4b0;
      border: 2px solid rgba(0,0,0,.18);
    }
    .sprite .body {
      position: absolute;
      width: 54%;
      height: 40%;
      left: 23%;
      top: 30%;
      border-radius: 10px 10px 6px 6px;
      background: var(--cyan);
      border: 2px solid rgba(255,255,255,.18);
    }
    .sprite .arm, .sprite .leg {
      position: absolute;
      background: #f2d4b0;
      border-radius: 99px;
      transform-origin: top center;
    }
    .sprite .arm { width: 14%; height: 32%; top: 34%; }
    .sprite .arm.left { left: 12%; transform: rotate(20deg); }
    .sprite .arm.right { right: 12%; transform: rotate(-20deg); }
    .sprite .leg { width: 15%; height: 34%; top: 66%; background: #42546d; }
    .sprite .leg.left { left: 30%; transform: rotate(10deg); }
    .sprite .leg.right { right: 30%; transform: rotate(-10deg); }
    .sprite.agent-sprite .body { background: var(--red); }
    .sprite.agent-sprite .head { background: #e7edf3; }
    .sprite.agent-sprite:before {
      content: "";
      position: absolute;
      width: 52%;
      height: 8%;
      left: 24%;
      top: 24%;
      border-radius: 999px;
      background: #26323a;
      z-index: 3;
    }
    .user { box-shadow: inset 0 0 0 2px var(--cyan), 0 0 0 4px rgba(50,169,232,.16); }
    .agent { box-shadow: inset 0 0 0 2px var(--red), 0 0 0 4px rgba(239,104,104,.14); }
    .sprite.strike { animation: melee-strike .28s ease-out 1; }
    .sprite.ranged-cast { animation: ranged-cast .36s ease-out 1; }
    .tile.hit-melee { animation: melee-hit .38s ease-out 1; }
    .tile.hit-ranged { animation: ranged-hit .46s ease-out 1; }
    .projectile {
      position: fixed;
      left: 0;
      top: 0;
      z-index: 20;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: radial-gradient(circle at 35% 35%, #fff8bc, #39b9ff 55%, #1767c9 100%);
      box-shadow: 0 0 16px rgba(47,168,231,.85);
      pointer-events: none;
      animation: projectile-fade .42s linear 1;
    }
    @keyframes walk {
      0% { transform: translateY(0) rotate(-1deg); }
      50% { transform: translateY(-4%) rotate(1deg); }
      100% { transform: translateY(0) rotate(-1deg); }
    }
    @keyframes melee-strike {
      0% { transform: translateX(0) scale(1); }
      45% { transform: translateX(18%) scale(1.12); }
      100% { transform: translateX(0) scale(1); }
    }
    @keyframes ranged-cast {
      0% { transform: translateY(0) rotate(-1deg); }
      45% { transform: translateY(-10%) scale(.95); }
      100% { transform: translateY(0) rotate(-1deg); }
    }
    @keyframes melee-hit {
      0% { transform: scale(1); filter: brightness(1); }
      35% { transform: scale(1.12) rotate(-2deg); filter: brightness(1.3); }
      70% { transform: scale(.96) rotate(2deg); }
      100% { transform: scale(1); filter: brightness(1); }
    }
    @keyframes ranged-hit {
      0% { box-shadow: inset 0 -5px 0 rgba(60,128,52,.2), 0 0 0 rgba(47,168,231,0); }
      45% { box-shadow: inset 0 -5px 0 rgba(60,128,52,.2), 0 0 0 8px rgba(47,168,231,.34); filter: brightness(1.24); }
      100% { box-shadow: inset 0 -5px 0 rgba(60,128,52,.2), 0 0 0 rgba(47,168,231,0); filter: brightness(1); }
    }
    @keyframes projectile-fade {
      0% { opacity: 0; transform: translate(-50%, -50%) scale(.75); }
      15% { opacity: 1; }
      100% { opacity: 0; transform: translate(-50%, -50%) scale(1.05); }
    }
    .hud {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .fighter {
      display: grid;
      gap: 9px;
      padding: 10px;
      border-radius: 8px;
      background: var(--panel-2);
      border: 1px solid var(--line);
    }
    .fighter h2 {
      margin: 0;
      font-size: 15px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .bars { display: grid; gap: 7px; }
    .bar {
      height: 10px;
      border-radius: 99px;
      background: #f3dfab;
      overflow: hidden;
      border: 1px solid #d3aa5d;
    }
    .fill { height: 100%; border-radius: inherit; width: 0%; }
    .hp-fill { background: linear-gradient(90deg, #dc5d66, #86db83); }
    .energy-fill { background: linear-gradient(90deg, #efa63a, #ffe07a); }
    .meta {
      color: var(--muted);
      font-size: 13px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }
    .side { display: grid; gap: 10px; align-content: start; }
    .actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; }
    .actions button {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      text-align: left;
    }
    .actions button.primary { background: linear-gradient(180deg, #d7f4ff, #8dd9ff); border-color: #5baed6; }
    .key {
      display: inline-grid;
      place-items: center;
      min-width: 26px;
      height: 24px;
      border-radius: 6px;
      border: 1px solid #c09b58;
      background: #fff8d8;
      color: #4b3c2d;
      font-size: 12px;
      font-weight: 800;
    }
    .instructions {
      display: grid;
      gap: 10px;
    }
    .keys {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
    }
    .log {
      display: grid;
      gap: 8px;
      max-height: 210px;
      overflow: auto;
      padding-right: 4px;
    }
    .log-entry {
      padding: 8px 9px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: #415056;
      font-size: 13px;
      white-space: pre-wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 13px;
      background: #fff8d8;
    }
    .top-actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    .loading {
      pointer-events: none;
    }
    .thinking-overlay {
      position: fixed;
      left: 50%;
      top: 18px;
      z-index: 30;
      display: none;
      min-width: 260px;
      max-width: min(420px, calc(100vw - 28px));
      transform: translateX(-50%);
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border: 1px solid #d3a85f;
      border-radius: 8px;
      background: #fff8d8;
      color: var(--ink);
      box-shadow: 0 8px 0 #c8954e, 0 14px 26px rgba(88, 66, 32, .22);
      font-weight: 700;
    }
    .thinking-overlay small {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-weight: 600;
    }
    .spinner {
      width: 26px;
      height: 26px;
      border: 4px solid #ffe1a0;
      border-top-color: var(--cyan);
      border-radius: 50%;
      animation: spin .7s linear infinite;
      flex: 0 0 auto;
    }
    body.loading .thinking-overlay {
      display: flex;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @media (max-width: 880px) {
      main { grid-template-columns: 1fr; width: min(700px, calc(100vw - 14px)); padding-top: 8px; }
      header { align-items: start; flex-direction: column; }
      .top-actions { justify-content: start; }
      .hud { grid-template-columns: 1fr; }
      .keys { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="thinking-overlay" role="status" aria-live="polite">
    <div class="spinner" aria-hidden="true"></div>
    <div>
      <div id="thinking-title">Resolving turn...</div>
      <small id="thinking-hint">Please wait for the arena update.</small>
    </div>
  </div>
  <main>
    <header>
      <div>
        <h1>DuelGrid</h1>
        <p class="subtitle">Spend energy, cut across the arena, and defeat the red agent.</p>
      </div>
      <div class="top-actions">
        <span class="pill">Agent: LLM</span>
        <span class="pill" id="turn-pill">Turn 0</span>
        <button id="new-map">New Map</button>
      </div>
    </header>

    <section class="panel arena">
      <div class="board-wrap">
        <div id="board" class="board" aria-label="DuelGrid board"></div>
      </div>
      <div class="hud">
        <article class="fighter" id="user-card"></article>
        <article class="fighter" id="agent-card"></article>
      </div>
    </section>

    <aside class="side">
      <section class="panel instructions">
        <strong>Controls</strong>
        <div class="keys">
          <span><span class="key">W</span> <span class="key">A</span> <span class="key">S</span> <span class="key">D</span> / arrows: move</span>
          <span><span class="key">F</span> melee attack</span>
          <span><span class="key">R</span> ranged attack</span>
          <span><span class="key">E</span> pickup</span>
          <span><span class="key">H</span> shield</span>
          <span><span class="key">X</span> wait</span>
        </div>
        <span class="pill">Keys and clicks apply immediately; the agent acts when your energy is spent.</span>
      </section>
      <section class="panel">
        <strong>Legal Actions</strong>
        <div class="actions" id="actions"></div>
      </section>
      <section class="panel">
        <strong>Battle Log</strong>
        <div class="log" id="log"></div>
      </section>
    </aside>
  </main>

  <script>
    let current = null;
    let busy = false;
    const glyphs = {
      '#': ['wall', ''],
      '.': ['floor', ''],
      'H': ['health', '💚'],
      'E': ['energy', '⚡'],
      'X': ['trap', '💥'],
      'A': ['agent', 'sprite-agent'],
      'U': ['user', 'sprite-user']
    };
    const actionLabels = {
      MOVE: 'Move',
      ATTACK: 'Attack',
      RANGED_ATTACK: 'Ranged',
      PICKUP: 'Pickup',
      SHIELD: 'Shield',
      WAIT: 'Wait'
    };
    const dirGlyphs = { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' };
    const keyLabels = {
      MOVE_UP: 'W',
      MOVE_DOWN: 'S',
      MOVE_LEFT: 'A',
      MOVE_RIGHT: 'D',
      ATTACK: 'F',
      RANGED_ATTACK: 'R',
      PICKUP: 'E',
      SHIELD: 'H',
      WAIT: 'X'
    };

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function actionKey(action) {
      return action.direction ? `${action.action}_${action.direction}` : action.action;
    }

    async function submitAction(action) {
      if (busy || !action) return;
      busy = true;
      try {
        const afterUser = await api('api/action', {
          method: 'POST',
          body: JSON.stringify({ action })
        });
        render(afterUser);
        if (afterUser.awaiting_agent) {
          await runAgentTurn(action);
        }
      } catch (error) {
        renderError(error);
      } finally {
        busy = false;
        document.body.classList.remove('loading');
      }
    }

    async function runAgentTurn(action) {
      showAgentThinking(action);
      document.body.classList.add('loading');
      render(await api('api/agent', { method: 'POST' }));
    }

    function showAgentThinking(action) {
      const title = document.getElementById('thinking-title');
      const hint = document.getElementById('thinking-hint');
      const label = `${actionLabels[action.action] || action.action} ${dirGlyphs[action.direction] || ''}`.trim();
      title.textContent = 'LLM is thinking...';
      hint.textContent = `You chose ${label}. Waiting for A to choose a tool-call action.`;
    }

    function render(data) {
      current = data;
      document.getElementById('turn-pill').textContent = `Turn ${data.state.turn}/${data.state.max_turns}`;
      renderBoard(data.map);
      renderFighter('user-card', 'You', data.state.user, 'user');
      renderFighter('agent-card', 'Agent', data.state.agent, 'agent');
      renderActions(data);
      renderLog(data.events, data.done, data.terminal_score);
      playAnimations(data.animations || []);
    }

    function renderBoard(rows) {
      const board = document.getElementById('board');
      board.innerHTML = '';
      rows.forEach((row, rowIndex) => {
        [...row].forEach((cell, colIndex) => {
          const spec = glyphs[cell] || ['floor', cell];
          const tile = document.createElement('div');
          tile.className = `tile ${spec[0]}`;
          tile.dataset.row = rowIndex;
          tile.dataset.col = colIndex;
          if (spec[1] === 'sprite-user' || spec[1] === 'sprite-agent') {
            tile.appendChild(makeSprite(spec[1] === 'sprite-agent'));
          } else if (spec[1]) {
            const item = document.createElement('span');
            item.className = 'item';
            item.textContent = spec[1];
            tile.appendChild(item);
          }
          board.appendChild(tile);
        });
      });
    }

    function makeSprite(agent) {
      const sprite = document.createElement('div');
      sprite.className = `sprite ${agent ? 'agent-sprite' : 'user-sprite'}`;
      for (const part of ['head', 'body', 'arm left', 'arm right', 'leg left', 'leg right']) {
        const div = document.createElement('div');
        div.className = part;
        sprite.appendChild(div);
      }
      return sprite;
    }

    function renderFighter(id, label, player, klass) {
      const card = document.getElementById(id);
      const hpPct = Math.max(0, Math.min(100, player.hp * 10));
      const enPct = player.max_energy ? Math.max(0, Math.min(100, player.energy / player.max_energy * 100)) : 0;
      card.innerHTML = `
        <h2><span>${label}</span><span>${klass === 'user' ? 'Blue Scout' : 'Red Agent'}</span></h2>
        <div class="bars">
          <div class="meta"><span>HP ${player.hp}/10</span><span>Shield ${player.shield}</span></div>
          <div class="bar"><div class="fill hp-fill" style="width:${hpPct}%"></div></div>
          <div class="meta"><span>Energy ${player.energy}/${player.max_energy}</span><span>${'■'.repeat(player.energy)}${'□'.repeat(Math.max(0, player.max_energy - player.energy))}</span></div>
          <div class="bar"><div class="fill energy-fill" style="width:${enPct}%"></div></div>
        </div>`;
    }

    function renderActions(data) {
      const actions = document.getElementById('actions');
      actions.innerHTML = '';
      if (data.done) {
        actions.innerHTML = '<button disabled>Game Over</button>';
        return;
      }
      for (const action of data.legal_actions) {
        const button = document.createElement('button');
        button.className = action.action === 'ATTACK' || action.action === 'RANGED_ATTACK' ? 'primary' : '';
        const label = `${actionLabels[action.action] || action.action} ${dirGlyphs[action.direction] || ''}`.trim();
        const key = keyLabels[actionKey(action)] || keyLabels[action.action] || '';
        button.innerHTML = `<span>${label}</span>${key ? `<span class="key">${key}</span>` : ''}`;
        button.onclick = () => submitAction(action);
        actions.appendChild(button);
      }
    }

    function renderLog(events, done, score) {
      const log = document.getElementById('log');
      const lines = done ? [`Game over. Score ${Number(score).toFixed(2)}`, ...events] : events;
      log.innerHTML = '';
      for (const event of lines) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.textContent = event;
        log.appendChild(entry);
      }
    }

    function tileAt(pos) {
      if (!pos) return null;
      return document.querySelector(`.tile[data-row="${pos.row}"][data-col="${pos.col}"]`);
    }

    function playAnimations(animations) {
      for (const animation of animations) {
        const source = tileAt(animation.source);
        const target = tileAt(animation.target);
        if (animation.kind === 'melee') {
          playMelee(source, target);
        } else if (animation.kind === 'ranged') {
          playRanged(source, target);
        }
      }
    }

    function playMelee(source, target) {
      const sprite = source ? source.querySelector('.sprite') : null;
      if (sprite) {
        sprite.classList.remove('strike');
        void sprite.offsetWidth;
        sprite.classList.add('strike');
      }
      if (target) {
        target.classList.remove('hit-melee');
        void target.offsetWidth;
        target.classList.add('hit-melee');
        setTimeout(() => target.classList.remove('hit-melee'), 460);
      }
    }

    function playRanged(source, target) {
      const sprite = source ? source.querySelector('.sprite') : null;
      if (sprite) {
        sprite.classList.remove('ranged-cast');
        void sprite.offsetWidth;
        sprite.classList.add('ranged-cast');
      }
      if (!source || !target) return;
      const from = source.getBoundingClientRect();
      const to = target.getBoundingClientRect();
      const projectile = document.createElement('div');
      projectile.className = 'projectile';
      projectile.style.left = `${from.left + from.width / 2}px`;
      projectile.style.top = `${from.top + from.height / 2}px`;
      document.body.appendChild(projectile);
      requestAnimationFrame(() => {
        projectile.style.transition = 'transform .34s ease-out, left .34s ease-out, top .34s ease-out';
        projectile.style.left = `${to.left + to.width / 2}px`;
        projectile.style.top = `${to.top + to.height / 2}px`;
      });
      target.classList.remove('hit-ranged');
      void target.offsetWidth;
      target.classList.add('hit-ranged');
      setTimeout(() => {
        projectile.remove();
        target.classList.remove('hit-ranged');
      }, 480);
    }

    function renderError(error) {
      const log = document.getElementById('log');
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.textContent = error.message;
      log.prepend(entry);
    }

    function findKeyboardAction(event) {
      if (!current || current.done) return null;
      const key = event.key.toLowerCase();
      const map = {
        w: { action: 'MOVE', direction: 'UP' },
        arrowup: { action: 'MOVE', direction: 'UP' },
        s: { action: 'MOVE', direction: 'DOWN' },
        arrowdown: { action: 'MOVE', direction: 'DOWN' },
        a: { action: 'MOVE', direction: 'LEFT' },
        arrowleft: { action: 'MOVE', direction: 'LEFT' },
        d: { action: 'MOVE', direction: 'RIGHT' },
        arrowright: { action: 'MOVE', direction: 'RIGHT' },
        e: { action: 'PICKUP' },
        h: { action: 'SHIELD' },
        x: { action: 'WAIT' }
      };
      if (key === 'f') return current.legal_actions.find(action => action.action === 'ATTACK') || null;
      if (key === 'r') return current.legal_actions.find(action => action.action === 'RANGED_ATTACK') || null;
      const wanted = map[key];
      if (!wanted) return null;
      return current.legal_actions.find(action => action.action === wanted.action && (wanted.direction || '') === (action.direction || '')) || null;
    }

    document.addEventListener('keydown', event => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const action = findKeyboardAction(event);
      if (!action) return;
      event.preventDefault();
      submitAction(action);
    });

    document.getElementById('new-map').onclick = async () => {
      if (busy) return;
      render(await api('api/new', { method: 'POST' }));
    };
    api('api/state').then(render).catch(renderError);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
