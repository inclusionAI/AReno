"""Calendar scheduling web UI server for the agentic example.

Run from the repository root after starting ``areno serve``:

    python examples/agentic/calendar/web_ui.py --base-url http://127.0.0.1:8000/v1 --api-key token

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

QUERY_AVAILABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_availability",
        "description": "Query the available time slots for a participant, returned in UTC.",
        "parameters": {
            "type": "object",
            "properties": {
                "participant": {
                    "type": "string",
                    "description": "The name of the participant to query.",
                },
            },
            "required": ["participant"],
            "additionalProperties": False,
        },
    },
}

PROPOSE_SLOT_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_slot",
        "description": "Propose a meeting time in UTC hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {"type": "string"},
                "utc_start_hour": {"type": "integer", "minimum": 0, "maximum": 23},
                "utc_end_hour": {"type": "integer", "minimum": 1, "maximum": 24},
            },
            "required": ["meeting_id", "utc_start_hour", "utc_end_hour"],
            "additionalProperties": False,
        },
    },
}

CONFIRM_SLOT_TOOL = {
    "type": "function",
    "function": {
        "name": "confirm_slot",
        "description": "Confirm a proposed meeting slot to finalize the booking.",
        "parameters": {
            "type": "object",
            "properties": {
                "meeting_id": {"type": "string"},
                "utc_start_hour": {"type": "integer", "minimum": 0, "maximum": 23},
                "utc_end_hour": {"type": "integer", "minimum": 1, "maximum": 24},
            },
            "required": ["meeting_id", "utc_start_hour", "utc_end_hour"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [QUERY_AVAILABILITY_TOOL, PROPOSE_SLOT_TOOL, CONFIRM_SLOT_TOOL]
TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}

SYSTEM_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Your task is to schedule a meeting by calling the provided tools. "
    "First, call query_availability for each required participant to learn their available times in UTC. "
    "Then, find a UTC time range that overlaps all participants' availability and fits the meeting duration. "
    "Call propose_slot with the meeting_id and the UTC time range. "
    "After seeing the proposal result, call confirm_slot to finalize the booking. "
    "Use exactly one tool call per turn. Do not write free text."
)


class CalendarServer(ThreadingHTTPServer):
    """Stateful HTTP server for one calendar scheduling session."""

    def __init__(self, server_address, request_handler, *, seed: int | None = None, args):
        super().__init__(server_address, request_handler)
        self.rng = random.Random(seed)
        self.args = args
        self.openai_client = None
        self.state: game.CalendarState | None = None
        self.meeting_id: str = ""
        self.messages: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.tool_calls_log: list[dict[str, Any]] = []
        self.agent_busy = False
        self.finished = False
        self.result: str = ""
        self.reward: float | None = None
        _new_scenario(self)


class CalendarHandler(BaseHTTPRequestHandler):
    server: CalendarServer

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            self._send_json(_payload(self.server))
        elif route == "scenario":
            self._send_json(_scenario_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        route = _route_path(self.path)
        if route == "new":
            _new_scenario(self.server)
            self._send_json(_payload(self.server))
        elif route == "agent":
            self._send_json(_agent_step(self.server))
        elif route == "reset":
            _new_scenario(self.server)
            self._send_json(_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("calendar-web: " + fmt % args + "\n")

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
    if path.endswith("/api/scenario"):
        return "scenario"
    if path.endswith("/api/new"):
        return "new"
    if path.endswith("/api/agent"):
        return "agent"
    if path.endswith("/api/reset"):
        return "reset"
    if "/api/" in path:
        return "missing"
    if path == "/" or not path.rsplit("/", 1)[-1].count("."):
        return "index"
    return "missing"


def _new_scenario(server: CalendarServer) -> None:
    """Generate a new random calendar scenario."""
    participant_names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
    timezones = ["UTC", "UTC+1", "UTC+2", "UTC+3", "UTC+8", "UTC-5", "UTC-8", "UTC+5:30"]
    num_participants = server.rng.randint(2, 3)
    chosen = server.rng.sample(participant_names, num_participants)

    participants: dict[str, game.Participant] = {}
    for name in chosen:
        tz = server.rng.choice(timezones)
        num_slots = server.rng.randint(1, 2)
        slots: list[game.TimeSlot] = []
        cursor = server.rng.randint(0, 8)
        for _ in range(num_slots):
            start = cursor
            duration = server.rng.randint(2, 6)
            end = min(start + duration, 24)
            if start < end:
                slots.append(game.TimeSlot(start, end))
            cursor = end + server.rng.randint(1, 3)
        if not slots:
            slots.append(game.TimeSlot(9, 17))
        participants[name] = game.Participant(name=name, timezone=tz, available_slots=tuple(slots))

    meeting_id = f"meeting-{server.rng.randint(1000, 9999)}"
    duration_hours = server.rng.randint(1, 2)
    meeting = game.Meeting(
        id=meeting_id,
        duration_hours=duration_hours,
        required_participants=tuple(chosen),
    )

    confirmed: dict[str, tuple[int, int]] = {}
    if server.rng.random() < 0.3:
        other_id = f"{meeting_id}-prior"
        other_start = server.rng.randint(0, 20)
        other_end = min(other_start + server.rng.randint(1, 3), 24)
        confirmed[other_id] = (other_start, other_end)

    server.state = game.CalendarState(
        participants=participants,
        meetings=(meeting,),
        confirmed=confirmed,
    )
    server.meeting_id = meeting_id
    server.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": game.format_prompt(server.state, meeting_id)},
    ]
    server.events = ["New scenario generated. Click 'Run Agent Step' to let the LLM schedule the meeting."]
    server.tool_calls_log = []
    server.agent_busy = False
    server.finished = False
    server.result = ""
    server.reward = None


def _agent_step(server: CalendarServer) -> dict[str, Any]:
    """Run one LLM agent step (one tool call)."""
    if server.finished:
        return _payload(server)
    if not server.args.base_url:
        server.events.insert(0, "LLM mode requires --base-url. Start areno serve first.")
        server.events = server.events[:10]
        return _payload(server)

    if server.openai_client is None:
        server.openai_client = _make_openai_client(server.args)

    # Determine which tool to use based on conversation progress.
    tool_name = _next_tool(server)
    tool = TOOL_BY_NAME[tool_name]
    tool_choice = {"type": "function", "function": {"name": tool_name}}

    # Add a turn prompt to guide the model.
    turn_prompt = _turn_prompt(server, tool_name)
    messages = [*server.messages, {"role": "user", "content": turn_prompt}]

    try:
        response = server.openai_client.chat.completions.create(
            model=server.args.model,
            messages=messages,
            tools=[tool],
            tool_choice=tool_choice,
        )
    except Exception as exc:
        server.events.insert(0, f"LLM call failed: {exc}")
        server.events = server.events[:10]
        return _payload(server)

    raw = response.model_dump() if hasattr(response, "model_dump") else response
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    message = choices[0].get("message", {}) if choices else {}
    tool_calls = message.get("tool_calls", []) if message else []

    if not tool_calls:
        server.events.insert(0, f"LLM did not return a tool call for {tool_name}.")
        server.events = server.events[:10]
        return _payload(server)

    call = tool_calls[0]
    func = call.get("function", {})
    name = func.get("name", tool_name)
    args_str = func.get("arguments", "{}")
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        args = {}

    # Execute the tool.
    result = _execute_tool(server, name, args)

    # Record the tool call.
    call_record = {"name": name, "arguments": args, "result": result}
    server.tool_calls_log.append(call_record)

    # Build assistant + tool messages for conversation history.
    assistant_msg = {
        "role": "assistant",
        "content": message.get("content", ""),
        "tool_calls": [
            {
                "id": call.get("id", f"call_{len(server.tool_calls_log)}"),
                "type": "function",
                "function": {"name": name, "arguments": args_str if isinstance(args_str, str) else json.dumps(args)},
            }
        ],
    }
    tool_msg = {
        "role": "tool",
        "tool_call_id": call.get("id", f"call_{len(server.tool_calls_log)}"),
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }
    server.messages.append(assistant_msg)
    server.messages.append(tool_msg)

    server.events.insert(0, f"Agent called {name}({json.dumps(args, ensure_ascii=False)}): {json.dumps(result, ensure_ascii=False)}")
    server.events = server.events[:10]

    # Check if finished.
    if name == "confirm_slot":
        server.finished = True
        _evaluate(server)
    elif name == "propose_slot" and result.get("valid") is False:
        # Let the model try again after a failed proposal.
        pass

    return _payload(server)


def _next_tool(server: CalendarServer) -> str:
    """Determine which tool the agent should call next based on progress."""
    called_names = [c["name"] for c in server.tool_calls_log]
    queried = [c for c in called_names if c == "query_availability"]
    proposed = [c for c in called_names if c == "propose_slot"]
    confirmed = [c for c in called_names if c == "confirm_slot"]

    meeting = server.state.meeting_by_id(server.meeting_id) if server.state else None
    num_participants = len(meeting.required_participants) if meeting else 2

    if len(queried) < num_participants:
        return "query_availability"
    if not proposed:
        return "propose_slot"
    if not confirmed:
        return "confirm_slot"
    return "confirm_slot"


def _turn_prompt(server: CalendarServer, tool_name: str) -> str:
    """Generate a guiding prompt for the current turn."""
    if tool_name == "query_availability":
        meeting = server.state.meeting_by_id(server.meeting_id) if server.state else None
        queried = []
        for c in server.tool_calls_log:
            if c["name"] != "query_availability":
                continue
            cargs = c["arguments"]
            if isinstance(cargs, str):
                try:
                    cargs = json.loads(cargs)
                except json.JSONDecodeError:
                    continue
            if isinstance(cargs, dict):
                queried.append(cargs.get("participant", ""))
        remaining = [p for p in (meeting.required_participants if meeting else []) if p not in queried]
        if remaining:
            return f"Call query_availability for participant '{remaining[0]}' to learn their UTC availability."
        return "Call query_availability for the next required participant."
    if tool_name == "propose_slot":
        return "Based on the availability results above, call propose_slot with a UTC time range that works for all participants."
    if tool_name == "confirm_slot":
        return "Based on the proposal result above, call confirm_slot to finalize the booking."
    return "Call the next tool."


def _execute_tool(server: CalendarServer, name: str, args: dict) -> dict:
    """Execute a tool call against the calendar state."""
    if server.state is None:
        return {"error": "no scenario loaded"}
    if name == "query_availability":
        participant = str(args.get("participant", ""))
        return game.execute_query_availability(server.state, participant)
    if name == "propose_slot":
        meeting_id = str(args.get("meeting_id", ""))
        utc_start = int(args.get("utc_start_hour", -1))
        utc_end = int(args.get("utc_end_hour", -1))
        return game.execute_propose_slot(server.state, meeting_id, utc_start, utc_end)
    if name == "confirm_slot":
        meeting_id = str(args.get("meeting_id", ""))
        utc_start = int(args.get("utc_start_hour", -1))
        utc_end = int(args.get("utc_end_hour", -1))
        return game.execute_confirm_slot(server.state, meeting_id, utc_start, utc_end)
    return {"error": f"unknown tool: {name}"}


def _evaluate(server: CalendarServer) -> None:
    """Evaluate the final result and compute reward."""
    if server.state is None:
        return
    # Extract the last confirmed/proposed slot.
    confirmed_slot = None
    for call in reversed(server.tool_calls_log):
        if call["name"] in ("confirm_slot", "propose_slot"):
            args = call["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if not isinstance(args, dict):
                continue
            if args.get("meeting_id") != server.meeting_id:
                continue
            utc_start = int(args.get("utc_start_hour", -1))
            utc_end = int(args.get("utc_end_hour", -1))
            if utc_start >= 0 and utc_end > utc_start:
                confirmed_slot = (utc_start, utc_end)
                break

    if confirmed_slot is None:
        server.result = "No valid slot proposed"
        server.reward = -1.0
        server.events.insert(0, "Agent failed to propose a valid slot. Reward: -1.0")
        return

    utc_start, utc_end = confirmed_slot
    error = game.validate_proposal(server.state, server.meeting_id, utc_start, utc_end)
    if error:
        server.result = f"Invalid: {error}"
        server.reward = -1.0
        server.events.insert(0, f"Agent's proposal was invalid: {error}. Reward: -1.0")
        return

    server.result = f"Scheduled {server.meeting_id} at UTC {utc_start:02d}:00-{utc_end:02d}:00"
    server.reward = game.compute_reward(server.state, server.meeting_id, utc_start, utc_end, server.tool_calls_log)
    server.events.insert(0, f"Successfully scheduled! Reward: {server.reward:.2f}")


def _payload(server: CalendarServer) -> dict[str, Any]:
    """Build the API response payload."""
    meeting = server.state.meeting_by_id(server.meeting_id) if server.state else None
    participants_info = []
    if server.state and meeting:
        for name in meeting.required_participants:
            p = server.state.participants.get(name)
            if p:
                utc_slots = []
                for slot in p.available_slots:
                    utc_start, utc_end = game.to_utc(slot, p.timezone)
                    utc_slots.append({"local_start": slot.start_hour, "local_end": slot.end_hour, "utc_start": utc_start, "utc_end": utc_end})
                participants_info.append({
                    "name": name,
                    "timezone": p.timezone,
                    "local_slots": [{"start": s.start_hour, "end": s.end_hour} for s in p.available_slots],
                    "utc_slots": utc_slots,
                })

    confirmed_info = []
    if server.state:
        for mid, (cs, ce) in server.state.confirmed.items():
            if mid != server.meeting_id:
                confirmed_info.append({"meeting_id": mid, "utc_start": cs, "utc_end": ce})

    common_slots = []
    if server.state and meeting:
        common_slots = game.find_common_slots(meeting, server.state.participants)

    return {
        "meeting_id": server.meeting_id,
        "duration_hours": meeting.duration_hours if meeting else 0,
        "required_participants": list(meeting.required_participants) if meeting else [],
        "participants": participants_info,
        "confirmed_meetings": confirmed_info,
        "common_slots": [{"start": s, "end": e} for s, e in common_slots],
        "messages": server.messages,
        "events": server.events,
        "tool_calls_log": server.tool_calls_log,
        "finished": server.finished,
        "result": server.result,
        "reward": server.reward,
        "next_tool": _next_tool(server) if not server.finished else None,
    }


def _scenario_payload(server: CalendarServer) -> dict[str, Any]:
    return _payload(server)


def _make_openai_client(args):
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
<title>Calendar Scheduling Agent</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#1a2733;background:#e8f0fe}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#667eea76,#764ba280 60%,#f093fb60);display:grid;place-items:center;padding:20px 0}
.app{width:min(1100px,96vw);display:grid;grid-template-columns:minmax(360px,1fr) minmax(360px,1fr);gap:20px;align-items:start}
.panel{background:#ffffff;border:3px solid #2d3748;border-radius:20px;box-shadow:6px 6px 0 rgba(45,55,72,.15);padding:20px}
h1{font-size:32px;line-height:1.1;margin:0 0 6px;color:#5b21b6}
h2{font-size:22px;margin:16px 0 8px;color:#5b21b6}
.subtitle{font-weight:700;color:#4a5568;margin-bottom:14px}
.meeting-card{background:#f7fafc;border:2px solid #cbd5e0;border-radius:14px;padding:14px;margin-bottom:12px}
.meeting-card .id{font-size:18px;font-weight:800;color:#2d3748}
.meeting-card .meta{color:#718096;font-size:14px;margin-top:4px}
.participant{background:#ebf8ff;border:2px solid #bee3f8;border-radius:12px;padding:10px 12px;margin-bottom:8px}
.participant .name{font-weight:800;color:#2c5282;font-size:15px}
.participant .tz{color:#718096;font-size:13px}
.slot{display:inline-block;background:#c6f6d5;border:1px solid #9ae6b4;border-radius:8px;padding:3px 8px;margin:3px 4px 0 0;font-size:12px;font-weight:700}
.slot.utc{background:#fef3c7;border-color:#fcd34e;color:#92400e}
.slot.local{background:#dbeafe;border-color:#93c5fd;color:#1e40af}
.confirmed{background:#fed7d7;border:1px solid #feb2b2;border-radius:8px;padding:3px 8px;margin:3px 4px 0 0;font-size:12px;font-weight:700;display:inline-block}
.common{background:#d6f997;border:1px solid #84cc16;border-radius:8px;padding:3px 8px;margin:3px 4px 0 0;font-size:12px;font-weight:700;display:inline-block}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
button{border:3px solid #2d3748;border-radius:14px;background:#a78bfa;box-shadow:3px 3px 0 #2d3748;color:#fff;font-weight:800;padding:10px 16px;cursor:pointer;font-size:14px}
button:hover{transform:translateY(-1px)}button:disabled{filter:grayscale(.7);opacity:.5;cursor:not-allowed}
button.success{background:#48bb78}button.danger{background:#f56565}
.status{margin-top:12px;padding:10px 14px;border-radius:12px;font-weight:800;font-size:14px}
.status.pending{background:#ebf8ff;color:#2c5282;border:2px solid #bee3f8}
.status.success{background:#f0fff4;color:#22543d;border:2px solid #9ae6b4}
.status.fail{background:#fff5f5;color:#9b2c2c;border:2px solid #feb2b2}
.reward{font-size:28px;font-weight:900;margin-top:8px}
.reward.good{color:#48bb78}.reward.bad{color:#f56565}.reward.neutral{color:#718096}
.tool-log{margin-top:12px}
.tool-entry{background:#f7fafc;border:2px solid #e2e8f0;border-radius:12px;padding:10px 12px;margin-bottom:8px}
.tool-entry .tool-name{font-weight:800;color:#5b21b6;font-size:13px}
.tool-entry .tool-args{color:#4a5568;font-size:12px;margin-top:4px;word-break:break-all}
.tool-entry .tool-result{font-size:12px;margin-top:4px;word-break:break-all}
.tool-entry .tool-result.valid{color:#48bb78}.tool-entry .tool-result.invalid{color:#f56565}
.events{display:grid;gap:6px;margin-top:12px}
.event{background:#fff;border:2px solid #e2e8f0;border-radius:10px;padding:8px 10px;font-size:13px;font-weight:600;color:#4a5568}
.timeline{margin-top:12px}
.timeline-step{display:flex;align-items:center;gap:8px;padding:6px 0;font-size:13px;font-weight:700}
.timeline-step.done{color:#48bb78}.timeline-step.current{color:#a78bfa}.timeline-step.pending{color:#a0aec0}
.timeline-dot{width:12px;height:12px;border-radius:50%;border:2px solid #2d3748;flex-shrink:0}
.timeline-step.done .timeline-dot{background:#48bb78}.timeline-step.current .timeline-dot{background:#a78bfa;animation:pulse 1.5s infinite}.timeline-step.pending .timeline-dot{background:#e2e8f0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.thinking{display:none;margin:10px 0;padding:10px 14px;border:2px solid #a78bfa;border-radius:12px;background:#faf5ff;font-weight:700;color:#5b21b6}.thinking.on{display:block}
.dots::after{content:"";animation:dots 1s steps(4,end) infinite}@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}100%{content:""}}
@media(max-width:820px){.app{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="app">
  <section class="panel">
    <h1>Calendar Scheduling Agent</h1>
    <div class="subtitle">Watch an LLM schedule meetings across time zones using tool calls.</div>
    <div id="meetingCard" class="meeting-card"></div>
    <div id="participants"></div>
    <div id="confirmedMeetings"></div>
    <h2>Common Available Slots (UTC)</h2>
    <div id="commonSlots"></div>
    <h2>Agent Timeline</h2>
    <div id="timeline" class="timeline"></div>
    <div id="thinking" class="thinking">Agent is thinking<span class="dots"></span></div>
    <div class="actions">
      <button id="agent" onclick="agentStep()">Run Agent Step</button>
      <button id="auto" onclick="toggleAuto()">Auto Run</button>
      <button id="new" class="success" onclick="newScenario()">New Scenario</button>
    </div>
    <div id="statusBar" class="status pending"></div>
    <div id="rewardDisplay" class="reward neutral" style="display:none"></div>
  </section>
  <aside class="panel">
    <h2 style="margin-top:0">Tool Call Log</h2>
    <div id="toolLog" class="tool-log"></div>
    <h2>Event Log</h2>
    <div id="events" class="events"></div>
  </aside>
</main>
<script>
let state = null, autoRun = false, autoTimer = null;
async function fetchState(){
  const res = await fetch(api("api/state"));
  state = await res.json();
  render();
}
function api(path){return new URL(path, window.location.href).toString();}
async function agentStep(){
  if(state && state.finished) return;
  document.getElementById("thinking").classList.add("on");
  document.getElementById("agent").disabled = true;
  try{
    const res = await fetch(api("api/agent"), {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})});
    state = await res.json();
    render();
  }catch(e){
    console.error(e);
  }finally{
    document.getElementById("thinking").classList.remove("on");
    document.getElementById("agent").disabled = false;
  }
}
async function newScenario(){
  autoRun = false;
  if(autoTimer){clearTimeout(autoTimer);autoTimer=null;}
  document.getElementById("auto").textContent = "Auto Run";
  const res = await fetch(api("api/new"), {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})});
  state = await res.json();
  render();
}
function toggleAuto(){
  autoRun = !autoRun;
  document.getElementById("auto").textContent = autoRun ? "Stop Auto" : "Auto Run";
  document.getElementById("auto").className = autoRun ? "danger" : "";
  if(autoRun && state && !state.finished){
    agentStep().then(()=>{
      if(autoRun && state && !state.finished){
        autoTimer = setTimeout(()=>{if(autoRun) agentStep();}, 800);
      }
    });
  } else if(!autoRun && autoTimer){
    clearTimeout(autoTimer); autoTimer = null;
  }
}
function render(){
  if(!state) return;
  // Meeting card
  const mc = document.getElementById("meetingCard");
  mc.innerHTML = `
    <div class="id">Meeting: ${esc(state.meeting_id)}</div>
    <div class="meta">Duration: ${state.duration_hours} hour(s) | Participants: ${state.required_participants.join(", ")}</div>
  `;
  // Participants
  const ph = document.getElementById("participants");
  ph.innerHTML = state.participants.map(p=>`
    <div class="participant">
      <div class="name">${esc(p.name)} <span class="tz">(${esc(p.timezone)})</span></div>
      <div>
        ${p.local_slots.map(s=>`<span class="slot local">Local ${s.start}:00-${s.end}:00</span>`).join("")}
      </div>
      <div style="margin-top:4px">
        ${p.utc_slots.map(s=>`<span class="slot utc">UTC ${s.utc_start}:00-${s.utc_end}:00</span>`).join("")}
      </div>
    </div>
  `).join("");
  // Confirmed meetings
  const cm = document.getElementById("confirmedMeetings");
  cm.innerHTML = state.confirmed_meetings.length
    ? `<h2 style="font-size:18px">Already Confirmed</h2>` + state.confirmed_meetings.map(m=>`<span class="confirmed">${esc(m.meeting_id)}: UTC ${m.utc_start}:00-${m.utc_end}:00</span>`).join("")
    : "";
  // Common slots
  const cs = document.getElementById("commonSlots");
  cs.innerHTML = state.common_slots.length
    ? state.common_slots.map(s=>`<span class="common">UTC ${s.start}:00-${s.end}:00</span>`).join("")
    : `<span style="color:#a0aec0;font-size:13px">No overlapping slots found.</span>`;
  // Timeline
  const queriedCount = state.tool_calls_log.filter(c=>c.name==="query_availability").length;
  const proposedCount = state.tool_calls_log.filter(c=>c.name==="propose_slot").length;
  const confirmedCount = state.tool_calls_log.filter(c=>c.name==="confirm_slot").length;
  const numParticipants = state.required_participants.length;
  const tl = document.getElementById("timeline");
  const steps = [
    {label:`Query Availability (${queriedCount}/${numParticipants})`,done:queriedCount>=numParticipants,current:queriedCount<numParticipants},
    {label:`Propose Slot (${proposedCount})`,done:proposedCount>=1,current:queriedCount>=numParticipants&&proposedCount===0&&!state.finished},
    {label:`Confirm Slot (${confirmedCount})`,done:confirmedCount>=1,current:proposedCount>=1&&confirmedCount===0&&!state.finished},
  ];
  tl.innerHTML = steps.map(s=>`
    <div class="timeline-step ${s.done?"done":s.current?"current":"pending"}">
      <div class="timeline-dot"></div>
      <span>${esc(s.label)}</span>
    </div>
  `).join("");
  // Tool log
  const tl2 = document.getElementById("toolLog");
  tl2.innerHTML = state.tool_calls_log.map(c=>{
    const resultCls = c.result && (c.result.valid===true || c.result.confirmed===true) ? "valid" : (c.result && c.result.error ? "invalid" : "");
    return `
      <div class="tool-entry">
        <div class="tool-name">${esc(c.name)}</div>
        <div class="tool-args">Args: ${esc(JSON.stringify(c.arguments))}</div>
        <div class="tool-result ${resultCls}">Result: ${esc(JSON.stringify(c.result))}</div>
      </div>
    `;
  }).join("") || `<span style="color:#a0aec0;font-size:13px">No tool calls yet.</span>`;
  // Events
  const ev = document.getElementById("events");
  ev.innerHTML = state.events.map(e=>`<div class="event">${esc(e)}</div>`).join("");
  // Status
  const sb = document.getElementById("statusBar");
  const rd = document.getElementById("rewardDisplay");
  if(state.finished){
    sb.className = "status " + (state.reward !== null && state.reward >= 0 ? "success" : "fail");
    sb.textContent = state.result || "Finished";
    rd.style.display = "block";
    rd.className = "reward " + (state.reward !== null && state.reward > 0 ? "good" : state.reward !== null && state.reward < 0 ? "bad" : "neutral");
    rd.textContent = `Reward: ${state.reward !== null ? state.reward.toFixed(2) : "N/A"}`;
    document.getElementById("agent").disabled = true;
  } else {
    sb.className = "status pending";
    sb.textContent = state.next_tool ? `Next step: ${state.next_tool}` : "Ready";
    rd.style.display = "none";
    document.getElementById("agent").disabled = false;
  }
  // Auto-run continuation
  if(autoRun && state && !state.finished){
    autoTimer = setTimeout(()=>{if(autoRun) agentStep();}, 800);
  }
}
function esc(text){return String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));}
fetchState();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Calendar scheduling web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (e.g. http://127.0.0.1:8000/v1)")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = CalendarServer((args.host, args.port), CalendarHandler, seed=args.seed, args=args)
    url = f"http://{args.host}:{args.port}"
    print(f"Calendar scheduling web UI running at {url}")
    if args.base_url:
        print(f"LLM endpoint: {args.base_url} (model: {args.model})")
    else:
        print("Warning: --base-url not set; agent steps will fail. Start `areno serve` first.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
