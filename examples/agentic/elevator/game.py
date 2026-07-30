"""Deterministic single-cab elevator dispatch environment for agentic RL.

The environment models one elevator cab serving N floors under a discrete
time-step clock. Passengers arrive at floors with destinations; the cab moves,
opens/closes its door, and picks up or drops off passengers subject to a
capacity limit. Everything is pure-Python and deterministic so episodes can be
reproduced without external services.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# defaults
DEFAULT_FLOORS = 6
DEFAULT_CAPACITY = 4
DEFAULT_HORIZON = 64

# action tools exposed to the model
MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move",
        "description": "Move the cab one floor up (+1) or down (-1). Door must be closed.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "integer",
                    "enum": [-1, 1],
                    "description": "+1 to move up one floor, -1 to move down one floor.",
                }
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
    },
}

OPEN_DOOR_TOOL = {
    "type": "function",
    "function": {
        "name": "open_door",
        "description": "Open the door at the current floor to let passengers board or alight.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}

CLOSE_DOOR_TOOL = {
    "type": "function",
    "function": {
        "name": "close_door",
        "description": "Close the door so the cab can move.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}

DONE_TOOL = {
    "type": "function",
    "function": {
        "name": "done",
        "description": "End the episode early once all reachable passengers are delivered.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}

TOOLS = [MOVE_TOOL, OPEN_DOOR_TOOL, CLOSE_DOOR_TOOL, DONE_TOOL]


@dataclass
class Passenger:
    """A passenger waiting or riding the elevator."""

    pid: int
    origin: int
    dest: int
    arrive_time: int
    board_time: int | None = None
    deliver_time: int | None = None

    @property
    def wait(self) -> int:
        """Waiting time in discrete steps before boarding."""

        return max(self.board_time if self.board_time is not None else 0, 0) - self.arrive_time

    @property
    def ride(self) -> int:
        """Ride time in discrete steps from boarding to delivery."""

        if self.board_time is None or self.deliver_time is None:
            return 0
        return self.deliver_time - self.board_time


@dataclass
class ElevatorState:
    """Full deterministic state of the single-cab elevator environment."""

    floors: int
    capacity: int
    horizon: int
    floor: int = 0
    direction: int = 0
    door_open: bool = False
    passengers: list[Passenger] = field(default_factory=list)
    waiting: dict[int, list[Passenger]] = field(default_factory=dict)
    delivered: int = 0
    total_wait: int = 0
    invalid_actions: int = 0
    overload_refused: int = 0
    time: int = 0
    terminated: bool = False
    scenario: str = "mixed"
    total_passengers_total: int = 0

    def total_passengers(self) -> int:
        """Initial passenger count (does not shrink as passengers deliver)."""

        return self.total_passengers_total


def validate_config(*, floors: int, capacity: int, horizon: int) -> None:
    """Validate environment configuration before expensive work."""

    if floors < 2:
        raise ValueError(f"floors must be >= 2, got {floors}")
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")


def build_state(
    record: dict[str, Any],
) -> ElevatorState:
    """Construct an initial ElevatorState from a dataset record."""

    floors = int(record.get("floors", DEFAULT_FLOORS))
    capacity = int(record.get("capacity", DEFAULT_CAPACITY))
    horizon = int(record.get("horizon", DEFAULT_HORIZON))
    validate_config(floors=floors, capacity=capacity, horizon=horizon)
    state = ElevatorState(
        floors=floors,
        capacity=capacity,
        horizon=horizon,
        waiting={i: [] for i in range(floors)},
        scenario=str(record.get("scenario", "mixed")),
    )
    for raw in record.get("passengers", []):
        pid = int(raw["pid"])
        origin = int(raw["origin"])
        dest = int(raw["dest"])
        arrive_time = int(raw.get("arrive_time", 0))
        if not 0 <= origin < floors:
            raise ValueError(f"passenger {pid} origin {origin} out of range")
        if not 0 <= dest < floors:
            raise ValueError(f"passenger {pid} dest {dest} out of range")
        if origin == dest:
            raise ValueError(f"passenger {pid} origin equals dest")
        passenger = Passenger(pid=pid, origin=origin, dest=dest, arrive_time=arrive_time)
        state.passengers.append(passenger)
        state.waiting.setdefault(origin, []).append(passenger)
    # initial door state for the empty-door scenario
    state.door_open = bool(record.get("door_open", False))
    state.total_passengers_total = len(state.passengers)
    return state


def is_terminal(state: ElevatorState) -> bool:
    """Return whether the episode is over."""

    if state.terminated:
        return True
    if state.time >= state.horizon:
        return True
    return state.delivered >= state.total_passengers() and state.total_passengers() > 0


def step(state: ElevatorState, action: dict[str, Any]) -> dict[str, Any]:
    """Apply one action and return an observation dict.

    Invalid actions increment ``invalid_actions`` and leave state unchanged.
    The returned dict is the tool result fed back to the model.
    """

    if is_terminal(state):
        return {"terminated": True, "state": format_state(state)}

    name = str(action.get("name", ""))
    result: dict[str, Any]
    if name == "move":
        result = _step_move(state, action)
    elif name == "open_door":
        result = _step_open(state)
    elif name == "close_door":
        result = _step_close(state)
    elif name == "done":
        state.terminated = True
        result = {"done": True, "state": format_state(state)}
        return result
    else:
        state.invalid_actions += 1
        result = {"invalid": True, "error": f"unknown action: {name}"}

    # advance clock and waiting times only on a successful state-changing action
    if not result.get("invalid"):
        state.time += 1
        _elapse_waiting(state)
    if is_terminal(state):
        result["terminated"] = True
    result["state"] = format_state(state)
    return result


def _step_move(state: ElevatorState, action: dict[str, Any]) -> dict[str, Any]:
    direction = action.get("direction")
    try:
        direction = int(direction)
    except (TypeError, ValueError):
        state.invalid_actions += 1
        return {"invalid": True, "error": "direction must be -1 or 1"}
    if state.door_open:
        state.invalid_actions += 1
        return {"invalid": True, "error": "cannot move while door is open"}
    if direction not in (-1, 1):
        state.invalid_actions += 1
        return {"invalid": True, "error": "direction must be -1 or 1"}
    next_floor = state.floor + direction
    if not 0 <= next_floor < state.floors:
        state.invalid_actions += 1
        return {"invalid": True, "error": f"floor {next_floor} out of range 0..{state.floors - 1}"}
    state.floor = next_floor
    state.direction = direction
    return {"moved": True, "floor": state.floor}


def _step_open(state: ElevatorState) -> dict[str, Any]:
    if state.door_open:
        state.invalid_actions += 1
        return {"invalid": True, "error": "door already open"}
    state.door_open = True
    alighted = _alight(state)
    boarded, refused = _board(state)
    state.overload_refused += refused
    return {"opened": True, "alighted": alighted, "boarded": boarded, "refused": refused}


def _step_close(state: ElevatorState) -> dict[str, Any]:
    if not state.door_open:
        state.invalid_actions += 1
        return {"invalid": True, "error": "door already closed"}
    state.door_open = False
    return {"closed": True}


def _alight(state: ElevatorState) -> int:
    delivered_here = 0
    for passenger in state.passengers:
        if passenger.board_time is not None and passenger.deliver_time is None and passenger.dest == state.floor:
            passenger.deliver_time = state.time
            state.delivered += 1
            delivered_here += 1
    # drop delivered passengers from the riding list; keep waiting and riding ones
    state.passengers = [p for p in state.passengers if p.deliver_time is None]
    return delivered_here


def _board(state: ElevatorState) -> tuple[int, int]:
    boarded = 0
    refused = 0
    riding = [p for p in state.passengers if p.board_time is not None and p.deliver_time is None]
    queue = state.waiting.get(state.floor, [])
    keep: list[Passenger] = []
    for passenger in queue:
        if passenger.arrive_time > state.time:
            keep.append(passenger)
            continue
        if len(riding) + boarded < state.capacity:
            passenger.board_time = state.time
            state.total_wait += passenger.wait
            boarded += 1
        else:
            refused += 1
            keep.append(passenger)
    state.waiting[state.floor] = keep
    return boarded, refused


def _elapse_waiting(state: ElevatorState) -> None:
    """Advance time so newly-arrived passengers become eligible to board.

    The discrete clock increments by one step; passengers whose arrive_time is
    now in the past are visible in the waiting queues. No per-passenger counter
    is needed because ``wait`` derives from board_time - arrive_time.
    """


def format_state(state: ElevatorState) -> str:
    """Render the state as a compact prompt fragment for the model."""

    riding = [p.dest for p in state.passengers if p.board_time is not None and p.deliver_time is None]
    waiting_counts = [
        sum(1 for p in q if p.arrive_time <= state.time) for q in (state.waiting.get(i, []) for i in range(state.floors))
    ]
    direction_label = {1: "up", -1: "down", 0: "idle"}[state.direction]
    door_label = "OPEN" if state.door_open else "CLOSED"
    return (
        f"floor={state.floor}/{state.floors - 1} dir={direction_label} door={door_label} "
        f"cab=[{','.join(str(d) for d in riding) or 'empty'}] cab_load={len(riding)}/{state.capacity} "
        f"waiting=[{','.join(str(c) for c in waiting_counts)}] "
        f"delivered={state.delivered}/{state.total_passengers()} "
        f"time={state.time}/{state.horizon}"
    )


def make_prompt(record: dict[str, Any]) -> str:
    """Build the initial user prompt describing the scenario."""

    floors = int(record.get("floors", DEFAULT_FLOORS))
    capacity = int(record.get("capacity", DEFAULT_CAPACITY))
    horizon = int(record.get("horizon", DEFAULT_HORIZON))
    passengers = record.get("passengers", [])
    scenario = str(record.get("scenario", "mixed"))
    lines = [
        f"Run an elevator with {floors} floors (0..{floors - 1}), cab capacity {capacity}, "
        f"and {horizon} discrete time steps.",
        f"Scenario: {scenario}.",
        f"{len(passengers)} passengers need to be delivered to their destinations.",
        "",
        "On each turn call exactly one tool: move (door must be closed), open_door, close_door, or done.",
        "- move(direction): travel one floor; door must be closed.",
        "- open_door: alight riders whose dest is the current floor, board waiting passengers up to capacity.",
        "- close_door: required before moving.",
        "- done: end the episode once you believe the job is finished.",
        "Maximize delivered passengers, minimize waiting, and avoid invalid actions.",
    ]
    return "\n".join(lines)


def state_to_dict(state: ElevatorState) -> dict[str, Any]:
    """Serialize state for JSON fixtures and tests."""

    data = asdict(state)
    # waiting dict uses int keys that JSON converts to str; normalize for round-trip
    data["waiting"] = {str(k): [asdict(p) for p in v] for k, v in state.waiting.items()}
    data["passengers"] = [asdict(p) for p in state.passengers]
    return data


def parse_action(response_message: Any) -> dict[str, Any] | None:
    """Extract the first actionable tool call from a model response message.

    Accepts either a raw OpenAI-style message object or a dict produced by
    ``_assistant_message`` helpers. Returns ``{"name": ..., **args}`` or None.
    """

    calls = _extract_tool_calls(response_message)
    if not calls:
        return None
    call = calls[0]
    name = call.get("function", {}).get("name") or call.get("name")
    raw_args = call.get("function", {}).get("arguments") or call.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except (json.JSONDecodeError, TypeError):
        args = {}
    if name is None:
        return None
    return {"name": name, **args}


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Normalize tool_calls from either an object or a dict."""

    raw = getattr(message, "tool_calls", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("tool_calls")
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for call in raw:
        if isinstance(call, dict):
            out.append(call)
        else:
            out.append(
                {
                    "id": getattr(call, "id", None),
                    "type": getattr(call, "type", "function"),
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
    return out


def fcfs_policy(state: ElevatorState) -> dict[str, Any]:
    """First-come-first-served baseline action for comparison.

    Greedily serves the earliest-arriving waiting passenger: move toward that
    passenger's origin, open to board, move toward destination, open to alight.
    Falls back to closing the door or ending when idle.
    """

    if is_terminal(state):
        return {"name": "done"}
    # close door if needed to move
    if state.door_open:
        riding = [p for p in state.passengers if p.board_time is not None and p.deliver_time is None]
        needs_move = bool(riding) or _has_pending_target(state)
        if needs_move:
            return {"name": "close_door"}
        # nothing to do; close then done
        return {"name": "close_door"}
    # pick the earliest waiting passenger or the earliest riding passenger destination
    target = _fcfs_target(state)
    if target is None:
        return {"name": "done"}
    if target == state.floor:
        return {"name": "open_door"}
    direction = 1 if target > state.floor else -1
    return {"name": "move", "direction": direction}


def _has_pending_target(state: ElevatorState) -> bool:
    riding = [p for p in state.passengers if p.board_time is not None and p.deliver_time is None]
    if riding:
        return True
    for queue in state.waiting.values():
        if any(p.arrive_time <= state.time for p in queue):
            return True
    return False


def _fcfs_target(state: ElevatorState) -> int | None:
    """Return the next target floor for the FCFS policy."""

    riding = [p for p in state.passengers if p.board_time is not None and p.deliver_time is None]
    eligible_waiters = [
        p
        for queue in state.waiting.values()
        for p in queue
        if p.arrive_time <= state.time and p.board_time is None
    ]
    if not riding and not eligible_waiters:
        return None
    # if riding, go to the earliest boarded passenger's destination
    if riding:
        earliest = min(riding, key=lambda p: p.board_time if p.board_time is not None else 0)
        return earliest.dest
    earliest = min(eligible_waiters, key=lambda p: p.arrive_time)
    return earliest.origin


def run_fcfs_episode(record: dict[str, Any]) -> dict[str, Any]:
    """Run a full episode under FCFS and return summary metrics."""

    state = build_state(record)
    steps_taken = 0
    while not is_terminal(state) and steps_taken <= state.horizon + 2:
        action = fcfs_policy(state)
        if action["name"] == "done":
            break
        step(state, action)
        steps_taken += 1
    return episode_metrics(state)


def episode_metrics(state: ElevatorState) -> dict[str, Any]:
    """Compute comparable metrics for an episode."""

    total = state.total_passengers()
    mean_wait = state.total_wait / max(total, 1)
    return {
        "delivered": state.delivered,
        "total_passengers": total,
        "total_wait": state.total_wait,
        "mean_wait": mean_wait,
        "invalid_actions": state.invalid_actions,
        "overload_refused": state.overload_refused,
        "time": state.time,
        "horizon": state.horizon,
        "scenario": state.scenario,
    }
