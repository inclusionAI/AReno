"""Small elevator-dispatch helpers for agentic examples.

A building has ``floors`` levels and one car with ``capacity`` passenger slots.
Passengers arrive at floors with a destination and wait in a per-floor hall
queue; the car dispatches them by moving (``U``/``D``), opening its door to let
passengers off and on (``O``), and closing again (``C``). The event queue
advances one tick per action, so the same building + sequence + seed replays
bit-identically.

Action letters
---------------
``U`` move the car up one floor, ``D`` move down one floor, ``O`` open the door
(lets passengers off then on, bounded by ``capacity``), ``C`` close the door.
Invalid actions -- moving past the top/bottom floor, opening when the door is
open, closing when it is closed -- are skipped but counted as ``n_invalid``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

ACTIONS = ("U", "D", "O", "C")
DEFAULT_FLOORS = 6
DEFAULT_CAPACITY = 4
DEFAULT_MAX_STEPS = 200
DEFAULT_SEED = 2026

# Built via concatenation so the raw source never carries a literal think tag,
# which keeps some editors/tooling from mis-parsing the module.
_THINK_OPEN = r"<think" + r"\b[^>]*>"
_THINK_CLOSE = r"</" + r"think>"
_THINK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.IGNORECASE | re.DOTALL)
_XML_ACTIONS_RE = re.compile(r"<dispatch>\s*([UDROC]+)\s*</dispatch>", re.IGNORECASE | re.DOTALL)
_CHAT_SPECIAL_RE = re.compile(r"<\|[^>]+?\|>|</?s>", re.IGNORECASE)


def normalize_building(building: dict[str, Any]) -> dict[str, Any]:
    """Return a validated elevator building.

    Required keys: ``floors`` (>=2), ``capacity`` (>=1, the integer slot count
    of the car), ``arrivals`` (a list of ``(tick, from_floor, to_floor)``).
    Optional: ``car`` (overridden position/door), ``seed``. Malformed values
    raise ``ValueError`` before any model or worker initialization.
    """

    floors = int(building["floors"])
    capacity = int(building["capacity"])
    if floors < 2:
        raise ValueError("floors must be >= 2")
    if capacity < 1:
        raise ValueError("capacity must be >= 1")

    car = dict(building.get("car") or {})
    car_floor = int(car.get("floor", 0))
    if not 0 <= car_floor < floors:
        raise ValueError(f"car floor out of range: {car_floor}")

    hall_queues: list[list[dict[str, int]]] = [[] for _ in range(floors)]
    raw_hall = building.get("hall_queues")
    if isinstance(raw_hall, dict):
        for raw_from, queue in raw_hall.items():
            from_floor = int(raw_from)
            if not 0 <= from_floor < floors:
                raise ValueError(f"hall queue floor out of range: {from_floor}")
            for passenger in queue:
                hall_queues[from_floor].append(_normalize_passenger(passenger, floors))
    elif isinstance(raw_hall, list):
        # Already normalized form (one list per floor).
        for from_floor, queue in enumerate(raw_hall):
            for passenger in queue:
                hall_queues[from_floor].append(_normalize_passenger(passenger, floors))

    car_passengers = [_normalize_passenger(p, floors) for p in (car.get("passengers") or [])]
    if len(car_passengers) > capacity:
        raise ValueError("car carries more passengers than capacity")

    arrivals = _normalize_arrivals(building["arrivals"], floors)

    return {
        "floors": floors,
        "capacity": capacity,
        "car": {
            "floor": car_floor,
            "direction": str(car.get("direction", "U")).upper(),
            "door_open": bool(car.get("door_open", False)),
            "passengers": car_passengers,
        },
        "hall_queues": hall_queues,
        "arrivals": arrivals,
        "tick": int(building.get("tick", 0)),
        "seed": int(building.get("seed", DEFAULT_SEED)),
    }


def _normalize_passenger(passenger: Any, floors: int) -> dict[str, int]:
    if isinstance(passenger, dict):
        to_floor = int(passenger["to_floor"])
        arrive_tick = int(passenger.get("arrive_tick", 0))
    else:
        values = list(passenger)
        to_floor = int(values[0])
        arrive_tick = int(values[1]) if len(values) > 1 else 0
    if not 0 <= to_floor < floors:
        raise ValueError(f"passenger to_floor out of range: {to_floor}")
    return {"to_floor": to_floor, "arrive_tick": arrive_tick}


def _normalize_arrivals(arrivals: Iterable, floors: int) -> list[dict[str, int]]:
    normalized: list[dict[str, int]] = []
    for idx, raw in enumerate(arrivals):
        if isinstance(raw, dict):
            tick = int(raw["tick"])
            from_floor = int(raw["from_floor"])
            to_floor = int(raw["to_floor"])
        else:
            tick, from_floor, to_floor = (int(v) for v in raw)
        if not 0 <= from_floor < floors or not 0 <= to_floor < floors:
            raise ValueError(f"arrival floor out of range: {raw}")
        if from_floor == to_floor:
            raise ValueError(f"arrival source equals destination: {raw}")
        normalized.append({"tick": tick, "seq": idx, "from_floor": from_floor, "to_floor": to_floor})
    normalized.sort(key=lambda e: (e["tick"], e["seq"]))
    return normalized


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    """Deep-ish copy of mutable simulation state for safe replay."""

    car = dict(state["car"])
    car["passengers"] = [dict(p) for p in car["passengers"]]
    return {
        "floors": state["floors"],
        "capacity": state["capacity"],
        "car": car,
        "hall_queues": [[dict(p) for p in queue] for queue in state["hall_queues"]],
        "arrivals": [dict(e) for e in state["arrivals"]],
        "tick": state["tick"],
        "seed": state["seed"],
    }


def advance_events(state: dict[str, Any], until_tick: int) -> None:
    """Land every arrival whose scheduled tick is <= ``until_tick`` in place.

    Mutates the building state: pending arrivals enter the hall queue of their
    floor so the same sequence of ticks always yields the same queues.
    """

    remaining: list[dict[str, int]] = []
    for event in state["arrivals"]:
        if event["tick"] <= until_tick:
            state["hall_queues"][event["from_floor"]].append(
                {"to_floor": event["to_floor"], "arrive_tick": event["tick"]}
            )
        else:
            remaining.append(event)
    state["arrivals"] = remaining


def step(state: dict[str, Any], action: str, stats: dict[str, int]) -> bool:
    """Apply one action to ``state`` and return whether it was valid.

    ``stats`` accumulates counters (``delivered``/``total_wait``/``max_wait``).
    Every action advances the clock by one tick and lands any due arrivals, so
    action ordering fully determines the simulation.
    """

    action = str(action).upper()
    if action not in ACTIONS:
        return False
    car = state["car"]
    floors = state["floors"]

    # The clock always advances by one tick per action; arrivals land first so a
    # passenger arriving on this tick can be served by this tick's door open.
    state["tick"] += 1
    advance_events(state, state["tick"])

    if action == "U":
        if car["door_open"] or car["floor"] + 1 >= floors:
            return False
        car["floor"] += 1
        car["direction"] = "U"
        return True
    if action == "D":
        if car["door_open"] or car["floor"] - 1 < 0:
            return False
        car["floor"] -= 1
        car["direction"] = "D"
        return True
    if action == "C":
        if not car["door_open"]:
            return False
        car["door_open"] = False
        return True
    # action == "O": open the door and exchange passengers.
    if car["door_open"]:
        return False
    car["door_open"] = True
    _exchange_passengers(state, stats)
    return True


def _exchange_passengers(state: dict[str, Any], stats: dict[str, int]) -> None:
    """Let passengers off, then on up to ``capacity`` (overload prevention)."""

    car = state["car"]
    floor = car["floor"]
    tick = state["tick"]

    staying: list[dict[str, int]] = []
    for passenger in car["passengers"]:
        if passenger["to_floor"] == floor:
            wait = tick - passenger["arrive_tick"]
            stats["delivered"] += 1
            stats["total_wait"] += wait
            if wait > stats["max_wait"]:
                stats["max_wait"] = wait
        else:
            staying.append(passenger)
    car["passengers"] = staying

    # Board from this floor's waiting queue, FIFO, up to remaining capacity.
    queue = state["hall_queues"][floor]
    boarded: list[dict[str, int]] = []
    leftover: list[dict[str, int]] = []
    for passenger in queue:
        if len(boarded) + len(car["passengers"]) < state["capacity"]:
            boarded.append(passenger)
        else:
            leftover.append(passenger)
    car["passengers"].extend(boarded)
    state["hall_queues"][floor] = leftover


def remaining_passengers(state: dict[str, Any]) -> int:
    """Count passengers still waiting or aboard -- those not yet delivered."""

    aboard = len(state["car"]["passengers"])
    waiting = sum(len(queue) for queue in state["hall_queues"])
    pending = len(state["arrivals"])
    return aboard + waiting + pending


def is_terminal(state: dict[str, Any]) -> bool:
    """Return whether nothing remains to dispatch (game over)."""

    car = state["car"]
    return (
        not car["passengers"]
        and all(len(queue) == 0 for queue in state["hall_queues"])
        and not state["arrivals"]
    )


def play(
    building: dict[str, Any],
    sequence: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Replay an action sequence deterministically and return metrics.

    Each action consumes one tick and advances the clock, so the same building
    + sequence is bit-identical. Invalid actions (no-ops) are counted but do not
    stop the episode. The episode also stops at ``max_steps`` valid actions or a
    terminal building. The metric dict matches the fields checked by tests and
    the reward function.
    """

    state = clone_state(normalize_building(building))
    sequence = "".join(ch for ch in str(sequence).upper() if ch in ACTIONS)
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    stats = {"delivered": 0, "total_wait": 0, "max_wait": 0}
    n_valid = 0
    n_invalid = 0
    terminal = False
    for action in sequence:
        if n_valid >= max_steps:
            break
        if step(state, action, stats):
            n_valid += 1
        else:
            n_invalid += 1
        if is_terminal(state):
            terminal = True
            break
    if not terminal:
        terminal = is_terminal(state)

    delivered = stats["delivered"]
    counts = n_valid + n_invalid
    return {
        "state": state,
        "delivered_passengers": delivered,
        "mean_wait": (stats["total_wait"] / delivered) if delivered else 0.0,
        "max_wait": stats["max_wait"],
        "total_wait": stats["total_wait"],
        "n_steps": n_valid,
        "n_attempts": counts,
        "n_invalid": n_invalid,
        "invalid_rate": (n_invalid / counts) if counts else 0.0,
        "remaining_passengers": remaining_passengers(state),
        "terminal": terminal,
    }


def building_to_text(building: dict[str, Any]) -> str:
    """Render the building for the policy prompt.

    Shows floors with the waiting queue sizes and destinations, the car's
    position/door state, and the names of pending arrivals so the model can
    plan without guessing.
    """

    state = normalize_building(building)
    lines = []
    floors = state["floors"]
    for floor in reversed(range(floors)):
        queue = state["hall_queues"][floor]
        marker = "[CAR>" if state["car"]["floor"] == floor else "      "
        door = "open ]" if state["car"]["door_open"] else "closed]"
        aboard = [str(p["to_floor"]) for p in state["car"]["passengers"]] if state["car"]["floor"] == floor else []
        waiting = ",".join(str(p["to_floor"]) for p in queue) if queue else "-"
        lines.append(f"F{floor} {marker}{door} aboard=[{','.join(aboard)}] waiting->{waiting}")
    pending = ",".join(f"t{e['tick']}:F{e['from_floor']}->F{e['to_floor']}" for e in state["arrivals"]) or "-"
    lines.append(f"pending arrivals: {pending}")
    lines.append(f"tick={state['tick']} capacity={state['capacity']}")
    return "\n".join(lines)


def format_prompt(building: dict[str, Any]) -> str:
    """Build the episode prompt for the tool-call agent."""

    return (
        "You are dispatching an elevator. Minimize passenger wait and deliver "
        "everyone, never exceeding capacity.\n\n"
        "Output format:\n"
        "- Call the dispatch tool with a string of single-letter actions.\n"
        "- U move up one floor, D move down one floor, O open the door to let "
        "passengers off and on, C close the door.\n"
        "- The door must be open to exchange passengers and closed to move.\n"
        "- Each action takes one tick; invalid actions (wrong door state, "
        "moving past the top/bottom floor) are penalized.\n\n"
        f"Building:\n{building_to_text(building)}\n\nDispatch:"
    )


def format_xml_prompt(building: dict[str, Any]) -> str:
    """Build the episode prompt for the XML no-tool agent."""

    return (
        "You are dispatching an elevator. Minimize passenger wait and deliver "
        "everyone, never exceeding capacity.\n\n"
        "Output format:\n"
        "- Answer with exactly one XML tag such as <dispatch>UDOCUD</dispatch>.\n"
        "- U move up, D move down, O open the door, C close the door.\n"
        "- The door must be open to exchange passengers and closed to move.\n"
        "- Each action takes one tick; invalid actions are penalized.\n\n"
        f"Building:\n{building_to_text(building)}\n\nDispatch:"
    )


def parse_action_sequence(text: str) -> str:
    """Return the final action sequence from a model response.

    Strips reasoning spans and chat-template sentinels first, then takes the
    last ``<dispatch>UDOC</dispatch>`` tag. Empty result means nothing parsed;
    the reward function maps that to a failing score -- mirroring how the
    other agentic examples parse without a raw-token fallback.
    """

    text = _CHAT_SPECIAL_RE.sub(" ", _THINK_RE.sub(" ", text)).strip()
    matches = list(_XML_ACTIONS_RE.finditer(text))
    return matches[-1].group(1).upper() if matches else ""


def fresh_building(rng, *, floors: int = DEFAULT_FLOORS, capacity: int = DEFAULT_CAPACITY) -> dict[str, Any]:
    """Return a new building with a couple of seeded starter passengers.

    Uses random.Random directly so generation stays deterministic given a seed
    and free of any heavy dependency.
    """

    if floors < 2:
        raise ValueError("floors must be >= 2")
    arrivals = []
    for _ in range(2):
        from_floor = rng.randrange(floors)
        to_floor = rng.randrange(floors - 1)
        if to_floor >= from_floor:
            to_floor += 1  # keep from == to impossible
        arrivals.append({"tick": 0, "from_floor": from_floor, "to_floor": to_floor})
    return {
        "floors": floors,
        "capacity": capacity,
        "arrivals": arrivals,
        "car": {"floor": 0, "direction": "U", "door_open": False, "passengers": []},
    }


def fcfs_actions(building: dict[str, Any]) -> str:
    """Build a first-come-first-served action string for a building.

    Greedily drives the car to the lowest floor that has waiting passengers or
    the lowest destination of an aboard passenger, opens to exchange, closes,
    and repeats. Routing every move through :func:`step` keeps the tick and
    event-queue advancement identical to :func:`play`, so a FCFS string replays
    cleanly. Side-effect free -- only reads ``building``.
    """

    state = clone_state(normalize_building(building))
    actions: list[str] = []
    stats = {"delivered": 0, "total_wait": 0, "max_wait": 0}
    guard = state["floors"] * 8 + 120
    while not is_terminal(state) and len(actions) < guard:
        if state["car"]["door_open"]:
            actions.append("C")
            step(state, "C", stats)
            continue
        # Land any arrivals already due so target selection sees them.
        advance_events(state, state["tick"])
        target = _fcfs_target(state)
        if target is None:
            # Nothing ready yet; coast one tick in the car's direction to keep
            # the clock advancing and land future arrivals.
            if not state["arrivals"] and not state["car"]["passengers"]:
                break
            coast = _coast(state)
            if coast is not None:
                actions.append(coast)
                step(state, coast, stats)
            continue
        if state["car"]["floor"] == target:
            actions.append("O")
            step(state, "O", stats)
            continue
        action = "U" if target > state["car"]["floor"] else "D"
        actions.append(action)
        step(state, action, stats)
    return "".join(actions)


def _coast(state: dict[str, Any]) -> str | None:
    """Pick a valid idle action when only future arrivals remain.

    Keeps the car moving in its current direction without running past an edge;
    returns None if no legal coasting move exists (car pinned at an edge), so the
    caller can drive the clock another way.
    """

    if state["car"]["door_open"]:
        return "C"
    floor = state["car"]["floor"]
    direction = state["car"]["direction"] or "U"
    if direction == "U" and floor + 1 < state["floors"]:
        return "U"
    if direction == "D" and floor - 1 >= 0:
        return "D"
    # Flip direction and try again.
    if floor + 1 < state["floors"]:
        state["car"]["direction"] = "U"
        return "U"
    if floor - 1 >= 0:
        state["car"]["direction"] = "D"
        return "D"
    return None


def _fcfs_target(state: dict[str, Any]) -> int | None:
    """Pick the lowest floor worth stopping at for FCFS dispatch.

    Deliver aboard passengers first (their drop-off floor), because a full car
    cannot board anyone; once the car is empty, pick up the lowest waiting hall
    passenger. This keeps the policy first-come-first-served while respecting
    capacity.
    """

    aboard = state["car"]["passengers"]
    if aboard:
        return min(p["to_floor"] for p in aboard)
    floors = state["floors"]
    for floor in range(floors):
        if state["hall_queues"][floor]:
            return floor
    return None