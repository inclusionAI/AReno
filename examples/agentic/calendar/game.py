"""Calendar scheduling helpers for the agentic RL example.

A self-contained calendar scheduling engine with participant availability,
fixed-offset time zones, meeting durations, and conflict detection. No
external calendars or databases are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeSlot:
    """A time range in *local* hours, 0-24."""

    start_hour: int
    end_hour: int

    def __post_init__(self) -> None:
        if self.start_hour < 0 or self.end_hour > 24:
            raise ValueError(f"TimeSlot out of range: {self.start_hour}-{self.end_hour}")
        if self.start_hour >= self.end_hour:
            raise ValueError(f"TimeSlot start must precede end: {self.start_hour}-{self.end_hour}")

    @property
    def duration(self) -> int:
        return self.end_hour - self.start_hour


@dataclass(frozen=True)
class Participant:
    name: str
    timezone: str  # "UTC+8", "UTC-5", "UTC", "UTC+5:30"
    available_slots: tuple[TimeSlot, ...]


@dataclass(frozen=True)
class Meeting:
    id: str
    duration_hours: int
    required_participants: tuple[str, ...]


@dataclass(frozen=True)
class CalendarState:
    participants: dict[str, Participant]
    meetings: tuple[Meeting, ...]
    confirmed: dict[str, tuple[int, int]] = field(default_factory=dict)
    # confirmed: meeting_id → (utc_start, utc_end)

    def meeting_by_id(self, meeting_id: str) -> Meeting | None:
        for m in self.meetings:
            if m.id == meeting_id:
                return m
        return None


# ---------------------------------------------------------------------------
# Time-zone helpers (fixed offset only, no DST)
# ---------------------------------------------------------------------------

_OFFSET_RE = re.compile(r"^UTC([+-]\d{1,2})(?::(\d{2}))?$")


def parse_offset(tz: str) -> int:
    """Return the UTC offset in *minutes* for a fixed-offset time zone string.

    >>> parse_offset("UTC+8")
    480
    >>> parse_offset("UTC-5")
    -300
    >>> parse_offset("UTC")
    0
    >>> parse_offset("UTC+5:30")
    330
    """
    if tz == "UTC":
        return 0
    m = _OFFSET_RE.match(tz)
    if m is None:
        raise ValueError(f"invalid timezone format: {tz!r}")
    hours = int(m.group(1))
    minutes = int(m.group(2)) if m.group(2) else 0
    if abs(hours) > 14:
        raise ValueError(f"invalid timezone offset: {tz!r} (max ±14)")
    return hours * 60 + (minutes if hours >= 0 else -minutes)


def to_utc(slot: TimeSlot, timezone: str) -> tuple[int, int]:
    """Convert a local TimeSlot to UTC hours, handling midnight wrap.

    Returns (utc_start, utc_end) where both are in [0, 24).
    A slot that wraps past midnight is split and the end is reported as 24+end.
    """
    offset_hours = parse_offset(timezone) / 60
    utc_start = (slot.start_hour - offset_hours) % 24
    utc_end_raw = slot.end_hour - offset_hours
    # Keep utc_end in a comparable range: if it wrapped, add 24.
    if utc_end_raw <= utc_start:
        utc_end = utc_end_raw + 24
    else:
        utc_end = utc_end_raw
    return int(utc_start), int(utc_end)


def format_utc_range(utc_start: int, utc_end: int) -> str:
    """Format a UTC range for human-readable output."""
    s = utc_start % 24
    e = utc_end % 24 if utc_end != 24 else 24
    return f"UTC {s:02d}:00-{e:02d}:00"


# ---------------------------------------------------------------------------
# Scheduling logic
# ---------------------------------------------------------------------------


def find_common_slots(
    meeting: Meeting,
    participants: dict[str, Participant],
) -> list[tuple[int, int]]:
    """Return all UTC time ranges where every required participant is available.

    Each range is (utc_start, utc_end) in UTC hours. Ranges that wrap past
    midnight are represented with utc_end > 24.
    """
    required = [participants[name] for name in meeting.required_participants if name in participants]
    if len(required) != len(meeting.required_participants):
        return []

    # Collect each participant's UTC availability as a list of (start, end).
    all_avails: list[list[tuple[int, int]]] = []
    for p in required:
        avails: list[tuple[int, int]] = []
        for slot in p.available_slots:
            avails.append(to_utc(slot, p.timezone))
        all_avails.append(avails)

    # Intersect all participants' availability.
    common = all_avails[0]
    for avails in all_avails[1:]:
        common = _intersect_ranges(common, avails)
        if not common:
            return []

    # Filter by minimum duration.
    result = [(s, e) for s, e in common if e - s >= meeting.duration_hours]
    return result


def _intersect_ranges(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Intersect two lists of (start, end) ranges."""
    result: list[tuple[int, int]] = []
    for sa, ea in a:
        for sb, eb in b:
            start = max(sa, sb)
            end = min(ea, eb)
            if start < end:
                result.append((start, end))
    return result


def check_conflict(
    proposed_start: int,
    proposed_end: int,
    confirmed: dict[str, tuple[int, int]],
    exclude_meeting_id: str | None = None,
) -> str | None:
    """Return the meeting_id of a conflicting confirmed meeting, or None."""
    for mid, (cs, ce) in confirmed.items():
        if exclude_meeting_id and mid == exclude_meeting_id:
            continue
        if proposed_start < ce and proposed_end > cs:
            return mid
    return None


def validate_proposal(
    state: CalendarState,
    meeting_id: str,
    utc_start: int,
    utc_end: int,
) -> str | None:
    """Return an error string if the proposal is invalid, or None if valid."""
    meeting = state.meeting_by_id(meeting_id)
    if meeting is None:
        return f"meeting {meeting_id!r} not found"

    if utc_start < 0 or utc_start >= 24:
        return f"utc_start {utc_start} out of range [0, 24)"

    if utc_end <= utc_start:
        return f"utc_end {utc_end} must be greater than utc_start {utc_start}"

    if utc_end - utc_start != meeting.duration_hours:
        return (
            f"duration mismatch: proposed {utc_end - utc_start}h, "
            f"required {meeting.duration_hours}h"
        )

    # Check all required participants are available.
    common = find_common_slots(meeting, state.participants)
    found = any(s <= utc_start and utc_end <= e for s, e in common)
    if not found:
        return "proposed slot is not within any common available range"

    # Check conflict with confirmed meetings.
    conflict = check_conflict(utc_start, utc_end, state.confirmed, exclude_meeting_id=meeting_id)
    if conflict is not None:
        return f"conflicts with confirmed meeting {conflict!r}"

    return None


def score_proposal(
    state: CalendarState,
    meeting_id: str,
    utc_start: int,
    utc_end: int,
) -> float:
    """Score a proposed slot: -1.0 (invalid) to 1.0 (perfect)."""
    error = validate_proposal(state, meeting_id, utc_start, utc_end)
    if error is not None:
        return -1.0
    return 1.0


def score_tool_efficiency(tool_calls: list[dict[str, Any]], meeting_id: str) -> float:
    """Score tool-call efficiency: 0.0 (redundant) to 1.0 (minimal).

    Ideal flow: query each required participant once, propose once, confirm once.
    """
    query_count = 0
    propose_count = 0
    confirm_count = 0
    queried_participants: set[str] = set()

    for call in tool_calls:
        name = call.get("name", "") if isinstance(call, dict) else ""
        args = call.get("arguments", {}) if isinstance(call, dict) else {}
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        if name == "query_availability":
            participant = args.get("participant", "")
            if participant not in queried_participants:
                queried_participants.add(participant)
            query_count += 1
        elif name == "propose_slot":
            propose_count += 1
        elif name == "confirm_slot":
            confirm_count += 1

    # Ideal: 1 propose, 1 confirm, queries == unique participants (no repeats).
    efficiency = 1.0
    # Penalise redundant queries.
    if query_count > len(queried_participants):
        efficiency -= 0.2 * (query_count - len(queried_participants))
    # Penalise multiple proposes.
    if propose_count > 1:
        efficiency -= 0.3 * (propose_count - 1)
    # Penalise multiple confirms.
    if confirm_count > 1:
        efficiency -= 0.3 * (confirm_count - 1)
    # Reward confirm present.
    if confirm_count == 0:
        efficiency -= 0.3

    return max(0.0, min(1.0, efficiency))


def compute_reward(
    state: CalendarState,
    meeting_id: str,
    utc_start: int,
    utc_end: int,
    tool_calls: list[dict[str, Any]],
) -> float:
    """Combined reward: constraint satisfaction (0.7) + tool efficiency (0.3)."""
    constraint_score = score_proposal(state, meeting_id, utc_start, utc_end)
    if constraint_score < 0:
        return -1.0
    efficiency = score_tool_efficiency(tool_calls, meeting_id)
    return 0.7 * constraint_score + 0.3 * efficiency


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(state: CalendarState, meeting_id: str) -> str:
    """Render a human-readable scheduling scenario for the LLM.

    Note: participant availability details are intentionally NOT included
    in the prompt. The model must discover them by calling the
    query_availability tool, which returns UTC-converted slots.
    """
    meeting = state.meeting_by_id(meeting_id)
    if meeting is None:
        raise ValueError(f"meeting {meeting_id!r} not found")

    lines = [
        "You are a calendar scheduling assistant.",
        "Your task is to schedule the meeting described below.",
        "Use the provided tools to query availability, propose a slot, and confirm it.",
        "",
        f"Meeting ID: {meeting.id}",
        f"Duration: {meeting.duration_hours} hour(s)",
        f"Required participants: {', '.join(meeting.required_participants)}",
        "",
        "Participants:",
    ]

    # Only list participant names and timezones, NOT their availability.
    # The model must call query_availability to learn the actual slots.
    for name in meeting.required_participants:
        p = state.participants.get(name)
        if p is None:
            lines.append(f"  {name}: (not found)")
            continue
        lines.append(f"  {name}: timezone={p.timezone}")

    if state.confirmed:
        lines.append("")
        lines.append("Already confirmed meetings:")
        for mid, (cs, ce) in state.confirmed.items():
            if mid == meeting_id:
                continue
            lines.append(f"  {mid}: {format_utc_range(cs, ce)}")

    lines.append("")
    lines.append("Steps:")
    lines.append("1. Call query_availability for each required participant to learn their UTC availability.")
    lines.append("2. Find a UTC time range that overlaps all participants' availability and fits the meeting duration.")
    lines.append("3. Call propose_slot with the meeting_id and the UTC time range.")
    lines.append("4. Call confirm_slot to finalize the booking.")
    lines.append("")
    lines.append("Answer by calling the tools. Do not write free text.")

    return "\n".join(lines)


def format_prompt_for_record(raw: dict, index: int) -> str:
    """Build a prompt from a raw JSONL record."""
    state = record_to_state(raw)
    meeting_id = raw.get("target_meeting_id", state.meetings[0].id if state.meetings else "")
    return format_prompt(state, meeting_id)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def state_to_record(state: CalendarState, target_meeting_id: str = "") -> dict:
    """Serialize a CalendarState to a JSON-able dict."""
    return {
        "id": target_meeting_id or (state.meetings[0].id if state.meetings else ""),
        "participants": {
            name: {
                "name": p.name,
                "timezone": p.timezone,
                "available_slots": [
                    {"start_hour": s.start_hour, "end_hour": s.end_hour} for s in p.available_slots
                ],
            }
            for name, p in state.participants.items()
        },
        "meetings": [
            {
                "id": m.id,
                "duration_hours": m.duration_hours,
                "required_participants": list(m.required_participants),
            }
            for m in state.meetings
        ],
        "confirmed": {mid: list(rng) for mid, rng in state.confirmed.items()},
        "target_meeting_id": target_meeting_id or (state.meetings[0].id if state.meetings else ""),
    }


def record_to_state(raw: dict) -> CalendarState:
    """Deserialize a JSON dict into a CalendarState."""
    participants: dict[str, Participant] = {}
    for name, pdata in raw.get("participants", {}).items():
        slots = tuple(
            TimeSlot(s["start_hour"], s["end_hour"])
            for s in pdata.get("available_slots", [])
        )
        participants[name] = Participant(
            name=pdata.get("name", name),
            timezone=pdata["timezone"],
            available_slots=slots,
        )

    meetings = tuple(
        Meeting(
            id=m["id"],
            duration_hours=m["duration_hours"],
            required_participants=tuple(m["required_participants"]),
        )
        for m in raw.get("meetings", [])
    )

    confirmed = {
        mid: (rng[0], rng[1])
        for mid, rng in raw.get("confirmed", {}).items()
    }

    return CalendarState(
        participants=participants,
        meetings=meetings,
        confirmed=confirmed,
    )


# ---------------------------------------------------------------------------
# Tool execution functions (called by run_agent.py during multi-turn rollout)
# ---------------------------------------------------------------------------


def execute_query_availability(state: CalendarState, participant_name: str) -> dict:
    """Execute query_availability: return a participant's availability in UTC.

    This is called by run_agent.py after the model requests availability,
    so the model sees the real UTC-converted slots before proposing a time.
    """
    p = state.participants.get(participant_name)
    if p is None:
        return {"error": f"participant {participant_name!r} not found"}
    utc_slots = []
    for slot in p.available_slots:
        utc_start, utc_end = to_utc(slot, p.timezone)
        utc_slots.append({
            "local_start": slot.start_hour,
            "local_end": slot.end_hour,
            "timezone": p.timezone,
            "utc_start": utc_start,
            "utc_end": utc_end,
        })
    return {"participant": participant_name, "timezone": p.timezone, "available_slots_utc": utc_slots}


def execute_propose_slot(state: CalendarState, meeting_id: str, utc_start: int, utc_end: int) -> dict:
    """Execute propose_slot: validate the proposed slot and return the result.

    The model sees whether the proposal is valid before confirming.
    """
    error = validate_proposal(state, meeting_id, utc_start, utc_end)
    if error is not None:
        return {"valid": False, "error": error}
    return {"valid": True, "meeting_id": meeting_id, "utc_start": utc_start, "utc_end": utc_end}


def execute_confirm_slot(state: CalendarState, meeting_id: str, utc_start: int, utc_end: int) -> dict:
    """Execute confirm_slot: finalize the booking if valid."""
    error = validate_proposal(state, meeting_id, utc_start, utc_end)
    if error is not None:
        return {"confirmed": False, "error": error}
    return {"confirmed": True, "meeting_id": meeting_id, "utc_start": utc_start, "utc_end": utc_end}