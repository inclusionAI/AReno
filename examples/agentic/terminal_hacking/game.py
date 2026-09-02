"""Deterministic retro-terminal word hacking environment."""

from __future__ import annotations

import copy
import json
import random
from typing import Any
from urllib.parse import urlparse

DEFAULT_ATTEMPTS = 3
MAX_TURNS = 8
FILTER_SYSTEM_PROMPT = """You are a terminal password candidate filter.

For every word in active_candidates, compare it with every previous guess position by position. Count a match only when both words have the same letter in the same position. Keep a word if and only if its match count equals the observed likeness for every guess clue. Preserve active_candidates order. Exclude guessed and removed words. If there are no guess clues, keep every active candidate.

Probe history does not add likeness constraints. Call submit_candidates exactly once with every and only consistent candidate. Output no prose outside the tool call."""


def tool_choice_for_base_url(base_url: object) -> str | dict[str, Any]:
    """Use auto tool selection for DeepSeek thinking-mode API compatibility."""

    hostname = (urlparse(str(base_url)).hostname or "").lower()
    if hostname == "api.deepseek.com":
        return "auto"
    return "required"


def extra_body_for_base_url(base_url: object) -> dict[str, Any] | None:
    """Disable DeepSeek thinking mode so tool selection remains concise and compatible."""

    hostname = (urlparse(str(base_url)).hostname or "").lower()
    if hostname == "api.deepseek.com":
        return {"thinking": {"type": "disabled"}}
    return None


def likeness(left: str, right: str) -> int:
    """Return the number of equal characters in equal positions."""

    if len(left) != len(right):
        raise ValueError("words must have the same length")
    return sum(a == b for a, b in zip(left, right, strict=True))


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy a hidden puzzle record."""

    record = copy.deepcopy(raw)
    candidates = [str(word).upper() for word in record.get("candidates", [])]
    if len(candidates) < 6 or len(candidates) != len(set(candidates)):
        raise ValueError("candidates must contain at least six unique words")
    word_length = len(candidates[0])
    if word_length < 4 or any(len(word) != word_length or not word.isalpha() for word in candidates):
        raise ValueError("candidate words must be alphabetic and have one shared length")
    password = str(record.get("password", "")).upper()
    if password not in candidates:
        raise ValueError("password must be one of the candidates")
    attempts = int(record.get("attempts", DEFAULT_ATTEMPTS))
    if attempts < 1:
        raise ValueError("attempts must be positive")
    probes = []
    seen_probe_ids: set[str] = set()
    for raw_probe in record.get("probes", []):
        probe = dict(raw_probe)
        probe_id = str(probe.get("id", ""))
        effect = str(probe.get("effect", ""))
        token = str(probe.get("token", ""))
        if not probe_id or probe_id in seen_probe_ids:
            raise ValueError("probe ids must be unique and non-empty")
        if effect not in {"remove_dud", "replenish"}:
            raise ValueError(f"invalid probe effect: {effect}")
        if len(token) < 2 or (token[0], token[-1]) not in {("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")}:
            raise ValueError(f"invalid bracket token: {token}")
        target = probe.get("target")
        if target is not None and str(target).upper() not in candidates:
            raise ValueError("probe target must be a candidate")
        probes.append(
            {
                "id": probe_id,
                "token": token,
                "effect": effect,
                "target": str(target).upper() if target is not None else None,
            }
        )
        seen_probe_ids.add(probe_id)
    dump_rows = [dict(row) for row in record.get("dump_rows", [])]
    if not dump_rows:
        raise ValueError("dump_rows must not be empty")
    record.update(
        {
            "candidates": candidates,
            "password": password,
            "attempts": attempts,
            "probes": probes,
            "dump_rows": dump_rows,
            "word_length": word_length,
        }
    )
    return record


def candidate_tool(active_candidates: list[str]) -> dict[str, Any]:
    """Build a closed tool schema for the predicted consistent candidate set."""

    return {
        "type": "function",
        "function": {
            "name": "submit_candidates",
            "description": "Submit every active word that satisfies every observed likeness clue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string", "enum": active_candidates},
                        "minItems": 1,
                        "uniqueItems": True,
                        "description": "All and only currently consistent candidates, in active_candidates order.",
                    }
                },
                "required": ["candidates"],
                "additionalProperties": False,
            },
        },
    }


def probe_tool(available_probes: list[dict[str, str]]) -> dict[str, Any]:
    """Build the tool schema for a model-selected bracket probe."""

    actions = [f"probe:{probe['id']}" for probe in available_probes]
    return {
        "type": "function",
        "function": {
            "name": "access_terminal",
            "description": "Use one currently available bracket probe instead of submitting candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": actions,
                        "description": "One available probe action.",
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }


def policy_tools(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the candidate and optional probe tools for one compact policy turn."""

    tools = [candidate_tool(state["active_candidates"])]
    if state["available_probes"]:
        tools.append(probe_tool(state["available_probes"]))
    return tools


class TerminalHackingSession:
    """Stateful deterministic puzzle session."""

    def __init__(self, raw_record: dict[str, Any]) -> None:
        self.record = normalize_record(raw_record)
        self.password = self.record["password"]
        self.attempts_max = self.record["attempts"]
        self.attempts_remaining = self.attempts_max
        self.guessed: list[str] = []
        self.removed: list[str] = []
        self.used_probes: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.solved = False
        self.locked = False

    @property
    def done(self) -> bool:
        return self.solved or self.locked

    def allowed_actions(self) -> list[str]:
        if self.done:
            return []
        actions = [
            f"guess:{word}"
            for word in self.record["candidates"]
            if word not in self.guessed and word not in self.removed
        ]
        actions.extend(f"probe:{probe['id']}" for probe in self.record["probes"] if probe["id"] not in self.used_probes)
        return actions

    def consistent_candidates(self) -> list[str]:
        """Return selectable candidates satisfying every observable likeness clue."""

        guesses = [event for event in self.history if event["kind"] == "guess"]
        return [
            word
            for word in self.record["candidates"]
            if word not in self.guessed
            and word not in self.removed
            and all(likeness(word, event["word"]) == event["likeness"] for event in guesses)
        ]

    def public_state(self) -> dict[str, Any]:
        active = [word for word in self.record["candidates"] if word not in self.guessed and word not in self.removed]
        return {
            "attempts_max": self.attempts_max,
            "attempts_remaining": self.attempts_remaining,
            "word_length": self.record["word_length"],
            "active_candidates": active,
            "guessed": list(self.guessed),
            "removed": list(self.removed),
            "available_probes": [
                {"id": probe["id"], "token": probe["token"]}
                for probe in self.record["probes"]
                if probe["id"] not in self.used_probes
            ],
            "history": copy.deepcopy(self.history),
            "dump_rows": copy.deepcopy(self.record["dump_rows"]),
            "solved": self.solved,
            "locked": self.locked,
            "done": self.done,
        }

    def execute(self, action: object) -> dict[str, Any]:
        if self.done:
            return self._result(False, str(action), "terminal session has already ended")
        normalized = str(action)
        if normalized not in self.allowed_actions():
            return self._result(False, normalized, "action is not currently available")
        if normalized.startswith("guess:"):
            return self._guess(normalized.removeprefix("guess:"))
        return self._probe(normalized.removeprefix("probe:"))

    def _guess(self, word: str) -> dict[str, Any]:
        self.guessed.append(word)
        matched = likeness(word, self.password)
        self.solved = word == self.password
        if not self.solved:
            self.attempts_remaining -= 1
            self.locked = self.attempts_remaining <= 0
        event = {
            "kind": "guess",
            "word": word,
            "likeness": matched,
            "out_of": len(self.password),
            "solved": self.solved,
        }
        self.history.append(event)
        return self._result(True, f"guess:{word}", None, event=event)

    def _probe(self, probe_id: str) -> dict[str, Any]:
        probe = next(probe for probe in self.record["probes"] if probe["id"] == probe_id)
        self.used_probes.append(probe_id)
        removed = None
        if probe["effect"] == "replenish":
            self.attempts_remaining = self.attempts_max
        else:
            preferred = probe.get("target")
            candidates = [
                word
                for word in self.record["candidates"]
                if word != self.password and word not in self.guessed and word not in self.removed
            ]
            if preferred in candidates:
                removed = preferred
            elif candidates:
                removed = candidates[0]
            if removed is not None:
                self.removed.append(removed)
        event = {
            "kind": "probe",
            "probe_id": probe_id,
            "token": probe["token"],
            "effect": probe["effect"],
            "removed": removed,
            "attempts_remaining": self.attempts_remaining,
        }
        self.history.append(event)
        return self._result(True, f"probe:{probe_id}", None, event=event)

    def _result(
        self,
        valid: bool,
        action: str,
        error: str | None,
        *,
        event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "valid": valid,
            "action": action,
            "error": error,
            "event": event,
            "current_state": self.public_state(),
        }


def candidate_filter_session(raw_record: dict[str, Any]) -> TerminalHackingSession:
    """Build one deterministic nonterminal state for single-turn candidate training."""

    session = TerminalHackingSession(raw_record)
    duds = [word for word in session.record["candidates"] if word != session.password]
    rng = random.Random(int(session.record.get("seed", 0)) ^ 0x43414E4449444154)
    rng.shuffle(duds)
    clue_count = rng.randint(1, min(session.attempts_max - 1, len(duds)))
    for word in duds[:clue_count]:
        session.execute(f"guess:{word}")
    return session


def controller_rng(record: dict[str, Any]) -> random.Random:
    """Return the deterministic controller RNG used by rollout and reward."""

    return random.Random(int(record.get("seed", 0)) ^ 0x43414E4449444154)


def candidate_controller_action(
    session: TerminalHackingSession,
    candidates: list[str],
    rng: random.Random,
) -> str:
    """Choose the hidden environment action from a model-produced candidate set."""

    if session.done:
        raise ValueError("cannot choose an action for a completed terminal session")
    return f"guess:{rng.choice(candidates)}"


def probe_is_preferred(session: TerminalHackingSession) -> bool:
    """Return whether the expert policy should choose a probe on this turn."""

    state = session.public_state()
    return (
        session.attempts_remaining == 1 and len(session.consistent_candidates()) > 1 and bool(state["available_probes"])
    )


def expert_decision(
    session: TerminalHackingSession,
    rng: random.Random,
) -> tuple[dict[str, Any], str]:
    """Return the expert tool call and the hidden environment action it induces."""

    if probe_is_preferred(session):
        probe = rng.choice(session.public_state()["available_probes"])
        action = f"probe:{probe['id']}"
        return {"name": "access_terminal", "arguments": {"action": action}}, action
    candidates = session.consistent_candidates()
    return (
        {"name": "submit_candidates", "arguments": {"candidates": candidates}},
        candidate_controller_action(session, candidates, rng),
    )


def candidate_filter_request_messages(prompt: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one compact state-to-candidates training request."""

    return [
        {"role": "system", "content": FILTER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "user", "content": "Current state: " + compact_state(state)},
    ]


def make_prompt(record: dict[str, Any]) -> str:
    """Render the initial visible puzzle without revealing hidden metadata."""

    session = TerminalHackingSession(record)
    state = session.public_state()
    dump = "\n".join(
        f"{row['left_address']} {row['left']}  {row['right_address']} {row['right']}" for row in state["dump_rows"]
    )
    probes = ", ".join(f"{probe['id']}={probe['token']}" for probe in state["available_probes"])
    return (
        "Filter the possible passwords in this retro data terminal. Candidate words all have the same length. "
        "A rejected guess returns likeness: the count of letters matching the password in the same positions. "
        "For example, PAPER with likeness 0 means a possible password must differ from PAPER at all five positions. "
        "Bracket probes remove a dud or restore attempts but do not create likeness clues. "
        "Normally submit every and only active candidate whose same-position match count equals every observed clue; "
        "the controller will randomly guess one submitted word. If only one attempt remains, multiple candidates are "
        "still consistent, and a probe is available, choose one probe instead.\n\n"
        f"Attempts: {state['attempts_remaining']}/{state['attempts_max']}\n"
        f"Bracket probes: {probes}\n"
        "Memory dump:\n"
        f"{dump}"
    )


def make_filter_prompt(record: dict[str, Any]) -> str:
    """Render the immutable puzzle context for state-to-candidates training."""

    session = TerminalHackingSession(record)
    state = session.public_state()
    dump = "\n".join(
        f"{row['left_address']} {row['left']}  {row['right_address']} {row['right']}" for row in state["dump_rows"]
    )
    return (
        "Filter the possible passwords in this retro data terminal. Candidate words all have the same length. "
        "For every rejected guess, keep a word only when its same-position match count equals the observed "
        "likeness. Return every and only consistent active candidate in active_candidates order. "
        "The memory dump only identifies visible words; it does not encode extra clues.\n\n"
        "Memory dump:\n"
        f"{dump}"
    )


def replay_actions(record: dict[str, Any], actions: list[object]) -> TerminalHackingSession:
    """Replay a trajectory for deterministic reward calculation."""

    session = TerminalHackingSession(record)
    for action in actions:
        result = session.execute(action)
        if not result["valid"] or session.done:
            break
    return session


def compact_state(state: dict[str, Any]) -> str:
    """Serialize the changing state for compact multi-turn prompts."""

    visible = {key: value for key, value in state.items() if key != "dump_rows"}
    return json.dumps(visible, separators=(",", ":"))
