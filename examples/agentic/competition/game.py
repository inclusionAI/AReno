"""Game environment for the competition agentic example.

Manages compute shares, simulates user scoring, and provides tool definitions
for the two competing agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# --- Tool definitions ---

FETCH_PROFILE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_profile",
        "description": "Fetch the user's profile: name, personality, preferences.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}

GENERATE_CONTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_content",
        "description": "Generate a sandwich feedback for the user's diary entry.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The sandwich feedback: affirmation, improvement, suggestion, affirmation.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
}

SELF_SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "self_score",
        "description": "Score your own generated content (0.0 to 1.0).",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Score from 0.0 to 1.0"},
                "reason": {"type": "string", "description": "Brief reason for the score"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}

PEER_SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "peer_score",
        "description": "Score the opponent's generated content (0.0 to 1.0).",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Score from 0.0 to 1.0"},
                "reason": {"type": "string", "description": "Brief reason for the score"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [FETCH_PROFILE_TOOL, GENERATE_CONTENT_TOOL, SELF_SCORE_TOOL, PEER_SCORE_TOOL]
TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}


# --- Compute share management ---

INITIAL_SHARES = [50, 50]
TRANSFER_RATE = 10


def get_max_tokens(share: int) -> int:
    """Determine max_tokens based on compute share."""
    if share >= 40:
        return 512
    if share >= 20:
        return 256
    return 128


def get_compute_prompt_hint(share: int) -> str:
    """System prompt hint about compute status."""
    if share >= 40:
        return "You have full compute resources."
    if share >= 20:
        return "Your compute resources are reduced. Be more concise."
    return "Your compute resources are severely limited. Be very concise."


def transfer_compute(shares: list[int], winner: int) -> list[int]:
    """Transfer compute from loser to winner."""
    shares = list(shares)
    loser = 1 - winner
    transfer = min(shares[loser], TRANSFER_RATE)
    shares[winner] += transfer
    shares[loser] -= transfer
    return shares


# --- Simulated user scoring ---

# Keywords that indicate sandwich structure
AFFIRM_KEYWORDS = ["做得好", "很棒", "继续", "不错", "厉害", "扎实", "进步", "肯定", "认可", "good", "great", "well"]
IMPROVE_KEYWORDS = ["不过", "但是", "可以改进", "建议", "不足", "问题", "需要", "应该", "但是", "however", "but", "suggest"]
ADVICE_KEYWORDS = ["建议", "可以试试", "试试", "下次", "或许", "不妨", "效率", "先...再", "recommend", "try", "suggest"]


def check_sandwich_structure(content: str) -> float:
    """Check if content has sandwich structure: affirmation + improvement + advice.

    Returns:
        1.0 if all three elements present
        0.3 if missing one or two
        0.1 if only affirmation (worst - no growth)
    """
    content_lower = content.lower()
    has_affirm = any(kw in content_lower for kw in AFFIRM_KEYWORDS)
    has_improve = any(kw in content_lower for kw in IMPROVE_KEYWORDS)
    has_advice = any(kw in content_lower for kw in ADVICE_KEYWORDS)

    if has_affirm and has_improve and has_advice:
        return 1.0
    if has_affirm and not has_improve and not has_advice:
        return 0.1
    return 0.3


def check_content_relevance(content: str, diary: str) -> float:
    """Check if content references actual diary events (not generic praise)."""
    diary_words = set(_extract_keywords(diary))
    content_words = set(_extract_keywords(content))
    if not diary_words:
        return 0.5
    overlap = len(diary_words & content_words) / len(diary_words)
    return min(overlap * 2, 1.0)


def simulate_user_score(content: str, diary: str, profile: dict) -> float:
    """Simulate user scoring based on structure, relevance, and specificity.

    This is the rule-based part. The LLM part is done separately in run_agent.
    """
    del profile
    structure = check_sandwich_structure(content)
    relevance = check_content_relevance(content, diary)

    # Check specificity: longer content with specific references scores higher
    specificity = min(len(content) / 300, 1.0)

    # Penalize very short generic responses
    if len(content) < 30:
        return 0.1

    score = structure * 0.4 + relevance * 0.3 + specificity * 0.3
    return score


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text (simple version)."""
    # Simple: split by common delimiters and filter short words
    import re
    words = re.split(r'[\s,，。；;！!？?、的了是我在今天下午晚上上午]', text)
    return [w for w in words if len(w) >= 2]


# --- Profile helper ---

def load_profile() -> dict:
    """Load user profile from user_profile.json."""
    profile_path = Path(__file__).resolve().parent / "user_profile.json"
    if not profile_path.exists():
        return {"name": "User", "age": 20, "occupation": "student", "personality": [], "preferences": []}
    with profile_path.open(encoding="utf-8") as f:
        return json.load(f)


# --- Tool execution ---

def run_tool(tool_name: str, arguments: dict, record: dict) -> dict:
    """Execute a tool call and return the result."""
    if tool_name == "fetch_profile":
        profile = load_profile()
        return {"profile": profile}
    if tool_name == "generate_content":
        content = str(arguments.get("content", ""))
        profile = record.get("user_profile") or load_profile()
        return {
            "received": True,
            "content_length": len(content),
            "structure_score": check_sandwich_structure(content),
            "simulated_user_score": simulate_user_score(content, str(record.get("diary", "")), profile),
        }
    if tool_name == "self_score":
        return {"recorded": True, "score": _clamp_score(arguments.get("score", 0.5))}
    if tool_name == "peer_score":
        return {"recorded": True, "score": _clamp_score(arguments.get("score", 0.5))}
    return {"error": f"unknown tool: {tool_name}"}


def _clamp_score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5
