"""Normalize conversation roles and pair tool messages.

Converts common role aliases (human, bot, function, etc.) to AReno standard
roles (user, assistant, tool, system), normalizes supported tool-call shapes,
and verifies each tool call has exactly one matching tool response in the
expected ordering.  Never guesses when conversion is ambiguous.

The implementation reuses AReno's existing public contracts: the standard
role names already used by ``areno.api.openai_chat.normalize_messages`` and
the ``tool_calls`` / ``tool_call_id`` pairing convention from the agentic
rollout path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public role constants
# ---------------------------------------------------------------------------

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

STANDARD_ROLES = frozenset({ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL})

# ---------------------------------------------------------------------------
# Role alias mapping
# ---------------------------------------------------------------------------

ROLE_ALIASES: dict[str, str] = {
    # user aliases
    "human": ROLE_USER,
    "person": ROLE_USER,
    "speaker": ROLE_USER,
    "user": ROLE_USER,

    # assistant aliases
    "bot": ROLE_ASSISTANT,
    "gpt": ROLE_ASSISTANT,
    "model": ROLE_ASSISTANT,
    "chatbot": ROLE_ASSISTANT,
    "ai": ROLE_ASSISTANT,
    "assistant": ROLE_ASSISTANT,

    # tool aliases
    "function": ROLE_TOOL,
    "tool": ROLE_TOOL,
    "tool_result": ROLE_TOOL,
    "function_response": ROLE_TOOL,
    "tool_response": ROLE_TOOL,

    # system aliases
    "system": ROLE_SYSTEM,
    "instruction": ROLE_SYSTEM,
    "developer": ROLE_SYSTEM,
}

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ConversationValidationError(Exception):
    """Raised when a conversation cannot be normalized safely.

    ``sample_index`` and ``turn_index`` locate the problem precisely.
    The ``detail`` string avoids dumping full message content so that
    sensitive training data is not leaked in logs.

    设计意图：Issue 要求 "Failure identifies the affected stage and input
    without exposing full training samples or hiding the original error"。
    所以错误信息只报 sample #N, turn #N，不打印对话内容。
    """

    def __init__(
        self,
        detail: str,
        *,
        sample_index: int | None = None,
        turn_index: int | None = None,
        error_type: str = "validation_error",
    ) -> None:
        self.detail = detail
        self.sample_index = sample_index
        self.turn_index = turn_index
        self.error_type = error_type
        location = ""
        parts: list[str] = []
        if sample_index is not None:
            parts.append(f"sample #{sample_index}")
        if turn_index is not None:
            parts.append(f"turn #{turn_index}")
        if parts:
            location = " at " + ", ".join(parts)
        super().__init__(f"{error_type}{location}: {detail}")


class UnknownRoleError(ConversationValidationError):
    """An unrecognized role string that cannot be mapped safely."""

    def __init__(self, raw_role: str, **kwargs: Any) -> None:
        super().__init__(
            f"unknown role '{raw_role}'; cannot normalize automatically",
            error_type="unknown_role",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Per-conversation normalization
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NormalizeResult:
    """Outcome of normalizing one conversation.

    ``messages`` is the normalized conversation when validation passes,
    or ``None`` when errors are collected instead of raised.
    ``errors`` is a list of :class:`ConversationValidationError` describing
    every problem found (only populated when ``raise_on_error`` is False).

    设计意图：支持两种模式——raise_on_error=True 时第一个问题就抛异常
    （适合在数据加载阶段 fail-fast），raise_on_error=False 时收集所有
    问题（适合批量处理时一次性看到全部错误）。
    """

    messages: list[dict[str, Any]] | None
    errors: list[ConversationValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.messages is not None


def normalize_role(raw_role: str) -> str:
    """Map a role alias to the AReno standard role.

    Raises :class:`UnknownRoleError` for roles that are not in the alias
    table.  The caller is responsible for catching the error and attaching
    sample/turn indices.

    设计意图：不猜测、不默认映射。遇到表里没有的角色直接报错，
    因为错误的角色映射可能导致训练数据被静默污染，比报错更危险。
    """

    if not isinstance(raw_role, str):
        raise UnknownRoleError(str(raw_role))
    # 大小写不敏感：先 strip 再 lower，比如 "Human" 和 "human" 都能匹配
    key = raw_role.strip().lower()
    if key not in ROLE_ALIASES:
        raise UnknownRoleError(raw_role)
    return ROLE_ALIASES[key]


def _extract_tool_call_ids(message: dict[str, Any]) -> list[str]:
    """Return the list of tool-call ids declared on an assistant message."""

    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    ids: list[str] = []
    for call in calls:
        if isinstance(call, dict):
            call_id = call.get("id")
            if isinstance(call_id, str):
                ids.append(call_id)
    return ids


def _normalize_tool_call(call: Any) -> dict[str, Any]:
    """Normalize a single tool-call dict to the OpenAI shape.

    Accepts both ``{"function": {"name": ..., "arguments": ...}}`` and
    the flatter ``{"name": ..., "arguments": ...}`` variant.
    """

    if not isinstance(call, dict):
        return {"id": "", "type": "function", "function": {"name": str(call), "arguments": {}}}

    item: dict[str, Any] = dict(call)
    item.setdefault("type", "function")

    # Ensure an id exists (may be set later by the caller).
    item.setdefault("id", "")

    func = item.get("function")
    if isinstance(func, dict):
        func = dict(func)
        args = func.get("arguments")
        if isinstance(args, str):
            try:
                func["arguments"] = json.loads(args or "{}")
            except json.JSONDecodeError:
                func["arguments"] = {"_raw": args}
        elif args is None:
            func["arguments"] = {}
        item["function"] = func
    elif "name" in item:
        # Flatten {"name": ..., "arguments": ...} into the function sub-dict.
        args = item.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {"_raw": args}
        elif args is None:
            args = {}
        item["function"] = {"name": item["name"], "arguments": args}
        item.pop("name", None)
        item.pop("arguments", None)

    return item


def normalize_conversation(
    messages: list[dict[str, Any]],
    *,
    sample_index: int | None = None,
    raise_on_error: bool = True,
) -> NormalizeResult:
    """Normalize a single conversation's roles and tool-message pairing.

    Steps:
      1. Map every ``role`` to a standard AReno role.
      2. Normalize ``tool_calls`` shape on assistant messages.
      3. Verify each ``tool_call`` id has exactly one matching ``tool``
         response, in the expected position.
      4. Verify role alternation rules (no consecutive users, system only
         first, tool only after assistant-with-tool-calls, etc.).

    When *raise_on_error* is ``True`` the first problem raises
    :class:`ConversationValidationError`.  When ``False`` all problems are
    collected in :attr:`NormalizeResult.errors` and ``messages`` is ``None``.
    """

    errors: list[ConversationValidationError] = []

    def record(error: ConversationValidationError) -> None:
        if raise_on_error:
            raise error
        errors.append(error)

    if not isinstance(messages, list):
        record(ConversationValidationError(
            "messages must be a list",
            sample_index=sample_index,
            error_type="invalid_input",
        ))
        return NormalizeResult(messages=None, errors=errors)

    if not messages:
        # An empty conversation is valid (nothing to normalize).
        return NormalizeResult(messages=[], errors=errors)

    # -- Phase 1: 角色映射 + tool_calls 格式归一化 -------------------------
    # 遍历每条消息，把 role 别名映射为 AReno 标准角色，
    # 同时把 tool_calls 的 arguments 从 JSON 字符串解析为 dict，
    # 把 content=None 替换为空字符串（兼容本地 chat template）。

    normalized: list[dict[str, Any]] = []
    for turn_index, raw_msg in enumerate(messages):
        if not isinstance(raw_msg, dict):
            record(ConversationValidationError(
                f"message at turn {turn_index} is not a dict",
                sample_index=sample_index,
                turn_index=turn_index,
                error_type="invalid_message",
            ))
            if raise_on_error:
                return NormalizeResult(messages=None, errors=errors)
            continue

        msg = dict(raw_msg)
        raw_role = msg.get("role")
        try:
            msg["role"] = normalize_role(raw_role)
        except UnknownRoleError as exc:
            exc.sample_index = sample_index
            exc.turn_index = turn_index
            record(exc)
            if raise_on_error:
                return NormalizeResult(messages=None, errors=errors)
            # Keep the raw role so downstream checks can still report.
            msg["role"] = raw_role

        # Normalize tool_calls on assistant messages.
        if msg["role"] == ROLE_ASSISTANT and isinstance(msg.get("tool_calls"), list):
            msg["tool_calls"] = [
                _normalize_tool_call(tc) for tc in msg["tool_calls"]
            ]

        # OpenAI assistant tool-call messages commonly carry content=null;
        # some local chat templates require a string.
        if msg.get("content") is None:
            msg["content"] = ""

        normalized.append(msg)

    if errors:
        return NormalizeResult(messages=None, errors=errors)

    # -- Phase 2: tool-call / tool-response 配对验证 -----------------------
    # 维护一个 pending_tool_call_ids 列表，追踪还没有收到 response 的 tool_call id。
    # 遇到 assistant 带 tool_call → 把 id 加入 pending
    # 遇到 tool 消息 → 检查 id 是否在 pending 里，在则移除，不在则报"孤立 response"
    # 遇到 user 或 assistant(无 tool_call) → 如果 pending 非空，报"pending 未回答"
    # 对话结束 → 如果 pending 非空，报"末尾缺少 response"

    pending_tool_call_ids: list[str] = []
    """IDs from assistant tool_calls awaiting a matching tool response."""

    for turn_index, msg in enumerate(normalized):
        role = msg["role"]

        if role == ROLE_ASSISTANT:
            call_ids = _extract_tool_call_ids(msg)
            if call_ids:
                if pending_tool_call_ids:
                    # Previous tool calls were not fully answered.
                    record(ConversationValidationError(
                        f"assistant at turn {turn_index} issued new tool calls "
                        f"while previous calls {pending_tool_call_ids} have no response",
                        sample_index=sample_index,
                        turn_index=turn_index,
                        error_type="orphan_tool_call",
                    ))
                    if raise_on_error:
                        return NormalizeResult(messages=None, errors=errors)
                pending_tool_call_ids = list(call_ids)
            else:
                # Assistant without tool calls — pending must be empty.
                if pending_tool_call_ids:
                    record(ConversationValidationError(
                        f"assistant at turn {turn_index} has no tool calls "
                        f"but pending tool calls {pending_tool_call_ids} are unanswered",
                        sample_index=sample_index,
                        turn_index=turn_index,
                        error_type="missing_tool_response",
                    ))
                    if raise_on_error:
                        return NormalizeResult(messages=None, errors=errors)
                    pending_tool_call_ids = []

        elif role == ROLE_TOOL:
            call_id = msg.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                record(ConversationValidationError(
                    "tool message missing 'tool_call_id'",
                    sample_index=sample_index,
                    turn_index=turn_index,
                    error_type="missing_tool_call_id",
                ))
                if raise_on_error:
                    return NormalizeResult(messages=None, errors=errors)
                continue

            if call_id not in pending_tool_call_ids:
                record(ConversationValidationError(
                    f"tool response '{call_id}' has no matching tool call",
                    sample_index=sample_index,
                    turn_index=turn_index,
                    error_type="orphan_tool_response",
                ))
                if raise_on_error:
                    return NormalizeResult(messages=None, errors=errors)
                continue

            pending_tool_call_ids.remove(call_id)

        elif role == ROLE_USER:
            if pending_tool_call_ids:
                record(ConversationValidationError(
                    f"user message at turn {turn_index} interrupts "
                    f"pending tool calls {pending_tool_call_ids}",
                    sample_index=sample_index,
                    turn_index=turn_index,
                    error_type="interrupted_tool_call",
                ))
                if raise_on_error:
                    return NormalizeResult(messages=None, errors=errors)
                pending_tool_call_ids = []

    # After all turns: any pending tool calls are missing responses.
    if pending_tool_call_ids:
        record(ConversationValidationError(
            f"conversation ended with unanswered tool calls {pending_tool_call_ids}",
            sample_index=sample_index,
            error_type="missing_tool_response",
        ))
        if raise_on_error:
            return NormalizeResult(messages=None, errors=errors)

    if errors:
        return NormalizeResult(messages=None, errors=errors)

    # -- Phase 3: 角色交替规则验证 -----------------------------------------
    # 检查角色出现的顺序是否合法：
    # - system 只能在对话开头
    # - 不能连续两个 user
    # - 不能连续两个 assistant（如果中间没有 tool_call 的话）
    # - tool 消息必须跟在 assistant 或另一个 tool 后面（支持并行 response）

    _validate_role_sequence(normalized, sample_index=sample_index, record=record)

    _validate_role_sequence(normalized, sample_index=sample_index, record=record)

    if errors:
        return NormalizeResult(messages=None, errors=errors)

    return NormalizeResult(messages=normalized, errors=errors)


def _validate_role_sequence(
    messages: list[dict[str, Any]],
    *,
    sample_index: int | None,
    record,
) -> None:
    """Check that roles follow the legal alternation pattern."""

    if not messages:
        return

    # system may only appear as the first message.
    for i, msg in enumerate(messages):
        role = msg["role"]
        if role == ROLE_SYSTEM and i > 0:
            record(ConversationValidationError(
                f"system message at turn {i} must be the first message",
                sample_index=sample_index,
                turn_index=i,
                error_type="misplaced_system",
            ))
            # Don't return — collect all errors when not raising.

    prev_role: str | None = None
    prev_had_tool_calls = False

    for i, msg in enumerate(messages):
        role = msg["role"]

            # 并行 tool response：tool 后面可以跟 tool（同一条 assistant 发起的多个 tool call 的 response）
        if role == ROLE_TOOL and prev_role is not None and prev_role not in (ROLE_ASSISTANT, ROLE_TOOL):
            record(ConversationValidationError(
                f"tool message at turn {i} must follow an assistant or tool message "
                f"(previous role: {prev_role})",
                sample_index=sample_index,
                turn_index=i,
                error_type="invalid_tool_position",
            ))

        if (
            prev_role == ROLE_ASSISTANT
            and not prev_had_tool_calls
            and role == ROLE_ASSISTANT
        ):
            record(ConversationValidationError(
                f"consecutive assistant messages at turn {i} without tool calls",
                sample_index=sample_index,
                turn_index=i,
                error_type="consecutive_assistant",
            ))

        if prev_role == ROLE_USER and role == ROLE_USER:
            record(ConversationValidationError(
                f"consecutive user messages at turn {i}",
                sample_index=sample_index,
                turn_index=i,
                error_type="consecutive_user",
            ))

        prev_role = role
        prev_had_tool_calls = bool(_extract_tool_call_ids(msg)) if role == ROLE_ASSISTANT else False


# ---------------------------------------------------------------------------
# Batch normalization with structured reporting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchNormalizeReport:
    """Aggregated result of normalizing a batch of conversations.

    Use :meth:`to_human_string` for CLI pretty-print and :meth:`to_dict`
    for structured (JSON) output.

    设计意图：同时提供两种输出格式，满足 Issue 要求的
    "human-readable and structured output when the feature is exposed
    through the CLI"。to_human_string 给人看，to_json 给程序解析。
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    normalized: list[list[dict[str, Any]]] = field(default_factory=list)

    def to_human_string(self) -> str:
        lines: list[str] = [
            f"Total: {self.total}  Passed: {self.passed}  "
            f"Failed: {self.failed}  Skipped: {self.skipped}",
        ]
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for err in self.errors:
                loc_parts: list[str] = []
                if err.get("sample") is not None:
                    loc_parts.append(f"sample #{err['sample']}")
                if err.get("turn") is not None:
                    loc_parts.append(f"turn #{err['turn']}")
                loc = ", ".join(loc_parts) if loc_parts else "unknown"
                lines.append(f"  [{err.get('type', 'error')}] {loc}: {err.get('detail', '')}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def normalize_dataset(
    samples: list[list[dict[str, Any]]],
    *,
    raise_on_error: bool = False,
    skip_invalid: bool = True,
) -> BatchNormalizeReport:
    """Normalize a batch of conversations and return a structured report.

    Each *sample* is a list of message dicts.  Invalid samples are reported
    in the error list; when *skip_invalid* is True they are omitted from the
    normalized output.  When *skip_invalid* is False a
    :class:`ConversationValidationError` is raised for the first invalid
    sample (equivalent to ``raise_on_error=True`` per-sample).
    """

    report = BatchNormalizeReport(total=len(samples))

    for sample_index, messages in enumerate(samples):
        result = normalize_conversation(
            messages,
            sample_index=sample_index,
            raise_on_error=False,
        )
        if result.ok:
            report.passed += 1
            report.normalized.append(result.messages)  # type: ignore[arg-type]
        else:
            report.failed += 1
            for err in result.errors:
                report.errors.append({
                    "sample": err.sample_index,
                    "turn": err.turn_index,
                    "type": err.error_type,
                    "detail": err.detail,
                })
            if not skip_invalid:
                # Re-raise with full context.
                first = result.errors[0]
                raise ConversationValidationError(
                    first.detail,
                    sample_index=first.sample_index,
                    turn_index=first.turn_index,
                    error_type=first.error_type,
                )

    return report


def normalize_dataset_iter(
    samples: Iterator[list[dict[str, Any]]],
    *,
    raise_on_error: bool = False,
) -> Iterator[tuple[int, list[dict[str, Any]] | None, list[ConversationValidationError]]]:
    """Stream-normalize samples one at a time.

    Yields ``(sample_index, normalized_messages_or_None, errors)`` tuples
    so callers can process large datasets without holding everything in
    memory.
    """

    for sample_index, messages in enumerate(samples):
        result = normalize_conversation(
            messages,
            sample_index=sample_index,
            raise_on_error=raise_on_error,
        )
        yield sample_index, result.messages, result.errors