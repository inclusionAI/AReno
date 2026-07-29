"""Side-effect-free fake tools and task logic for the multi-tool agentic example.

Tools: contacts, notes, calculator, unit_convert, parcel_lookup.
Tasks require two or more correctly ordered tool calls, with all state held
in module-level constants and in-memory dictionaries.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# In-memory data stores
# ---------------------------------------------------------------------------

# In-memory contact list. Each entry is a dict with keys:
#   name, email, phone, city — all string values.
# Serves as the data source for the lookup_contact() tool.
CONTACTS: list[dict[str, str]] = [
    {"name": "Alice Chen", "email": "alice@example.com", "phone": "13800001111", "city": "Shanghai"},
    {"name": "Bob Smith", "email": "bob@example.com", "phone": "13900002222", "city": "Beijing"},
    {"name": "Carol Lee", "email": "carol@example.com", "phone": "13700003333", "city": "Shenzhen"},
    {"name": "David Wong", "email": "david@example.com", "phone": "13600004444", "city": "Shanghai"},
]

# In-memory note store. Key = note topic (e.g. "meeting"), value = note text.
# Serves as the data source for the read_note() tool.
NOTES: dict[str, str] = {
    "meeting": "Team sync at 2pm on Tuesday in Room 3.",
    "budget": "Q3 travel budget is 8000 CNY.",
    "shipping": "Standard shipping takes 3-5 business days.",
    "project": "Deadline for the API migration is next Friday.",
}

# In-memory parcel tracking store. Key = tracking id (e.g. "P001"),
# value = dict with keys: status, address, eta.
# Serves as the data source for the lookup_parcel() tool.
PARCELS: dict[str, dict[str, str]] = {
    "P001": {"status": "delivered", "address": "Shanghai", "eta": "2026-07-20"},
    "P002": {"status": "in_transit", "address": "Beijing", "eta": "2026-07-30"},
    "P003": {"status": "pending", "address": "Shenzhen", "eta": "2026-08-02"},
    "P004": {"status": "delivered", "address": "Shanghai", "eta": "2026-07-18"},
}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def lookup_contact(name: str) -> dict[str, str] | None:
    """Return contact info for the first match by partial name (case-insensitive).

    Args:
        name: A (possibly partial) name to search for. Matching is case-insensitive
              and substring-based — e.g. "alice" matches "Alice Chen".

    Returns:
        A copy of the matched contact dict (name, email, phone, city), or None
        if no contact matches.
    """

    needle = name.strip().lower()
    for contact in CONTACTS:
        if needle in contact["name"].lower():
            return dict(contact)
    return None


def read_note(note_key: str) -> dict[str, str] | None:
    """Return a note by exact key match.

    Args:
        note_key: The exact note topic key (e.g. "meeting", "budget").
                  Whitespace is stripped before lookup.

    Returns:
        A dict {"key": ..., "content": ...} if found, or None.
    """

    key = note_key.strip()
    if key in NOTES:
        return {"key": key, "content": NOTES[key]}
    return None


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate a safe arithmetic expression containing numbers and + - * / ( ).

    Uses a custom tokenizer + recursive-descent parser instead of Python's eval(),
    so arbitrary code execution is impossible.

    Args:
        expression: A string arithmetic expression, e.g. "3 * (4 + 5)".

    Returns:
        On success: {"expression": ..., "result": float}.
        On error: {"error": error_message}.
    """

    expr = expression.strip()
    if not expr:
        return {"error": "empty expression"}
    try:
        value = _safe_eval(expr)
    except _CalcError as exc:
        return {"error": str(exc)}
    return {"expression": expr, "result": value}


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert between supported length and weight units.

    Supported length units: m, cm, mm, km.
    Supported weight units: g, kg, mg.
    Conversion across categories (e.g. cm to g) is not supported.

    Args:
        value: The numeric value to convert.
        from_unit: The source unit (e.g. "cm").
        to_unit: The target unit (e.g. "m").

    Returns:
        On success: {"value": ..., "from_unit": ..., "to_unit": ..., "result": converted_value}.
        On error: {"error": ...} if the unit pair is unsupported.
    """

    value = float(value)
    from_u = from_unit.strip().lower()
    to_u = to_unit.strip().lower()
    factor = _conversion_factor(from_u, to_u)
    if factor is None:
        return {"error": f"unsupported conversion: {from_unit} -> {to_unit}"}
    return {"value": value, "from_unit": from_u, "to_unit": to_u, "result": value * factor}


def lookup_parcel(tracking_id: str) -> dict[str, str] | None:
    """Return parcel tracking info by exact tracking id.

    Args:
        tracking_id: The exact tracking id (e.g. "P001"). Whitespace is stripped.

    Returns:
        A dict with tracking_id, status, address, eta if found, or None.
    """

    tid = tracking_id.strip()
    if tid in PARCELS:
        return {"tracking_id": tid, **PARCELS[tid]}
    return None


def search_notes(keyword: str) -> list[dict[str, str]]:
    """Search notes by keyword (case-insensitive substring match).

    Unlike read_note which requires an exact key, search_notes finds all notes
    whose content contains the keyword. This adds complexity: the agent must
    first search to discover the right key, then read the full note.

    Args:
        keyword: A keyword to search for in note content (case-insensitive).

    Returns:
        A list of {"key": ..., "snippet": ...} dicts for matching notes.
        Empty list if no matches.
    """

    kw = keyword.strip().lower()
    if not kw:
        return []
    results: list[dict[str, str]] = []
    for key, content in NOTES.items():
        if kw in content.lower():
            snippet = content if len(content) <= 80 else content[:77] + "..."
            results.append({"key": key, "snippet": snippet})
    return results


def list_contacts_by_city(city: str) -> list[dict[str, str]]:
    """List all contacts in a given city.

    Unlike lookup_contact which finds one person by name, this returns all
    contacts in a city. This adds complexity: the agent must determine the city
    from a parcel or note, then list contacts to find the right person.

    Args:
        city: City name to filter by (case-insensitive).

    Returns:
        A list of contact dicts (name, email, phone, city) in that city.
        Empty list if no contacts found.
    """

    c = city.strip().lower()
    if not c:
        return []
    return [dict(contact) for contact in CONTACTS if contact["city"].lower() == c]


# ---------------------------------------------------------------------------
# Task generation and scoring
# ---------------------------------------------------------------------------

# Task definitions for the multi-tool agentic game.
# Each task specifies:
#   - id: unique identifier (used to route scoring logic)
#   - description: the natural-language task shown to the agent
#   - required_tools: the expected tool-call sequence (by function name)
#   - expected_* fields: expected values used by the scoring helpers to verify
#     that the agent called the right tools with the right arguments.
TASKS: list[dict[str, Any]] = [
    {
        "id": "contact-meeting",
        "description": "Find Alice's phone number, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    },
    {
        "id": "budget-shipping",
        "description": "Read the budget note, then read the shipping note.",
        "required_tools": ["read_note", "read_note"],
        "expected_note_keys": ["budget", "shipping"],
    },
    {
        "id": "parcel-city",
        "description": "Look up parcel P002, then find a contact in the same city.",
        "required_tools": ["lookup_parcel", "lookup_contact"],
        "expected_parcel": "P002",
        "expected_contact_city": "Beijing",
    },
    {
        "id": "calc-shipping",
        "description": "Calculate 3 * 15, then read the shipping note.",
        "required_tools": ["calculate", "read_note"],
        "expected_expression": "3 * 15",
        "expected_note_key": "shipping",
    },
    {
        "id": "convert-parcel",
        "description": "Convert 100 cm to m, then look up parcel P003.",
        "required_tools": ["unit_convert", "lookup_parcel"],
        "expected_value": 100,
        "expected_from_unit": "cm",
        "expected_to_unit": "m",
        "expected_parcel": "P003",
    },
    {
        "id": "search-meeting-contact",
        "description": "Search notes for 'Team', read the meeting note, then list contacts in Shanghai.",
        "required_tools": ["search_notes", "read_note", "list_contacts_by_city"],
        "expected_search_keyword": "Team",
        "expected_note_key": "meeting",
        "expected_city": "Shanghai",
    },
    {
        "id": "parcel-calc-note",
        "description": "Look up parcel P002, calculate the ETA days from today (2026-07-29 to 2026-07-30 = 1 day) as 7 - 6, then read the shipping note.",
        "required_tools": ["lookup_parcel", "calculate", "read_note"],
        "expected_parcel": "P002",
        "expected_expression": "7 - 6",
        "expected_note_key": "shipping",
    },
    {
        "id": "convert-search-contact-parcel",
        "description": "Convert 1000 mm to m, search notes for 'shipping', then list contacts in Shanghai and look up parcel P001.",
        "required_tools": ["unit_convert", "search_notes", "list_contacts_by_city", "lookup_parcel"],
        "expected_value": 1000,
        "expected_from_unit": "mm",
        "expected_to_unit": "m",
        "expected_search_keyword": "shipping",
        "expected_city": "Shanghai",
        "expected_parcel": "P001",
    },
]


def make_prompt(record: dict[str, Any]) -> str:
    """Build the user request string for one multi-tool task.

    Args:
        record: A task dict from TASKS (or a copy generated by generate_records).
                Must contain a "description" key.

    Returns:
        A formatted prompt string instructing the agent to use tools in order.
    """

    return (
        f"Task: {record['description']} "
        "Use the available tools in the correct order to complete the task. "
        "Do not answer in plain text."
    )


def generate_records(count: int, *, seed: int = 42) -> list[dict]:
    """Generate deterministic multi-tool records by sampling from TASKS.

    Uses a seeded RNG so the same seed always produces the same task sequence,
    ensuring reproducible training/evaluation data.

    Args:
        count: Number of task records to generate.
        seed: Random seed for reproducibility (keyword-only).

    Returns:
        A list of `count` task dicts, each with a unique "id" suffix "-<idx>".
    """

    import random

    rng = random.Random(seed)
    records: list[dict] = []
    for idx in range(count):
        task = dict(rng.choice(TASKS))
        task["id"] = f"{task['id']}-{idx}"
        records.append(task)
    return records


def score_task(record: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a tool-call trajectory against the task requirements.

    Evaluates four dimensions:
      1. tool_selection — did the agent use all required tools?
      2. arguments       — did each tool call pass the correct arguments?
      3. order           — were the tools called in the required sequence?
      4. final_answer    — did the last relevant call produce a valid result?

    The overall reward is:
      - 1.0  if all four dimensions are perfect,
      - -1.0 if both tool_selection and arguments are 0 (agent failed completely),
      - otherwise a partial score based on the average of the four dimensions.

    Args:
        record: The task definition dict (from TASKS or generate_records).
        tool_calls: A list of tool-call dicts, each with keys "name" and "arguments".
                    "arguments" may be a dict or a JSON string.

    Returns:
        A dict with keys: overall, tool_selection, arguments, order,
        final_answer, failures (list of dimension names that scored < 1.0).
    """

    names = [call.get("name") for call in tool_calls]
    required = list(record.get("required_tools", []))

    tool_selection = _score_tool_selection(names, required)
    arguments = _score_arguments(record, tool_calls)
    order = _score_order(names, required)
    final_answer = _score_final_answer(record, tool_calls)

    failures: list[str] = []
    if tool_selection < 1.0:
        failures.append("tool_selection")
    if arguments < 1.0:
        failures.append("arguments")
    if order < 1.0:
        failures.append("order")
    if final_answer < 1.0:
        failures.append("final_answer")

    overall = 0.0
    if not failures:
        overall = 1.0
    elif tool_selection == 0.0 and arguments == 0.0:
        overall = -1.0
    else:
        overall = 0.5 * (tool_selection + arguments + order + final_answer) / 4.0 - 0.25

    return {
        "overall": overall,
        "tool_selection": tool_selection,
        "arguments": arguments,
        "order": order,
        "final_answer": final_answer,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------


def _score_tool_selection(names: list[str | None], required: list[str]) -> float:
    """1.0 if all required tool names appear, partial credit otherwise.

    Args:
        names: The sequence of tool names actually called by the agent
               (entries may be None if a call had no "name" key).
        required: The list of tool names the task expects to see.

    Returns:
        1.0 if every required tool was used at least once;
        otherwise the fraction of required tools that appeared (0.0 if none).
    """

    if not required:
        return 1.0
    present = {n for n in names if n is not None}
    needed = set()
    for r in required:
        needed.add(r)
    if needed.issubset(present):
        return 1.0
    ratio = len(needed & present) / len(needed)
    return ratio if ratio > 0 else 0.0


def _score_arguments(record: dict[str, Any], tool_calls: list[dict[str, Any]]) -> float:
    """Check that tool call arguments match expected values.

    For each tool call, verifies the arguments against the expected_* fields
    in the task record. Different tools are checked differently:
      - lookup_contact: name should match expected_contact or resolve to
        expected_contact_city.
      - read_note: note_key should match expected_note_key or one of
        expected_note_keys.
      - calculate: expression should match expected_expression exactly.
      - unit_convert: value, from_unit, to_unit should all match expected_*.
      - lookup_parcel: tracking_id should match expected_parcel.

    Args:
        record: The task definition dict containing expected_* fields.
        tool_calls: A list of tool-call dicts with "name" and "arguments" keys.
                    "arguments" may be a dict or a JSON string (auto-parsed).

    Returns:
        Fraction of checked calls that passed (0.0 if no calls were checked).
    """

    import json as _json

    checks: list[bool] = []
    for call in tool_calls:
        name = call.get("name")
        raw_args = call.get("arguments")
        # Parse arguments: accept dict directly, or decode JSON string
        if isinstance(raw_args, str):
            try:
                args = _json.loads(raw_args)
            except _json.JSONDecodeError:
                checks.append(False)
                continue
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            checks.append(False)
            continue

        if name == "lookup_contact":
            if record.get("expected_contact") and record["expected_contact"].split()[0].lower() in str(args.get("name", "")).lower():
                checks.append(True)
            elif record.get("expected_contact_city"):
                # For city-based contact tasks, any name that resolves to the city is acceptable
                city = record["expected_contact_city"]
                contact = lookup_contact(str(args.get("name", "")))
                checks.append(contact is not None and contact["city"] == city)
            else:
                checks.append(False)
        elif name == "read_note":
            key = str(args.get("note_key", "")).strip()
            expected_keys = record.get("expected_note_keys")
            if expected_keys is None:
                expected_keys = [record.get("expected_note_key", "")]
            if expected_keys is None:
                expected_keys = []
            if key in expected_keys:
                checks.append(True)
            else:
                checks.append(False)
        elif name == "calculate":
            if record.get("expected_expression") and str(args.get("expression", "")).strip() == record["expected_expression"]:
                checks.append(True)
            else:
                checks.append(False)
        elif name == "unit_convert":
            try:
                value_ok = record.get("expected_value") is not None and float(args.get("value", -1)) == float(record["expected_value"])
            except (ValueError, TypeError):
                value_ok = False
            if (
                value_ok
                and str(args.get("from_unit", "")).lower() == record.get("expected_from_unit", "")
                and str(args.get("to_unit", "")).lower() == record.get("expected_to_unit", "")
            ):
                checks.append(True)
            else:
                checks.append(False)
        elif name == "lookup_parcel":
            if record.get("expected_parcel") and str(args.get("tracking_id", "")).strip() == record["expected_parcel"]:
                checks.append(True)
            else:
                checks.append(False)
        elif name == "search_notes":
            kw = str(args.get("keyword", "")).strip().lower()
            expected_kw = str(record.get("expected_search_keyword", "")).strip().lower()
            checks.append(bool(kw and kw == expected_kw))
        elif name == "list_contacts_by_city":
            city = str(args.get("city", "")).strip().lower()
            expected_city = str(record.get("expected_city", record.get("expected_contact_city", ""))).strip().lower()
            checks.append(bool(city and city == expected_city))

    if not checks:
        return 0.0
    return sum(1 for c in checks if c) / len(checks)


def _score_order(names: list[str | None], required: list[str]) -> float:
    """1.0 if the required tools appear in the correct relative order.

    Checks whether the required tool sequence appears as a subsequence of the
    actual call sequence (ignoring None entries and extra calls in between).

    Args:
        names: The actual sequence of tool names called (may contain None).
        required: The expected ordered sequence of tool names.

    Returns:
        Fraction of required tools matched in order (1.0 if all matched).
    """

    if not required:
        return 1.0
    filtered = [n for n in names if n is not None]
    idx = 0
    matched = 0
    for req in required:
        while idx < len(filtered):
            if filtered[idx] == req:
                matched += 1
                idx += 1
                break
            idx += 1
    return matched / len(required) if required else 1.0


def _score_final_answer(record: dict[str, Any], tool_calls: list[dict[str, Any]]) -> float:
    """Check whether the last relevant tool call produces the expected result.

    Looks at the last tool call in the trajectory and verifies it is the
    correct final tool for the task type and that it returns a valid result.
    Task type is inferred from the "id" prefix (before the first "-").

    Args:
        record: The task definition dict; its "id" field determines which
                final tool is expected.
        tool_calls: A list of tool-call dicts with "name" and "arguments".

    Returns:
        1.0 if the last call is the expected final tool and it succeeds;
        0.0 otherwise.
    """

    if not tool_calls:
        return 0.0

    import json as _json

    last = tool_calls[-1]
    name = last.get("name")
    raw_args = last.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = _json.loads(raw_args)
        except _json.JSONDecodeError:
            return 0.0
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return 0.0

    # Infer task type from the id prefix (e.g. "contact-meeting-0" -> "contact")
    task_id = str(record.get("id", "")).split("-")[0]
    if task_id == "contact" and name == "read_note":
        key = str(args.get("note_key", "")).strip()
        note = read_note(key)
        return 1.0 if note is not None else 0.0
    if task_id == "budget" and name == "read_note":
        key = str(args.get("note_key", "")).strip()
        note = read_note(key)
        return 1.0 if note is not None and key == record.get("expected_note_keys", [""])[-1] else 0.0
    if task_id == "parcel" and name == "lookup_contact":
        contact = lookup_contact(str(args.get("name", "")))
        return 1.0 if contact is not None else 0.0
    if task_id == "calc" and name == "read_note":
        note = read_note(str(args.get("note_key", "")))
        return 1.0 if note is not None else 0.0
    if task_id == "convert" and name == "lookup_parcel":
        parcel = lookup_parcel(str(args.get("tracking_id", "")))
        return 1.0 if parcel is not None else 0.0
    if task_id == "search" and name == "list_contacts_by_city":
        contacts = list_contacts_by_city(str(args.get("city", "")))
        return 1.0 if contacts else 0.0
    if task_id == "parcel" and name == "read_note":
        # parcel-calc-note ends with read_note
        note = read_note(str(args.get("note_key", "")))
        return 1.0 if note is not None else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator
# ---------------------------------------------------------------------------


class _CalcError(Exception):
    """Custom exception for calculation errors in the safe arithmetic evaluator.

    Used by _tokenize, _safe_eval, and _Parser to signal malformed expressions,
    invalid characters, division by zero, etc. The calculate() tool catches this
    specifically so that only evaluator errors are returned as {"error": ...},
    while unexpected runtime bugs propagate normally.
    """
    pass


def _safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression without eval().

    Pipeline: tokenize -> parse -> verify all tokens consumed.
    Does NOT use Python's built-in eval(), so arbitrary code execution
    is impossible.

    Args:
        expr: A pre-stripped arithmetic expression string (e.g. "3 + 5 * 2").
              Supports numbers, + - * /, and parentheses.

    Returns:
        The computed float result.

    Raises:
        _CalcError: If the expression is malformed or contains trailing characters
                    after a valid sub-expression.
    """

    tokens = _tokenize(expr)
    parser = _Parser(tokens)
    value = parser.parse_expr()
    # After parsing, all tokens should be consumed; leftover tokens mean
    # the expression had extra content after a valid sub-expression (e.g. "3 + 5 6")
    if parser.pos != len(tokens):
        raise _CalcError("unexpected trailing characters")
    return float(value)


def _tokenize(expr: str) -> list[tuple[str, str | float]]:
    """Convert an arithmetic expression string into a list of tokens.

    Each token is a (type, value) pair:
      - ("op", operator_char)  for + - * / ( )
      - ("num", float_value)   for numeric literals

    Whitespace is skipped. Any character that is not whitespace, a digit,
    a decimal point, or a supported operator triggers _CalcError.

    Args:
        expr: The arithmetic expression string to tokenize.

    Returns:
        A list of (type, value) tuples.

    Raises:
        _CalcError: If an invalid number or unsupported character is encountered.
    """

    tokens: list[tuple[str, str | float]] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/()":
            tokens.append(("op", ch))
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            # Consume the full numeric literal (digits and decimal points)
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            num_str = expr[i:j]
            try:
                tokens.append(("num", float(num_str)))
            except ValueError:
                raise _CalcError(f"invalid number: {num_str}")
            i = j
            continue
        raise _CalcError(f"unexpected character: {ch}")
    return tokens


class _Parser:
    """Recursive-descent arithmetic parser: turns the token stream from _tokenize
    into a numeric value, respecting operator precedence.

    Precedence (low to high): parse_expr -> parse_add_sub(+,-) ->
    parse_mul_div(*,/) -> parse_atom(number/parenthesized sub-expr).
    Each layer only handles its own operators and delegates higher-precedence
    tokens to the next layer down, naturally yielding "multiplication before
    addition" and parenthesized sub-expressions first.
    """

    def __init__(self, tokens: list[tuple[str, str | float]]):
        # tokens: token list produced by _tokenize; pos: current read cursor
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str | float] | None:
        # Look at the current token without advancing; return None at end of input
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> tuple[str, str | float]:
        # Return the current token and advance the cursor; error if already at end
        tok = self.peek()
        if tok is None:
            raise _CalcError("unexpected end of expression")
        self.pos += 1
        return tok

    def parse_expr(self) -> float:
        # Entry point: delegate to the lowest-precedence add/sub layer
        return self.parse_add_sub()

    def parse_add_sub(self) -> float:
        # Handle + and - (lowest precedence): parse one mul/div term, then loop
        # over any following +/- operators
        left = self.parse_mul_div()
        while True:
            tok = self.peek()
            # Stop on any non +/- operator (or end of input), returning control upward
            if tok is None or tok[0] != "op" or tok[1] not in "+-":
                break
            self.consume()
            right = self.parse_mul_div()
            if tok[1] == "+":
                left += right
            else:
                left -= right
        return left

    def parse_mul_div(self) -> float:
        # Handle * and / (medium precedence): parse one atom, then loop over any
        # following * / operators
        left = self.parse_atom()
        while True:
            tok = self.peek()
            # Stop on any non * / operator so multiplication binds tighter than addition
            if tok is None or tok[0] != "op" or tok[1] not in "*/":
                break
            self.consume()
            right = self.parse_atom()
            if tok[1] == "*":
                left *= right
            else:
                # Division by zero is undefined; report it as a calculation error
                if right == 0:
                    raise _CalcError("division by zero")
                left /= right
        return left

    def parse_atom(self) -> float:
        # Highest precedence: a numeric literal, or a parenthesized sub-expression
        tok = self.consume()
        if tok[0] == "num":
            # Numeric token: return its float value directly
            return float(tok[1])
        if tok[0] == "op" and tok[1] == "(":
            # Opening paren: recursively parse the sub-expression, then require a
            # matching closing paren
            value = self.parse_expr()
            close = self.consume()
            if close[0] != "op" or close[1] != ")":
                raise _CalcError("missing closing parenthesis")
            return value
        # Neither a number nor an opening paren: the expression structure is invalid
        raise _CalcError(f"unexpected token: {tok}")


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

# Length units converted to meters (the common base unit for length).
_LENGTH_TO_M = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "km": 1000.0,
}

# Weight units converted to grams (the common base unit for weight).
_WEIGHT_TO_G = {
    "g": 1.0,
    "kg": 1000.0,
    "mg": 0.001,
}


def _conversion_factor(from_u: str, to_u: str) -> float | None:
    """Return the multiplier to convert from from_u to to_u, or None if unsupported.

    Both units must belong to the same category (both length or both weight).
    Cross-category conversion (e.g. "cm" to "g") returns None.

    The formula is: result = value * (base[from_u] / base[to_u])
    where base is the conversion factor to the common base unit (meters or grams).

    Args:
        from_u: The source unit string (e.g. "cm").
        to_u: The target unit string (e.g. "m").

    Returns:
        The multiplication factor, or None if the unit pair is unsupported.
    """

    if from_u in _LENGTH_TO_M and to_u in _LENGTH_TO_M:
        return _LENGTH_TO_M[from_u] / _LENGTH_TO_M[to_u]
    if from_u in _WEIGHT_TO_G and to_u in _WEIGHT_TO_G:
        return _WEIGHT_TO_G[from_u] / _WEIGHT_TO_G[to_u]
    return None