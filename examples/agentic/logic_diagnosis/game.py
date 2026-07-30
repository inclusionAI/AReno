"""Deterministic logic-circuit diagnosis rules and tool schemas."""

from __future__ import annotations

import random
from typing import Any

# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------
MIN_INPUTS = 2
MAX_INPUTS = 6
MIN_GATES = 3
MAX_GATES = 15
BRUTE_FORCE_GATE_LIMIT = 8
MAX_PROBES = 5  # per-episode inspect_node limit; keep total trajectory within context window

# Gate type weights (AND : OR : NOT)
GATE_TYPE_WEIGHTS = [("and", 4), ("or", 4), ("not", 2)]

# ---------------------------------------------------------------------------
# OpenAI function-calling tool schemas
# ---------------------------------------------------------------------------
SET_INPUT_VECTOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_input_vector",
        "description": (
            "Set all primary input values (list of booleans, one per input in order). "
            "Free action — use this first to observe the faulty circuit's output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "description": "Boolean vector, one value per primary input in order.",
                }
            },
            "required": ["inputs"],
            "additionalProperties": False,
        },
    },
}

INSPECT_NODE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_node",
        "description": (
            "Probe the actual output value of one internal (AND/OR/NOT) gate node. "
            "Costs 1 probe. You must call set_input_vector before probing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "Numeric id of the gate node to probe (not input or output nodes).",
                }
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_DIAGNOSIS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_diagnosis",
        "description": "Submit your final diagnosis. Ends the episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "integer",
                    "description": "Id of the faulty gate node.",
                },
                "fault_type": {
                    "type": "string",
                    "enum": ["stuck_at_0", "stuck_at_1"],
                    "description": "Fault type: stuck_at_0 means the gate always outputs False, stuck_at_1 means always True.",
                },
            },
            "required": ["node_id", "fault_type"],
            "additionalProperties": False,
        },
    },
}

ALL_TOOLS: list[dict[str, Any]] = [SET_INPUT_VECTOR_TOOL, INSPECT_NODE_TOOL, SUBMIT_DIAGNOSIS_TOOL]


# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------
def _pick_gate_type(rng: random.Random) -> str:
    population, weights = zip(*GATE_TYPE_WEIGHTS)
    return rng.choices(population, weights=weights, k=1)[0]


def generate_circuit(
    n_inputs: int | None = None,
    n_gates: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate a random acyclic logic circuit.

    Parameters
    ----------
    n_inputs : int | None
        Number of primary input nodes. Clamped to [MIN_INPUTS, MAX_INPUTS].
    n_gates : int | None
        Number of internal gate nodes. Clamped to [MIN_GATES, MAX_GATES].
    seed : int
        Reproducibility seed.

    Returns
    -------
    list[dict]
        Node list sorted by ``id`` ascending. Edges always point from smaller
        to larger ``id``, so topological order is the natural index order.
    """
    n_in = max(MIN_INPUTS, min(MAX_INPUTS, int(n_inputs if n_inputs is not None else 4)))
    n_g = max(MIN_GATES, min(MAX_GATES, int(n_gates if n_gates is not None else 8)))
    rng = random.Random(seed)

    nodes: list[dict[str, Any]] = []

    # 1. Primary inputs (layer 0)
    for i in range(n_in):
        nodes.append({"id": i, "type": "input", "inputs": []})

    # 2. Internal gates
    for g_idx in range(n_g):
        gate_id = n_in + g_idx
        gate_type = _pick_gate_type(rng)
        arity = 1 if gate_type == "not" else 2

        # Candidates: all nodes with smaller id (guarantees acyclicity)
        candidates = list(range(gate_id))
        if len(candidates) < arity:
            # Edge case: first gate with arity 2 but only 1 input — force arity 1
            arity = 1
            gate_type = "not"
        inputs = sorted(rng.sample(candidates, k=arity))
        nodes.append({"id": gate_id, "type": gate_type, "inputs": inputs})

    # 3. Output node — single input from the deepest gate (or a random one)
    output_id = n_in + n_g
    gate_ids = [n["id"] for n in nodes if n["type"] in ("and", "or", "not")]
    if gate_ids:
        id_map = {n["id"]: n for n in nodes}
        max_depth = max(_node_depth_by_id(gid, id_map) for gid in gate_ids)
        deepest = [gid for gid in gate_ids if _node_depth_by_id(gid, id_map) == max_depth]
        output_inputs = [rng.choice(deepest)]
    else:
        output_inputs = [rng.choice(list(range(n_in)))] if n_in > 0 else [0]
    nodes.append({"id": output_id, "type": "output", "inputs": output_inputs})

    # 4. Prune dead nodes (backward BFS from output)
    nodes = _prune_unreachable(nodes)

    return nodes


def _prune_unreachable(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove gates that do not affect the output node.

    Input nodes are always kept (even if currently unused — they represent
    available inputs the agent can set). Gate and output nodes are kept only
    if reachable from the output via backward traversal.
    """
    if not nodes:
        return nodes

    id_to_idx = {n["id"]: idx for idx, n in enumerate(nodes)}
    output_id = nodes[-1]["id"]

    # Backward BFS from output
    reachable: set[int] = set()
    stack = [output_id]
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        if nid not in id_to_idx:
            continue
        for inp in nodes[id_to_idx[nid]].get("inputs", []):
            if inp not in reachable:
                stack.append(inp)

    # Keep input nodes unconditionally + reachable gates + output
    kept = []
    old_to_new: dict[int, int] = {}
    for node in nodes:
        if node["type"] == "input" or node["id"] in reachable:
            new_id = len(kept)
            old_to_new[node["id"]] = new_id
            kept.append(dict(node))

    # Remap inputs
    for node in kept:
        node["id"] = old_to_new[node["id"]]
        node["inputs"] = [old_to_new[inp] for inp in node["inputs"] if inp in old_to_new]

    return kept


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------
def inject_fault(nodes: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Pick a random internal gate and assign a stuck-at fault.

    Returns ``{"node": <id>, "stuck_value": 0 | 1}``.
    """
    gate_nodes = [n for n in nodes if n["type"] in ("and", "or", "not")]
    if not gate_nodes:
        raise ValueError("circuit has no internal gates to fault")
    rng = random.Random(seed)
    target = rng.choice(gate_nodes)
    return {"node": target["id"], "stuck_value": rng.choice([0, 1])}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    nodes: list[dict[str, Any]],
    input_vector: list[bool] | None = None,
    fault: dict[str, Any] | None = None,
) -> dict[int, bool]:
    """Topological pass: compute every node's value, optionally with a fault.

    Parameters
    ----------
    nodes : list[dict]
        Circuit node list (sorted by ``id``).
    input_vector : list[bool] | None
        One bool per input node. Missing entries default to ``False``.
    fault : dict | None
        ``{"node": <id>, "stuck_value": 0|1}``.

    Returns
    -------
    dict[int, bool]
        ``node_id → value`` for every node.
    """
    inp = list(input_vector or [])
    values: dict[int, bool] = {}

    for node in nodes:
        nid = node["id"]
        ntype = node["type"]

        if ntype == "input":
            idx = nid  # inputs are numbered 0..n-1
            values[nid] = bool(inp[idx]) if idx < len(inp) else False

        elif ntype == "and":
            ins = [values[i] for i in node["inputs"]]
            values[nid] = all(ins) if ins else False

        elif ntype == "or":
            ins = [values[i] for i in node["inputs"]]
            values[nid] = any(ins) if ins else False

        elif ntype == "not":
            ins = [values[i] for i in node["inputs"]]
            values[nid] = not ins[0] if ins else False

        elif ntype == "output":
            ins = [values[i] for i in node["inputs"]]
            values[nid] = ins[0] if ins else False

        # Fault override
        if fault is not None and nid == fault["node"]:
            values[nid] = bool(fault["stuck_value"])

    return values


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_diagnosis(
    nodes: list[dict[str, Any]],
    fault: dict[str, Any],
    suspected_node: int,
    suspected_fault_type: str,
) -> bool:
    """Check if the agent's diagnosis matches the actual fault."""
    del nodes  # unused in verification
    return (
        suspected_node == fault["node"]
        and (
            (suspected_fault_type == "stuck_at_0" and fault["stuck_value"] == 0)
            or (suspected_fault_type == "stuck_at_1" and fault["stuck_value"] == 1)
        )
    )


def brute_force_verify(nodes: list[dict[str, Any]], fault: dict[str, Any]) -> bool:
    """Verify that the fault has *exactly one* distinguishing I/O signature.

    For circuits with ≤ ``BRUTE_FORCE_GATE_LIMIT`` gates, exhaustively checks
    every possible ``(gate × {stuck-0, stuck-1})`` candidate. Returns ``True``
    iff exactly one candidate matches the observed output across *all* input
    vectors. Larger circuits are **not** verified (returns ``True``).
    """
    gate_nodes = [n for n in nodes if n["type"] in ("and", "or", "not")]
    if len(gate_nodes) > BRUTE_FORCE_GATE_LIMIT:
        return True  # skip — too expensive

    n_in = sum(1 for n in nodes if n["type"] == "input")
    output_id = next(n["id"] for n in nodes if n["type"] == "output")

    # Pre-compute expected outputs for all input vectors under the true fault
    expected: dict[tuple[bool, ...], bool] = {}
    for bits in _all_input_vectors(n_in):
        values = evaluate(nodes, list(bits), fault)
        expected[bits] = values[output_id]

    # Enumerate all candidate faults
    match_count = 0
    for gate in gate_nodes:
        for sv in (0, 1):
            candidate = {"node": gate["id"], "stuck_value": sv}
            if candidate == fault:
                continue  # don't compare against itself
            if _matches_all(nodes, candidate, expected):
                return False  # ambiguous — another fault produces identical I/O
            match_count += 1  # just tracking, not used

    return True


def _all_input_vectors(n_inputs: int):
    """Yield every ``n_inputs``-length bool tuple."""
    for v in range(1 << n_inputs):
        yield tuple(bool(v >> i & 1) for i in range(n_inputs))


def _matches_all(
    nodes: list[dict[str, Any]],
    candidate: dict[str, Any],
    expected: dict[tuple[bool, ...], bool],
) -> bool:
    output_id = next(n["id"] for n in nodes if n["type"] == "output")
    for bits, exp_val in expected.items():
        values = evaluate(nodes, list(bits), candidate)
        if values[output_id] != exp_val:
            return False
    return True


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _node_depth(node: dict[str, Any], nodes: list[dict[str, Any]]) -> int:
    """Longest path distance from any input to this node."""
    id_map = {n["id"]: n for n in nodes}
    return _node_depth_by_id(node["id"], id_map)


def _node_depth_by_id(nid: int, id_map: dict[int, dict[str, Any]]) -> int:
    node = id_map.get(nid)
    if node is None or node["type"] == "input":
        return 0
    inputs = node.get("inputs", [])
    if not inputs:
        return 1
    return 1 + max(_node_depth_by_id(inp, id_map) for inp in inputs)


def _node_label(node: dict[str, Any]) -> str:
    ntype = node["type"]
    nid = node["id"]
    if ntype == "input":
        return f"IN{nid}"
    if ntype == "output":
        return f"OUT"
    return f"{ntype.upper()}{nid}"


def node_list_text(nodes: list[dict[str, Any]]) -> str:
    """Render the circuit topology as a compact layered listing.

    Format: one layer per line, each node as ``LABEL=TYPE(input1,input2)``.
    Input nodes are listed bare on the first line; output node on the last.
    """
    # Group nodes by depth
    layers: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        d = _node_depth(node, nodes)
        layers.setdefault(d, []).append(node)

    lines = []
    for depth in sorted(layers.keys()):
        parts = []
        for node in layers[depth]:
            label = _node_label(node)
            ntype = node["type"]
            if ntype == "input":
                parts.append(label)
            elif ntype == "output":
                ins = ",".join(_node_label(n) for n in nodes if n["id"] in node.get("inputs", []))
                parts.append(f"OUT({ins})")
            else:
                ins = ",".join(_node_label(n) for n in nodes if n["id"] in node.get("inputs", []))
                parts.append(f"{label}={ntype.upper()}({ins})")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


def make_prompt(record: dict[str, Any]) -> str:
    """Build a prompt describing the task without revealing the fault."""
    nodes = record.get("nodes", [])
    n_in = sum(1 for n in nodes if n["type"] == "input")
    n_g = sum(1 for n in nodes if n["type"] in ("and", "or", "not"))
    max_probes = int(record.get("max_probes", MAX_PROBES))

    topology = node_list_text(nodes)

    return (
        "Diagnose the faulty gate in this logic circuit. "
        f"Exactly one internal gate is stuck-at-0 or stuck-at-1. "
        "Inputs (IN*) and output (OUT) are never faulty.\n\n"
        "Circuit (nodes grouped by depth, left-to-right within each line):\n"
        f"{topology}\n\n"
        f"Tools: set_input_vector (free), inspect_node (1 probe, max {max_probes}), "
        "submit_diagnosis(node_id, fault_type). Fewer probes is better."
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_episode(
    *,
    correct_diagnosis: bool,
    probes_used: int,
    max_probes: int = MAX_PROBES,
    submitted: bool,
) -> float:
    """Compute scalar reward for one episode.

    Returns
    -------
    float
        - No submission → -1.0
        - Wrong diagnosis → 0.0
        - Correct diagnosis → 0.5 + 0.5 × (1 − probes_used / max_probes)
    """
    del submitted  # handled by caller; -1.0 returned for no submit in reward.py
    if not correct_diagnosis:
        return 0.0
    denom = max(max_probes, 1)
    efficiency = max(0.0, 1.0 - probes_used / denom)
    return 0.5 + 0.5 * efficiency