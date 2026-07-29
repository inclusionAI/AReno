#!/usr/bin/env python3
"""Scaffold a runnable, dependency-light AReno agentic project.

Generates a deterministic project under ``examples/agentic/<name>/`` containing
an environment state machine, dataset generator/loader, JSON tool definitions,
reward function, OpenAI-compatible agent entrypoint, and a no-model smoke
episode runner. The output uses only AReno's existing public contracts and
introduces no external database or mandatory sandbox.

Usage:
    python .agents/skills/areno-build-agentic-workflow/scripts/generate_agentic_project.py \
        --name my-gridworld \
        --out examples/agentic/my-gridworld \
        [--force] [--seed 2026]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DEFAULT_SEED = 2026


def _module_name(name: str) -> str:
    """Python module-safe name (hyphens -> underscores)."""
    return name.replace("-", "_")


def _header(name: str, seed: int, desc: str = "") -> str:
    """Build a single module docstring (keeps `from __future__` legal)."""
    body = f"Generated agentic project: {name} (seed={seed})."
    if desc:
        body += f"\n\n{desc}"
    body += (
        "\n\nDependency-light, self-contained, no external database or sandbox.\n"
        "Re-run the generator with a different --seed to vary fixtures deterministically."
    )
    return f'"""\n{body}\n"""\n'


# ---------------------------------------------------------------------------
# Templates for the generated project files.
# Each is a function of (name, seed) so output is fully deterministic.
# ---------------------------------------------------------------------------


def _game_py(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''{_header(name, seed, "Deterministic environment state machine with clear reset/step/error/reward placeholders.")}

from __future__ import annotations

import random
from typing import Any


class Env:
    """A tiny deterministic gridwalk used as a placeholder task.

    The agent starts at position 0 on a line of {mod.upper()}_SIZE cells and
    must reach the last cell. Each ``step`` moves left/right by one cell.
    Rewards are +1 for reaching the goal, -0.1 otherwise, and a clear error
    is raised on an illegal action.
    """

    SIZE = 5
    GOAL_REWARD = 1.0
    STEP_PENALTY = 0.1

    def __init__(self, seed: int = {seed}) -> None:
        self._rng = random.Random(seed)
        self.position: int = 0
        self._steps: int = 0
        self._done: bool = False

    # --- reset ---
    def reset(self) -> dict[str, Any]:
        """Reset the environment to its initial state and return an observation."""
        self.position = 0
        self._steps = 0
        self._done = False
        return self._observe()

    # --- step ---
    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply one action and return (observation, reward, done, info).

        action: -1 to move left, +1 to move right.
        """
        # --- error handling ---
        if not isinstance(action, int) or action not in (-1, 1):
            raise ValueError(f"illegal action {{action!r}}; expected -1 or +1")
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        self._steps += 1
        self.position = max(0, min(self.SIZE - 1, self.position + action))
        done = self.position == self.SIZE - 1
        # --- reward ---
        reward = self.GOAL_REWARD if done else -self.STEP_PENALTY
        self._done = done
        return self._observe(), reward, done, self._info()

    def legal_actions(self) -> list[int]:
        return [-1, 1]

    def render(self) -> str:
        cells = ["." for _ in range(self.SIZE)]
        cells[self.position] = "A"
        cells[-1] = "G"
        return "[" + " ".join(cells) + "]"

    def _observe(self) -> dict[str, Any]:
        return {{"position": self.position, "size": self.SIZE}}

    def _info(self) -> dict[str, Any]:
        return {{"steps": self._steps, "position": self.position, "done": self._done}}


def make_env(seed: int = {seed}) -> Env:
    """Factory used by dataset_loader / run_agent."""
    return Env(seed=seed)
'''


def _dataset_generator_py(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''{_header(name, seed, "Generate reproducible JSONL task records.")}

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def generate_records(count: int = 8, seed: int = {seed}) -> list[dict]:
    """Produce ``count`` deterministic task records.

    Each record is a starting position for the gridwalk placeholder.
    """
    rng = random.Random(seed)
    size = 5
    records = []
    for i in range(count):
        start = rng.randrange(0, size - 1)
        records.append({{"id": f"{mod}-{{i:05d}}", "start": start, "size": size}})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JSONL task records.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default={seed})
    args = parser.parse_args()
    records = generate_records(count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\\n")
    print(json.dumps({{"ok": True, "path": str(args.output), "count": len(records)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _dataset_loader_py(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''{_header(name, seed, f"Dataset loader for the {name} agentic example. Processor/tokenizer independent.")}

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: Any) -> list[dict]:
    """Load JSONL records and convert them to Areno prompt records."""
    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "tasks.jsonl"
    if not path.exists():
        import dataset_generator  # local fallback
        return dataset_generator.generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    size = int(raw.get("size", game.Env.SIZE))
    start = int(raw.get("start", 0))
    return {{
        "id": raw.get("id", f"{mod}-{{index:05d}}"),
        "prompt": (
            f"Reach the goal cell of a {{size}}-cell line. You start at cell {{start}}. "
            f"Reply with a single action: -1 (left) or +1 (right)."
        ),
        "start": start,
        "size": size,
        "legal_actions": [-1, 1],
    }}
'''


def _tool_defs_py(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''{_header(name, seed, "Strict, bounded JSON tool schemas for the agent.")}

from __future__ import annotations

ACT_TOOL = {{
    "type": "function",
    "function": {{
        "name": "act",
        "description": "Move the agent one cell left (-1) or right (+1) on the line.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "action": {{
                    "type": "integer",
                    "enum": [-1, 1],
                    "description": "Direction to move: -1 = left, +1 = right.",
                }}
            }},
            "required": ["action"],
            "additionalProperties": False,
        }},
    }},
}}


def tools() -> list[dict]:
    return [ACT_TOOL]


def tool_choice() -> dict:
    return {{"type": "function", "function": {{"name": "act"}}}}
'''


def _run_agent_py(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''{_header(name, seed, f"Agent entrypoint for {name} tool-call rollouts.")}

from __future__ import annotations

import asyncio
import logging

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

import tool_defs  # local

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    f"You are an agent on a 1D line. Always call the `act` tool exactly once "
    f"with action -1 (left) or +1 (right) to move toward the goal."
)


async def run_agent(ctx, batch):
    """Run one tool-call model request for each task record."""
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            f"The {name} agentic example requires `openai` and `httpx`. "
            f"Install with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    tools = tool_defs.tools()
    choice = tool_defs.tool_choice()
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        messages = [
            {{"role": "system", "content": SYSTEM_PROMPT}},
            {{"role": "user", "content": item.prompt}},
        ]
        response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=tools,
            tool_choice=choice,
            stream=False,
        )
        return AgentTrajectoryTurn(item=item, messages=messages, response=response, tools=tools, tool_choice=choice)

    try:
        return AgentTrajectory(turns=list(await asyncio.gather(*(run_one(item) for item in items))))
    finally:
        await client.close()
'''


def _reward_py(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''{_header(name, seed, f"Reward function for the {name} tool-call example.")}

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by replaying its `act` tool call against the env."""
    source = record.source_record
    env = game.Env()
    env.position = int(source.get("start", 0))
    action = _tool_action(record)
    if action is None:
        return -1.0  # malformed / missing tool call
    try:
        _, reward, done, _ = env.step(action)
    except ValueError:
        return -1.0  # illegal action
    return reward


def _tool_action(record: Any) -> int | None:
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "act":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            try:
                return int(arguments.get("action"))
            except (TypeError, ValueError):
                return None
    return None
'''


def _run_episode_py(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''{_header(name, seed, "Deterministic no-model smoke: run ONE fixed episode. No LLM, no network, no GPU.")}

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

# Fixed, deterministic action sequence: always reach the goal in {mod.upper()}_SIZE-1 steps.
FIXED_ACTION_SEQ = [1] * (game.Env.SIZE - 1)


def main() -> int:
    env = game.Env()
    obs = env.reset()  # --- reset ---
    print(f"reset obs={{obs}} render={{env.render()}}")
    total = 0.0
    for i, action in enumerate(FIXED_ACTION_SEQ):
        obs, reward, done, info = env.step(action)  # --- step ---
        total += reward                              # --- reward ---
        print(f"step i={{i}} action={{action}} reward={{reward}} done={{done}} info={{info}} render={{env.render()}}")
        if done:
            break
    print(f"episode_total_reward={{total}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _readme_md(name: str, seed: int) -> str:
    mod = _module_name(name)
    return f'''# Agentic {name} Example (generated)

A dependency-light, self-contained AReno agentic project scaffolded by
`.agents/skills/areno-build-agentic-workflow/scripts/generate_agentic_project.py`.

No external database, hosted service, or mandatory sandbox is required.

## Files

- `game.py` — environment state machine with clear `reset` / `step` / error / reward.
- `dataset_generator.py` — reproducible JSONL task records (`--output --count --seed`).
- `dataset_loader.py` — `load_training_dataset(...) -> list[dict]` (processor-independent).
- `tool_defs.py` — strict, bounded JSON tool schema for the `act` tool.
- `run_agent.py` — `async def run_agent(ctx, batch) -> AgentTrajectory` (OpenAI-compatible).
- `reward.py` — `reward_fn(record) -> float`, replays the tool call against the env.
- `run_episode.py` — no-model smoke that runs one fixed episode immediately.

## Minimal runnable example

```bash
# 1. Run the no-model smoke (no LLM / no network / no GPU)
python run_episode.py
```

Expected observable output:

```
reset obs={{'position': 0, 'size': 5}} render=[A . . . G]
step i=0 action=1 reward=-0.1 done=False ...
...
episode_total_reward=0.6
```

## Train with AReno

```bash
python dataset_generator.py --output /tmp/{mod}.jsonl --count 16 --seed {seed}

areno train \\
  --ckpt Qwen/Qwen3-0.6B \\
  --dataset-path /tmp/{mod}.jsonl \\
  --dataset-loader-fn examples/agentic/{name}/dataset_loader.py \\
  --reward-fn-path examples/agentic/{name}/reward.py \\
  --agent-fn examples/agentic/{name}/run_agent.py \\
  --algo gspo \\
  --tp-size 1 --world-size 1
```

## Limitations

- The environment is a placeholder gridwalk; replace `game.py` with your real task.
- `run_agent.py` requires `pip install openai` and a reachable OpenAI-compatible endpoint.
- `run_episode.py` needs no model and is the CI smoke entrypoint.
'''


# ---------------------------------------------------------------------------
# Scaffold writer
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "game.py": _game_py,
    "dataset_generator.py": _dataset_generator_py,
    "dataset_loader.py": _dataset_loader_py,
    "tool_defs.py": _tool_defs_py,
    "run_agent.py": _run_agent_py,
    "reward.py": _reward_py,
    "run_episode.py": _run_episode_py,
    "README.md": _readme_md,
}


def write_scaffold(out: Path, name: str, seed: int) -> list[Path]:
    """Write all template files into ``out`` and return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, fn in _TEMPLATES.items():
        path = out / filename
        path.write_text(fn(name, seed), encoding="utf-8")
        written.append(path)
    return written


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold a runnable, dependency-light AReno agentic project.",
    )
    parser.add_argument("--name", required=True, help="Project name (lowercase hyphen-case).")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: examples/agentic/<name>).")
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed (default: %(default)s).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not NAME_RE.fullmatch(args.name):
        print(f"error: --name must match {NAME_RE.pattern}", file=sys.stderr)
        return 1

    out = args.out or Path("examples/agentic") / args.name

    if out.exists() and any(out.iterdir()) and not args.force:
        print(
            f"error: {out} is non-empty; pass --force to overwrite (existing edits will be lost)",
            file=sys.stderr,
        )
        return 1

    files = write_scaffold(out, args.name, args.seed)

    # Human-readable progress.
    for f in files:
        print(f"created {f}")

    # Structured summary (single JSON line on stdout for tooling).
    summary = {
        "ok": True,
        "name": args.name,
        "seed": args.seed,
        "path": str(out),
        "files": [str(f) for f in files],
        "episode_cmd": f"python {out / 'run_episode.py'}",
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())