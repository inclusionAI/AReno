"""Prepare short static visual-design tasks from ModelScope MiMo Webdev.

Other datasets can supply the same record contract or their own loader; no
dataset ID, field names, rubric or browser policy is registered in core code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import deque
from pathlib import Path

DATASET = "FineEnvs/MiMo-V2.6-RL-harbor-webdev"
REVISION = "d76448127f2a9e448b6bb6cd7102796d165e24ab"
METADATA_FILE = "data/tasks.jsonl"
BRIEF_MARKER = "Build a website for the following request:"
FRAMEWORKS = re.compile(
    r"\b(react|next\.?js|nuxt|vue|svelte\w*|astro|gsap|three\.?js|bootstrap|tailwind|angular|"
    r"element\s*(?:ui|plus)|mui|flutter|streamlit)\b",
    re.I,
)
COMPLEX_TASKS = re.compile(
    r"\b(attached|reference image|screenshot|video|youtube|webgl|websocket|3d|dashboard|calculator|"
    r"editor|game|multi-page|multipage|checkout|drag|drop|chatbot|infinite)\b|complex layouts|"
    r"multiple pages|several pages|booking system|drag.and.drop|видео|فيديو|视频|拖拽|拖放",
    re.I,
)


def select_tasks(rows: list[dict], *, seed: int, min_words: int = 5, max_words: int = 180) -> list[dict]:
    """Mix page types while choosing briefs supported by this static runtime."""
    candidates = []
    seen = set()
    for row in rows:
        brief = row["instruction"].split(BRIEF_MARKER, 1)[-1].strip()
        if not min_words <= len(brief.split()) <= max_words:
            continue
        if FRAMEWORKS.search(brief) or COMPLEX_TASKS.search(brief):
            continue
        fingerprint = " ".join(brief.casefold().split())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append(row)
    candidates.sort(key=lambda row: row["task_id"])
    random.Random(seed).shuffle(candidates)
    families: dict[str, deque] = {}
    for row in candidates:
        brief = row["instruction"].split(BRIEF_MARKER, 1)[-1].lower()
        family = next(
            (
                name
                for name in ["dashboard", "documentation", "portfolio", "store", "restaurant", "event"]
                if name in brief
            ),
            "landing",
        )
        families.setdefault(family, deque()).append(row)
    active, mixed = deque(families.values()), []
    while active:
        group = active.popleft()
        mixed.append(group.popleft())
        if group:
            active.append(group)
    return mixed


def convert_row(raw: dict, *, source: dict, max_turns: int = 20) -> dict:
    brief = raw["instruction"].split(BRIEF_MARKER, 1)[-1].strip()
    preview = {
        "workspace": "/workspace/dist",
        "output": "/workspace/preview",
        "entrypoint": "index.html",
        "viewports": {
            "desktop": {"width": 1280, "height": 960},
        },
    }
    return {
        "prompt": (
            "Build the following website and deliver /workspace/dist/index.html.\n\n"
            + brief
            + "\n\nExecution contract for this environment:\n"
            "- Keep the implementation focused on the requested page. Prioritize typography, color harmony, "
            "spacing, visual hierarchy and desktop composition. Do not invent unrelated features. "
            "Only the 1280 by 960 desktop viewport is evaluated; mobile layouts are outside this evaluation.\n"
            "- Work in /workspace. Only dist/ is served by a static HTTP server; no build step or backend is run. "
            "Use plain HTML, CSS and JavaScript. Every stylesheet, script and image must be inside dist/ or embedded. "
            "Do not reference /src/ files, CDNs, remote fonts or external image URLs; network access is blocked. "
            "Use CSS or local/embedded SVG for illustrations.\n"
            "- First write a small working index.html with the requested content. "
            "Then refine it with separate tool calls for HTML, CSS and JavaScript files. "
            "Keep individual writes modest so tool calls finish completely.\n"
            "- Node and Chromium are preinstalled. /workspace/preview.mjs and /workspace/preview.json are "
            "provided Node-only test helpers: do not replace them or include them in the webpage. "
            "After writing the website, run `node /workspace/preview.mjs /workspace/preview.json`, "
            "then read /workspace/preview/report.json to check JavaScript errors, assets and desktop overflow. "
            "Read the JSON report rather than screenshot files. Fix problems and validate again before finishing.\n"
            "- Missing, blank or unrenderable pages receive zero reward. Visible pages receive continuous scores "
            "for requirements, layout, style, viewport fit and usability. Broken assets, runtime errors, layout "
            "defects and failed interactions reduce the relevant scores rather than forcing the whole reward to zero."
        ),
        "grading_instruction": brief,
        "files": {
            "dist/index.html": "",
            "preview.mjs": Path(__file__).with_name("browser_capture.mjs").read_text(),
            "preview.json": json.dumps(preview),
        },
        "serve_dir": "dist",
        "entrypoint": "index.html",
        "source": {
            **source,
            "task_id": raw["task_id"],
            "source_id": raw.get("source_id"),
            "original_step_limit": raw.get("step_limit"),
        },
        "max_turns": max_turns,
        "timeout": 1800,
        "verify_timeout": 180,
    }


def generate(
    metadata: Path,
    output: Path,
    *,
    dataset_id: str = DATASET,
    revision: str = REVISION,
    limit: int = 200,
    seed: int = 0,
    max_turns: int = 20,
) -> dict:
    if limit < 1 or max_turns < 1:
        raise ValueError("limit and max_turns must be positive")
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    candidates = select_tasks(rows, seed=seed)
    if len(candidates) < limit:
        raise ValueError("not enough compatible frontend briefs")
    source = {
        "dataset": dataset_id,
        "revision": revision,
        "hub": "modelscope",
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    # Choose the shortest original briefs, then mix their page families for training.
    shortest = sorted(
        candidates,
        key=lambda row: (len(row["instruction"].split(BRIEF_MARKER, 1)[-1].split()), row["task_id"]),
    )[:limit]
    selected = [convert_row(row, source=source, max_turns=max_turns) for row in select_tasks(shortest, seed=seed)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    report = {
        **source,
        "source_rows": len(rows),
        "compatible_rows": len(candidates),
        "selected_task_ids": [row["source"]["task_id"] for row in selected],
        "seed": seed,
        "max_turns": max_turns,
        "selection": "short static visual-design briefs without prescribed frameworks or complex interactions",
        "selected_word_range": [
            min(len(row["grading_instruction"].split()) for row in selected),
            max(len(row["grading_instruction"].split()) for row in selected),
        ],
    }
    output.with_suffix(output.suffix + ".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--dataset-id", default=DATASET)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=20)
    args = parser.parse_args()
    metadata = args.metadata
    if metadata is None:
        from modelscope.hub.file_download import dataset_file_download

        metadata = Path(dataset_file_download(args.dataset_id, METADATA_FILE, revision=args.revision))
    print(
        json.dumps(
            generate(
                metadata,
                args.output,
                dataset_id=args.dataset_id,
                revision=args.revision,
                limit=args.limit,
                seed=args.seed,
                max_turns=args.max_turns,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
