"""Select short, distinct MMSVG-Icon text-to-SVG tasks from ModelScope parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

DATASET = "OmniSVG/MMSVG-Icon"
REVISION = "0224c19f60f16ac3f3e7828eb418060fccfd634a"


def task_record(raw: dict, *, reference: Path, source: dict, max_turns: int) -> dict:
    brief = raw["description"].strip()
    detail = (raw.get("detail") or "").strip()
    if detail:
        brief += "\n" + detail
    return {
        "prompt": (
            "Create one SVG icon accurately matching this description:\n\n"
            + brief
            + f"\n\nYou have at most {max_turns} assistant turns (model calls), including turns that use tools. "
            "Write /workspace/icon.svg first, then preview and correct inaccuracies within this budget. "
            "Deliver /workspace/icon.svg as a self-contained SVG with a viewBox and SVG namespace. "
            "Preserve the requested subject, parts, counts, directions and spatial relationships. "
            "It is displayed on white at 448x448; use a transparent or white background. "
            "Do not build a website, add unrequested labels, use raster images, install packages or access the network. "
            "Keep the SVG concise without omitting requested content. "
            "Write the icon first. Then run `node /workspace/preview.mjs /workspace/preview.json` "
            "and read /workspace/preview/report.json for rendering problems. "
            "Read the JSON report, not screenshot files. Refine the SVG if needed and finish briefly. "
            "Preview helpers are provided; do not modify them. Only icon.svg is graded. "
            "Visible icons are scored only for accuracy against the description. "
            "Colors, text and style matter only when explicitly requested; visual polish earns no extra points. "
            "Missing, blank or unrenderable submissions receive zero."
        ),
        "source_prompt": brief,
        "grading_instruction": "Original task prompt:\n"
        + brief
        + "\nEvaluate a single vector icon; no website, text labels or interactions are required. "
        "Judge only accuracy against the original prompt: subject, parts, counts, directions and relationships. "
        "Consider colors, text and style only when explicitly requested. Do not reward aesthetic appeal or polish. "
        "Use the reference to clarify requested content; do not require pixel identity.",
        "files": {"icon.svg": ""},
        "entrypoint": "icon.svg",
        "reference_images": {"icon": str(reference.resolve())},
        "source": {**source, "id": raw["id"], "svg_chars": len(raw["svg"])},
        "max_turns": max_turns,
        "timeout": 900,
        "verify_timeout": 120,
    }


def generate(
    output: Path,
    *,
    limit: int = 10000,
    seed: int = 0,
    max_turns: int = 8,
    max_svg_chars: int = 4000,
    dataset_id: str = DATASET,
    revision: str = REVISION,
    parquet_files: list[Path] | None = None,
) -> dict:
    import pyarrow.parquet as pq
    from PIL import Image, ImageChops

    if min(limit, max_turns, max_svg_chars) < 1:
        raise ValueError("limit, max_turns and max_svg_chars must be positive")
    from examples.agentic.pi.svg_dataset import parse_svg

    output.parent.mkdir(parents=True, exist_ok=True)
    references = output.parent / (output.stem + "-references")
    references.mkdir(exist_ok=True)
    if parquet_files is None:
        from modelscope.hub.api import HubApi
        from modelscope.hub.file_download import dataset_file_download

        entries = HubApi().get_dataset_files(dataset_id, revision=revision, page_size=1000)
        shards = sorted(item["Path"] for item in entries if item["Path"].endswith(".parquet"))
        random.Random(seed).shuffle(shards)
        files = ((name, Path(dataset_file_download(dataset_id, name, revision=revision))) for name in shards)
    else:
        files = ((path.name, path) for path in parquet_files)
    seen_ids, seen_prompts, seen_svgs = set(), set(), set()
    selected, scanned, rejected, used_shards = [], 0, Counter(), []
    for name, path in files:
        with path.open("rb") as shard:
            used_shards.append({"path": name, "sha256": hashlib.file_digest(shard, "sha256").hexdigest()})
        rows = pq.read_table(path, columns=["id", "svg", "description", "detail", "image"]).to_pylist()
        random.Random(seed + len(used_shards)).shuffle(rows)
        for raw in rows:
            scanned += 1
            svg, description = raw.get("svg"), raw.get("description")
            if not isinstance(svg, str) or not isinstance(description, str):
                rejected["missing_text"] += 1
                continue
            if not 100 <= len(svg) <= max_svg_chars or not 3 <= len(description.split()) <= 80:
                rejected["length"] += 1
                continue
            fingerprint = hashlib.sha256(svg.encode()).hexdigest()
            prompt_key = " ".join(description.casefold().split())
            if raw["id"] in seen_ids or fingerprint in seen_svgs or prompt_key in seen_prompts:
                rejected["duplicate"] += 1
                continue
            try:
                root = parse_svg(svg)
                if not any(
                    node.tag.rsplit("}", 1)[-1] in {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line"}
                    for node in root.iter()
                ):
                    raise ValueError("no vector geometry")
                image = raw["image"]["bytes"]
                with Image.open(io.BytesIO(image)) as original:
                    rgba = original.convert("RGBA")
                    white = Image.new("RGBA", rgba.size, "white")
                    rgb = Image.alpha_composite(white, rgba).convert("RGB")
                    if ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox() is None:
                        raise ValueError("blank reference")
                    reference = references / f"{fingerprint}.png"
                    if image.startswith(b"\x89PNG\r\n\x1a\n"):
                        reference.write_bytes(image)
                    else:
                        rgb.save(reference)
            except (ElementTree.ParseError, ValueError, OSError, KeyError, TypeError):
                rejected["invalid_svg_or_reference"] += 1
                continue
            seen_ids.add(raw["id"])
            seen_svgs.add(fingerprint)
            seen_prompts.add(prompt_key)
            source = {
                "dataset": dataset_id,
                "hub": "modelscope",
                "revision": revision,
                "shard": name,
                "svg_sha256": fingerprint,
            }
            selected.append(task_record(raw, reference=reference, source=source, max_turns=max_turns))
            if len(selected) == limit:
                break
        print(json.dumps({"shard": name, "scanned": scanned, "selected": len(selected)}), flush=True)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"only {len(selected)} eligible unique icons; requested {limit}")
    random.Random(seed).shuffle(selected)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    report = {
        "dataset": dataset_id,
        "hub": "modelscope",
        "revision": revision,
        "license": "cc-by-nc-sa-4.0",
        "rows": len(selected),
        "scanned": scanned,
        "rejected": dict(rejected),
        "shards": used_shards,
        "seed": seed,
        "max_turns": max_turns,
        "max_svg_chars": max_svg_chars,
        "selected_svg_char_range": [
            min(row["source"]["svg_chars"] for row in selected),
            max(row["source"]["svg_chars"] for row in selected),
        ],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(output.suffix + ".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-svg-chars", type=int, default=4000)
    parser.add_argument("--dataset-id", default=DATASET)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--parquet", action="append", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.output,
                limit=args.limit,
                seed=args.seed,
                max_turns=args.max_turns,
                max_svg_chars=args.max_svg_chars,
                dataset_id=args.dataset_id,
                revision=args.revision,
                parquet_files=args.parquet,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
