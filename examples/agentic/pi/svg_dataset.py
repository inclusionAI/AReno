"""SVG icon tasks using the shared workspace, Chromium and VLM graders."""

from __future__ import annotations

import io
import json
import shlex
from dataclasses import replace
from html import escape
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from areno.agent.harness_base import write_workspace_files
from areno.api.harness import list_datasets, register_dataset
from examples.agentic.pi.browser_capture import BrowserCapture
from examples.agentic.pi.frontend_dataset import FrontendDataset
from examples.agentic.pi.visual_reward import RubricDimension

VIEWPORTS = {"icon": {"width": 448, "height": 448}}
PREVIEW = (
    '<!doctype html><html><head><meta charset="utf-8">'
    "<style>html,body{margin:0;width:100%;height:100%;background:white}"
    "img{display:block;width:100%;height:100%;object-fit:contain}</style>"
    '</head><body><img src="icon.svg"></body></html>'
)
DIMENSIONS = {
    "semantic_match": RubricDimension(
        "Requested subject, parts and relationships are recognizable and accurate.", 0.30
    ),
    "composition": RubricDimension("Balance, proportions, visual hierarchy, centering and negative space.", 0.20),
    "geometry": RubricDimension("Clean contours, consistent strokes, smooth curves and deliberate joins.", 0.20),
    "style": RubricDimension("Coherent requested style, palette, contrast and consistency with the reference.", 0.15),
    "aesthetics": RubricDimension(
        "Polish, visual clarity and legibility as a small icon, without unnecessary detail.", 0.15
    ),
}


def parse_svg(text: str):
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("SVG must not contain XML document types or entities")
    root = ElementTree.fromstring(text)
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("submission must be an SVG document with the SVG namespace")
    return root


class SVGIconCapture(BrowserCapture):
    """Render only the delivered SVG, never an agent-controlled HTML wrapper."""

    async def capture(self, attempt, *, entrypoint, viewports, **kwargs):
        from PIL import Image, ImageChops

        handle = attempt.environment
        assert handle is not None
        candidate = str(PurePosixPath(handle.workspace) / entrypoint)
        exists = await handle.environment.exec(shlex.join(["test", "-f", candidate]), cwd="/", timeout_sec=10)
        if exists.return_code == 1:
            return {"capture_error": "missing delivered SVG"}, {}
        if exists.return_code != 0:
            raise RuntimeError("could not inspect the delivered SVG")
        files = attempt.root / "svg-control"
        files.mkdir()
        await handle.environment.download_file(candidate, files / "icon.svg")
        try:
            parse_svg((files / "icon.svg").read_text())
        except (ValueError, ElementTree.ParseError):
            return {"capture_error": "invalid SVG document"}, {}
        write_workspace_files(files, {"index.html": PREVIEW})
        workspace = str(PurePosixPath(handle.control_dir) / "svg-submission")
        await handle.environment.upload_dir(files, workspace)
        isolated = replace(attempt, environment=replace(handle, workspace=workspace))
        report, images = await super().capture(isolated, entrypoint="index.html", viewports=viewports, serve_dir=".")
        for name, view in report.get("views", {}).items():
            image = Image.open(io.BytesIO(images[f"Candidate {name}"])).convert("RGB")
            # An empty/transparent SVG still creates an <img> element; inspect its pixels too.
            visible = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
            view["visible_content"] = not view["broken_images"] and visible is not None
        return report, images


class SVGIconDataset(FrontendDataset):
    def __init__(self, **kwargs):
        kwargs.setdefault("renderer", SVGIconCapture())
        kwargs.setdefault("dimensions", DIMENSIONS)
        kwargs.setdefault("viewports", VIEWPORTS)
        super().__init__(**kwargs)

    def normalize(self, record):
        record = super().normalize(record)
        record.setdefault("entrypoint", "icon.svg")
        if record.get("serve_dir", ".") != ".":
            raise ValueError("SVG tasks deliver their entrypoint in the workspace root")
        return record

    def workspace_files(self, record):
        return {
            **super().workspace_files(record),
            "preview.html": PREVIEW.replace("icon.svg", escape(record["entrypoint"], quote=True)),
            "preview.mjs": Path(__file__).with_name("browser_capture.mjs").read_text(),
            "preview.json": json.dumps(
                {
                    "workspace": "/workspace",
                    "output": "/workspace/preview",
                    "entrypoint": "preview.html",
                    "viewports": self.viewports,
                }
            ),
        }


if "svg_icon" not in list_datasets():
    register_dataset("svg_icon", SVGIconDataset)
