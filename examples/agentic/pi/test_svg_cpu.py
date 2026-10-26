"""SVG task boundaries: private references, isolated rendering and data selection."""

import asyncio
import base64
import io
import json
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

pytest.importorskip("harbor", reason="Harbor integration requires Python 3.12+ and Harbor 0.24")

from PIL import Image

from areno.api.agentic import AgentBatch
from areno.api.harness import HarnessAttempt, HarnessEnvironment
from examples.agentic.pi.browser_capture import BrowserCapture
from examples.agentic.pi.generate_dataset import generate, task_record
from examples.agentic.pi.svg_dataset import SVGIconCapture, SVGIconDataset, parse_svg
from examples.agentic.pi.visual_reward import VLMJudge

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><rect x="20" y="20" width="160" height="160"/></svg>'
)


def png(color):
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


def test_task_keeps_reference_and_solution_out_of_agent_workspace(tmp_path):
    raw = {
        "id": "square",
        "svg": SVG,
        "description": "A simple black square icon.",
        "detail": "Keep its corners sharp. " * 35,
    }
    record = task_record(raw, reference=tmp_path / "private.png", source={}, max_turns=8)
    dataset = SVGIconDataset()
    normalized = dataset.normalize(record)
    files = dataset.workspace_files(normalized)
    assert normalized["max_turns"] == 8
    assert files["icon.svg"] == ""
    assert SVG not in record["prompt"]
    assert str(tmp_path) not in "".join(files.values()) + record["prompt"]
    assert set(files) == {"icon.svg", "preview.html", "preview.mjs", "preview.json"}
    preview = json.loads(files["preview.json"])
    assert preview["workspace"] == "." and preview["output"] == "preview"
    assert "/workspace/" not in record["prompt"]
    assert record["reference_images"] == {"icon": str((tmp_path / "private.png").resolve())}
    source_prompt = raw["description"] + "\n" + raw["detail"].strip()
    assert record["source_prompt"] == source_prompt
    assert source_prompt in record["prompt"]
    item = next(AgentBatch([normalized], ["p"], [[0]], 1).iter_samples())
    assert source_prompt in dataset.visual_task(HarnessAttempt(item, tmp_path, 900, 120)).instruction


def test_custom_svg_task_can_supply_files_viewport_and_rubric(tmp_path):
    dataset = SVGIconDataset(dimensions={"custom": {"description": "A clear geometric symbol", "weight": 2}})
    record = dataset.normalize(
        {
            "prompt": "Draw a symbol and save it to custom/icon.svg.",
            "source_prompt": "A black square with sharp corners.",
            "entrypoint": "custom/icon.svg",
            "files": {"custom/icon.svg": SVG},
            "binary_files": {"asset.png": base64.b64encode(png("black")).decode()},
            "viewports": {"small": {"width": 128, "height": 128}},
            "custom_id": "user-task",
        }
    )
    files = dataset.workspace_files(record)
    assert files["custom/icon.svg"] == SVG and files["asset.png"] == png("black")
    item = next(AgentBatch([record], [record["prompt"]], [[0]], 1).iter_samples())
    task = dataset.visual_task(HarnessAttempt(item, tmp_path, 900, 120))
    assert task.instruction == record["source_prompt"]
    assert task.entrypoint == "custom/icon.svg" and task.viewports == record["viewports"]
    assert list(task.dimensions) == ["custom"] and task.dimensions["custom"].weight == 2
    assert record["custom_id"] == "user-task"


@pytest.mark.parametrize(
    "text", ["", "<html/>", '<svg xmlns="http://www.w3.org/2000/svg">', '<!DOCTYPE svg [<!ENTITY x "x">]><svg/>']
)
def test_rejects_non_svg_or_unsafe_xml(text):
    with pytest.raises((ValueError, ElementTree.ParseError)):
        parse_svg(text)


def test_vlm_request_contains_the_original_prompt(monkeypatch, tmp_path):
    requests = []

    def request(self, payload):
        requests.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "dimensions": {
                                    name: {"score": 50, "reason": "visible evidence"}
                                    for name in SVGIconDataset().dimensions
                                },
                                "checks": {},
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(VLMJudge, "_request", request)
    raw = {"id": "square", "svg": SVG, "description": "A centered black square icon.", "detail": "Four sharp corners."}
    record = task_record(raw, reference=tmp_path / "reference.png", source={}, max_turns=8)
    item = next(AgentBatch([record], ["p"], [[0]], 1).iter_samples())
    task = SVGIconDataset().visual_task(HarnessAttempt(item, tmp_path, 900, 120))
    asyncio.run(
        VLMJudge("https://judge.invalid/v1", "fixture", "fixture").score(
            instruction=task.instruction, images={"Candidate icon": png("black")}, dimensions=task.dimensions
        )
    )
    data = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert record["source_prompt"] in data["requirements"]
    assert "original task prompt" in data["rubric"]["semantic_match"]["description"]


@pytest.mark.parametrize("submission", [None, "malformed", SVG])
@pytest.mark.parametrize("visible", [False, True])
def test_grader_uses_only_svg_and_rejects_blank_pixels(tmp_path, monkeypatch, submission, visible):
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "index.html").write_text("forged beautiful page")
    if submission is not None:
        (workspace / "icon.svg").write_text(submission)

    class Environment:
        async def exec(self, command, **kwargs):
            if shlex.split(command)[:2] == ["mkdir", "-p"]:
                Path(shlex.split(command)[-1]).mkdir(parents=True, exist_ok=True)
                return SimpleNamespace(return_code=0)
            return SimpleNamespace(return_code=0 if Path(shlex.split(command)[-1]).is_file() else 1)

        async def download_file(self, source, target):
            shutil.copyfile(source, target)

        async def upload_dir(self, source, target):
            assert Path(target).is_dir(), "create the remote grading directory before uploading"
            shutil.copytree(source, target, dirs_exist_ok=True)

    called = []

    async def capture(self, attempt, **kwargs):
        root = Path(attempt.environment.workspace)
        called.append(root)
        assert root != workspace
        assert "forged" not in (root / "index.html").read_text()
        assert (root / "icon.svg").read_text() == SVG
        return {"views": {"icon": {"broken_images": 0}}, "interactions": []}, {
            "Candidate icon": png("black" if visible else "white")
        }

    monkeypatch.setattr(BrowserCapture, "capture", capture)
    item = next(AgentBatch([{}], ["p"], [[0]], 1).iter_samples())
    attempt = HarnessAttempt(
        item, tmp_path, 900, 120, HarnessEnvironment(Environment(), str(workspace), str(tmp_path / "control"))
    )
    report, _ = asyncio.run(
        SVGIconCapture().capture(attempt, entrypoint="icon.svg", viewports={"icon": {"width": 64, "height": 64}})
    )
    if submission != SVG:
        assert report["capture_error"] and not called
    else:
        assert report["views"]["icon"]["visible_content"] == visible


def test_generation_deduplicates_and_fails_if_insufficient_rows(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    rows = [
        {
            "id": "one",
            "svg": SVG,
            "description": "A simple black square icon.",
            "detail": "",
            "image": {"bytes": png("black")},
        },
        {
            "id": "copy",
            "svg": SVG,
            "description": "Another black square icon.",
            "detail": "",
            "image": {"bytes": png("black")},
        },
        {
            "id": "blank",
            "svg": SVG.replace('x="20"', 'x="21"'),
            "description": "A blank square icon.",
            "detail": "",
            "image": {"bytes": png("white")},
        },
    ]
    parquet = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    output = tmp_path / "tasks.jsonl"
    report = generate(output, limit=1, parquet_files=[parquet])
    assert report["rows"] == 1
    row = json.loads(output.read_text())
    assert row["source"]["hub"] == "modelscope"
    assert Path(row["reference_images"]["icon"]).is_file()
    assert row["files"] == {"icon.svg": ""}
    with pytest.raises(ValueError, match="requested 2"):
        generate(tmp_path / "insufficient.jsonl", limit=2, parquet_files=[parquet])
