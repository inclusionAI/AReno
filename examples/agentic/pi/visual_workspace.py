"""Reusable file-workspace grading with browser evidence and a VLM rubric."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from weakref import WeakValueDictionary

from areno.agent.harness_base import FileWorkspaceDataset, workspace_path
from areno.api.harness import HarnessAttempt
from examples.agentic.pi.browser_capture import BrowserCapture
from examples.agentic.pi.visual_reward import RubricDimension, VLMJudge

_GRADING_SLOTS: WeakValueDictionary[tuple[asyncio.AbstractEventLoop, int], asyncio.Semaphore] = WeakValueDictionary()


@dataclass(slots=True)
class VisualTask:
    instruction: str
    entrypoint: str
    viewports: dict[str, dict]
    dimensions: dict[str, RubricDimension]
    references: dict[str, Path] = field(default_factory=dict)
    interactions: list[dict] = field(default_factory=list)
    validity_checks: dict[str, str] = field(default_factory=dict)
    serve_dir: str = "."


class VisualWorkspaceDataset(FileWorkspaceDataset):
    """Adapters own task schemas; capture, hard gates and scoring are shared.

    Browser/transport infrastructure failures invalidate a trajectory. Missing,
    blank or unrenderable pages earn zero; visible pages receive rubric scores
    with page defects included as evidence for proportional deductions.
    """

    def __init__(
        self,
        *,
        renderer: BrowserCapture | None = None,
        judge: VLMJudge | None = None,
        grading_concurrency: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.renderer = renderer or BrowserCapture()
        self.judge = judge
        capacity = int(
            os.environ.get("ARENO_PI_GRADING_CONCURRENCY", "8") if grading_concurrency is None else grading_concurrency
        )
        if capacity < 1:
            raise ValueError("grading_concurrency must be positive")
        self._grading_capacity = capacity

    def normalize(self, record):
        record = dict(record)
        prompt = record.get("prompt") or record.get("problem_statement")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("visual task requires a prompt")
        record["prompt"] = prompt
        if "grading_instruction" in record and (
            not isinstance(record["grading_instruction"], str) or not record["grading_instruction"].strip()
        ):
            raise ValueError("grading_instruction must be nonempty text")
        for field_name in ["files", "binary_files"]:
            files = record.get(field_name, {})
            if not isinstance(files, dict):
                raise ValueError(f"{field_name} must map relative paths to file contents")
            for name, content in files.items():
                workspace_path(name, self.reserved_roots)
                if not isinstance(content, str):
                    raise ValueError("visual file content must be text or base64")
                if field_name == "binary_files":
                    base64.b64decode(content, validate=True)
        if record.get("files", {}).keys() & record.get("binary_files", {}).keys():
            raise ValueError("text and binary task file paths overlap")
        if "entrypoint" in record:
            workspace_path(record["entrypoint"])
        if record.get("serve_dir", ".") != ".":
            workspace_path(record["serve_dir"])
        return record

    def workspace_files(self, record):
        return {
            **record.get("files", {}),
            **{name: base64.b64decode(data, validate=True) for name, data in record.get("binary_files", {}).items()},
        }

    @abstractmethod
    def visual_task(self, attempt: HarnessAttempt) -> VisualTask: ...

    async def verify(self, attempt: HarnessAttempt) -> dict:
        # HarnessRunner constructs one dataset adapter per sample. All adapters
        # in a rollout must share the same grading capacity.
        key = (asyncio.get_running_loop(), self._grading_capacity)
        slots = _GRADING_SLOTS.get(key)
        if slots is None:
            slots = asyncio.Semaphore(self._grading_capacity)
            _GRADING_SLOTS[key] = slots
        async with slots:
            return await self._verify(attempt)

    async def _verify(self, attempt: HarnessAttempt) -> dict:
        task = self.visual_task(attempt)
        report, images = await self.renderer.capture(
            attempt,
            entrypoint=task.entrypoint,
            viewports=task.viewports,
            interactions=task.interactions,
            serve_dir=task.serve_dir,
        )
        issues = list(report.get("errors", [])) + list(report.get("resource_failures", []))
        faults = []
        if report.get("capture_error"):
            faults.append(report["capture_error"])
        if not report.get("views"):
            faults.append("no rendered views")
        elif not any(view.get("visible_content") for view in report["views"].values()):
            faults.append("no visible page content")
        for name, view in report.get("views", {}).items():
            if not view.get("visible_content"):
                issues.append(f"{name}: blank page")
            if view.get("broken_images"):
                issues.append(f"{name}: broken images")
        for action in report.get("interactions", []):
            if not action.get("passed"):
                issues.append(f"interaction failed: {action['action']}")
        result = {
            "grading_complete": True,
            "page_valid": not faults,
            "zero_reasons": faults,
            "page_issues": issues,
            "browser_evidence": report,
        }
        if faults:
            result["reward"] = 0.0
        else:
            references = {f"Reference {label}": path.read_bytes() for label, path in task.references.items()}
            result.update(
                await (self.judge or VLMJudge.from_env()).score(
                    instruction=task.instruction,
                    images={**references, **images},
                    dimensions=task.dimensions,
                    evidence=report,
                    validity_checks=task.validity_checks,
                )
            )
            failures = [f"{name}: {check['reason']}" for name, check in result["checks"].items() if not check["passed"]]
            result.update(page_valid=not failures, zero_reasons=failures)
        artifacts = attempt.root / "artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "grading.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        handle = attempt.environment
        assert handle is not None
        exists = await handle.environment.exec("test -d " + shlex.quote(handle.workspace), cwd="/", timeout_sec=10)
        if exists.return_code == 0:
            await handle.environment.download_dir(handle.workspace, artifacts / "submission")
        elif exists.return_code != 1:
            raise RuntimeError("could not inspect the submitted workspace")
        return result
