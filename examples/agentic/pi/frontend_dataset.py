"""Example adapter for frontend tasks; the core runner has no visual policy."""

from __future__ import annotations

import base64
from pathlib import Path

from areno.agent.harness_base import workspace_path
from areno.api.harness import list_datasets, register_dataset
from examples.agentic.pi.visual_reward import RubricDimension
from examples.agentic.pi.visual_workspace import VisualTask, VisualWorkspaceDataset

DIMENSIONS = {
    "requirements": RubricDimension(
        "Verify every requested section, component, label and content requirement. Deduct proportionally for omissions, generic substitutes and placeholder copy.",
        0.25,
    ),
    "layout": RubricDimension(
        "Composition, alignment, hierarchy, spacing and density satisfy the requested design throughout the full page, without rough or unfinished sections.",
        0.25,
    ),
    "visual_style": RubricDimension(
        "Typography, color, surfaces and visual details consistently match the requested style. Generic default styling earns little credit.",
        0.20,
    ),
    "responsive": RubricDimension(
        "At the captured viewports, content remains readable, well arranged and free of overflow or clipping.", 0.20
    ),
    "usability": RubricDimension(
        "Controls are recognizable, text has adequate contrast, labels are clear and targets are usable. Score actual interactions only from browser evidence.",
        0.10,
    ),
}
VIEWPORTS = {"desktop": {"width": 1280, "height": 960}}


class FrontendDataset(VisualWorkspaceDataset):
    def __init__(self, *, dimensions=None, validity_checks=None, viewports=None, reserved_roots=(".pi",), **kwargs):
        super().__init__(reserved_roots=reserved_roots, **kwargs)
        self.dimensions = {
            name: value if isinstance(value, RubricDimension) else RubricDimension(**value)
            for name, value in (DIMENSIONS if dimensions is None else dimensions).items()
        }
        self.validity_checks = dict(validity_checks or {})
        self.viewports = dict(VIEWPORTS if viewports is None else viewports)

    def normalize(self, record):
        record = dict(record)
        prompt = record.get("prompt") or record.get("problem_statement")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("frontend task requires a prompt")
        record["prompt"] = prompt
        if "grading_instruction" in record and (
            not isinstance(record["grading_instruction"], str) or not record["grading_instruction"].strip()
        ):
            raise ValueError("grading_instruction must be nonempty text")
        for field in ["files", "binary_files"]:
            files = record.get(field, {})
            if not isinstance(files, dict):
                raise ValueError(f"{field} must map relative paths to file contents")
            for name, content in files.items():
                workspace_path(name, self.reserved_roots)
                if not isinstance(content, str):
                    raise ValueError("frontend file content must be text or base64")
                if field == "binary_files":
                    base64.b64decode(content, validate=True)
        if record.get("files", {}).keys() & record.get("binary_files", {}).keys():
            raise ValueError("text and binary task file paths overlap")
        workspace_path(record.get("entrypoint", "index.html"))
        if record.get("serve_dir", ".") != ".":
            workspace_path(record["serve_dir"])
        return record

    def workspace_files(self, record):
        return {
            **record.get("files", {}),
            **{name: base64.b64decode(data, validate=True) for name, data in record.get("binary_files", {}).items()},
        }

    def visual_task(self, attempt):
        record = attempt.item.record
        return VisualTask(
            instruction=record.get("grading_instruction", record["prompt"]),
            entrypoint=record.get("entrypoint", "index.html"),
            viewports=record.get("viewports", self.viewports),
            dimensions=self.dimensions,
            references={label: Path(path) for label, path in record.get("reference_images", {}).items()},
            interactions=record.get("interactions", []),
            validity_checks=self.validity_checks,
            serve_dir=record.get("serve_dir", "."),
        )


if "frontend" not in list_datasets():
    register_dataset("frontend", FrontendDataset)
