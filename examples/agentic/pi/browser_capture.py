"""Trusted browser evidence collection through any Harbor environment."""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath

from areno.agent.harness_base import workspace_path, write_workspace_files
from areno.api.harness import HarnessAttempt


class BrowserCapture:
    def __init__(self, *, node: str = "node", chromium: str | None = None) -> None:
        self.node = node
        self.chromium = chromium

    async def capture(
        self,
        attempt: HarnessAttempt,
        *,
        entrypoint: str,
        viewports: dict[str, dict],
        interactions: list[dict] | None = None,
        serve_dir: str = ".",
    ) -> tuple[dict, dict[str, bytes]]:
        handle = attempt.environment
        assert handle is not None
        workspace_path(entrypoint)
        if serve_dir != ".":
            workspace_path(serve_dir)
        delivered_root = PurePosixPath(handle.workspace) / serve_dir
        exists = await handle.environment.exec(
            shlex.join(["test", "-f", str(delivered_root / entrypoint)]), cwd="/", timeout_sec=10
        )
        if exists.return_code == 1:
            return {"capture_error": "missing delivered entrypoint"}, {}
        if exists.return_code != 0:
            raise RuntimeError("could not inspect the delivered entrypoint")
        control = PurePosixPath(handle.control_dir)
        files = attempt.root / "browser-control"
        files.mkdir()
        request = {
            "workspace": str(delivered_root),
            "output": str(control / "rendered"),
            "entrypoint": entrypoint,
            "viewports": viewports,
            "interactions": interactions or [],
            "chromium": self.chromium,
            "timeout_ms": int(attempt.verify_timeout * 1000) - 1000,
        }
        write_workspace_files(
            files,
            {
                "browser_capture.mjs": Path(__file__).with_suffix(".mjs").read_text(),
                "browser-request.json": json.dumps(request),
            },
        )
        await handle.environment.upload_dir(files, handle.control_dir)
        result = await handle.environment.exec(
            shlex.join([self.node, str(control / "browser_capture.mjs"), str(control / "browser-request.json")]),
            cwd=handle.workspace,
            timeout_sec=attempt.verify_timeout,
        )
        if result.return_code == 2:
            return {"capture_error": "page did not respond within the browser time budget"}, {}
        if result.return_code != 0:
            raise RuntimeError(f"browser capture failed (exit {result.return_code}): {(result.stderr or '')[-1500:]}")
        output = attempt.root / "artifacts" / "browser"
        output.mkdir(parents=True)
        await handle.environment.download_dir(str(control / "rendered"), output)
        report = json.loads((output / "report.json").read_text())
        images = {}
        for name, view in report["views"].items():
            images[f"Candidate {name}"] = (output / workspace_path(view["screenshot"])).read_bytes()
        for index, action in enumerate(report["interactions"]):
            images[f"Candidate interaction {index}"] = (output / workspace_path(action["screenshot"])).read_bytes()
        return report, images
