"""Read-only repository tools for the dashboard operations agent."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any


class AgentFileBrowser:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._cwd = self.root
        self._lock = threading.Lock()

    def cwd(self) -> str:
        with self._lock:
            cwd = self._cwd
        return str(cwd.relative_to(self.root) if cwd != self.root else ".")

    def resolve(self, raw_path: str | None = None) -> Path:
        with self._lock:
            base = self._cwd
        candidate = base if not raw_path else Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path must stay inside the AReno repository") from exc
        return resolved

    def list_folder(self, path: str | None = None) -> dict[str, Any]:
        folder = self.resolve(path)
        if not folder.is_dir():
            raise ValueError(f"not a directory: {folder.relative_to(self.root)}")
        entries = []
        for child in sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if child.name == ".git":
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(self.root)),
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return {
            "cwd": self.cwd(),
            "path": str(folder.relative_to(self.root) if folder != self.root else "."),
            "entries": entries[:250],
        }

    def cd(self, path: str) -> dict[str, Any]:
        folder = self.resolve(path)
        if not folder.is_dir():
            raise ValueError(f"not a directory: {folder.relative_to(self.root)}")
        with self._lock:
            self._cwd = folder
        return {"cwd": self.cwd()}

    def read_file(self, path: str, *, start_line: int = 1, max_lines: int = 200) -> dict[str, Any]:
        file_path = self.resolve(path)
        if not file_path.is_file():
            raise ValueError(f"not a file: {file_path.relative_to(self.root)}")
        max_lines = max(1, min(max_lines, 500))
        start_line = max(1, start_line)
        selected = []
        end_line = start_line + max_lines - 1
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for idx, line in enumerate(handle, start=1):
                    if idx >= start_line:
                        selected.append(line.rstrip("\r\n"))
                    if idx >= end_line:
                        break
        except Exception as exc:
            raise ValueError(f"failed to read file: {exc}") from exc
        content = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start_line))
        return {
            "path": str(file_path.relative_to(self.root)),
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line,
            "content": content,
        }

    def rg(self, pattern: str, *, path: str | None = None, max_matches: int = 100) -> dict[str, Any]:
        if not pattern:
            raise ValueError("pattern is required")
        search_path = self.resolve(path)
        max_matches = max(1, min(max_matches, 500))
        command = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-count",
            str(max_matches),
            "--",
            pattern,
            str(search_path),
        ]
        try:
            proc = subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=20, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError("rg is not installed") from exc
        output = proc.stdout.strip()
        matches = output.splitlines()[:max_matches] if output else []
        return {
            "cwd": self.cwd(),
            "path": str(search_path.relative_to(self.root) if search_path != self.root else "."),
            "pattern": pattern,
            "returncode": proc.returncode,
            "matches": matches,
        }

    def path_info(self) -> dict[str, Any]:
        try:
            import areno

            package_path = str(Path(areno.__file__).resolve().parent)
        except Exception:
            package_path = str(self.root / "areno")
        return {
            "repo_root": str(self.root),
            "agent_cwd": self.cwd(),
            "areno_package_path": package_path,
            "examples_path": str(self.root / "examples"),
        }
