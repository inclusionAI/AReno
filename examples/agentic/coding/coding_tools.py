"""Constrained coding-agent tools for the agentic coding example."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 4000
MAX_READ_CHARS = 6000
DEFAULT_TIMEOUT_S = 10.0


class ToolError(ValueError):
    """User-facing tool error with a compact message."""


@dataclass(slots=True)
class CodingWorkspace:
    """Isolated workspace for one coding task."""

    task: dict[str, Any]
    root: Path
    cleanup_on_close: bool = True
    submitted: dict[str, Any] | None = None
    command_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> CodingWorkspace:
        # Dataset-backed training samples run in temp repos created from the
        # task's file map; this keeps generated patches isolated and repeatable.
        workspace = Path(tempfile.mkdtemp(prefix="areno-coding-"))
        try:
            files = task.get("files")
            if not isinstance(files, dict) or not files:
                raise ToolError("task must define a non-empty files object")
            for rel_path, content in files.items():
                target = _safe_path(workspace, str(rel_path))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(content), encoding="utf-8")
            return cls(task=task, root=workspace)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise

    @classmethod
    def from_existing_repo(cls, task: dict[str, Any], repo_path: str | os.PathLike[str]) -> CodingWorkspace:
        workspace = Path(tempfile.mkdtemp(prefix="areno-coding-"))
        source = Path(repo_path).expanduser().resolve()
        if not source.is_dir():
            shutil.rmtree(workspace, ignore_errors=True)
            raise ToolError(f"repo path is not a directory: {source}")
        try:
            for item in sorted(source.rglob("*")):
                rel = item.relative_to(source)
                if _ignored_repo_path(rel):
                    continue
                target = workspace / rel
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif item.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
            return cls(task=task, root=workspace)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise

    @classmethod
    def from_current_repo(cls, task: dict[str, Any], repo_path: str | os.PathLike[str]) -> CodingWorkspace:
        # The interactive CLI is intentionally Codex-like: it operates on the
        # current checkout, so close() must not delete the workspace.
        source = Path(repo_path).expanduser().resolve()
        if not source.is_dir():
            raise ToolError(f"repo path is not a directory: {source}")
        return cls(task=task, root=source, cleanup_on_close=False)

    def close(self) -> None:
        if self.cleanup_on_close:
            shutil.rmtree(self.root, ignore_errors=True)

    def list_files(self, path: str = ".") -> dict[str, Any]:
        base = _safe_path(self.root, path)
        if not base.exists():
            raise ToolError(f"path does not exist: {path}")
        if base.is_file():
            return {"files": [_relative(self.root, base)]}
        files = [
            _relative(self.root, item)
            for item in sorted(base.rglob("*"))
            if item.is_file() and _is_visible_source(item)
        ]
        return {"files": files[:200]}

    def inspect_tree(self, path: str = ".", max_depth: int = 3) -> dict[str, Any]:
        base = _safe_path(self.root, path)
        if not base.exists():
            raise ToolError(f"path does not exist: {path}")
        depth_limit = min(max(int(max_depth), 1), 6)
        rows = []
        for item in sorted(base.rglob("*")):
            if not _is_visible_source(item):
                continue
            rel = Path(_relative(self.root, item))
            depth = len(rel.parts) - len(Path(_relative(self.root, base)).parts)
            if depth > depth_limit:
                continue
            suffix = "/" if item.is_dir() else ""
            rows.append(f"{'  ' * max(depth - 1, 0)}{item.name}{suffix}")
            if len(rows) >= 200:
                return {"tree": rows, "truncated": True}
        return {"tree": rows, "truncated": False}

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 80) -> dict[str, Any]:
        target = _safe_path(self.root, path)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        start = max(int(start_line), 1)
        count = min(max(int(max_lines), 1), 200)
        lines = target.read_text(encoding="utf-8").splitlines()
        selected = lines[start - 1 : start - 1 + count]
        text = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))
        return {
            "path": _relative(self.root, target),
            "start_line": start,
            "end_line": start + len(selected) - 1 if selected else start - 1,
            "content": _truncate(text, MAX_READ_CHARS),
        }

    def rg(self, pattern: str, path: str = ".", case_sensitive: bool = True) -> dict[str, Any]:
        if not pattern:
            raise ToolError("pattern must be non-empty")
        base = _safe_path(self.root, path)
        if not base.exists():
            raise ToolError(f"path does not exist: {path}")
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(str(pattern), flags)
        except re.error as exc:
            raise ToolError(f"invalid regex pattern: {exc}") from exc
        matches = []
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for item in candidates:
            if not item.is_file() or not _is_visible_source(item):
                continue
            for lineno, line in enumerate(item.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": _relative(self.root, item), "line": lineno, "text": line[:240]})
                    if len(matches) >= 50:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def search(self, query: str, path: str = ".") -> dict[str, Any]:
        return self.rg(pattern=re.escape(str(query)), path=path)

    def apply_patch(self, patch: str) -> dict[str, Any]:
        if not patch.strip():
            raise ToolError("patch must be non-empty")
        touched = apply_unified_patch(self.root, patch)
        return {"applied": True, "files": touched}

    def replace_text(self, path: str, old_text: str, new_text: str, count: int = 0) -> dict[str, Any]:
        target = _safe_path(self.root, path)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        if not old_text:
            raise ToolError("old_text must be non-empty")
        content = target.read_text(encoding="utf-8")
        available = content.count(old_text)
        if available == 0:
            raise ToolError("old_text was not found in target file")
        limit = max(int(count), 0)
        replacements = available if limit == 0 else min(available, limit)
        updated = content.replace(old_text, new_text, replacements)
        target.write_text(updated, encoding="utf-8")
        return {"replaced": replacements, "path": _relative(self.root, target)}

    def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        target = _safe_path(self.root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return {
            "path": _relative(self.root, target),
            "bytes": len(content.encode("utf-8")),
            "append": bool(append),
        }

    def run_command(self, command: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
        if _is_dangerous_rm_command(command):
            raise ToolError(f"dangerous rm command is not allowed: {command}")
        timeout = min(max(float(timeout_s), 0.1), DEFAULT_TIMEOUT_S)
        proc = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        result = {
            "command": command,
            "returncode": int(proc.returncode),
            "stdout": _truncate(proc.stdout, MAX_OUTPUT_CHARS),
            "stderr": _truncate(proc.stderr, MAX_OUTPUT_CHARS),
        }
        self.command_history.append(result)
        return result

    def submit(self, status: str, summary: str = "") -> dict[str, Any]:
        self.submitted = {"status": str(status), "summary": str(summary)[:500]}
        return {"submitted": self.submitted}

    def run_all_tests(self) -> list[dict[str, Any]]:
        results = []
        for command in self.task.get("test_commands") or []:
            try:
                results.append(self.run_command(str(command)))
            except (subprocess.TimeoutExpired, ToolError) as exc:
                results.append({"command": command, "returncode": 124, "stdout": "", "stderr": str(exc)})
        return results


def run_tool(workspace: CodingWorkspace, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tool call and convert tool exceptions into model-visible errors."""

    try:
        if name == "list_files":
            return workspace.list_files(path=str(arguments.get("path", ".")))
        if name == "inspect_tree":
            return workspace.inspect_tree(
                path=str(arguments.get("path", ".")), max_depth=int(arguments.get("max_depth", 3))
            )
        if name == "read_file":
            return workspace.read_file(
                path=str(arguments.get("path", "")),
                start_line=int(arguments.get("start_line", 1)),
                max_lines=int(arguments.get("max_lines", 80)),
            )
        if name == "rg":
            return workspace.rg(
                pattern=str(arguments.get("pattern", "")),
                path=str(arguments.get("path", ".")),
                case_sensitive=bool(arguments.get("case_sensitive", True)),
            )
        if name == "search":
            return workspace.search(query=str(arguments.get("query", "")), path=str(arguments.get("path", ".")))
        if name == "apply_patch":
            return workspace.apply_patch(patch=str(arguments.get("patch", "")))
        if name == "replace_text":
            return workspace.replace_text(
                path=str(arguments.get("path", "")),
                old_text=str(arguments.get("old_text", "")),
                new_text=str(arguments.get("new_text", "")),
                count=int(arguments.get("count", 0)),
            )
        if name == "write_file":
            return workspace.write_file(
                path=str(arguments.get("path", "")),
                content=str(arguments.get("content", "")),
                append=bool(arguments.get("append", False)),
            )
        if name == "run_command":
            return workspace.run_command(
                command=str(arguments.get("command", "")),
                timeout_s=float(arguments.get("timeout_s", DEFAULT_TIMEOUT_S)),
            )
        if name == "submit":
            return workspace.submit(status=str(arguments.get("status", "")), summary=str(arguments.get("summary", "")))
        return {"error": f"unknown tool: {name}"}
    except subprocess.TimeoutExpired as exc:
        return {"error": f"command timed out after {exc.timeout}s", "returncode": 124}
    except (OSError, ToolError, UnicodeError, ValueError) as exc:
        return {"error": str(exc)}


def apply_unified_patch(root: Path, patch: str) -> list[str]:
    """Apply a small unified patch without invoking external patch tools."""

    lines = patch.splitlines()
    idx = 0
    touched = []
    while idx < len(lines):
        if not lines[idx].startswith("--- "):
            idx += 1
            continue
        if idx + 1 >= len(lines) or not lines[idx + 1].startswith("+++ "):
            raise ToolError("invalid patch: missing +++ file header")
        old_path = _patch_path(lines[idx][4:].strip())
        new_path = _patch_path(lines[idx + 1][4:].strip())
        rel_path = new_path if new_path != "/dev/null" else old_path
        if rel_path == "/dev/null":
            raise ToolError("deleting files is not supported")
        target = _safe_path(root, rel_path)
        original = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
        idx += 2
        updated, idx = _apply_file_hunks(original, lines, idx)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")
        touched.append(_relative(root, target))
    if not touched:
        raise ToolError("invalid patch: no file headers found")
    return touched


def _apply_file_hunks(original: list[str], lines: list[str], idx: int) -> tuple[list[str], int]:
    output = []
    cursor = 0
    saw_hunk = False
    while idx < len(lines):
        if lines[idx].startswith("--- "):
            break
        if not lines[idx].startswith("@@"):
            idx += 1
            continue
        saw_hunk = True
        old_start = _parse_hunk_start(lines[idx])
        output.extend(original[cursor : old_start - 1])
        cursor = old_start - 1
        idx += 1
        while idx < len(lines) and not lines[idx].startswith("@@") and not lines[idx].startswith("--- "):
            line = lines[idx]
            if line.startswith("\\"):
                idx += 1
                continue
            if not line:
                raise ToolError("invalid patch: empty hunk line")
            marker, text = line[0], line[1:]
            if marker == " ":
                if cursor >= len(original) or original[cursor] != text:
                    raise ToolError("patch context does not match target file")
                output.append(text)
                cursor += 1
            elif marker == "-":
                if cursor >= len(original) or original[cursor] != text:
                    raise ToolError("patch removal does not match target file")
                cursor += 1
            elif marker == "+":
                output.append(text)
            else:
                raise ToolError(f"invalid patch hunk marker: {marker}")
            idx += 1
    if not saw_hunk:
        raise ToolError("invalid patch: missing hunk")
    output.extend(original[cursor:])
    return output, idx


def _parse_hunk_start(header: str) -> int:
    match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", header)
    if match is None:
        raise ToolError(f"invalid patch hunk header: {header}")
    return int(match.group(1))


def _patch_path(raw: str) -> str:
    path = raw.split("\t", 1)[0].split(" ", 1)[0]
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _is_dangerous_rm_command(command: str) -> bool:
    # The interactive coding example permits general shell/test commands, but
    # keeps destructive removals out of the tool surface.
    return re.search(r"(^|[;&|]\s*)(?:sudo\s+)?rm(?:\s|$)", command.strip()) is not None


def _safe_path(root: Path, rel_path: str) -> Path:
    if not rel_path:
        raise ToolError("path must be non-empty")
    root_resolved = root.resolve()
    raw = Path(rel_path).expanduser()
    if raw.is_absolute():
        # Models often echo absolute paths from prompts; accept them only when
        # they still resolve inside the active workspace.
        resolved = raw.resolve()
        if resolved == root_resolved:
            return root_resolved
        if root_resolved in resolved.parents:
            return resolved
        raise ToolError(f"absolute path is outside workspace: {rel_path}")
    candidate = (root / rel_path).resolve()
    if candidate == root_resolved or root_resolved in candidate.parents:
        return candidate
    raise ToolError(f"unsafe path outside workspace: {rel_path}")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_visible_source(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".git" in parts:
        return False
    return not path.name.startswith(".")


def _ignored_repo_path(path: Path) -> bool:
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv"}
    return any(part in ignored for part in path.parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated {len(text) - limit} chars ..."
