"""Constrained coding-agent tools for the agentic coding example."""

from __future__ import annotations

import os
import pty
import queue
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 4000
MAX_READ_CHARS = 6000
DEFAULT_TIMEOUT_S = 10.0
SCREENED_OUTPUT_CHARS = 6000
SCREENED_HEAD_LINES = 80
SCREENED_TAIL_LINES = 160
STREAM_HEAD_LINES = 120
IMPORTANT_OUTPUT_PATTERNS = (
    "error",
    "exception",
    "traceback",
    "failed",
    "failure",
    "oom",
    "out of memory",
    "cuda",
    "nan",
    "warning",
    "epoch=",
    "stage=",
    "metric=",
    "train_stats",
    "smoke",
    "selected",
    "saved",
    "listening",
    "running on",
)
_IGNORED_REPO_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
}


class ToolError(ValueError):
    """User-facing tool error with a compact message."""


@dataclass(slots=True)
class CodingWorkspace:
    """Isolated workspace for one coding task."""

    task: dict[str, Any]
    root: Path
    cleanup_on_close: bool = True
    max_command_timeout_s: float = DEFAULT_TIMEOUT_S
    submitted: dict[str, Any] | None = None
    command_history: list[dict[str, Any]] = field(default_factory=list)
    command_output_callback: Callable[[dict[str, Any]], None] | None = None
    user_input_callback: Callable[[str], str] | None = None
    response_callback: Callable[[str], None] | None = None
    interrupt_requested: bool = False

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
        timeout = min(max(float(timeout_s), 0.1), max(float(self.max_command_timeout_s), 0.1))
        self._emit_command_output({"kind": "start", "command": command, "timeout_s": timeout})
        stdout, stderr, timed_out, streamed_lines, skipped_stream_lines, returncode_value = _run_command_streaming(
            command,
            cwd=self.root,
            timeout_s=timeout,
            on_output=self._emit_command_output,
            should_interrupt=lambda: self.interrupt_requested,
        )
        interrupted = self.interrupt_requested and timed_out
        returncode = 130 if interrupted else 124 if timed_out else int(returncode_value or 0)
        screened = _screen_command_output(stdout, stderr)
        result = {
            "command": command,
            "returncode": returncode,
            "output": screened["output"],
            "stdout": screened["stdout"],
            "stderr": screened["stderr"],
            "screened": screened["screened"],
            "timed_out": timed_out,
            "interrupted": interrupted,
            "streamed_lines": streamed_lines,
            "skipped_stream_lines": skipped_stream_lines,
        }
        self._emit_command_output(
            {
                "kind": "end",
                "command": command,
                "returncode": returncode,
                "timed_out": timed_out,
                "interrupted": interrupted,
                "streamed_lines": streamed_lines,
                "skipped_stream_lines": skipped_stream_lines,
            }
        )
        self.command_history.append(result)
        return result

    def _emit_command_output(self, event: dict[str, Any]) -> None:
        if self.command_output_callback is not None:
            self.command_output_callback(event)

    def submit(self, status: str, summary: str = "") -> dict[str, Any]:
        self.submitted = {"status": str(status), "summary": str(summary)[:500]}
        return {"submitted": self.submitted}

    def request_user_input(self, prompt: str) -> dict[str, Any]:
        if self.user_input_callback is None:
            return {"error": "user input is not available in this environment"}
        return {"response": self.user_input_callback(str(prompt))}

    def response_to_user(self, message: str) -> dict[str, Any]:
        if self.response_callback is not None:
            self.response_callback(str(message))
        return {"delivered": True, "message": str(message)}

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
        if name in {"request_user_input", "require_user_input", "requre_userinput"}:
            return workspace.request_user_input(prompt=str(arguments.get("prompt", "")))
        if name == "response_to_user":
            return workspace.response_to_user(message=str(arguments.get("message", "")))
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
        old_index = max(old_start - 1, 0)
        output.extend(original[cursor:old_index])
        cursor = old_index
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
    if parts & _IGNORED_REPO_PARTS:
        return False
    return not path.name.startswith(".")


def _ignored_repo_path(path: Path) -> bool:
    return any(part in _IGNORED_REPO_PARTS for part in path.parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated {len(text) - limit} chars ..."


def _communicate_streaming(
    proc: subprocess.Popen[str],
    *,
    timeout_s: float,
    on_output: Callable[[dict[str, Any]], None],
    should_interrupt: Callable[[], bool] | None = None,
) -> tuple[str, str, bool, int, int]:
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    line_counts = {"stdout": 0, "stderr": 0}
    streamed_lines = 0
    skipped_stream_lines = 0

    def read_stream(name: str, stream: Any) -> None:
        try:
            if stream is not None:
                for line in stream:
                    output_queue.put((name, line))
        finally:
            output_queue.put((name, None))

    threads = [
        threading.Thread(target=read_stream, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=read_stream, args=("stderr", proc.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    finished_streams: set[str] = set()
    deadline = time.monotonic() + timeout_s
    timed_out = False
    while len(finished_streams) < 2:
        if should_interrupt is not None and should_interrupt():
            timed_out = True
            proc.kill()
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            proc.kill()
            break
        try:
            stream_name, line = output_queue.get(timeout=min(0.1, remaining))
        except queue.Empty:
            continue
        if line is None:
            finished_streams.add(stream_name)
            continue
        if stream_name == "stdout":
            stdout_parts.append(line)
        else:
            stderr_parts.append(line)
        line_counts[stream_name] += 1
        if _should_stream_line(line_counts[stream_name], line):
            streamed_lines += 1
            on_output({"kind": "line", "stream": stream_name, "line": line, "line_index": line_counts[stream_name]})
        else:
            skipped_stream_lines += 1

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=0.2)
    while not output_queue.empty():
        stream_name, line = output_queue.get_nowait()
        if line is None:
            continue
        if stream_name == "stdout":
            stdout_parts.append(line)
        else:
            stderr_parts.append(line)
    return "".join(stdout_parts), "".join(stderr_parts), timed_out, streamed_lines, skipped_stream_lines


def _run_command_streaming(
    command: str,
    *,
    cwd: Path,
    timeout_s: float,
    on_output: Callable[[dict[str, Any]], None],
    should_interrupt: Callable[[], bool] | None = None,
) -> tuple[str, str, bool, int, int, int]:
    if os.name == "posix":
        return _run_command_streaming_pty(
            command,
            cwd=cwd,
            timeout_s=timeout_s,
            on_output=on_output,
            should_interrupt=should_interrupt,
        )
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    stdout, stderr, timed_out, streamed_lines, skipped_stream_lines = _communicate_streaming(
        proc,
        timeout_s=timeout_s,
        on_output=on_output,
        should_interrupt=should_interrupt,
    )
    return stdout, stderr, timed_out, streamed_lines, skipped_stream_lines, int(proc.returncode or 0)


def _run_command_streaming_pty(
    command: str,
    *,
    cwd: Path,
    timeout_s: float,
    on_output: Callable[[dict[str, Any]], None],
    should_interrupt: Callable[[], bool] | None = None,
) -> tuple[str, str, bool, int, int, int]:
    master_fd, slave_fd = pty.openpty()
    proc: subprocess.Popen[bytes] | None = None
    output_parts: list[str] = []
    streamed_chunks = 0
    skipped_chunks = 0
    timed_out = False
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "TERM": os.environ.get("TERM") or "xterm-256color"},
        )
        os.close(slave_fd)
        slave_fd = -1
        deadline = time.monotonic() + timeout_s
        while True:
            if should_interrupt is not None and should_interrupt():
                timed_out = True
                proc.kill()
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                proc.kill()
                break
            ready, _, _ = select.select([master_fd], [], [], min(0.1, remaining))
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            output_parts.append(text)
            if _should_stream_chunk(streamed_chunks + skipped_chunks + 1, text):
                streamed_chunks += 1
                on_output({"kind": "chunk", "stream": "stdout", "text": text, "chunk_index": streamed_chunks})
            else:
                skipped_chunks += 1
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    finally:
        if slave_fd >= 0:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        try:
            os.close(master_fd)
        except OSError:
            pass
    return (
        "".join(output_parts),
        "",
        timed_out,
        streamed_chunks,
        skipped_chunks,
        int((proc.returncode if proc else 1) or 0),
    )


def _should_stream_line(line_index: int, line: str) -> bool:
    if line_index <= STREAM_HEAD_LINES:
        return True
    lowered = line.lower()
    return any(pattern in lowered for pattern in IMPORTANT_OUTPUT_PATTERNS)


def _should_stream_chunk(chunk_index: int, text: str) -> bool:
    if chunk_index <= STREAM_HEAD_LINES:
        return True
    lowered = text.lower()
    return any(pattern in lowered for pattern in IMPORTANT_OUTPUT_PATTERNS)


def _screen_command_output(stdout: str, stderr: str) -> dict[str, Any]:
    combined_parts = []
    if stdout:
        combined_parts.append("STDOUT:\n" + stdout)
    if stderr:
        combined_parts.append("STDERR:\n" + stderr)
    combined = "\n\n".join(combined_parts)
    screened_output = _screen_text(combined, SCREENED_OUTPUT_CHARS)
    return {
        "output": screened_output,
        "stdout": _screen_text(stdout, MAX_OUTPUT_CHARS),
        "stderr": _screen_text(stderr, MAX_OUTPUT_CHARS),
        "screened": len(combined) > len(screened_output)
        or len(stdout) > MAX_OUTPUT_CHARS
        or len(stderr) > MAX_OUTPUT_CHARS,
    }


def _screen_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    important = [line for line in lines if any(pattern in line.lower() for pattern in IMPORTANT_OUTPUT_PATTERNS)]
    selected = [
        *lines[:SCREENED_HEAD_LINES],
        *([] if not important else ["... important lines ..."]),
        *important[-SCREENED_TAIL_LINES:],
        "... tail ...",
        *lines[-SCREENED_TAIL_LINES:],
    ]
    compact = "\n".join(_dedupe_adjacent(selected))
    return _truncate_middle(compact, limit)


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n... screened {len(text) - limit} chars ...\n"
    head = max((limit - len(marker)) // 2, 0)
    tail = max(limit - len(marker) - head, 0)
    return text[:head] + marker + text[-tail:]


def _dedupe_adjacent(lines: list[str]) -> list[str]:
    out = []
    previous: str | None = None
    for line in lines:
        if line == previous:
            continue
        out.append(line)
        previous = line
    return out
