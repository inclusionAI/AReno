from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "coding"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous = {key: sys.modules.pop(key, None) for key in ("agent_loop", "coding_tools")}
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_coding_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        for key in ("agent_loop", "coding_tools"):
            sys.modules.pop(key, None)
            if previous[key] is not None:
                sys.modules[key] = previous[key]


def _add_task() -> dict:
    return json.loads((EXAMPLE_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_coding_loader_builds_prompt_records():
    loader = _load_module("dataset_loader")
    task = _add_task()

    records = loader.load_training_dataset("unused", default_loader=lambda _: [task])

    assert records[0]["prompt"].startswith("Fix SWE-bench-style task local__calculator-001")
    assert records[0]["FAIL_TO_PASS"] == [
        "test_calculator.py::test_add_positive_numbers",
        "test_calculator.py::test_add_negative_numbers",
    ]
    assert records[0]["test_commands"] == ["python -m pytest test_calculator.py -q"]


def test_coding_tools_apply_patch_run_tests_and_reject_unsafe_paths():
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())
    try:
        (workspace.root / "node_modules").mkdir()
        (workspace.root / "node_modules" / "ignored.js").write_text("return a - b\n", encoding="utf-8")
        assert workspace.list_files()["files"] == ["calculator.py", "test_calculator.py"]
        assert "calculator.py" in "\n".join(workspace.inspect_tree()["tree"])
        assert "node_modules" not in "\n".join(workspace.inspect_tree()["tree"])
        assert workspace.inspect_tree(str(workspace.root), max_depth=1)["tree"]
        assert "return a - b" in workspace.read_file("calculator.py")["content"]
        assert workspace.rg("ignored.js")["matches"] == []
        assert workspace.rg("return a [-+] b")["matches"][0]["path"] == "calculator.py"
        replace = tools.run_tool(
            workspace,
            "replace_text",
            {"path": "calculator.py", "old_text": "return a - b", "new_text": "return a + b"},
        )
        assert replace == {"path": "calculator.py", "replaced": 1}
        tools.run_tool(
            workspace,
            "replace_text",
            {"path": "calculator.py", "old_text": "return a + b", "new_text": "return a - b"},
        )
        written = tools.run_tool(workspace, "write_file", {"path": "notes/result.txt", "content": "hello\n"})
        assert written == {"path": "notes/result.txt", "bytes": 6, "append": False}
        appended = tools.run_tool(
            workspace, "write_file", {"path": "notes/result.txt", "content": "world\n", "append": True}
        )
        assert appended == {"path": "notes/result.txt", "bytes": 6, "append": True}
        assert workspace.read_file("notes/result.txt")["content"] == "1: hello\n2: world"
        unsafe = tools.run_tool(workspace, "read_file", {"path": "../outside.py"})
        assert "unsafe path outside workspace" in unsafe["error"]
        unsafe_write = tools.run_tool(workspace, "write_file", {"path": "../outside.py", "content": "x"})
        assert "unsafe path outside workspace" in unsafe_write["error"]
        result = tools.run_tool(
            workspace,
            "apply_patch",
            {
                "patch": (
                    "--- a/calculator.py\n"
                    "+++ b/calculator.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def add(a, b):\n"
                    "-    return a - b\n"
                    "+    return a + b\n"
                )
            },
        )
        assert result == {"applied": True, "files": ["calculator.py"]}
        new_file = tools.run_tool(
            workspace,
            "apply_patch",
            {"patch": ("--- /dev/null\n+++ b/created.py\n@@ -0,0 +1,2 @@\n+def created():\n+    return 1\n")},
        )
        assert new_file == {"applied": True, "files": ["created.py"]}
        assert "1: def created()" in workspace.read_file("created.py")["content"]
        test_result = workspace.run_command("python -m pytest test_calculator.py -q")
        assert test_result["returncode"] == 0
    finally:
        workspace.close()


def test_coding_tools_reject_invalid_patch_and_dangerous_rm_command():
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())
    try:
        patch = tools.run_tool(workspace, "apply_patch", {"patch": "not a patch"})
        assert "no file headers found" in patch["error"]
        replace = tools.run_tool(
            workspace,
            "replace_text",
            {"path": "calculator.py", "old_text": "missing", "new_text": "new"},
        )
        assert "old_text was not found" in replace["error"]
        harmless = tools.run_tool(workspace, "run_command", {"command": "python -c 'print(123)'"})
        assert harmless["returncode"] == 0
        assert harmless["stdout"].strip() == "123"
        command = tools.run_tool(workspace, "run_command", {"command": "rm -rf ."})
        assert "dangerous rm command is not allowed" in command["error"]
    finally:
        workspace.close()


def test_coding_reward_scores_successful_submitted_patch():
    reward = _load_module("reward")
    task = _add_task()
    record = SimpleNamespace(
        source_record=task,
        tool_calls=[
            {"name": "apply_patch", "arguments": {"patch": "..."}},
            {"name": "run_command", "arguments": {"command": "python -m pytest test_calculator.py -q"}},
            {"name": "submit", "arguments": {"status": "solved"}},
        ],
        tool_results=[
            {
                "name": "run_command",
                "content": json.dumps({"command": "python -m pytest test_calculator.py -q", "returncode": 0}),
            }
        ],
    )

    assert reward.reward_fn(record) == 1.0

    record.tool_results[0]["content"] = json.dumps(
        {"command": "python -m pytest test_calculator.py -q", "returncode": 1}
    )
    assert reward.reward_fn(record) == -0.5


def test_coding_reward_allows_solved_patch_without_required_commands():
    reward = _load_module("reward")
    record = SimpleNamespace(
        source_record={"test_commands": []},
        tool_calls=[
            {"name": "apply_patch", "arguments": {"patch": "..."}},
            {"name": "submit", "arguments": {"status": "solved"}},
        ],
        tool_results=[],
    )

    assert reward.reward_fn(record) == 1.0

    record.tool_calls = [{"name": "submit", "arguments": {"status": "solved"}}]
    assert reward.reward_fn(record) == -0.5


def test_coding_cli_builds_interactive_task_from_args(tmp_path):
    code_cli = _load_module("code_cli")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('hi')\n", encoding="utf-8")
    args = SimpleNamespace(
        repo=str(repo),
        prompt="Fix the local app",
        test_command=["python -m pytest -q"],
        max_turns=3,
    )

    task = code_cli._task_from_args(args)
    workspace = code_cli._workspace_from_args(task, args)

    try:
        assert task["problem_statement"] == "Fix the local app"
        assert task["test_commands"] == ["python -m pytest -q"]
        assert workspace.read_file("app.py")["content"] == "1: print('hi')"
    finally:
        workspace.close()
    assert (repo / "app.py").exists()


def test_coding_cli_json_pretty_print_does_not_duplicate_colons():
    code_cli = _load_module("code_cli")
    colors = code_cli._Colors(enabled=True)

    rendered = code_cli._format_json({"path": ".", "max_depth": 1}, colors=colors)

    assert "::" not in rendered
    assert '"path"' in rendered


def test_coding_cli_auto_compacts_history_without_orphan_tool_messages():
    code_cli = _load_module("code_cli")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}],
        },
        {"role": "tool", "name": "read_file", "content": "x" * 1000},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "answer"},
    ]

    compacted = code_cli._compact_messages(messages, max_chars=200, keep_recent=2)

    assert compacted[0]["role"] == "system"
    assert compacted[1]["content"] == "initial task"
    assert compacted[2]["content"].startswith("Compacted prior conversation")
    assert compacted[3]["role"] != "tool"
    assert compacted[-1]["content"] == "answer"


def test_coding_cli_compaction_does_not_duplicate_head_when_keep_recent_is_large():
    code_cli = _load_module("code_cli")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "x" * 1000},
        {"role": "user", "content": "next"},
    ]

    compacted = code_cli._compact_messages(messages, max_chars=200, keep_recent=99)

    assert [message["content"] for message in compacted].count("system") == 1
    assert [message["content"] for message in compacted].count("initial task") == 1
    assert compacted[2]["content"].startswith("Compacted prior conversation")


def test_coding_loop_rejects_non_object_tool_arguments():
    agent_loop = _load_module("agent_loop")
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())

    result = agent_loop._execute_tool_call(
        workspace,
        {"id": "call_1", "function": {"name": "read_file", "arguments": json.dumps("README.md")}},
    )

    assert result == {"error": "tool arguments must be a JSON object"}


def test_coding_loop_normalizes_bad_tool_argument_quotes():
    agent_loop = _load_module("agent_loop")
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())

    read_by_bad_key = agent_loop._execute_tool_call(
        workspace,
        {
            "id": "call_1",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({'"path"': '"calculator.py"', '"start_line"': "1"}),
            },
        },
    )
    read_by_bad_value = agent_loop._execute_tool_call(
        workspace,
        {
            "id": "call_2",
            "function": {"name": "read_file", "arguments": json.dumps({"path": '"calculator.py"', "max_lines": "1}"})},
        },
    )
    search_by_bad_key = agent_loop._execute_tool_call(
        workspace,
        {
            "id": "call_3",
            "function": {
                "name": "rg",
                "arguments": json.dumps({'"path"': '".\\""', '"pattern"': '"return"', '{"case_sensitive"': False}),
            },
        },
    )

    assert read_by_bad_key["path"] == "calculator.py"
    assert read_by_bad_value["path"] == "calculator.py"
    assert read_by_bad_value["end_line"] == 1
    assert search_by_bad_key["matches"][0]["path"] == "calculator.py"


def test_coding_loop_write_file_preserves_escaped_content():
    agent_loop = _load_module("agent_loop")
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())
    content = '{"name": "areno", "items": ["{keep}", "\\"quoted\\""]}\n'
    try:
        result = agent_loop._execute_tool_call(
            workspace,
            {
                "id": "call_write",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({'"path"': '"data/config.json"', "content": json.dumps(content)}),
                },
            },
        )
        written = (workspace.root / "data" / "config.json").read_text(encoding="utf-8")
    finally:
        workspace.close()

    assert result["path"] == "data/config.json"
    assert written == content


def test_coding_loop_records_multiturn_trajectory_and_solves_task():
    agent_loop = _load_module("agent_loop")
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())
    item = SimpleNamespace(record=_add_task(), prompt="fix add", input_tokens=[1], prompt_index=0, sample_index=0)
    client = _FakeClient(
        [
            _tool_response("inspect_tree", {"path": ".", "max_depth": 2}),
            _tool_response("read_file", {"path": "calculator.py"}),
            _tool_response(
                "apply_patch",
                {
                    "patch": (
                        "--- a/calculator.py\n"
                        "+++ b/calculator.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def add(a, b):\n"
                        "-    return a - b\n"
                        "+    return a + b\n"
                    )
                },
            ),
            _tool_response("run_command", {"command": "python -m pytest test_calculator.py -q"}),
            _tool_response("submit", {"status": "solved", "summary": "tests pass"}),
        ]
    )
    try:
        messages, turns = asyncio.run(
            agent_loop.run_single_task(client=client, item=item, workspace=workspace, model="policy")
        )
    finally:
        workspace.close()

    assert len(turns) == 5
    assert workspace.submitted == {"status": "solved", "summary": "tests pass"}
    assert any(message.get("role") == "tool" and message.get("name") == "run_command" for message in messages)
    assert all(turn.item is item for turn in turns)


def test_coding_loop_plain_answer_requests_tool_call():
    agent_loop = _load_module("agent_loop")
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())
    item = SimpleNamespace(record=_add_task(), prompt="describe repo", input_tokens=[1], prompt_index=0, sample_index=0)
    client = _FakeClient([_text_response("This repository trains and serves AReno models.")])
    try:
        messages, turns = asyncio.run(
            agent_loop.run_single_task(client=client, item=item, workspace=workspace, model="policy", max_turns=1)
        )
    finally:
        workspace.close()

    assert len(turns) == 1
    assert workspace.submitted is None
    assert messages[-1]["role"] == "user"
    assert "did not include a tool call" in messages[-1]["content"]


def test_coding_loop_keeps_running_after_plain_answer_when_not_implicit_submit():
    agent_loop = _load_module("agent_loop")
    tools = _load_module("coding_tools")
    workspace = tools.CodingWorkspace.from_task(_add_task())
    item = SimpleNamespace(record=_add_task(), prompt="describe repo", input_tokens=[1], prompt_index=0, sample_index=0)
    messages = agent_loop.initial_messages(_add_task())
    client = _FakeClient(
        [
            _text_response("The repo contains a tiny calculator task."),
            _tool_response("submit", {"status": "solved", "summary": "reported repository contents"}),
        ]
    )
    try:
        turns = asyncio.run(
            agent_loop.run_conversation_turns(
                client=client,
                item=item,
                workspace=workspace,
                model="policy",
                messages=messages,
                max_turns=2,
            )
        )
    finally:
        workspace.close()

    assert len(turns) == 2
    assert any(
        message.get("role") == "user" and "did not include a tool call" in message.get("content", "")
        for message in messages
    )
    assert workspace.submitted == {"status": "solved", "summary": "reported repository contents"}


def _tool_response(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id=f"call_{name}",
                            type="function",
                            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
                        )
                    ],
                )
            )
        ],
        model_extra={"areno": {"response_tokens": [1, 2], "response_logprobs": [-0.1, -0.2]}},
    )


def _text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))],
        model_extra={"areno": {"response_tokens": [1, 2], "response_logprobs": [-0.1, -0.2]}},
    )


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._responses = list(responses)

    async def _create(self, **_):
        if not self._responses:
            raise AssertionError("unexpected model call")
        return self._responses.pop(0)
