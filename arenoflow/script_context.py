"""Repository demonstrations supplied to the script generator, never executed locally."""

from arenoflow.catalog import ROOT

DEMONSTRATIONS = (
    {
        "name": "math",
        "purpose": "GSPO/GRPO math: normalize question/answer into prompt and solutions; score record.answer against record.completion. Standard math rollout does not require an agent script.",
        "sample": {"question": "What is 2 + 3?", "answer": "2 + 3 = 5. #### 5"},
        "files": ("examples/math/dataset_loader.py", "examples/math/math_verify_reward.py"),
    },
    {
        "name": "tictactoe",
        "purpose": "Agentic GSPO/GRPO: preserve board metadata, request a legal choose_square tool call, return AgentTrajectory turns, and score the move from record.source_record and record.tool_calls. Game rules and helpers are included below.",
        "sample": {"id": "board-1", "board": [["X", "O", "."], [".", "X", "."], ["O", ".", "."]]},
        "files": (
            "examples/agentic/tictactoe/dataset_loader.py",
            "examples/agentic/tictactoe/run_agent.py",
            "examples/agentic/tictactoe/reward.py",
            "examples/agentic/tictactoe/game.py",
            "examples/agentic/tictactoe/dataset_generator.py",
        ),
    },
    {
        "name": "sft",
        "purpose": "SFT: normalize Alpaca instruction/input/output rows into prompt/response records. No reward function or agent rollout is used for SFT. DPO instead needs chosen/rejected responses; do not apply the SFT output schema to DPO.",
        "sample": {"instruction": "Translate into English.", "input": "你好", "output": "Hello."},
        "files": ("examples/sft/alpaca/dataset_loader.py",),
    },
)


def demonstration_context(root=ROOT):
    """Include complete files so imports, helpers and entrypoints remain reviewable."""
    return [
        {
            "name": demo["name"],
            "purpose": demo["purpose"],
            "sample": demo["sample"],
            "files": [{"reference": path, "source": (root / path).read_text()} for path in demo["files"]],
        }
        for demo in DEMONSTRATIONS
    ]
