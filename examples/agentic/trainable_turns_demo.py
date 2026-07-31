"""Demonstrate trainable-turn selection modes for agentic trajectories.

This script shows how ``trainable_turns`` and ``mask_tool_call_args`` affect
the loss mask on a fixed multi-tool transcript. It runs entirely on CPU with
fake tokens — no model, GPU, or network required.

Usage::

    python examples/agentic/trainable_turns_demo.py

The output prints the loss mask for each mode so you can verify which tokens
contribute to policy loss.
"""

from types import SimpleNamespace

from areno.api.agentic import LossMaskPolicy, ResponseSpan, RolloutSession, _AgentSample


def _make_session(mode: str, mask_args: bool = False) -> RolloutSession:
    return RolloutSession(
        None,
        sampling_params=None,
        loss_mask_policy=LossMaskPolicy(trainable_turns=mode, mask_tool_call_args=mask_args),
    )


def _make_sample(session: RolloutSession) -> _AgentSample:
    """Build a fixed 3-span trajectory: text -> tool_call -> text(after tool result).

    Prompt: [1, 2]
    Response: [10, 11, 20, 21, 22, 30, 31]
      span 0: assistant_text  [10, 11]           (turn 0, "let me search")
      span 1: assistant_tool_call [20, 21, 22]   (turn 1, tool call)
      span 2: assistant_text  [30, 31]            (turn 2, "the answer is 42")
    """
    item = SimpleNamespace(
        record={}, prompt="p", input_tokens=[1, 2], prompt_index=0, sample_index=0
    )
    sample = _AgentSample(
        item=item,
        messages=[
            {"role": "user", "content": "what is the answer"},
            {"role": "assistant", "content": "let me search"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "the answer is 42"},
        ],
        response_text="let me search\nthe answer is 42",
        last_response_text="the answer is 42",
        response_tokens=[10, 11, 20, 21, 22, 30, 31],
        response_logprobs=[0.0] * 7,
        trace=[],
        response_kind="assistant_text",
        response_spans=[
            ResponseSpan(kind="assistant_text", length=2),
            ResponseSpan(kind="assistant_tool_call", length=3),
            ResponseSpan(kind="assistant_text", length=2),
        ],
    )
    session._set_sample_training_row(sample, item.input_tokens)
    return sample


def _format_mask(mask: list[bool]) -> str:
    return "[" + ", ".join("T" if m else "." for m in mask) + "]"


def main():
    transcript = """
Fixed transcript:
  prompt:     [1, 2]
  span 0:     assistant_text     [10, 11]           turn 0
  span 1:     assistant_tool_call [20, 21, 22]      turn 1
  span 2:     assistant_text     [30, 31]            turn 2 (after tool result)
Token layout: [1, 2, 10, 11, 20, 21, 22, 30, 31]
               pp  ^span0^  ^--span1--^  ^span2^
"""
    print(transcript)

    modes = [
        ("all_assistant", False),
        ("last_assistant", False),
        ("final_answer", False),
    ]

    for mode, mask_args in modes:
        session = _make_session(mode, mask_args)
        sample = _make_sample(session)
        rows = session._train_rows_from_samples([sample])
        loss_mask = rows.loss_masks[0]
        print(f"trainable_turns={mode:<16s}  loss_mask={_format_mask(loss_mask)}")
        print(f"  trainable_tokens={rows.trainable_tokens}  masked_response_tokens={rows.masked_response_tokens}")
        print()

    # Invalid mode demonstration
    print("Invalid mode validation:")
    try:
        from areno.api.trainer_config import TrainerConfig

        TrainerConfig(algo="sft", ckpt="demo", dataset_path="demo", trainable_turns="bogus")
        print("  ERROR: should have raised ValueError")
    except ValueError as exc:
        print(f"  ValueError: {exc}")


if __name__ == "__main__":
    main()