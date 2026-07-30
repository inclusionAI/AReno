"""Deterministic demo of configurable trainable-turn loss masking.

No network, no sandbox, no GPU: builds an in-memory multi-tool trajectory with a
fake tokenizer and prints the per-token loss mask under each ``trainable_turns``
mode, plus one illegal-input (call without result) rejection.

This is a runnable reference for the ``--trainable-turns`` / ``--mask-tool-call-args``
options (see docs/cli/training.rst) and a fixture for the trainable-token
ablation statistics (issue #199). Run from a source checkout with AReno installed:

    python examples/agentic/trainable_turns_demo.py
"""

from __future__ import annotations

from areno.api.agentic import LossMaskPolicy, ResponseSpan, RolloutSession


class _FakeTokenizer:
    """Decodes token ids to their string form; encode is length-based."""

    def encode(self, text):
        return [len(text)]

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


class _FakeTrainer:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def dp_size(self):
        return 1


class _FakeSamplingParams:
    greedy = False
    max_new_tokens = 8
    temperature = 0.0
    top_p = 1.0
    top_k = -1
    stop_token_ids = None
    ignore_eos = False
    skip_special_tokens = True
    max_prompt_len = None
    max_context_len = None

    def model_copy(self):
        return _FakeSamplingParams()


def _spanned_sample(tokens, spans, base_mask):
    """Build a minimal _AgentSample with explicit response spans and base mask."""
    from areno.api.agentic import AgentBatch, RewardEvent, _AgentSample

    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    sample = _AgentSample(
        item=item,
        messages=[],
        response_text="",
        last_response_text="",
        response_tokens=list(tokens),
        response_logprobs=[0.0] * len(tokens),
        trace=[RewardEvent(type="assistant_text", text="")],
    )
    sample.response_spans = list(spans)
    sample.loss_mask_override = list(base_mask)
    return sample


def main() -> None:
    session = RolloutSession(_FakeTrainer(), sampling_params=_FakeSamplingParams(), loss_mask_policy=LossMaskPolicy())
    # Trajectory: assistant_text(2) -> tool_call(2) -> [tool_result] -> assistant_text(2)
    spans = [
        ResponseSpan("assistant_text", 2),
        ResponseSpan("assistant_tool_call", 2),
        ResponseSpan("assistant_text", 2),
    ]
    tokens = [10, 11, 12, 13, 20, 21]
    base = [True] * len(tokens)

    print("Trajectory: assistant_text(2) | tool_call(2) | assistant_text(2)")
    print(f"tokens: {tokens}\n")
    for mode in ("all_assistant", "last_assistant", "final_answer"):
        sess = RolloutSession(
            _FakeTrainer(),
            sampling_params=_FakeSamplingParams(),
            loss_mask_policy=LossMaskPolicy(trainable_turns=mode),
        )
        sample = _spanned_sample(tokens, spans, base)
        sess._apply_trainable_turn_mode(sample)
        trainable = sum(sample.loss_mask_override)
        print(f"  {mode:16s} loss_mask={sample.loss_mask_override}  trainable_tokens={trainable}")

    # Illegal input: a mid-trajectory tool call whose result never arrives.
    from areno.api.agentic import RewardEvent

    bad = _spanned_sample(
        [10, 11, 20], [ResponseSpan("assistant_tool_call", 1), ResponseSpan("assistant_text", 2)], [True] * 3
    )
    bad.trace = [
        RewardEvent(type="assistant_tool_call", name="search", arguments="{}"),
        RewardEvent(type="assistant_text", text="ok"),
    ]
    bad.messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
    try:
        session._validate_call_result_pairing([bad])
        print("\nvalidation: unexpectedly accepted malformed trajectory")
    except ValueError as exc:
        print(f"\nvalidation rejected malformed trajectory: {exc}")


if __name__ == "__main__":
    main()
