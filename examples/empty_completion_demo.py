"""Minimal runnable demo for completion validation (Issue #240).

Run without GPU or model loading:

    python examples/empty_completion_demo.py

The demo constructs fake completions (including all four invalid types),
runs them through ``validate_completions``, and prints the classification,
filtering result, metrics, and quarantine records.
"""

from __future__ import annotations

from areno.engine.runtime.completion_validator import (
    classify_completion,
    validate_completions,
)


def main() -> None:
    # Simulate a batch of 8 completions from one prompt (n_samples=8).
    # The EOS token id is 2; special token ids are {2, 100, 101}.
    eos_token_ids = (2,)
    special_token_ids = (2, 100, 101)

    completions = [
        "The answer is 42.",        # 0: valid
        "",                          # 1: empty string
        "   \n  ",                   # 2: whitespace-only
        "<|endoftext|>",             # 3: immediate EOS (single token id=2)
        "<|im_start|><|im_end|>",    # 4: special-token-only (ids=100,101)
        "The answer is 17.",        # 5: valid
        "",                          # 6: empty (no response tokens)
        "The answer is 9.",         # 7: valid
    ]
    resp_tokens = [
        [10, 20, 30],          # valid
        [],                     # empty
        [40, 41],              # whitespace
        [2],                   # immediate EOS
        [100, 101],            # special tokens
        [11, 21, 31],          # valid
        [],                     # empty
        [12, 22, 32],          # valid
    ]

    print("=== Step 1: Classify each completion ===")
    for i, (text, tokens) in enumerate(zip(completions, resp_tokens, strict=True)):
        check = classify_completion(
            completion=text,
            resp_tokens=tokens,
            eos_token_ids=eos_token_ids,
            special_token_ids=special_token_ids,
        )
        status = "VALID" if check.is_valid else f"INVALID ({check.invalid_type})"
        preview = text[:40] if text else "(empty)"
        print(f"  [{i}] {status:25s} completion={preview!r}")

    print("\n=== Step 2: Filter with policy='off' (default, no-op) ===")
    kept, _, vr = validate_completions(
        completions, resp_tokens,
        policy="off",
        eos_token_ids=eos_token_ids,
        special_token_ids=special_token_ids,
    )
    print(f"  kept: {len(kept)} / {len(completions)}")
    print(f"  dropped: {vr.dropped_indices}")
    print(f"  metrics: {vr.metrics}")

    print("\n=== Step 3: Filter with policy='filter' ===")
    kept, _, vr = validate_completions(
        completions, resp_tokens,
        policy="filter",
        eos_token_ids=eos_token_ids,
        special_token_ids=special_token_ids,
        prompt="What is the meaning of life?",
    )
    print(f"  kept: {len(kept)} / {len(completions)}")
    print(f"  kept_indices: {vr.kept_indices}")
    print(f"  dropped_indices: {vr.dropped_indices}")
    print(f"  metrics: {vr.metrics}")

    print("\n=== Step 4: Quarantine records ===")
    for record in vr.quarantine_records:
        print(f"  index={record['index']} type={record['invalid_type']} "
              f"tokens={record['resp_token_count']} policy={record['policy']}")

    print("\n=== Step 5: All-invalid boundary case ===")
    kept, _, vr = validate_completions(
        ["", "  "], [[], [40]],
        policy="filter",
        eos_token_ids=eos_token_ids,
        special_token_ids=special_token_ids,
    )
    print(f"  kept: {len(kept)}")
    print(f"  dropped: {vr.dropped_indices}")
    print(f"  metrics: {vr.metrics}")

    print("\nDone. Invalid completions never reach reward_fn or training code.")


if __name__ == "__main__":
    main()
