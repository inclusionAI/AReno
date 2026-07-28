# Hanoi Agentic Example — Design Notes & Self-Review

This is a human-authored companion to the PR for issue #186 (Towers of Hanoi
agentic RL demo). It exists because the code was written with AI assistance,
and AReno's CONTRIBUTING.md is explicit that **a human submitter must
understand and defend each line**. The notes below state what each piece does,
why it was written that way, what it does *not* do, and how it was verified —
so a reviewer (and the submitter) can sanity-check the reasoning, not just the
diff.

## 1. How this maps to the issue

Issue #186 asks for a focused, independently reviewable Hanoi agentic demo that
reuses AReno's existing public contracts and adds no external DB / sandbox /
heavyweight dependency. Its acceptance criteria map to code as follows.

| Issue acceptance criterion | Where it lives in this PR |
| --- | --- |
| Deterministic fixtures | `dataset_generator.py` — 4 scripted scenarios × `n=3..6` = 16 JSONL records, fully reproducible (no RNG) |
| Oracle shortest-step calculator | `game.optimal_steps` (`2**n-1`), `game.optimal_solution` (recursive), `game.validate_solution` |
| Text trace replay | `game.serialize_trace` / `game.parse_trace` / `game.replay` (+ `ReplayResult.as_text` / `as_dict`) |
| Illegal-action coverage | `game.step` rejects `empty_source`, `larger_on_smaller`, `out_of_range`, `no_op`, `malformed`; `illegal_policy` supports `penalize` (default) and `terminate` |
| Completion rate + excess moves over optimum | `game.evaluate` returns `completion_rate`, `avg_excess_moves`, `oracle_steps` |
| Uses existing contracts; no DB/sandbox | `run_agent` uses `areno.api.agentic.AgentTrajectory` / `AgentTrajectoryTurn` + `ctx.get_base_url()` proxy; only optional runtime dep is `openai` (same as DuelGrid) |
| Default behaviour backward compatible | Pure opt-in: no AReno CLI flag / config / public API is added or changed; `illegal_action_policy` defaults to `penalize` |
| Focused CPU tests (success / invalid / boundary-failure) | `tests/test_agentic_hanoi_example_cpu.py` — 65 cases, `importlib`-loaded, no `areno` import, no CUDA build |
| User docs with runnable example + observable output | `README.md` (rules, `2**n-1`, metrics, input contract, defaults, output fields, limitations, copyable example incl. a boundary input) |

The implementation stays narrow on purpose: it adds one example directory and
one test file, and touches no file under `areno/`. That is the "keep the change
narrow; reuse existing contracts rather than introducing a parallel subsystem"
line from the issue.

## 2. Per-file self-review

### `game.py` — rules engine (no AReno dependency)
- **Why frozen dataclass state**: rollout scoring often needs "pretend this
  move, then discard". Immutable `HanoiState` makes branching safe and cheap,
  and mirrors DuelGrid's `@dataclass(frozen=True)`.
- **Why `step` returns `(state, reward, done, info)`**: Gym/duelgrid convention;
  `info` carries `illegal`/`reason` so failures identify the affected move
  (issue: "identify the affected stage and input").
- **Why `illegal_policy="penalize"` by default**: a single bad move should not
  waste an episode; small penalty + unchanged state lets the policy keep
  exploring. `terminate` is opt-in. This is also the "safe default" the issue
  asks for. See the step docstring.
- **Why fixtures are scripted, not random**: Hanoi has a single canonical start,
  so a RNG gives no variety. The four scenarios are deliberately constructed to
  cover the issue's illegal/boundary/failure surface.
- **Test coverage**: `test_*_optimal_steps*`, `test_*_legal_move*`,
  `test_empty_source_rejected`, `test_larger_on_smaller_rejected`,
  `test_out_of_range_and_noop_rejected`, `test_malformed_action_rejected`,
  `test_illegal_terminate_policy_ends_episode`, `test_replay_*`,
  `test_evaluate_*`, `test_deterministic_output_same_inputs`,
  `test_default_illegal_policy_is_penalize`.

### `dataset_generator.py` — fixtures
- **Why `expected` is computed, not hand-written**: `expected` is filled by
  running the real `game.replay(trace, n)` and storing its output. Tests then
  only assert `expected == fresh_replay`, so there is no second oracle to keep
  in sync. See `_record_for_scenario`.
- **Why `count`/`seed` exist despite no RNG**: signature parity with DuelGrid's
  `generate_records(count, seed)` so CLI/test conventions line up; `seed` is
  deliberately ignored (`del seed`) — output is byte-identical regardless.
- **`_failure_trace` repeat count**: `2*(2**n-1)` stays under `max_moves`
  (`max(64, 4*2**n)`) and yields a clearly non-empty, never-completing trace.
- **Test coverage**: `test_default_fixture_count_*`, `test_count_truncates_*`,
  `test_every_record_has_required_fields_*`, `test_optimal_moves_are_valid_*`,
  `test_record_to_state_roundtrips_*`, `test_expected_outcomes_match_actual_replay`
  (parametrized over all 4 scenarios), per-scenario outcome tests.

### `dataset_loader.py` — JSONL → Areno prompt records
- **Why split from the generator**: mirrors DuelGrid — generation is offline
  fixture production, loading is the training-time step that builds the
  `prompt` text and best-action hint.
- **Why `best_action = optimal_moves[0]`**: Hanoi has a true optimum (unlike
  DuelGrid's heuristic baseline), so the hint is the genuine first optimal move.
- **Why swallow `default_loader`/`**_`**: AReno's CLI may inject a default
  loader or extra kwargs; absorbing them keeps the loader drop-in compatible
  without asserting on a specific call signature. See docstring.
- **Test coverage**: `test_load_training_dataset_returns_prompt_records`,
  `test_best_action_is_first_optimal_move`, `test_prompt_matches_game_format_prompt`,
  `test_load_from_explicit_file_path`, `test_missing_dataset_raises_with_hint`,
  `test_default_loader_arg_accepted_and_ignored`, `test_roundtrip_generator_to_loader_in_memory`,
  `test_loader_records_are_json_serializable`.

### `reward.py` — `reward_fn(record)`
- **Reading of the issue (lenient, chosen)**: the issue says "score completion
  with a small efficiency component relative to the known optimum". Read
  strictly that is 0-unsolved; this PR adopts the **lenient** reading —
  completion is still the dominant signal, but a small partial credit is given
  for legal moves on unsolved traces. The lenient choice is driven by a real
  training failure: under strict-sparse, a weak base model that cannot solve in
  one shot leaves every GSPO group at all-zero reward → zero variance → zero
  advantage → zero gradient (observed empirically: steps 0–7 all
  `grad_zero_ratio=1.0`). The partial credit gives GSPO non-zero variance so
  training can move. See the module docstring for the strict-vs-lenient trade.
- **Solved**: `COMPLETION_REWARD - EXCESS_STEP_PENALTY * excess` =
  `1.0 - 0.02 * (actual - 2**n-1)` — completion-led, efficiency is the small
  component, relative to the known optimum (exactly the issue phrase).
- **Unsolved**: `min(LEGAL_STEP_BONUS * legal_count + ILLEGAL_STEP_PENALTY *
  illegal_count, PARTIAL_CREDIT_CAP)` ≈ `min(0.02*legal - 0.05*illegal, 0.5)`.
  The **cap is essential** — without it a long legal-but-unsolved trace (the
  failure fixture oscillating for `2*(2**n-1)` steps) would accumulate >1.0
  and perversely reward *not* solving. The cap (0.5, strictly < 1.0) guarantees
  completing is always the globally optimal choice; the partial credit only
  exists to differ across samples. This invariant is enforced by tests.
- **Why the fallback to `completion` text**: if a model does not produce a
  `move_disk` tool call, we still try to parse a JSON move list from its text
  so reward is best-effort rather than crashing; the wrong-tool-name path is
  exercised by `test_wrong_tool_name_ignored_falls_back_to_completion`.
- **Test coverage**: `test_optimal_solution_rewards_full_score` (parametrized
  n=3..6), `test_longer_legal_solution_scores_lower`,
  `test_failure_sequence_scores_partial_credit` (unsolved = formula, < completion),
  `test_empty_moves_scores_zero`, `test_string_arguments_are_parsed`,
  `test_malformed_arguments_score_zero_not_crash`,
  `test_reward_matches_replay_outcome_for_every_fixture` (asserts solved ≤ 1.0
  and unsolved < 1.0 across all 16 fixtures — pins the "completion is dominant"
  invariant).

### `run_agent.py` — `async run_agent(ctx, batch)`
- **Why single-turn full solution**: one `move_disk` call carrying the whole
  move list, matching DuelGrid's single-call shape — short trajectory, one
  policy span to score, and `reward.py` scores the complete sequence at once.
  Multi-turn is a follow-up, not required by the issue. See module docstring.
- **Why `tool_choice` forces `move_disk`**: weak base models ramble in natural
  language and never emit the tool call; forcing it maximises the chance of a
  scoreable trajectory. The proxy backing the in-training policy honours
  `tool_choice`.
- **Why `model="policy"`**: routes the request to the in-training policy through
  `ctx.get_base_url()`'s local OpenAI-compatible proxy (same as DuelGrid).
- **Why the tuple-form schema (`prefixItems` + `items:false`)**: enforces "each
  move is exactly two ints" where supported; servers that ignore JSON Schema
  2020-12 are fine because `reward.py` re-validates every move in the engine.
  Schema validation is convenience, not a correctness boundary. See comment.
- **Not unit-tested in the CPU suite**: it imports `areno.api.agentic` and
  `openai`, so it cannot be loaded by the `importlib` CPU tests; this matches
  DuelGrid's `run_agent.py`, which is likewise exercised only at training time.

## 3. Run records (evidence, not claims)

### 3.1 CPU tests
Command (per CONTRIBUTING step 6):
```
pytest tests/ -k cpu -q tests/test_agentic_hanoi_example_cpu.py
```
Result (Python 3.9.6, CPU only):
```
collected 65 items
tests/test_agentic_hanoi_example_cpu.py ................................ [ 49%]
.................................                                        [100%]
============================== 65 passed in 0.47s ==============================
```

### 3.2 Lint / format (pre-commit parity)
```
ruff check examples/agentic/hanoi/ tests/test_agentic_hanoi_example_cpu.py   # All checks passed!
ruff format --check examples/agentic/hanoi/ tests/test_agentic_hanoi_example_cpu.py   # already formatted
```
Repo ruff config used: `target-version=py310`, `line-length=120`,
`select=E,F,W,I,UP`, `ignore=E501`.

### 3.3 Training smoke (Kaggle, 2× Tesla T4, GSPO)
Config printed correctly (ckpt=Qwen3.5-0.8B, dataset=hanoi_fixtures.jsonl,
loader/reward/agent all pointing at `examples/agentic/hanoi/`, `attn_backend=native`
fallback because T4 cc7.5 lacks flash-attn/BF16 — non-fatal). The agentic path
loaded and ran:
```
agentic rollout batch prompts=2 n_samples=4 expected_requests=8 max_running_prompts=8
Hanoi agent start requests=8
agentic train batch built samples=8 tokens=8105 messages=24 tool_calls=5 tool_results=0
```
So `run_agent.py` was invoked, the model produced `move_disk` tool calls, and
`reward.py` scored them — the end-to-end agentic chain works.

One full epoch (steps 0–7) of `train_stats`:
```
step  reward_mean  advantage_mean  grad_norm  grad_zero_ratio
 0      0.000000      0.0            0.0        1.0
 1      0.000000      0.0            0.0        1.0
 2      0.000000      0.0            0.0        1.0
 3      0.000000      0.0            0.0        1.0
 4      0.000000      0.0            0.0        1.0   (tool_calls=8/8, still 0)
 5      0.000000      0.0            0.0        1.0
 6      0.000000      0.0            0.0        1.0
 7      0.000000      0.0            0.0        1.0
```
> Note: the run above used an earlier **strict-sparse** reward build. It
> reproduces exactly the cold-start stall that motivated the lenient reward in
> this PR: with completion-only reward, a base model that cannot solve in one
> shot leaves every GSPO group at all-zero reward, so advantage/gradient stay
> 0. The committed (lenient) reward is expected to break this stall by giving
> unsolved traces a small partial credit — i.e. `reward_mean` / `advantage_mean`
> / `grad_norm` should become non-zero from step 0.

## 4. Known limitations (stated honestly)

1. **Cold-start reward stall — mitigated, not eliminated.** Under a strictly
   sparse reward a weak base model (Qwen3.5-0.8B) cannot solve Hanoi in one
   shot, so every GSPO group scores all-zero → zero advantage → zero gradient
   → no update (observed: steps 0–7 `grad_zero_ratio=1.0`). This PR mitigates
   it with the lenient partial-credit reward (unsolved traces get a small,
   capped signal so groups have variance); whether that is *enough* for the
   model to actually learn is a training-experiment question, not a demo-
   correctness one. AReno also has no early-stop today, so a stalled run must
   be bounded with `--max-steps`. The stall itself is the scenario H-version
   #203/#239 target. Demo correctness is fixed by the 65 CPU tests and the
   oracle/replay fixtures, independent of training outcome.
2. **`run_agent.py` has no CPU unit test.** It imports `areno` and `openai`, so
   it is only exercised at training time — consistent with DuelGrid's
   `run_agent.py`.
3. **Only 16 fixtures.** Scripted scenarios are great for verification but
   small for a long training run; a random-legal-trace generator is a natural
   follow-up, not required by the issue.

## 5. What I would change if a reviewer asked

- Add a multi-turn `run_agent` variant (move → observe → move) if the
  single-turn full-solution shape is deemed too far from "expose
  `move(source, target)`" in the issue. The current shape is a deliberate,
  documented choice (one policy span, one engine-scored sequence).
- Promote the dense reward to an opt-in flag behind `--reward-fn-path` if the
  team wants a training-friendly default while keeping the sparse version as the
  issue-faithful baseline.