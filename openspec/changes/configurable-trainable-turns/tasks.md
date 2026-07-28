## 1. Phase 0 — Maintainer alignment (ask-first gate)

- [x] 1.1 Post English plan comment to issue inclusionAI/AReno#199: mode names (`all_assistant`/`last_assistant`/`final_answer`), field names (`trainable_turns`/`mask_tool_call_args`), whether `mask_tool_call_args` is a standalone option, and the `final_assistant_text` dead-flag deprecation.
- [x] 1.2 Sync field naming to maintainer feedback; if renamed, replace across proposal.md/design.md/spec.md. — no rename requested; names retained.
- [x] 1.3 Resolve open question: whether core-logic Phase 2 may proceed before full maintainer ack (AGENTS.md ask-first list excludes `agentic.py`, but issue-199 plan Phase 0 gates everything). — user is assignee; config/CLI authorized alongside core logic.
- verify: maintainer explicit ack or assignment on #199. ✔ user is assignee.

## 2. Phase 1 — Core logic (`areno/api/agentic.py`, CPU-verifiable, not ask-first)

- [x] 2.1 Define `LossSelectionMode = Literal["all_assistant","last_assistant","final_answer"]` and add `trainable_turns: LossSelectionMode = "all_assistant"` + `mask_tool_call_args: bool = False` to `LossMaskPolicy` (L46-55); remove dead `final_assistant_text` (L53).
- [x] 2.2 Add `ResponseSpan` dataclass (`kind`, `length`) and `response_spans: list[ResponseSpan]` field to `_AgentSample` (L151-167); initialize in `_set_sample_training_row` (L668) as `[(response_kind, len(response_tokens))]`.
- [x] 2.3 In `_append_sample_response` (L627), append `(new_sample.response_kind, len(new_sample.response_tokens))` to `response_spans` when extending `response_tokens` (L640).
- [x] 2.4 Implement `_apply_trainable_turn_mode(sample)` with composition order: start from existing per-span mask (incl. `_tool_call_loss_mask` result suppression) → apply `mask_tool_call_args` (tool_call span arg sub-region narrowed to False, name/action kept) → apply turn-selection (all_assistant=no-op / last_assistant=only final assistant span / final_answer=only post-last-tool-result assistant_text span, degenerate to last span when no tool result). Write back `loss_mask_row` (response region via span offsets) + sync `loss_mask_override`.
- [x] 2.5 Hook `_apply_trainable_turn_mode` at top of `_train_rows_from_samples` loop (L393), the single chokepoint covering both HTTP-proxy (L607) and trajectory (L627) paths.
- [x] 2.6 Implement tool-call arg sub-region localization (chat-template boundary + tokenizer align; do NOT reuse `_tool_call_loss_mask` sentinel logic); mark as approximate.
- [x] 2.7 Implement call/result pairing validation function; call before `_train_rows_from_samples` (or after sample assembly in `_run_agentic_rollout`). tool_call without tool_result → `ValueError`; tool_result without preceding tool_call → `ValueError`; explicitly exclude empty `response_tokens`.
- verify: `pytest tests/test_agentic_cpu.py -k cpu` green with new cases. ✔ 64 passed.

## 3. Phase 2 — Config + CLI (ask-first, after 1.x ack)

- [x] 3.1 Add `trainable_turns: str = "all_assistant"` + `mask_tool_call_args: bool = False` to `TrainerConfig` (`trainer_config.py`); validate literal set in `__post_init__` (L64) → `ValueError`.
- [x] 3.2 Update `policy_only._loss_mask_policy()` (L179) to map `trainable_turns`/`mask_tool_call_args` from config → `LossMaskPolicy`; reuse existing `RolloutSession(loss_mask_policy=...)` path, no new trainer surface.
- [x] 3.3 Add `--trainable-turns` / `--mask-tool-call-args` to `cli/train.py` Rollout section + config summary; invalid mode → `click.UsageError` (early failure).
- verify: CLI `--help` shows options; invalid value fails early; defaults equivalent to current behavior. ✔ both options VISIBLE; invalid `--trainable-turns bogus` exits 2 with Usage error.

## 4. Phase 3 — Observability

- [x] 4.1 Emit per-batch metrics `trainable_tokens = sum(loss_mask)` and `masked_response_tokens = sum(response_mask) - sum(loss_mask)`.
- [x] 4.2 Log active `trainable_turns` mode + `mask_tool_call_args` state in rollout log.
- verify: CPU test asserts metric field values. ✔ `test_trainable_turns_last_assistant_full_row_and_metrics` + `test_trainable_turns_all_assistant_full_row_default_parity` + `test_rollout_log_records_active_mode_and_mask_state` (policy fields).

## 5. Phase 4 — Tests (`tests/test_agentic_cpu.py`)

- [x] 5.1 Fix multi-tool transcript fixture: `assistant_text → tool_call → tool_result → assistant_text`.
- [x] 5.2 Per-token `loss_mask` assertions for three modes × `mask_tool_call_args` on/off.
- [x] 5.3 Illegal input (missing tool result) → turn-level error; boundary (empty final answer / all tool_call / trailing bare tool_call zero-signal); default-off asserts pre-change parity.
- [x] 5.4 Assert `trainable_tokens` / `masked_response_tokens` numeric values.
- [x] 5.5 Dual-path coverage: explicit-trajectory path + HTTP-proxy path both masked correctly. — `test_explicit_trajectory_path_last_assistant_masks_correctly` (AgentTrajectoryTurn path) + `test_trainable_turns_last_assistant_full_row_and_metrics` (proxy path via `_pending_chat`).
- verify: `pytest tests/ -k cpu` green. ✔ 64 passed (21 new + 43 regression).

## 6. Phase 5 — Docs + example

- [x] 6.1 Add `--trainable-turns final_answer` copyable example + contract/default/output/limitation note to CLI guide and skills/troubleshooting. — `docs/cli/training.rst` + `docs/cli/observability.rst` log example updated.
- [x] 6.2 Update `docs/sdk/trainer.rst` (L443/450) to remove `final_assistant_text` declaration; document new `LossMaskPolicy` fields.
- [x] 6.3 Add minimal deterministic, no-network, no-sandbox fixture under `examples/` (incl. one illegal-input case). — `examples/agentic/trainable_turns_demo.py`. NOTE: requires AReno installed (`import areno`); not runnable in a bare CPU venv without `pip install -e .` (which needs CUDA). Runs offline once AReno is installed.
- verify: example runs offline; docs build clean. — docs edits are rst text; example import-verified structure, runtime needs installed AReno.

## 7. Phase 6 — Research extension (G8, compute-dependent)

- [ ] 7.1 Script: per same trajectory batch, count trainable_tokens across three modes (CPU-runnable). — demo fixture (`examples/agentic/trainable_turns_demo.py`) prints per-mode trainable_tokens; standalone ablation script deferred.
- [ ] 7.2 (if compute available) small-model + few-step ablation: compare convergence trend / final reward across modes. — deferred (needs GPU).
- [ ] 7.3 Short blog / appendix for application material. — deferred.
- verify: linkable artifact (blog URL or PDF); if no compute, stop at 7.1 and label scope honestly.