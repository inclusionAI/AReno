## ADDED Requirements

### Requirement: Trainable-turn selection modes

The system SHALL support three trainable-turn selection modes for agentic trajectories, applied at token granularity: `all_assistant`, `last_assistant`, and `final_answer`. The active mode SHALL be selected via `LossMaskPolicy.trainable_turns` and surfaced through `TrainerConfig.trainable_turns` plus the `--trainable-turns` CLI option. `last_assistant` SHALL mark only the final assistant span (regardless of kind) as trainable; all prior assistant spans SHALL be masked to zero. `final_answer` SHALL mark only the assistant_text span following the last tool result as trainable; when no tool result exists in the trajectory, it SHALL degenerate to the last assistant span. Mode evaluation SHALL happen at the trajectory level after span assembly is complete, not within the per-span `_response_loss_mask_for_span` function.

#### Scenario: all_assistant keeps all assistant spans trainable
- **WHEN** a multi-tool trajectory `assistant_text → tool_call → tool_result → assistant_text` is trained with `trainable_turns=all_assistant`
- **THEN** every assistant span's tokens retain their existing loss mask bits (behavior identical to pre-change default)

#### Scenario: last_assistant masks all but the final assistant span
- **WHEN** the same trajectory is trained with `trainable_turns=last_assistant`
- **THEN** only the final assistant span tokens are trainable; the first assistant_text span tokens are masked to zero

#### Scenario: final_answer targets the post-tool-result text
- **WHEN** the trajectory is trained with `trainable_turns=final_answer`
- **THEN** only the final assistant_text span (after the last tool result) is trainable; the first assistant_text and the tool_call span are masked to zero

#### Scenario: final_answer degenerates when no tool result exists
- **WHEN** a single assistant_text trajectory with no tool result is trained with `trainable_turns=final_answer`
- **THEN** that single assistant span is trainable (degenerates to last assistant span)

#### Scenario: final_answer with bare trailing tool_call yields zero trainable signal
- **WHEN** a trajectory ending in a tool_call span with no subsequent assistant_text is trained with `trainable_turns=final_answer`
- **THEN** no tokens are trainable (zero trainable signal); no error is raised

### Requirement: Tool-call argument masking

The system SHALL provide a `mask_tool_call_args` option (via `LossMaskPolicy.mask_tool_call_args`, `TrainerConfig.mask_tool_call_args`, and `--mask-tool-call-args` CLI flag). When enabled, within a tool_call span the JSON argument tokens SHALL be masked to zero while tool-name/action tokens remain trainable. Argument-span localization SHALL be treated as approximate (decode/encode non-round-trip) and SHALL NOT reuse the existing `_tool_call_loss_mask` result-region sentinel logic. The argument mask SHALL be composed on top of the existing per-span mask (including any `_tool_call_loss_mask` result-region suppression), never rebuilt from zero.

#### Scenario: tool-call args masked, name tokens kept
- **WHEN** a tool_call turn is trained with `mask_tool_call_args=True` and `trainable_turns=all_assistant`
- **THEN** the tool-name/action tokens within the tool_call span remain trainable; the JSON argument tokens are masked to zero; tool-result region suppression already applied by `_tool_call_loss_mask` is preserved

#### Scenario: disabled by default
- **WHEN** `mask_tool_call_args` is not set
- **THEN** the full tool_call span is trainable per the existing `assistant_tool_calls` policy bit

### Requirement: Dual-path coverage

The trainable-turn and argument-masking logic SHALL apply uniformly to both the HTTP-proxy path (`_sample_from_pending_chat`) and the explicit-trajectory path (`_append_sample_response`), by performing the trajectory-level rewrite at the single chokepoint `_train_rows_from_samples` that both paths flow through.

#### Scenario: explicit-trajectory path masked correctly
- **WHEN** an explicit `AgentTrajectory` with multiple turns is materialized and trained with `trainable_turns=last_assistant`
- **THEN** the loss mask matches the span-level expectation (only final assistant span trainable)

#### Scenario: HTTP-proxy path masked correctly
- **WHEN** samples are built via the HTTP proxy completion path and trained with `trainable_turns=last_assistant`
- **THEN** the loss mask matches the same span-level expectation as the trajectory path

### Requirement: Call/result pairing validation

The system SHALL validate tool-call/tool-result pairing at the turn level before worker initialization by pairing `assistant_tool_call` trace events against `tool` messages in the conversation history. A mid-trajectory tool call without a matching tool result SHALL raise `ValueError`. A trajectory ending in a bare tool call (final agent action, result never received) SHALL be allowed (the trailing call is exempt so `final_answer` can yield zero trainable signal without error). Orphan tool results (tool messages without a structured call) SHALL be tolerated, because the agentic data path permits tool messages as environment context (existing multi-turn fixtures rely on this). An empty `response_tokens` SHALL NOT be treated as invalid (the `_run_chat_request` empty fallback is a legal path).

#### Scenario: mid-trajectory tool call without result rejected
- **WHEN** a trajectory contains a mid-trajectory assistant tool_call turn whose result never arrives before a later assistant turn
- **THEN** a `ValueError` is raised before training begins

#### Scenario: bare trailing tool call allowed
- **WHEN** a trajectory ends in a tool_call turn with no subsequent tool result
- **THEN** no validation error is raised (the trailing call is exempt)

#### Scenario: orphan tool result tolerated
- **WHEN** a trajectory contains a tool result message with no structured preceding tool_call
- **THEN** no validation error is raised (tolerated as environment context)

#### Scenario: empty response tokens accepted
- **WHEN** a turn produces empty `response_tokens` via the empty fallback path
- **THEN** no validation error is raised

### Requirement: Default backward compatibility

The default `trainable_turns` SHALL be `all_assistant` and `mask_tool_call_args` SHALL default to `False`. With defaults, the loss mask for any trajectory SHALL be identical to the pre-change behavior.

#### Scenario: defaults reproduce prior behavior
- **WHEN** a trajectory is trained with no `trainable_turns` / `mask_tool_call_args` overrides
- **THEN** the per-token loss mask is identical to the pre-change implementation output

### Requirement: Early failure on invalid configuration

`TrainerConfig.__post_init__` SHALL reject `trainable_turns` values outside the literal set with `ValueError`. The CLI SHALL reject invalid `--trainable-turns` values with `click.UsageError` before any rollout begins.

#### Scenario: invalid mode rejected by config
- **WHEN** `TrainerConfig(trainable_turns="all_turns")` is constructed
- **THEN** `ValueError` is raised from `__post_init__`

#### Scenario: invalid mode rejected by CLI
- **WHEN** `areno train --trainable-turns bogus` is invoked
- **THEN** `click.UsageError` is raised before rollout starts

### Requirement: Trainable-token observability

The metrics path SHALL emit, per batch: `trainable_tokens = sum(loss_mask)` and `masked_response_tokens = sum(response_mask) - sum(loss_mask)`. These values SHALL be assertable in CPU tests. Rollout logs SHALL print the active `trainable_turns` mode and `mask_tool_call_args` state.

#### Scenario: metrics reflect mask counts
- **WHEN** a batch is trained with `trainable_turns=final_answer`
- **THEN** `trainable_tokens` equals the count of True loss-mask bits and `masked_response_tokens` equals the count of response-mask True bits minus trainable_tokens

#### Scenario: active mode logged
- **WHEN** rollout begins with `trainable_turns=last_assistant` and `mask_tool_call_args=True`
- **THEN** the rollout log line records both the mode and the argument-masking state

### Requirement: B-tier interface shape reservation

The `LossMaskPolicy` dataclass SHALL retain an extensible field shape so a future per-trajectory scorer callable can be injected without breaking A-tier contracts. The `response_spans` span list captured during assembly SHALL serve as the physical basis enabling a B-tier scorer to locate spans at the `policy_only._run_agentic_rollout` seam (after rewards are computed, before `_train_rows_from_samples`). This change SHALL NOT implement any dynamic scorer; it SHALL only ensure the field shape and span data do not preclude its later addition.

#### Scenario: response_spans populated for multi-turn trajectory
- **WHEN** a multi-turn trajectory is assembled
- **THEN** each assembled sample carries a `response_spans` list recording every assistant span's kind and length, available at the rollout seam before `_train_rows_from_samples`

#### Scenario: no dynamic scorer implemented
- **WHEN** the change is applied
- **THEN** no callable-based per-trajectory masking logic is present; only static rules run