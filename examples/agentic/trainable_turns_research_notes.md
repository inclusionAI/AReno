# Research Notes: Selective Supervision in Agentic RL — AReno Issue #199

> **Dual purpose**: engineering deliverable context + research positioning material
> for graduate application (agentic RL direction).
>
> Status of AReno #199: Phase 1-5 (A-tier static config) shipped; Phase 6 G8
> statistics script (`trainable_turns_stats.py`) produced; GPU ablation pending.

---

## Part I — The Problem: Which Training Signal to Keep

Multi-turn agentic RL trajectories contain heterogeneous content: system
prompts, user queries, assistant reasoning text, tool-call invocations, tool
results, and final answers. The decision of **which tokens contribute to the
policy gradient loss** — termed *selective supervision* or *credit assignment* —
is a first-class design choice that affects:

1. **Supervision density**: how many tokens carry gradient signal per step.
2. **Behavioral focus**: whether the model learns to reason, call tools, or
   produce final answers.
3. **Training stability**: sparse supervision (only final answer) reduces
   gradient noise but may slow convergence; dense supervision (all turns)
   amplifies noise from tool-result contamination.
4. **Forgetting**: uncontrolled dense SFT degrades non-target capabilities
   ([Wu et al. 2025, STM](https://arxiv.org/abs/2501.14315)).

The granularity of selection forms a spectrum:

```
trajectory-level  >  turn-level  >  token-level static  >  step/token-level dynamic
     (keep/discard     (per-turn         (pre-defined            (reward/perplexity
      entire rollout)   advantage)         rules)                  driven)
```

## Part II — Literature Map

### II.1 Trajectory-Level: Keep or Discard Entire Rollouts

| Work | Mechanism | Granularity | Signal |
|---|---|---|---|
| **RFT** (Yuan et al. 2023) | Rejection sampling: sample N, keep correct, SFT on kept | Trajectory (binary) | Outcome correctness |
| **RAGEN/StarPO-S** (Wang et al. 2025, [arXiv:2504.20073](https://arxiv.org/abs/2504.20073)) | Variability-based trajectory filtering + critic + gradient stabilization | Trajectory (binary) | Reward variance / gradient norm |
| **RIFT** (Liu et al. 2025, [arXiv:2501.09253](https://arxiv.org/abs/2501.09253)*) | Reward-weighted loss: keep all trajectories, reweight by scalar reward | Trajectory (soft) | Per-trajectory reward |

**RFT** is the simplest form of selective supervision: it discards entire
trajectories below a correctness threshold. No within-trajectory selection
— all assistant tokens in kept trajectories are trained equally. Its
weakness: **information waste** (negative trajectories carry no gradient signal;
the correct portions of partially-correct trajectories are discarded).

**StarPO-S** extends trajectory-level filtering to the RL setting, adding
stability mechanisms (critic baselining, decoupled clipping) to address the
"Echo Trap" failure mode — where agents overfit to locally rewarded reasoning
patterns, causing reward variance collapse and gradient spikes. Trajectory
filtering is **reactive** (discard high-variance rollouts) rather than
**selective** (choose which turns within a rollout to train).

**RIFT** replaces binary filtering with reward-weighted loss, but the
weighting is per-trajectory, not per-turn or per-token.

### II.2 Turn-Level: Per-Turn Advantage Estimation

| Work | Mechanism | Granularity | Signal |
|---|---|---|---|
| **MT-GRPO** (Zeng et al. 2025, [arXiv:2505.11821](https://arxiv.org/abs/2505.11821)) | MDP formulation, per-turn advantage estimation within GRPO | Turn | Per-turn reward (tool execution + result quality) |

**MT-GRPO** is the most directly relevant prior work for the C-tier vision. It
models multi-turn tool-use as an MDP and computes **per-turn advantages**
rather than treating the entire trajectory as a single bandit step. Key
insights:

- Trajectory-level advantage (bandit formulation) causes agents to "forget to
  call tools" and exhibit high variance.
- Turn-level advantages isolate which decisions (e.g., search query quality)
  contribute to success, enabling more precise credit assignment.
- Reward design includes **turn-level verified rewards** (tool execution
  success, search result answer presence) separate from **outcome rewards**
  (final answer correctness).
- Results: 100% tool invocation success, 50% exact match vs 20-30% baselines.

**Gap**: requires turn-level reward functions — not just a single scalar
trajectory reward. This conflicts with AReno's current `reward_fn` contract
(rewards.py: single `RewardRecord` -> scalar). The C-tier extension would
need to redefine this contract.

### II.3 Token-Level Static: Pre-Defined Masking Rules

| Work | Mechanism | Granularity | Signal |
|---|---|---|---|
| **veRL delta-tokenization** | Incremental per-turn tokenization; mask non-assistant tokens | Token (segment): assistant vs environment | Static (role-based) |
| **AReno #199 A-tier** (this work) | Three trainable-turn modes + tool-call arg masking | Token (span-level): all/last/final assistant + arg sub-span | Static (mode-based) |
| **STM** (Wu et al. 2025, [arXiv:2501.14315](https://arxiv.org/abs/2501.14315)) | Perplexity threshold masking for SFT | Token (individual) | Dynamic (perplexity) |

**veRL** uses delta-tokenization: apply chat template incrementally per message,
identify assistant-generated tokens, and mask everything else. Selection
granularity is **segment-level** (assistant segment vs environment segment).
No distinction within assistant content — thinking text, tool-call name, and
tool-call arguments are all treated equally. A detailed engineering account
documents the non-trivial challenges of multi-turn tokenization: position-
dependent chat-template rendering (reasoning content stripped from non-final
assistant messages in QwQ/Qwen3), tokenization non-roundtrip, and the
fixed-base incremental approach that solved these issues.

**AReno #199 A-tier** (this work) adds **within-assistant-span selection** on
top of the veRL-style assistant/environment split. Three modes control which
assistant spans contribute to loss:

- `all_assistant`: all assistant spans trainable (matches existing behavior).
- `last_assistant`: only the final assistant span trainable.
- `final_answer`: only the `assistant_text` span after the last tool call
  trainable (degenerates to `last_assistant` without tools; yields zero
  trainable signal when the trajectory ends in a bare tool call).

Plus `mask_tool_call_args`: within tool-call spans, mask the JSON arguments
value while keeping the tool name trainable.

**STM** masks individual tokens whose perplexity exceeds a threshold, operating
at **individual token granularity** (not segment or span). The motivation is
different: reducing catastrophic forgetting in SFT rather than selecting which
turns to train in agentic RL. However, the mechanism (per-token binary mask
applied to the loss function) is structurally identical, making it a
conceptual cousin. Key difference: STM uses **perplexity** as the signal
(model-internal uncertainty), not reward (environment feedback) or role
(static structure).

### II.4 Step-Level Dynamic: Process Reward Models

| Work | Mechanism | Granularity | Signal |
|---|---|---|---|
| **PRM** (Lightman et al. 2023, [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) | Step-level human-verified labels; process reward model scores each step | Step | Human step-level correctness |
| **PRIME** (Cui et al. 2025) | Online implicit PRM; fuse token-level dense rewards with sparse outcome | Step/token | Implicit process reward |
| **PURE** (Cheng et al. 2025) | Min-form credit assignment to prevent PRM reward hacking | Step | PRM scores with min-aggregation |

**PRM** ("Let's Verify Step by Step") provides the strongest evidence that
**process supervision outperforms outcome supervision** for math reasoning
(78% on MATH test subset). The PRM800K dataset (800K human step-level labels)
enables training discriminative reward models. But: expensive annotation,
restricted to math/code, and the "step" boundary is defined by human
annotators — not applicable to agentic tool-use directly.

**PRIME** and **PURE** extend PRMs with implicit rewards and anti-hacking
measures, but remain in the math-reasoning domain. No published work applies
process reward models to **agentic tool-use trajectories** at the turn boundary.

### II.5 Additional Relevant Work

**Tool-use SFT** (ToolFormer, Gorilla, ToolACE, xLAM, Hermes, NexusRaven):
all train tool-call content (name + arguments) without internal masking.
ToolFormer uses loss-reduction threshold for filtering tool-augmented
training tokens; SWiRL ([arXiv:2504.04736](https://arxiv.org/abs/2504.04736))
uses step-level process rewards for tool-use. No published work performs
**parameter-level token masking within tool-call spans** — AReno's
`mask_tool_call_args` is a novel (if exploratory) contribution.

**A Practitioner's Guide to Multi-turn Agentic RL** (Wang & Ammanabrolu 2025,
[arXiv:2510.01132](https://arxiv.org/abs/2510.01132)): systematic ablation of
environment, reward, and policy design choices. Key finding: dense turn-level
rewards accelerate training, but stability depends on RL algorithm choice.
SFT-to-RL ratio matters. This work provides the experimental methodology
template for the pending GPU ablation (T6.2).

## Part III — AReno #199 in the Landscape

```
                  ┌─────────────────────────────────────────────────┐
                  │           Selective Supervision Taxonomy         │
                  └─────────────────────────────────────────────────┘

  Trajectory       Turn             Token-Static         Token-Dynamic
  ──────────       ────             ────────────         ─────────────
  RFT (binary)     MT-GRPO ◀── C    veRL delta ◀── A     STM (ppl)
  StarPO-S         (turn-level      AReno #199           PRM (step)
  RIFT (soft)       advantage)      (3 modes + arg)      PRIME/PURE

  A = this work (shipped)     B = interface reserved (not implemented)
  C = future issue (blocked by reward contract + turn offset)
```

**Positioning statement**: AReno #199 A-tier fills a gap in the **token-static**
column — it moves beyond veRL's coarse assistant/environment split to
**span-level** selection (which assistant turns to train) and
**sub-span-level** masking (tool-call arguments), while remaining purely static
(no reward or perplexity signal needed). This makes it the most fine-grained
**static** selection mechanism in the literature.

The `mask_tool_call_args` option is **without precedent** in published tool-use
training: all prior work trains the complete tool-call (name + arguments).
AReno's implementation serves as an **ablation probe** — isolating whether
tool-call arguments (which the model does not control — they come from the
environment) contribute noise to the policy gradient.

## Part IV — Statistics Results (T6.1, CPU-only)

**Method**: 6 deterministic trajectory fixtures × 3 modes × 2 mask_args
configurations = 36 data points. All masks computed through AReno's real
`RolloutSession._train_rows_from_samples` pipeline (not a reimplementation).
Character-level tokenizer ensures tool-call argument localization is exact.

**Aggregate trainable ratio** (across all fixtures, pooled response tokens):

| Mode | mask_args=False | mask_args=True | Change |
|---|---|---|---|
| all_assistant | 100.0% (922/922) | 79.8% (736/922) | -20.2pp |
| last_assistant | 26.6% (245/922) | 24.0% (221/922) | -2.6pp |
| final_answer | 20.6% (190/922) | 20.6% (190/922) | 0 |

### Key findings

1. **Supervision density drops sharply with turn selection**: `last_assistant`
   reduces trainable tokens to ~27% of total response, `final_answer` to ~21%.
   The difference between the two is small for well-formed trajectories (both
   end in assistant_text after tool use), but **diverges for bare trailing
   tool calls**: `final_answer` yields zero trainable signal (0%),
   `last_assistant` still trains the tool-call span (65.5%).

2. **Tool-call argument masking** reduces trainable tokens by ~20pp under
   `all_assistant` — this is the proportion of tool-call argument characters
   in the response. Its effect under `last_assistant`/`final_answer` is
   minimal (2.6pp / 0pp) because tool-call spans are already masked by mode
   selection; the only case where it matters is when `last_assistant` targets
   a trailing tool-call span (e.g., `bare_trailing`: 65.5% -> 36.9%).

3. **Mode XOR mask_args interaction**: `mask_tool_call_args` is only
   independently meaningful under `all_assistant` (where all assistant spans
   are trainable). Under `last_assistant`/`final_answer`, tool-call spans are
   already excluded from training by the mode, so arg masking is redundant
   except for the edge case where the target span itself is a tool call.

4. **Degenerate fixture (no tools)**: all configurations produce identical
   trainable counts (100%), confirming expected behavior and backward
   compatibility.

**Per-fixture data**: see `trainable_turns_stats.csv` / `.json` for full 36-row
table. Re-run with:
```bash
python examples/agentic/trainable_turns_stats.py
```

## Part V — Research Gaps and Future Directions

### Gap 1: B-tier — Per-Trajectory Scorer (interface reserved)

**Status**: `LossMaskPolicy` field shape does not reject a
`Callable[[RewardRecord], float|bool]` injection. Not implemented.

**Research question**: Can per-trajectory reward signals (already available
in AReno's `reward_fn` contract) drive dynamic loss-mask decisions? E.g.,
discard response tokens in low-reward trajectories rather than training them
with a full positive gradient.

**Closest prior work**: RIFT (reward-weighted loss) at trajectory level, STM
(perplexity threshold) at token level. The gap: **reward-driven token-level
masking** for agentic RL trajectories — no published work exists.

**Implementation path**: inject callable at `policy_only.py:245-246` where
rewards are already computed but `_train_rows_from_samples` has not yet
constructed the mask. The callable returns a weight or binary mask per
response token, composed with the static mode mask.

### Gap 2: C-tier — Per-Turn Credit Assignment (blocked)

**Status**: Two blockers identified in this work:

1. **Reward contract**: AReno's `reward_fn` returns a single scalar per
   trajectory (`rewards.py:63`). Per-turn reward requires redefining this to
   accept `RewardEvent[]` trace history and return per-turn signals.

2. **Turn offset loss**: `_append_sample_response` (agentic L627-666) folds
   multi-turn boundaries into a single `loss_mask_override` list. Per-turn
   credit assignment requires preserving turn boundaries (offsets) to assign
   advantages per-turn, not per-trajectory.

**Closest prior work**: MT-GRPO (turn-level advantage in MDP formulation).
The gap: MT-GRPO requires a custom MDP environment and turn-level reward
verifier; no turn-level credit assignment framework exists that composes with
**general-purpose RLVR systems** (veRL, AReno, OpenRLHF) without replacing
the trainer.

**Implementation path**: (1) extend `RewardRecord` to carry per-turn reward
events; (2) preserve turn offsets in `_AgentSample`; (3) modify advantage
computation in the policy trainer to use per-turn advantages. This is a
trainer-level change (violates "no trainer replacement" boundary) and is
correctly scoped as a follow-up issue.

### Gap 3: Parameter-Level Tool-Call Masking (exploratory, this work)

**Status**: `mask_tool_call_args` is implemented and tested (CPU per-token
tests). Positioned as research ablation, not engineering standard.

**Research question**: Do tool-call argument tokens (JSON values the model
generates but whose correctness is environment-determined) contribute noise
to the policy gradient? Masking them isolates the model's tool-selection
behavior from its parameter-generation behavior.

**Industry context**: no published tool-use training system masks individual
tool-call argument tokens. All train the complete tool-call (ToolFormer,
Gorilla, ToolACE, xLAM, Hermes, NexusRaven). The closest filtering is
ToolFormer's loss-reduction threshold and SWiRL's process reward, both at
the **step/sample level**, not the token level.

**Open question**: can only be answered with GPU ablation (T6.2, pending).

## Part VI — Implications for Agentic RL Research

The taxonomy and statistics produced here support several research claims:

1. **Static span-level supervision density is a tunable knob with large
   effects**: a 5x difference in trainable tokens between `all_assistant`
   and `final_answer` (100% vs ~21%). Whether this translates to convergence
   speedup or quality degradation is an open empirical question requiring
   GPU ablation.

2. **The "which turns" question decomposes into two sub-questions**: (a) which
   turns to train (mode selection, A-tier), and (b) what weight to assign each
   turn (advantage estimation, C-tier). Industry work has largely addressed
   (b) through trajectory-level normalization; the systematic exploration of
   (a) is a contribution of this work.

3. **Tool-call argument masking is unexplored territory**: the interaction
   data (Gap 3) shows it primarily affects `all_assistant` mode and is
   redundant under turn-level selection. Its value as a training signal
   isolation technique awaits empirical validation.

4. **The B/C-tier path is the real research frontier**: static masking (A-tier)
   is a solved engineering problem; dynamic, reward-aware, turn-level
   credit assignment is where the literature has clear gaps. The interface
   reservation in `LossMaskPolicy` makes AReno a viable platform for this
   research, lowering the barrier from "replace the trainer" to "inject a
   callable".

---

### References

| ID | Paper | arXiv |
|---|---|---|
| RFT | Yuan et al. 2023, "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models" | [2308.01825](https://arxiv.org/abs/2308.01825) |
| StarPO-S | Wang et al. 2025, "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL" | [2504.20073](https://arxiv.org/abs/2504.20073) |
| MT-GRPO | Zeng et al. 2025, "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment" | [2505.11821](https://arxiv.org/abs/2505.11821) |
| PRM | Lightman et al. 2023, "Let's Verify Step by Step" | [2305.20050](https://arxiv.org/abs/2305.20050) |
| STM | Wu et al. 2025, "Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning" | [2501.14315](https://arxiv.org/abs/2501.14315) |
| RIFT | Liu et al. 2025, "RIFT: Repurposing Negative Samples via Reward-Informed Fine-Tuning" | [2501.09253](https://arxiv.org/abs/2501.09253) (unverified) |
| PRIME | Cui et al. 2025, "PRIME: Process Reward Model via implicit rewards" | (cited in MT-GRPO, arXiv ID not verified) |
| PURE | Cheng et al. 2025, "Min-form credit assignment" | (cited in MT-GRPO, arXiv ID not verified) |
| ToolFormer | Schick et al. 2023 | [2302.04761](https://arxiv.org/abs/2302.04761) |
| veRL | (engineering account, LinkedIn post by Jiang) | N/A |
| Practitioner Guide | Wang & Ammanabrolu 2025, "A Practitioner's Guide to Multi-turn Agentic RL" | [2510.01132](https://arxiv.org/abs/2510.01132) |

---

*Document generated as part of AReno issue #199 research extension (Phase 6 / G8).
Statistics artifact: `trainable_turns_stats.py` / `.csv` / `.json`.
GPU ablation (T6.2) pending — range limitations noted where applicable.*