---
license: apache-2.0
language:
- en
pretty_name: Typed Decisions
size_categories:
- n<1K
task_categories:
- text-classification
tags:
- structured-decisions
- calibration
- probabilistic-classification
- system-one
- workflow-evaluation
- synthetic
configs:
- config_name: agent_trace_observability
  data_files:
  - split: train
    path: agent_trace_observability/train-*.parquet
  - split: test
    path: agent_trace_observability/test-*.parquet
- config_name: customer_service
  data_files:
  - split: train
    path: customer_service/train-*.parquet
  - split: test
    path: customer_service/test-*.parquet
- config_name: invoice_processing
  data_files:
  - split: train
    path: invoice_processing/train-*.parquet
  - split: test
    path: invoice_processing/test-*.parquet
- config_name: security_incidents
  data_files:
  - split: train
    path: security_incidents/train-*.parquet
  - split: test
    path: security_incidents/test-*.parquet
- config_name: all
  data_files:
  - split: train
    path: all/train-*.parquet
  - split: test
    path: all/test-*.parquet
---

# Typed Decisions

A benchmark for typed probabilistic decisions over shared state. You give a
model one piece of unstructured state. It answers several typed questions about
that state at once, and every answer is a probability distribution rather than a
single label.

The schema follows the System One primitives used by
[TypeSafe AI](https://typesafe.ai): `noul`, `choice` and `score`. A row replays against any API that implements that shape. This benchmark is
independent. It is not affiliated with TypeSafe and it does not reproduce
their Jev model.

## Leaderboard

Scored on the `test` split: 400 cases, 2,000 decisions. General models are
scored zero-shot: they have never seen these workflows or their question schemas.

| # | Model | Kind | Accuracy ↑ | KL from gold ↓ | Brier ↓ | Latency, p50 end to end ↓ | Price per 1M input tokens ↓ |
|---|---|---|---|---|---|---|---|
| 1 | **[meraGPT Decider 1](https://meragpt.com/models/state-decider-1)** (`sd-1`) | general, zero-shot | **0.768** | **0.096** | **0.052** | **526 ms** | **$0.03** |
| 2 | TypeSafe Jev 1.13.0 | general, zero-shot | 0.727 | 1.442 | 0.148 | 710 ms | $0.042 |
| 3 | [Featherless Simple Jev](https://simple-jev.featherless.ai) (`Qwen3.6-35B-A3B-classifier`) | general, zero-shot | 0.716 | 0.488 | 0.176 | – (free demo endpoint, rate-limited) | free demo |
| 4 | ModernBERT-base (149M) | specialist, fitted per workflow | 0.646 | 0.223 | 0.119 | – | – |
| 5 | MiniLM-L6 (22M) | specialist, fitted per workflow | 0.587 | 0.262 | 0.143 | – | – |
| – | Prior (ignores the input) | reference | 0.470 | 0.347 | 0.189 | – | – |

**meraGPT's proprietary model, [Decider 1](https://meragpt.com/models/state-decider-1),
is state of the art on this benchmark** as of 2026-09-22. It beats TypeSafe's Jev
on accuracy and on every question type, it is faster end to end, and it costs
less per token. It answers `POST /v1/systemone` at
[meragpt.com](https://meragpt.com/docs/systemone), so the typesafe-sdk works
against it by setting `TYPESAFE_BASE_URL`. Full metrics are under
[Baseline results](#baseline-results).

To add a model, score it on `test` with the full distributions and open a
discussion with the numbers and the mode (specialist or generalist) it used.

## What it is for

One question sits behind the dataset. Does breaking a workflow into typed
probabilistic decisions, especially with a shared encoder, buy you a better
accuracy / calibration / latency trade-off than direct classification or than
prompting an LLM?

Answering that needs several typed questions over one input, answers that are
genuinely probabilistic, and latency you can measure per decision. That is what
this is.

## The three question types

| Type | Answer | Shape |
|---|---|---|
| `noul` | yes/no | a single probability that the statement is true |
| `choice` | one of N labels | a distribution over labels, plus confidence |
| `score` | an ordered rubric | a distribution over integer levels, plus an *expected* score that may fall between levels |

Every option carries a written description in `criteria`. Those descriptions are part of the input. Strip them to a bare label list and
you have a different, easier task.

## Workflows

| Workflow | Decision | Train | Test |
|---|---|---|---|
| `agent_trace_observability` | Assess an agent run to decide whether human review is needed, and how urgent it is. | 300 | 100 |
| `customer_service` | Determine the appropriate assistant response and action from a customer thread and account state. | 300 | 100 |
| `invoice_processing` | Review a vendor bill against the order and delivery to determine payment, hold or rejection. | 300 | 100 |
| `security_incidents` | Decide whether a security alert should be closed, investigated, or contained, from alert data and machine history. | 300 | 100 |

Each case asks 5 questions over one shared state.

## Columns

| Column | Description |
|---|---|
| `id` | Case identifier |
| `workflow` | Which workflow the case belongs to |
| `state` | JSON. The unstructured state, which is the model's input |
| `questions` | JSON. The question set, with instructions and criteria |
| `gold` | JSON. Full gold answers including every probability |
| `factors` | JSON. Latent factors used to build the case. Not model input |
| `label_agreement` | JSON. Per question, how much the teacher samples disagreed |
| `<question>__label` | Discrete gold answer |
| `<question>__confidence` | Gold confidence, on the System One scale |
| `<question>__probabilities` | JSON. Full gold distribution |
| `<question>__score` | Expected score, for `score` questions |
| `<question>__probability_true` | Probability of yes, for `noul` questions |

`state` and `questions` together are exactly the body of a `POST /v1/systemone`
request. You can replay a row without reshaping it.

## Usage

```python
from datasets import load_dataset
import json

ds = load_dataset("LocalLLaMA/typed-decisions", "customer_service", split="test")   # benchmark
tr = load_dataset("LocalLLaMA/typed-decisions", "customer_service", split="train")  # training data
row = ds[0]

state = json.loads(row["state"])
questions = json.loads(row["questions"])
gold = json.loads(row["gold"])

print(row["category__label"], row["category__confidence"])
print(gold["urgency"]["probabilities"])   # distribution over rubric levels
```

Score against the full distributions, not just the argmax. Calibration is the
point. Report log loss, Brier score and ECE next to accuracy.

## How it was built

1. Sample a latent skeleton. Each case starts from independently drawn factors:
   topic, tone, tenure, severity, discrepancy type, whether a constraint was
   violated, and so on. The spaces are large enough that states essentially
   never repeat.
2. Render the state. Free text is written by a model conditioned on the
   skeleton where the task is textual, as in customer threads and alert
   narratives. It stays structured where the artefact genuinely is structured,
   as in invoices and agent traces.
3. Label it with a teacher endpoint, sampled 3 times per case at
   temperature 0.7. The gold is the mean of the sampled
   distributions. Averaging distributions instead of argmax labels is what
   leaves the gold soft where a decision is genuinely ambiguous.
4. Check before release: state diversity, label balance, and whether the gold
   actually tracks the input.

## What a score here means

Gold is the mean of three samples from a teacher endpoint of roughly 4B-class
capability. A score measures agreement with that teacher. It does not measure
correctness. Three reference points, all measured on the 1600-case set:

| Reference | Accuracy | What it is |
|---|---|---|
| Majority baseline | 0.520 | ignore the input, always guess the commonest label |
| Perfect scenario understanding | 0.704 | a model fitted to the latent factors that generated each case, cross-validated |
| Teacher self-agreement | 0.735 | a fresh teacher sample scored against gold built from the others |

Read 0.52 as the floor. Around 0.70 is strong. Around 0.75 is saturation.

The factor ceiling sits below teacher self-agreement. That is not a mistake. The
teacher shares its own idiosyncrasies with the gold, and an outside model does
not get that advantage. A score much above 0.75 means a model has learned the
teacher's quirks rather than the task.

Per-question ceilings vary a lot, from 0.560 on `agent_trace/urgency` to 0.937
on `customer_service/category`. Read every score against its own question, not
against the mean.

A better model can score worse here. Anything right where the teacher is wrong
gets penalised. The teacher missed a duplicate invoice whose ID literally
matched a prior one.

## Two ways to be scored, and why the difference matters

The System One models this benchmark is shaped after are general pretrained
models. Their API is one call that takes an arbitrary question schema at request
time. No training step, no per-workflow setup. Anything scored here should say which
of the two modes it used. The two are not comparable.

| | What it is | What it needs | Can it answer an unseen question? |
|---|---|---|---|
| Specialist | fitted per workflow, label spaces fixed at training time | training data for *these* workflows | No |
| Generalist | one model, arbitrary question schemas, zero-shot | training data from *other* workflows | Yes |

Train a specialist on these four workflows, score it on them, and you have
measured architecture: how cheaply many typed decisions can come out of one
input. That is a real question and this benchmark answers it well. It is not a
comparison against a general System One model, which has never seen these
workflows.

To be scored as a generalist, train on other workflows entirely and evaluate
here zero-shot. Say which mode you used. A specialist number sitting next to a
generalist number, unlabelled, misleads the reader.

## Baseline results

Everything below is scored on the `test` split. The first column says what kind
of number it is, because they are not all the same kind.

| Model | Kind | Acc | Soft acc | Macro F1 | KL | TV | Brier | ECE | Score MAE | Within 1 level | ms/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Uniform | reference | 0.308 | 0.311 | 0.152 | 0.444 | 0.381 | 0.238 | 0.169 | - | - | 0 |
| Prior | reference | 0.470 | 0.430 | 0.207 | 0.347 | 0.317 | 0.189 | 0.088 | - | - | 0 |
| MiniLM-L6 (22M) | specialist | 0.587 | 0.506 | 0.424 | 0.262 | 0.267 | 0.143 | 0.108 | 0.515 | 0.864 | 22 |
| ModernBERT-base (149M) | specialist | 0.646 | 0.542 | 0.469 | 0.223 | 0.249 | 0.119 | 0.179 | 0.444 | 0.931 | 349 |
| Perfect scenario understanding | ceiling | 0.704 | - | - | - | - | - | - | - | - | - |
| TypeSafe Jev 1.13.0 | general | 0.727 | 0.580 | 0.613 | 1.442 | 0.251 | 0.148 | 0.144 | 0.391 | 0.952 | 710 |
| Teacher self-agreement | ceiling | 0.735 | - | - | - | - | - | - | - | - | - |
| **[meraGPT Decider 1](https://meragpt.com/models/state-decider-1)** | **general** | **0.768** | **0.608** | **0.641** | **0.096** | **0.149** | **0.052** | 0.180 | **0.219** | **0.984** | 526* |

\* p50 end to end from a client, one request at a time, like Jev's 710.

### What each row is

**Uniform** puts the same probability on every option. It reads nothing and
knows nothing. It is here to anchor the KL and Brier scale: 0.444 is what no
information costs in distribution terms.

**Prior** fits each question's label frequencies on the `train` split, then
answers those frequencies for every case, ignoring the state entirely. If 67% of
`needs_human` golds are true, it answers 0.67 true every time.

This is the row to check a learned model against. MiniLM beats it by 9 points
and ModernBERT by 18. That gap is how much of each score comes from reading the
input rather than from label frequency. A model that cannot clear it has learned
nothing about the state, which is the failure that sank the v0.1 prototype.

Prior also has the best ECE on the table, at 0.088, while knowing nothing.
Guessing the base rate is perfectly calibrated by construction. That is the
clearest argument for reading KL and Brier here instead of ECE.

**Perfect scenario understanding** is what a model would score if it recovered
the latent factors that generated each case exactly. Measured by fitting those
factors to the gold labels with cross-validation. It is optimistic, since the
factors are more than the text reveals.

**Teacher self-agreement** is a fresh teacher sample scored against gold built
from the other samples. It is the noise floor of the labelling process. Scoring
far above it means predicting the teacher's quirks rather than the task.

**TypeSafe Jev 1.13.0** is a measurement, taken on 2026-09-18 through the
TypeSafe API (`POST /v1/systemone`, `model: jev-latest`, which reported itself
as `jev-1.13.0`). All 400 cases, all 2,000 decisions, zero errors, p50 710ms per
case, $0.016 total at the published $0.042/1M input rate. Earlier revisions of
this card carried an estimated range here instead; that estimate is gone.

Jev scores **0.727 against a 0.735 ceiling**, so it has effectively saturated
this benchmark. It also clears the 0.704 factor ceiling, meaning it reads these
scenarios better than a model that recovers the generating factors exactly.

**Its distributions are a different story.** Jev's KL from gold is 1.442 against
ModernBERT's 0.223 -- six times worse -- while scoring 8 points higher on
accuracy. Jev picks the right label and commits to it; the specialist is right
less often but its uncertainty tracks the teacher's spread much more closely.
Jev is not badly calibrated in absolute terms (ECE 0.144, overconfidence
+0.023); it is confident because it is usually correct. The KL gap is mostly
that this gold is a three-sample teacher spread and Jev does not reproduce that
spread. Which number matters depends on whether you consume the argmax or the
distribution.

**meraGPT Decider 1** (`state-decider-1`, alias `sd-1`) is meraGPT's proprietary
System One model, scored zero-shot: it had never seen these four workflows or
any of the twenty question schemas. It leads every question type — `noul` 0.840
against Jev's 0.775, `choice` 0.733 against 0.720, `score` 0.739 against 0.696 —
with KL 0.096 against Jev's 1.442. End to end it answers in about half a second
at the median, against Jev's 710 ms, at $0.03 per million input tokens against
Jev's $0.042. [meragpt.com/models/state-decider-1](https://meragpt.com/models/state-decider-1).

### Specialist and generalist are not comparable

Both learned rows are specialists, fitted on the `train` split of the same four
workflows they are scored on. Neither can answer a question it was not fitted
for, so neither can be run zero-shot.

Jev at 0.727 against the specialist's 0.646 has not beaten it by eight points.
Jev answered all twenty question schemas cold, having never seen this benchmark;
the specialists were fitted on the `train` split of the very workflows they are
scored on and cannot answer anything else at all. Read the gap as the price of
generality, not as a quality ranking. A general model can also score lower while being
the better model, since anything it gets right where the teacher is wrong counts
against it.

### Reproducing the specialist rows

Both use [Adaptive Classifier](https://github.com/codelion/adaptive-classifier)
0.2.0, one classifier per question, encoder frozen. Configuration was tuned on a
held-out quarter of `train` and never on `test`: mean pooling, `max_length` 512,
30 epochs, `prototype_weight` 0.3.

The gap between the two encoders is the trade-off this benchmark exists to
measure. Six points of accuracy cost 16x the latency.

One harness detail matters for reproducing these. Adaptive Classifier trains on
hard labels, so the gold distribution is normally thrown away at fit time. Each
case is instead entered four times, apportioned across labels in proportion to
its gold, which carries the soft target into a learner that cannot represent one
directly. That single change cut KL by a third and score MAE by 15%, while
barely moving accuracy. The argmax was already right. What improved was the
shape of the predicted distribution, which is what this benchmark is for.

## Splits

Two splits, generated independently. `test` is the benchmark. `train` comes from
a separate run at a different seed, with prefixed case ids. Packaging verifies
that no case id and no state appears in both, and refuses to build if either
does.
