# A Trainable Scoring Head for Generalist Business Decisions on a Single-Device Language Model

**Authors:** [Author list to be supplied]

**Affiliations:** [Affiliations to be supplied]

**Correspondence:** [Corresponding author to be supplied]

---

## Abstract

Generalist decision benchmarks such as `typed-decisions` require a model to answer heterogeneous business questions, including forced choices, boolean judgements, and ordered scores, under a single serving contract. Whether a compact instruction model can be improved on this benchmark by task-specific training, rather than by scale, remains unclear. Here we train a JevForge-style trainable per-sequence scoring head on `Ling-3.0-tiny` (24 layers, 128 experts, 1536 hidden units) inside AReno, using a grouped softmax that shares one normalization across the candidate answers of a question and a loss that combines cross-entropy with a Brier term. All training and inference use the same packed variable-length forward pass, so no train-inference gap is introduced. We ran three controlled training runs and a zero-shot baseline on one NVIDIA DGX Spark (GB10, 128 GB unified memory). The head reached 0.582 accuracy on `typed-decisions` test, statistically indistinguishable from the 0.567 accuracy of the untrained base model's language-model head. Training did not raise accuracy but did improve calibration: Kullback-Leibler divergence fell from 0.893 to 0.316-0.443 and the Brier score from 0.326 to 0.178-0.229, with overconfidence moving from +0.228 to between -0.030 and +0.103. The clear benefit of training was in-distribution, where accuracy on Open-Jev v1.1 rose by 24 points. Adding missing answer-format structure did not help, and an inappropriate score data source degraded score accuracy. These results indicate that on this benchmark the micro-fine-tuning route is bounded by the base model's zero-shot decision ability or by its agreement with the benchmark's teacher, and that its reliable value is calibration rather than capability.

**Keywords:** decision scoring, calibration, parameter-efficient fine-tuning, mixture-of-experts, LLM evaluation, group softmax

---

## 1. Introduction

Language models are increasingly asked to make business decisions rather than generate free text. In applications such as invoice handling, security triage, customer-service routing, and agent-trace review, a model receives a structured state and must produce one of three answer types: a choice among described options, a boolean judgement, or a value on an ordered scale. The `typed-decisions` benchmark formalizes this setting within the Jev decision evaluation family. It contains 400 cases and 2000 decisions, and current leaderboard entries span both dedicated systems and generalist models, from Jev 1.13 at 0.727 accuracy to generalist entries near 0.716.

Two questions motivate this study. The first is whether a compact model can be brought close to that level by task-specific training rather than by scale. The second is what such training actually changes. A decision model can improve in at least two distinct ways: it can become more accurate, or it can become better calibrated about how confident it should be. These outcomes have different practical value and are not interchangeable.

We test a route that mirrors the JevForge architecture: a backbone language model paired with a separate trainable head that scores each candidate answer path. We implement this route in AReno as an experimental classification algorithm, applied to `Ling-3.0-tiny`, a hybrid mixture-of-experts model. We then evaluate on `typed-decisions`, which is held out and used in a strictly zero-shot manner with respect to our training data. The study asks a single question: does training a candidate-path scoring head improve generalist decision accuracy on `typed-decisions` beyond the base model's own zero-shot ability?

To answer it, we compare three training runs under matched hyperparameters: a first data mix built from Open-Jev v1.1, a second mix that adds the answer-format structure the first lacks, and a zero-shot baseline that uses the base model's language-model head directly. We isolate the contribution of each factor and report the calibration behaviour of every checkpoint. We also verify that our local evaluation reproduces the published benchmark's uniform baseline, and we screen for overlap between training data and the test set.

---

## 2. Related Work

The benchmark we study belongs to a recent line of work that reframes language-model evaluation from label prediction to typed probabilistic decision making. `typed-decisions` is an independent benchmark built on the System One primitives used by TypeSafe AI [1] and shaped around the `noul`, `choice`, and `score` answer types [2]. Its construction is unusual in two respects that matter for interpreting our results. Gold labels are the mean of three samples from a roughly 4B-class teacher endpoint at temperature 0.7, so a score measures agreement with that teacher rather than correctness, and the dataset card reports reference points of 0.520 for a majority baseline, 0.704 for a model fitted to the latent generative factors, 0.735 for teacher self-agreement, and saturation near 0.75 [2]. The card further warns that a model which is right where the teacher is wrong is penalised, and that per-question ceilings range widely, from 0.560 on `agent_trace/urgency` to 0.937 on `customer_service/category` [2]. Our evaluation inherits these properties, and we return to their consequences in Section 5.

The benchmark distinguishes generalist models, which answer arbitrary question schemas zero-shot, from specialists fitted per workflow with fixed label spaces, and it states that the two modes are not comparable [2]. Published generalist entries include TypeSafe Jev 1.13.0 at 0.727 accuracy, Featherless Simple Jev at 0.716, and the proprietary meraGPT Decider 1 at 0.768 [2,3,4]. Published specialists include ModernBERT-base at 0.646 and MiniLM-L6 at 0.587, both fitted per workflow [2,5,6]. Our work is a generalist evaluation in the first sense: the model is trained on other workflows and applied zero-shot, so we compare only against generalist entries.

Candidate-path scoring with a shared normalization is related to grouped softmax objectives in classification and language modeling [7] and to listwise ranking objectives in information retrieval and recommendation, where candidates for a single query are scored jointly. We adopt the grouping explicitly, so that candidate answers belonging to the same question share one softmax. This is the property that makes the head a decision model rather than an independent token classifier, and it is also what makes cross-entropy and Brier loss jointly well defined on the candidate simplex.

The Brier score originates in probabilistic forecasting [8], and combining it with a proper scoring rule such as cross-entropy follows standard practice in evaluating probabilistic predictions [9]. Calibration of neural network confidence is a long-studied problem [10], and post-hoc temperature scaling is a widely used and effective remedy [11]. Our work differs from the standalone calibration literature in two ways: we study how a trainable head changes calibration over the course of training rather than only applying a post-hoc correction, and we test whether an in-distribution temperature transfers to an out-of-distribution decision benchmark.

Supervised fine-tuning of instruction-tuned language models is the standard route to task adaptation [12], and parameter-efficient variants such as LoRA reduce its cost [13]. Its benefit depends on the distance between the target task and the pretraining distribution, and gains on an in-distribution set do not necessarily transfer to a differently distributed held-out set [14]. Our design follows that concern directly by separating an in-distribution measure (Open-Jev) from an out-of-distribution measure (`typed-decisions`) and reporting both.

Distillation of soft targets from a stronger teacher is an established way to transfer knowledge beyond the hard labels a student observes [15], and it is the natural candidate for injecting decision knowledge that a compact base model does not encode. We did not test distillation here, but we identify it in Section 5 as the one route we can see that could change the ceiling we observed.

Mixture-of-experts language models such as `Ling-3.0-tiny` add a further consideration. Sparse routing is sensitive to input distribution, and an adaptation procedure that shifts the input distribution may change such a model in ways that a dense model would not exhibit [16]. We report the observed limits of our route for this model family and do not generalize across architectures.

---

## 3. Method

### 3.1 Task formulation

A task instance pairs a business `state` with a question whose type is one of three: `choice`, in which the answer is a soft distribution over described options and is not required to be one-hot; `noul` (no-utility-label), in which the answer is true or false; and `score`, in which the answer is a distribution over ordered levels and is evaluated by its expected level. A question may carry `criteria` that describe how answers should be judged.

For each question we render a prompt from the state and question, then render every candidate answer as the prompt followed by an answer marker and the candidate text. The model scores each candidate path, and the scores for all candidates of one question are normalized together. This formulation supplies the supervision signal and the evaluation procedure simultaneously.

### 3.2 Overview

The system has four parts that share one forward pass: a backbone language model, a trainable scoring head that replaces the language-model head, a grouped-softmax loss, and an evaluation and serving path that reuses the same code. Figure 1 illustrates the pipeline. The design goal was to eliminate every difference between training, evaluation, and serving so that no train-inference gap can confound the comparison.

**Figure 1 | The scoring-head pipeline.** A business state and question are rendered to a prompt. Each candidate answer path is appended with an answer marker and tokenized. The backbone produces final-norm hidden states with the language-model head deferred, and a candidate-path scoring head reads the last token of each path. Scores for candidates of the same question share one softmax, producing a distribution that is compared against the target distribution during training and read out for choice, boolean, or score answers during evaluation and serving. All three stages use the same packed variable-length forward pass.

![The scoring-head pipeline.](_static/figures/fig1_pipeline.png)

### 3.3 Candidate-path scoring head

The head is a small multilayer perceptron, `Linear(1536,1536) - GELU - Linear(1)`, that reads the final-norm hidden state of the last token of each candidate path and returns one scalar score per path. The architecture follows the JevForge reference implementation [28]. This module exists because the language-model head predicts the next token over the full vocabulary, which is the wrong inductive bias for comparing complete candidate paths. Reading only the last token of a path and projecting to a scalar gives a per-candidate score that can be normalized against its siblings.

The backbone is called with `defer_lm_head=True`, so the language-model head is skipped entirely. The head is initialized with a fixed seed so that multiple data-parallel ranks agree, and it is stored as `score_head.safetensors` with keys `0.weight`, `0.bias`, `2.weight`, and `2.bias`. Deferral of the language-model head is implemented for four model families in our engine, and this implementation boundary marks which backbones the head can and cannot be attached to.

### 3.4 Grouped-softmax objective

For a question with candidate scores, we compute a softmax within the question rather than across the whole batch. The implementation uses a `scatter_reduce` maximum followed by an `index_add` normalization, so that the segmentation is exact and does not depend on candidate ordering. The per-question loss combines cross-entropy against the target distribution with a Brier term,

`L = CE + brier_weight * Brier`,

with `brier_weight` set to 0.5. The Brier term penalizes miscalibrated probability mass that cross-entropy alone can tolerate. Rows that belong to no question, which arise from packing, are assigned a group index of -1 and are excluded from the loss while a computation-graph anchor is retained.

The choice of a grouped softmax is deliberate. Scoring candidates independently would let a model express confidence that does not depend on the alternatives it faced, which is inconsistent with a decision task. Sharing one normalization across the candidates of a question makes the output a proper distribution over the actual alternatives, and it makes cross-entropy and Brier both well defined on the same simplex.

### 3.5 Step organization and packing

A question is never split across a microbatch or a data-parallel rank, because splitting it would break the grouped normalization. The trainer packs questions into a step by sorting them on a first-order decreasing token count, which keeps microbatch token counts near the target and avoids padding. Training and inference both use packed variable-length sequences, so the forward pass is identical in both modes.

Gradients are normalized by question weight before the data-parallel and microbatch averaging, so that the resulting update equals the mean over all questions rather than the mean over microbatches. This detail matters because questions differ substantially in token count.

### 3.6 Optimization settings

We group the learning rates. The backbone uses 2e-5 with cosine decay to zero, and the head uses 2e-4 with a head-only warmup of 12 steps. During warmup the backbone learning rate is zero, but AdamW continues to accumulate backbone momentum. We use 4-bit AdamW, a microbatch of 8000 tokens, `MAX_LEN=1536`, flash attention, and seed 17. Each run trains for 500 steps at 32 questions per step and archives a checkpoint every 100 steps.

### 3.7 Evaluation protocol

Evaluation uses the same packed forward pass as training through a `SequenceScorer` adapter, with `MAX_LEN=1536`. We report accuracy, cross-entropy, Kullback-Leibler divergence `KL(gold||pred)`, total variation distance, Brier score, mean confidence (`conf`), overconfidence (`overconf = conf - acc`), and expected calibration error with 10 bins.

The `typed-decisions` test set comprises 400 cases and 2000 decisions under an Apache-2.0 license and is vendored in our repository [2]. Each case asks five questions over one shared JSON state across four workflows: `agent_trace_observability`, `customer_service`, `invoice_processing`, and `security_incidents`. We treat it as a zero-shot generalist evaluation and compare only against generalist leaderboard entries. Gold labels are the mean of three samples from a roughly 4B-class teacher endpoint at temperature 0.7, so a score measures agreement with that teacher rather than correctness, and the dataset card reports a 0.520 majority floor, a 0.704 factor ceiling, 0.735 teacher self-agreement, and saturation near 0.75 [2]. Our `evaluate.py` reproduces the published uniform baseline exactly (KL 0.444, TV 0.381, Brier 0.238), so those three columns are on the same footing as the benchmark; accuracy is consistent for any model without ties. The prior baseline was not reproduced exactly, with differences of 0.01 to 0.09, so we treat it as indicative only. Temperature is fitted only on the Open-Jev calibration split and never on `typed-decisions`.

We note two provenance limitations. The token-length analysis that guided `MAX_LEN` was recorded during the experiment but its script output was not retained, so we could not independently re-verify those numbers. Likewise, a leakage screen compared Open-Jev v1.1 and Open-Jev v1 against `typed-decisions` test by literal and Jaccard matching and found no overlap; the screen also identified 400 segments in JevEmbed-Data `system_one_270m` that were identical to test criteria, and that source was excluded. The exclusion decision is recorded in `convert_datasets.py:JEVEMBED_SOURCES`, but the screening script itself was not retained, so this step likewise could not be independently re-verified.

### 3.8 Implementation

All changes are on the `feat/classify-jev` branch in ten commits. At the engine level we added `areno/engine/score_head.py` with `build_score_head`, `attach_score_head`, `packed_sequence_scores`, and `save_score_head`; we added a `score_head` switch to `RuntimeConfig`; we extended `TrainSequence` and `make_train_pack` to carry `sequence_labels`; and we added `defer_lm_head` to four model families. At the algorithm level we registered a `classify` algorithm with `register_algorithm` rather than modifying the factory, following the repository's extension rule, and added `grouped_softmax_loss` plus a `ClassifyTrainer` that handles encoding, whole-question packing, data-parallel interleaving, and weight normalization. The example layer adds a records loader, dataset converters, vendored `typed-decisions`, training and evaluation scripts, a temperature-fit script, a decisions API server, and a JevForge export script.

Three issues were found and handled. The official Ling model definition is based on transformers 4.45 and fails to import under transformers 5.15 or later because `is_torch_fx_available` is unavailable at import time, so evaluation and serving use our own adapter. The FastAPI server deliberately omits `from __future__ import annotations`, which would break FastAPI's runtime annotation resolution. One engine test hard-codes a four-GPU configuration and fails on a single-GPU machine; this is unrelated to our work, and the CPU test suite is run with `CUDA_VISIBLE_DEVICES=` unset. Reproducible commands and artifact paths are given in Appendix A.

---

## 4. Experiments

### 4.1 Setup and a smoke test

We ran all experiments on one NVIDIA DGX Spark with a GB10 GPU, 128 GB of unified memory, and a single process (`--world-size 1 --tp-size 1`). The base model was `Ling-3.0-tiny` in the `bailing_hybrid` configuration: 24 layers, 128 experts with top-8 routing plus one shared expert, MLA and KDA attention [17], and 1536 hidden units [27]. Training used 4-bit AdamW [24,25] with flash attention [26].

We first ran a smoke test on synthetic data (a button-colour choice task and a completion boolean task, 256 training and 40 development records) to confirm that the full path from encoding through the scoring head, grouped softmax, checkpointing, packed inference, and the decisions API executes end to end. Training for 20 steps drove cross-entropy from 0.81 to near zero with no out-of-memory failure, development accuracy reached 1.000, and the API returned valid answers for all three answer types with about 1 s for the first request including warmup. This test establishes only that the pipeline runs; it supports no capability claim.

### 4.2 Training runs under matched hyperparameters

We trained three models. Run v1 used Open-Jev v1.1 as the sole data source, with 147,139 training questions, of which about 82,000 were WANLI NLI [18,19] and the `open_jev_question` source used the option text itself as the option description for choice questions; about 99% of targets were one-hot. A token-length analysis under the Ling tokenizer gave a 95th percentile of 876 and a maximum of 1436 tokens for the longest candidate path, which set `MAX_LEN=1536`; a limit of 512 would have discarded 32% of examples. Run v1 trained for 500 steps in 4.85 h at about 37 s per step, and the training cross-entropy over the first ten and last ten windows moved from 1.2947 to 0.5767.

Run v2 kept every v1 hyperparameter and changed only the data. Because the Hugging Face Hub was unreachable from the machine (timeout before the TLS handshake, no proxy) while ModelScope delivered 20.8 MB/s, all data were fetched from ModelScope at pinned revisions with recorded SHA-256 digests. The v2 mix combined Open-Jev v1.1, Open-Jev v1 with `id: description` choice formatting, and a whitelist of JevEmbed-Data sources [20]. Each source was admitted through a license and contamination whitelist. It was built with a bucket-based water-filling mixer: 64,332 training questions distributed as 16,191 `choice_desc`, 5,131 `choice_nodesc`, 3,280 WANLI, 12,857 `noul_crit`, 8,887 `noul_nocrit` (of which 4,397 received generic criteria), 11,561 `score`, and 6,425 `soft`, with no duplicate identifiers. Run v2 trained for 500 steps in 5.88 h at about 43 s per step, with the first-to-last cross-entropy window moving from 1.2545 to 0.5418.

Run v3 was the zero-shot baseline. We left `Ling-3.0-tiny` untrained and answered each question with its language-model head [27], normalizing over the tokens that constitute legal answers (option letters, yes or no, or level digits) with thinking disabled. Probe results showed that no answer prefix was best: the label mass on legal answer tokens was 0.9646 for the empty prefix, 0.8745 for a newline, 0.3902 for `Answer: `, and 0.3888 for a newline followed by `Answer: `. Label mass was high for choice (0.995) and score (0.994) but only 0.548 for `noul`, because about 45% of probability mass fell on other tokens such as capitalized `Yes` or `No`. The zero-shot `noul` estimate is therefore likely conservative.

### 4.3 Training does not improve accuracy on `typed-decisions`

On the `typed-decisions` test set, the trained head and the untrained base model performed at the same level. The zero-shot baseline reached 0.567 accuracy, and v1 reached 0.582 at step 500. The difference of 0.015 is within noise: with 2000 decisions, the standard error of an accuracy near 0.58 is about 0.011, and the range across v1 checkpoints was 0.565 to 0.589. Training therefore did not clearly improve zero-shot accuracy on this benchmark, and it did not narrow the gap to the best generalist entries.

Table 1 places these results against published generalist entries. The trained head sits between the prior baseline and the generalist systems, but both the head and its base model remain well below Jev 1.13 at 0.727 and Featherless 35B-A3B at 0.716. Figure 2 shows the same contrast over training: in-distribution accuracy rises steadily, whereas out-of-distribution accuracy stays flat and tracks the zero-shot base model.

**Table 1 | Accuracy and calibration on `typed-decisions` test, compared with published generalist entries.**

| Model / stage | acc | KL | TV | Brier |
| --- | --- | --- | --- | --- |
| meraGPT Decider 1 | 0.768 | 0.096 | 0.149 | 0.052 |
| TypeSafe Jev 1.13.0 | 0.727 | 1.442 | 0.251 | 0.148 |
| Featherless Simple Jev (35B-A3B) | 0.716 | 0.488 | - | 0.176 |
| Teacher self-agreement | 0.735 | - | - | - |
| Factor-fitted ceiling | 0.704 | - | - | - |
| Zero-shot LM head (this work) | 0.567 | 0.893 | 0.383 | 0.326 |
| v1 scoring head, step 100 | 0.565 | 0.316 | 0.292 | 0.178 |
| v1 scoring head, step 500 | 0.582 | 0.443 | 0.305 | 0.229 |
| v2 scoring head, step 500 | 0.561 | 0.544 | 0.316 | 0.248 |
| Prior baseline | 0.470 | 0.347 | 0.317 | 0.189 |
| Uniform baseline | 0.308 | 0.444 | 0.381 | 0.238 |
| ModernBERT-base (149M)† | 0.646 | 0.223 | - | 0.119 |
| MiniLM-L6 (22M)† | 0.587 | 0.262 | - | 0.143 |

![Training and in-distribution accuracy rise together; out-of-distribution accuracy does not.](_static/figures/fig2_accuracy_trajectory.png)

**Figure 2 | Accuracy trajectories for the v1 and v2 runs.** Each point is read directly from the archived evaluation files. In-distribution accuracy (Open-Jev v1.1 dev) rises steadily for both runs, while out-of-distribution accuracy (typed-decisions test) stays within the 0.56 to 0.59 band and closely tracks the zero-shot base model (dotted line, 0.567). The annotated bracket at step 500 marks the in-distribution test split, where v1 rose from 0.557 zero-shot to 0.799. The shaded band marks the benchmark's 0.70 to 0.75 strong/saturation reference range.

Rows above the `Prior baseline` row are published entries and benchmark references; the zero-shot, v1, and v2 rows are from this study. The uniform row was reproduced exactly by our evaluation code. †Specialist models are fitted per workflow with fixed label spaces and are not directly comparable to generalist results [2]; we report them only to bound what a fitted classifier of comparable scale achieves on these four workflows.

The per-type breakdown explains the aggregate. Relative to the zero-shot baseline, v1 improved score accuracy from 0.482 to 0.534 and held choice accuracy nearly constant at 0.592 against 0.588, while `noul` accuracy decreased slightly from 0.658 to 0.635. The three types therefore moved in different directions and partially cancelled, which is consistent with the aggregate difference being noise rather than a uniform gain.

### 4.4 What training changes is calibration

Although accuracy did not move, the output distribution changed substantially. Cross-entropy fell from 0.893 to 0.443 and the Brier score from 0.326 to 0.229 at step 500 of v1, and the step-100 checkpoint reached 0.316 and 0.178 respectively. Overconfidence fell from +0.228 in the zero-shot baseline to +0.103 at step 500 and -0.030 at step 100. Expected calibration error followed the same pattern, falling from 0.230 to 0.106 at step 500.

Two features of this trajectory are notable. First, the best calibration occurred early and degraded with further training: step 100 was best on KL, Brier, overconfidence, and ECE, while step 500 was best on accuracy but worse on every calibration measure. Second, the miscalibration that training removed was concentrated out of distribution. At step 500, v1 was near-perfectly calibrated on the in-distribution Open-Jev v1.1 test set (overconfidence -0.006, ECE 0.018) while remaining overconfident on `typed-decisions` (overconfidence +0.103, ECE 0.106). The step-500 checkpoint thus behaves like a model that has learned the confidence structure of its training distribution and applies it to a different one.

This pattern also shows that the loss is not the source of the problem. A loss that produced globally miscalibrated outputs would leave the model miscalibrated on the training distribution as well; the near-zero in-distribution overconfidence indicates that the objective and the head are functioning as intended. Figure 3 makes both points visible at once: the trained model moves toward the diagonal only on the in-distribution split, while on typed-decisions the largest residual overconfidence sits in choice and noul. In the tables and figures that follow we report KL divergence from the gold distribution, Brier score, mean confidence (`conf`), overconfidence (`overconf = conf - acc`), and expected calibration error with 10 bins.

![Calibration before and after training.](_static/figures/fig3_calibration.png)

**Figure 3 | Calibration, aggregate and per answer type.** Left: mean confidence against accuracy on typed-decisions test. The diagonal is perfect calibration. The zero-shot base model is strongly overconfident; v1 at step 100 is close to the diagonal; continued training to step 500 moves back above it, and v2 is worse than v1. The v1 checkpoint on the in-distribution Open-Jev test set sits almost exactly on the diagonal. Right: overconfidence by answer type for the zero-shot baseline, v1, and v2 on typed-decisions. Training reduces overconfidence in all three types and the most in score, while the format-completed v2 run partially reverses the gain in choice and noul.

### 4.5 Temperature scaling does not transfer

We fitted a temperature on the Open-Jev calibration split, which contains 4000 questions. The fitted value was 1.0476 overall, with 1.0592 for choice, 1.0391 for `noul`, and 1.02 for score, and calibration cross-entropy moved from 0.5662 to 0.5658. Applying this temperature to `typed-decisions` changed accuracy not at all and reduced KL from 0.443 to 0.422 and the Brier score from 0.229 to 0.222, leaving overconfidence at +0.093. Per-type temperatures produced nearly the same result. A temperature fitted in distribution is therefore almost unity and largely ineffective out of distribution, which is the expected outcome when the miscalibration arises from distribution shift rather than from a monotone overconfidence of the training distribution. Figure 4 shows the fitted values and the small metric change they produce.

![Fitted temperatures and their limited effect.](_static/figures/fig4_temperature.png)

**Figure 4 | Temperature scaling does not transfer.** Left: per-type reliability on typed-decisions test before (circles) and after (squares) applying the globally fitted temperature; the two markers nearly coincide for every type. Right: temperatures fitted on the in-distribution calibration split are close to 1 for every label and answer type in both runs, which is why rescaling cannot repair an out-of-distribution shift.

### 4.6 Supplying missing answer-format structure does not help

Our initial hypothesis for the residual overconfidence was a format gap. In Open-Jev v1.1, choice options have no descriptions and `noul` questions have no criteria, whereas `score` questions have both and were, in the step-500 checkpoint, the best-calibrated type. We tested this by building the v2 mix, which adds described options and criteria.

The hypothesis was not supported. After correction of the format gap, out-of-distribution overconfidence increased rather than decreased: choice overconfidence rose from +0.146 to +0.223 and `noul` from +0.148 to +0.189, with aggregate accuracy falling from 0.582 to 0.561 and ECE rising from 0.106 to 0.173. Figure 5 shows the per-type direction of change and the bucket composition of the v2 mix that produced it.

![The format-gap hypothesis was tested and failed.](_static/figures/fig5_format_gap.png)

**Figure 5 | The format-gap hypothesis failed.** Left: out-of-distribution overconfidence by answer type for v1 and the format-completed v2 run; arrows show the direction of change. Adding option descriptions and criteria moved overconfidence up in choice and noul rather than down. Right: composition of the v2 training mix by bucket, with the buckets that supply option descriptions and criteria highlighted.

Score accuracy degraded from 0.534 to 0.480, which we attribute to negative transfer from language-model-judged score data of the kind found in sources such as prometheus [21], helpsteer [22], and ultrafeedback [23]. In-distribution behaviour was essentially unchanged: Open-Jev v1.1 test accuracy moved from 0.799 to 0.780, and the v2 in-distribution development accuracy was 0.753 at an overconfidence of -0.008. The temperature fitted on the calibration split was 1.0698 and likewise failed to transfer.

Across both runs, the same regularities held: `typed-decisions` accuracy stayed between 0.56 and 0.59 at every checkpoint, overconfidence rose monotonically with training, and greater data diversity raised overall confidence without raising accuracy.

We record the provenance of this hypothesis plainly, because it affects how the negative result should be read. The format-gap explanation was formed after observing the v1 calibration results and was not stated as a design hypothesis in a document before v2 was trained. It is therefore a post-hoc attribution, and we treat v2 as a test of that attribution rather than as a pre-registered prediction.

### 4.7 A bound on the training route

The shared explanation for the two negative results is that the interface was not the binding constraint. The score head improved calibration because the interface, a joint distribution over candidates with a Brier term, is exactly what calibration requires. It did not improve accuracy because the base model already answers at roughly its own ceiling, and a newly initialized head trained from scratch on a few thousand questions has no mechanism to acquire business decision knowledge that pretraining did not supply. This is consistent with the zero-shot baseline, which reached 0.567 with no training at all: that number appears to be the base model's generalist decision ability, and none of our runs exceeded it by more than noise. On a question this benchmark cannot itself settle, whether the ceiling belongs to model scale or to pretraining knowledge, we did not isolate the two factors.

The in-distribution result gives the complementary bound. On Open-Jev, v1 improved from 0.557 to 0.799, a gain of 24 points, and reached near-perfect calibration. The training route is therefore effective when the target distribution is the training distribution, and its failure on `typed-decisions` is a transfer failure rather than a training failure.

![Where this work sits against published systems and reference baselines.](_static/figures/fig6_leaderboard.png)

**Figure 6 | Accuracy against calibration on typed-decisions test, with published context.** Each trained configuration is plotted by accuracy (horizontal) and KL divergence from the teacher distribution (vertical, lower is better). Reference baselines and the factor-fitted ceiling are shown alongside published generalist and specialist systems. This work's trained heads and zero-shot baseline occupy the lower-accuracy region and, on KL, sit closer to the teacher than the prior baseline but do not approach the generalist leaders on accuracy. Specialist models marked † were fitted to the workflow and are not directly comparable.

---

## 5. Discussion

This study set out to test whether a trainable candidate-path scoring head could raise generalist decision accuracy on `typed-decisions`. It did not. The base model's zero-shot accuracy was 0.567, and every checkpoint of both training runs fell within 0.56 to 0.59, a spread consistent with the 0.011 standard error of the measurement. We therefore conclude that the route as configured is bounded by the base model on this benchmark, and that the reproducible benefit of training is calibration rather than capability.

This negative result is informative because it is specific, but one rival explanation must be addressed before we attribute it to the base model's decision knowledge. Because the benchmark's gold is the mean of three teacher samples, a score measures agreement with that teacher rather than correctness, and a model that is right where the teacher is wrong is penalised [2]. A model can therefore fail to improve for a reason that has nothing to do with its own capability: it may be learning the task while diverging from the teacher's idiosyncrasies. This explanation would also predict a ceiling on measured accuracy, and it is consistent with the observation that our accuracy stayed near 0.57 to 0.59 across every configuration. We cannot separate it from a genuine capability ceiling with the evidence we have, and we state it as an open ambiguity rather than dismissing it.

The rival explanation that we can exclude is that the training interface was at fault. The most plausible interface-level candidate, a missing answer-format structure, was tested directly and failed: adding described options and criteria increased out-of-distribution overconfidence. A miscalibrated objective is also excluded, because the same loss produced near-perfect in-distribution calibration. What remains, within the bounds of what the benchmark can measure, is the base model's decision knowledge or its agreement with the teacher. Under that reading the fitted head learns to express confidence over the candidate set but cannot create discriminative signal that the backbone does not encode, and it cannot acquire the teacher's specific idiosyncrasies from a different workflow distribution. This also explains why the zero-shot language-model head is a strong baseline here: the base model is not failing to answer, it is answering at its own level, and that level is close to the published 0.520 majority floor and well below the 0.704 factor ceiling and the 0.735 teacher self-agreement reference [2].

A further structural point bounds the comparison. The benchmark states that specialist and generalist scores are not comparable, because specialists are fitted per workflow with fixed label spaces while generalists answer unseen schemas zero-shot [2]. Our 0.582 is a generalist number, and it sits below the published specialists (ModernBERT-base 0.646, MiniLM-L6 0.587) and far below the generalist leaders. That does not change our conclusion, but it does mean our result should not be read as a statement about what a fitted classifier of this size could achieve on these four workflows.

The calibration result carries a practical nuance. The best-calibrated checkpoint in both runs was the earliest one we saved, at step 100, while the best accuracy was at step 500. Because the accuracy differences between these checkpoints are within noise but the calibration differences are not, a practitioner whose decisions consume confidence estimates should prefer an early checkpoint, whereas one who needs the head's nominal accuracy gains nothing measurable from training further. Temperature scaling is not a substitute, because a temperature fitted in distribution is close to 1.05 and does not correct the out-of-distribution shift.

Our conclusions are bounded in several ways. First, the study covers one model family (`bailing_hybrid`) and one model size; the observed ceiling may or may not persist at larger scales, and we did not test a larger model zero-shot, so the scale and knowledge explanations remain confounded. Second, each configuration was trained once, so we can establish that training did not significantly improve zero-shot accuracy, but we cannot rank step 300 against step 400 or one data mix against another. Third, our data coverage was thin: run v1 saw about 16,000 questions, roughly 0.11 epochs, over 500 steps. Fourth, two supporting analyses, the token-length analysis that set `MAX_LEN` and the leakage screen that excluded JevEmbed-Data `system_one_270m`, could not be independently re-verified because their scripts were not retained, though the exclusion decision itself is recorded in the code. Fifth, our evaluation reproduced the uniform baseline exactly but not the prior baseline, with differences of 0.01 to 0.09, so prior-based comparisons should be treated as indicative. Finally, we did not evaluate on Jevals ground-truth labels or on any in-house business workflow, which is where we would expect the route to be most useful.

These boundaries suggest three concrete directions. The first is to test a larger base model zero-shot on the same benchmark, which would separate scale from knowledge without any change to the pipeline. The second is to replace hard-label training with soft targets distilled from a teacher following the `typed-decisions` construction, which is the only route we can identify that could inject decision knowledge the base model lacks. The third is to apply the same recipe to in-house workflow data, where the in-distribution gain of 24 points suggests a substantially larger payoff, and to select the checkpoint on an in-distribution development split rather than on the out-of-distribution benchmark.

---

## 6. Conclusion

We implemented a JevForge-style trainable candidate-path scoring head on `Ling-3.0-tiny` in AReno and evaluated it on the generalist `typed-decisions` benchmark with a shared, packed forward pass across training, evaluation, and serving. Training did not improve decision accuracy: the head reached 0.582 against 0.567 for the untrained base model, a difference within measurement noise, and no configuration moved the benchmark above the base model's own zero-shot level. Training did improve calibration consistently, reducing KL divergence from 0.893 to 0.316-0.443 and the Brier score from 0.326 to 0.178-0.229, with the best-calibrated checkpoint being the earliest saved one. The strongest benefit was in distribution, where Open-Jev accuracy rose by 24 points. These results indicate that a candidate-path scoring head is an effective calibration and in-distribution adaptation interface for a compact mixture-of-experts model, and that its generalist accuracy on `typed-decisions` is limited by the base model's own decision ability or its agreement with the benchmark teacher, rather than by the head or the objective. The implication is bounded to a single device, a single model family, 500 training steps, and one zero-shot generalist benchmark.

---

## References

[1] TypeSafe AI. System One primitives and API. https://typesafe.ai (accessed 2026).

[2] LocalLLaMA. `typed-decisions`: a benchmark for typed probabilistic decisions over shared state. Hugging Face dataset, revision `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8`, Apache-2.0. https://huggingface.co/datasets/LocalLLaMA/typed-decisions

[3] TypeSafe. Jev 1.13.0 decision model. Reported on the `typed-decisions` leaderboard (accessed 2026).

[4] Featherless AI. Simple Jev (`Qwen3.6-35B-A3B-classifier`). https://simple-jev.featherless.ai

[5] Warner, B., et al. Smarter, better, faster, longer: a modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. arXiv:2412.13663, 2024. (ModernBERT)

[6] Reimers, N., Gurevych, I. Sentence-BERT: sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP*, 2019. (MiniLM is introduced in Wang, W., et al. Minilm: deep self-attention distillation for task-agnostic compression of pre-trained transformers. In *Advances in Neural Information Processing Systems*, 2020.)

[7] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., Hinton, G. E. Adaptive mixtures of local experts. *Neural Computation* 3, 79-87, 1991.

[8] Brier, G. W. Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78, 1-3, 1950.

[9] Gneiting, T., Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102, 359-378, 2007.

[10] Niculescu-Mizil, A., Caruana, R. Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning*, 625-632, 2005.

[11] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning*, 1321-1330, 2017.

[12] Ouyang, L., et al. Training language models to follow instructions with human feedback. In *Advances in Neural Information Processing Systems*, 2022.

[13] Hu, E. J., et al. LoRA: low-rank adaptation of large language models. In *International Conference on Learning Representations*, 2022.

[14] Recht, B., Roelofs, R., Schmidt, L., Shankar, V. Do ImageNet classifiers generalize to ImageNet? In *Proceedings of the 36th International Conference on Machine Learning*, 5389-5400, 2019.

[15] Hinton, G., Vinyals, O., Dean, J. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.

[16] Fedus, W., Zoph, B., Shazeer, N. Switch Transformers: scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research* 23, 1-39, 2022.

[17] Shazeer, N. Fast transformer decoding: one write-head is all you need. arXiv:1911.02150, 2019. (Multi-query attention, the basis of the MLA-style attention used by the base model.)

[18] Cai, Z., et al. Open-Jev v1.1. ModelScope dataset `ZefanCai/Open-Jev-v1.1`, config `community-hard-mix-v2-redistributable`. (Includes WANLI, distributed under CC BY 4.0.)

[19] Liu, A., et al. WANLI: worker and AI collaboration for natural language inference dataset creation. In *Findings of EMNLP*, 2022.

[20] HIT-TMG. JevEmbed-Data. ModelScope dataset `HIT-TMG/JevEmbed-Data`.

[21] Kim, S., et al. Prometheus 2: an open source language model specialized in evaluating other language models. In *Proceedings of EMNLP*, 2024.

[22] Wang, Z., et al. HelpSteer: multi-attribute helpfulness dataset for SteerLM. In *Proceedings of NAACL*, 2024.

[23] Cui, G., et al. UltraFeedback: boosting language models with scaled AI feedback. In *Proceedings of ICML*, 2024.

[24] Loshchilov, I., Hutter, F. Decoupled weight decay regularization. In *International Conference on Learning Representations*, 2019. (AdamW, used here in its 4-bit form.)

[25] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. QLoRA: efficient finetuning of quantized LLMs. In *Advances in Neural Information Processing Systems*, 2023. (Basis of 4-bit optimizer states.)

[26] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. FlashAttention: fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems*, 2022.

[27] InclusionAI. Ling-3.0-tiny. ModelScope model `inclusionai/ling-3.0-tiny`, architecture `bailing_hybrid`.

[28] zwliJay. jev-forge: reference implementation of the JevForge decision scorer. https://github.com/zwliJay/jev-forge

## Data and code availability

The `typed-decisions` test set is vendored in the repository under `examples/classify/jev/data/typed-decisions/` and is distributed under Apache-2.0. All data sources used for training are publicly available and were fetched at pinned revisions with recorded SHA-256 digests. The implementation is contained in ten commits on the `feat/classify-jev` branch of the AReno repository. Reproducible commands and artifact paths are listed in Appendix A.

## Author contributions

[To be supplied.]

## Competing interests

[To be supplied.]

## Acknowledgements

[To be supplied.]

---

## Appendix A. Reproduction

The following commands reproduce each experiment. Runs v2 and v3 require the `feat/classify-jev` branch, whose head is `c98359a`; the `feat/classify-head` branch does not contain the v2 or zero-shot scripts.

```bash
# Select the experiment branch (v2 and zero-shot scripts exist only here)
git checkout feat/classify-jev      # c98359a

# E0 smoke test on synthetic data
STEPS=20 DATA=smoke bash tmp.sh

# E1 v1 training, evaluation, and serving (default 500 steps, ~4.85 h)
bash tmp.sh
bash tmp2.sh ~/ling-jev-openjev.log      # training summary, eval comparison, latency

# D1 diagnostics (checkpoint trend, calibration, temperature)
bash tmp3.sh

# E2 v2 mixed-data training under matched hyperparameters and v1/v2 comparison (~5.9 h)
bash tmp5.sh

# E3 zero-shot baseline
bash tmp6.sh

# Data and network preflight
bash tmp4.sh
```

Artifact locations:

| Content | Path |
| --- | --- |
| v1 checkpoints | `~/areno-runs/ling-3.0-tiny-jev-run/step_000{100..500}` |
| v2 checkpoints | `~/areno-runs/ling-3.0-tiny-jev-v2-run/step_000{100..500}` |
| v1 diagnostics and temperature | `~/areno-runs/diag/` |
| v2 diagnostics and temperature | `~/areno-runs/diag-v2/` |
| Zero-shot results | `~/areno-runs/zero-shot/` |
| Training logs | `~/ling-jev-openjev.log` (v1), `~/ling-jev-2.log` (v2) |
| Diagnostic log | `~/ling-jev-dial.log` |
| Zero-shot log | `~/ling-jev-zeroshot.log` |
| Data records | `~/data/jev-records/{open-jev-v1.1,open-jev-v1,jevembed,mix-v2,typed-decisions}` |

---

## Appendix B. Additional results

### B.1 Checkpoint trajectory of the v1 run

Table B.1 reports the full step trajectory of the v1 run on `typed-decisions` test and on 1000 sampled Open-Jev development questions. Accuracy on `typed-decisions` peaks at step 300 (0.589) and then declines slightly, while KL divergence, Brier score, and overconfidence worsen monotonically. In-distribution accuracy rises throughout the run.

**Table B.1 | Step trajectory of run v1.**

| step | TD acc | TD KL | TD Brier | TD overconf | OJ dev acc | OJ dev overconf |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.565 | 0.316 | 0.178 | -0.030 | 0.670 | -0.034 |
| 200 | 0.570 | 0.381 | 0.209 | +0.075 | 0.704 | +0.029 |
| 300 | 0.589 | 0.393 | 0.210 | +0.071 | 0.741 | +0.000 |
| 400 | 0.585 | 0.439 | 0.228 | +0.101 | 0.757 | +0.004 |
| 500 | 0.582 | 0.443 | 0.229 | +0.103 | 0.746 | +0.015 |

### B.2 Calibration at step 500 of run v1

Table B.2 reports calibration at step 500. In-distribution calibration is near-perfect, and the entire miscalibration is out of distribution.

**Table B.2 | Calibration of run v1 at step 500 with temperature 1.**

| Split / type | acc | conf | overconf | ECE |
| --- | --- | --- | --- | --- |
| Open-Jev v1.1 test (in distribution) | 0.799 | 0.793 | -0.006 | 0.018 |
| `typed-decisions` (out of distribution) | 0.582 | 0.685 | +0.103 | 0.106 |
| - choice | 0.592 | 0.738 | +0.146 | 0.146 |
| - noul | 0.635 | 0.783 | +0.148 | 0.166 |
| - score | 0.534 | 0.571 | +0.037 | 0.048 |

### B.3 Per-type comparison of v1, v2, and zero-shot

Table B.3 gives the per-type breakdown that underlies the aggregate comparisons in the main text.

**Table B.3 | Per-type accuracy and overconfidence.**

| Metric | Zero-shot LM head | v1 step 500 | v2 step 500 |
| --- | --- | --- | --- |
| choice acc / overconf | 0.588 / +0.239 | 0.592 / +0.146 | 0.583 / +0.223 |
| noul acc / overconf | 0.658 / +0.212 | 0.635 / +0.148 | 0.647 / +0.189 |
| score acc / overconf | 0.482 / +0.232 | 0.534 / +0.037 | 0.480 / +0.122 |
| agent_trace acc | - | 0.456 | 0.426 |
| customer_service acc | - | 0.648 | 0.638 |
| invoice acc | - | 0.568 | 0.578 |
| security acc | - | 0.654 | 0.602 |
| Open-Jev v1.1 dev acc / overconf | - | 0.746 / +0.015 | 0.744 / +0.008 |
| Open-Jev v1.1 test acc / overconf | 0.557 / - | 0.799 / -0.006 | 0.780 / +0.004 |

### B.4 Full v1 and v2 comparison at step 500

**Table B.4 | v1 against v2 at step 500.**

| Metric | v1 | v2 |
| --- | --- | --- |
| TD acc / KL / TV / Brier | 0.582 / 0.443 / 0.305 / 0.229 | 0.561 / 0.544 / 0.316 / 0.248 |
| Fitted temperature | 1.0476 | 1.0698 |
| TD at fitted T: acc / KL / Brier / overconf | 0.582 / 0.422 / 0.222 / +0.093 | 0.561 / 0.502 / 0.237 / +0.158 |
| mix-v2 dev acc / overconf | - | 0.753 / -0.008 |

### B.5 Serving latency

After warmup, the decisions API served the official example request (3 questions, 8 candidate paths) with a median latency of 212 ms and a 90th percentile of 219 ms over 20 requests, with a minimum of 194 ms. These are local measurements and are not comparable to an end-to-end hosted API measurement. A single checkpoint re-evaluated on a different day differed by 0.001 in accuracy with all other metrics unchanged, which we attribute to bf16 numerical variation rather than a regression.

### B.6 Notes on differences from the original experiment record

The main tables in this paper were re-read from the archived metric files and agree with the original record. Two numbers were added from the logs because the record did not state them directly: the training cross-entropy windows (v1 1.2947 to 0.5767; v2 1.2545 to 0.5418). The record's estimate of about 37 s per step for v2 was superseded by the measured 43 s per step and 5.88 h. The token-length analysis and the leakage screen could not be independently re-verified, as noted in Section 3.7.