# AReno Community Meeting #7

Date: 2026-11-27
Recording duration: 27 minutes 44 seconds
Format: Online community meeting

## Meeting Summary

The meeting reviewed preparations for AReno 0.12.0, including training/rollout consistency with Native Attention, bidirectional attention for JEV, and OPD support. The team discussed extending AReno Flow so a small model can manage training, diagnosis, and deployment on a single machine, then explore further iteration through Recursive Self-Improvement (RSI). Community work will focus on hands-on tutorials, university outreach, and Agent talks, alongside a possible collaboration on single-machine TTS training.

## Agenda

### 1. AReno 0.12.0 Release Preparation

- Target this weekend for 0.12.0, moving the release beyond the target discussed at the previous meeting.
- Default training and rollout paths are moving to the optimized Native Attention backend, reducing installation dependence on FlashAttention. The team reported zero difference between rollout and training log probabilities in the integrated configurations validated so far.
- JEV-related training gains bidirectional attention support. The team reported evaluation gains of more than ten points in related experiments. The evaluation protocol and configurations still need to be documented.
- OPD (On-Policy Distillation), contributed through the Open Source Promotion Plan, has been merged.
- The unified installer still needs to follow the new backend defaults and update its FlashAttention installation logic.

### 2. AReno Flow and Small-Model Self-Improvement

The team discussed the next stage of AReno Flow, with an Agent launching tasks, adjusting parameters, diagnosing failures, and deploying services, then feeding the resulting checkpoints into further iterations.

- Start with a small model managing training on a single DGX Spark or Mac mini to validate the complete workflow.
- Explore learning from data collected while executing training tasks to improve the model's training and diagnostic capabilities.
- Possible reward signals include successfully completing a workflow and finding and fixing problems. The design needs further work.
- Review reusable workflows and capabilities from Qimeng's ongoing game-based RSI experiments.
- The proposal is at the discussion stage. A separate discussion will define the first task, resource budget, and evaluation metrics.

### 3. Developer Tutorials and Community Activities

- Publish more tutorials and examples that developers can run locally to solve concrete problems, introducing RL through familiar Agent use cases.
- Continue preparing the technical exchange at the Hong Kong University of Science and Technology, focusing on single-machine scaling and training/rollout consistency.
- Plan an open-source campus visit to Jinling Institute of Technology in Nanjing next Friday, with students running AReno on site.
- Explore participation in an Agent Meetup proposed for December 12 in Beijing. Candidate topics include RSI, Agent training, and standardized experiment workflows. The topic is still to be confirmed.
- Continue preparing the December 18 AICon talk in Beijing, with a rehearsal planned for next Wednesday and an article to follow.
- Invite contributors from the Open Source Promotion Plan to continue participating in community meetings.

### 4. Exploring TTS and Single-Machine Training Collaboration

- Yuhan shared work on full-duplex speech models, Text-to-Speech (TTS) inference optimization, and traffic scheduling across multiple network interfaces.
- The group discussed evaluating AReno for single-machine training once the first model release is ready for integration, with joint work on performance and memory optimization.
- Yuhan plans to raise the collaboration in an internal discussion next week, then clarify next steps based on model and training requirements.

## Decisions

- Continue working toward the 0.12.0 release this weekend and update the unified installer.
- Refine a proof of concept combining AReno Flow, small models, and RSI.
- Continue publishing local, hands-on tutorials for a broader developer audience and organize community talks around concrete Agent use cases.
- Follow up on the feasibility and prerequisites of the proposed TTS training collaboration.

## Action Items

- He Laoshi will update the unified installer to follow the default Native Attention path.
- Document configurations, evaluation protocols, and results for training/rollout consistency and bidirectional attention experiments.
- Su Le and other contributors will refine AReno Flow's task execution, diagnosis, training, and deployment workflow, and discuss an initial task with Qimeng's RSI work.
- Define rewards, resource requirements, and evaluation metrics for the single-machine small-model experiment.
- Prepare university hands-on materials, Agent Meetup topic proposals, and the AICon rehearsal.
- Yuhan will follow up on the TTS collaboration and clarify model integration, performance, and memory requirements.
- Invite relevant Open Source Promotion Plan contributors to future community discussions.

## Open Questions

- What should be the first self-improvement task for AReno Flow, and how should workflow completion and model capability gains be measured?
- How much data, time, and compute would iterative training require on a Mac mini or DGX Spark?
- Which Agent use cases are suitable for reproducible local training tutorials for a broader developer audience?
- Which training interfaces and runtime capabilities are needed to integrate the TTS model with AReno?

## Next Meeting

Date: TBD. Suggested topics include the 0.12.0 release, the AReno Flow and RSI proof of concept, tutorials, community events, and the proposed TTS collaboration.
