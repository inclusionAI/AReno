# AReno Community Meeting #3

Date: 2026-09-04
Time: 13:00-14:00 China Standard Time (UTC+8)
Attendance: Not counted
Format: Online community meeting

## Meeting Summary

The third AReno community meeting focused on the `0.0.8` release, the roadmap for foundational capabilities, contributor recognition, domestic compute collaboration, embodied intelligence, and follow-up work for the `MLX backend`.

The meeting reviewed capabilities planned for or under final validation in `0.0.8`, including LoRA support for the `MLX backend`, MiniCPM-V-4.6 multimodal improvements, training pipeline performance optimizations, R3 rollout routing replay, and the 8-bit AdamW optimizer. The discussion also emphasized that future foundational work should be organized by real usage scenarios rather than isolated technical items, with separate tracks for Mac Mini, consumer GPUs, DGX Spark, domestic accelerators, and embodied intelligence collaborations.

## Agenda

### 1. AReno 0.0.8 Release Progress

The meeting reviewed the main updates in `0.0.8`. The release was planned for September 4, 2026, with several capabilities still under final validation before publication.

Main progress:

- LoRA support for the `MLX backend` is being validated on Mac Mini.
- A community contributor completed additional MiniCPM-V-4.6 multimodal support, including freeze support for components such as the tower and projector.
- A new contributor submitted several PRs around trainers such as GSPO and GRPO, optimizing materialize overhead in the training pipeline.
- Training sequence packing and advantage computation received performance and memory optimizations.
- R3, or rollout routing replay, is now supported to improve routing consistency when training some MoE models.
- The 8-bit AdamW optimizer is supported. On environments such as DGX Spark, it can save more than ten GB of memory compared with regular AdamW while maintaining good results.
- In addition to major features, the release includes bug fixes, Docker improvements, and documentation updates.

### 2. Community Activities and Contributor Recognition

The meeting reviewed contributor evaluation and community swag distribution:

- More than 20 outstanding contributors were selected from recent community contributions.
- Backpacks, hiking bags, and sun-protection jackets have been sent out gradually.
- Stickers and other materials will continue to be tracked by the community operations team.
- The community will continue recognizing contributors based on PRs, model adaptation, performance optimization, documentation improvements, and other forms of contribution.

### 3. Roadmap for Foundational Capabilities

The meeting discussed future foundational capabilities for AReno. Initial candidates include:

- Inference-side MTP support. A contributor has started early work on Llama-related MTP support, but MTP involves multiple algorithms and model-specific MTP heads, so support will need to be added model by model.
- Version-aware prefix cache. When multiple rollouts share the same prompt or prefix, prefix computation can be reused to avoid repeated prefill. This is especially useful for agent-style multi-turn conversations and R1-like tasks.
- FP8 KV Cache. Compared with BF16 KV Cache, this may reduce memory usage by around half, but the accuracy impact still needs evaluation.
- Independent expert parallelism. Current expert parallelism is still coupled with tensor parallelism, and future work may split it into an independent capability.
- More complete context parallelism. AReno currently supports sequence parallelism, which is a weaker form of context parallelism. More complete support is needed for long-context training, such as 200K-context workloads.
- Multimodal encoder feature cache. Encoder features produced by multimodal models may be cacheable to reduce repeated computation.

The meeting concluded that these capabilities should not be advanced as scattered technical items. Instead, they should be organized around concrete tracks:

- Local training and inference on unified-memory devices such as Mac Mini.
- Low-barrier training on consumer GPUs.
- Training optimization for small workstation devices such as DGX Spark.
- Domestic accelerator adaptation.
- Embodied intelligence and external collaboration scenarios.

Memory and performance optimization on lower-end compute should receive high priority. Version-aware prefix cache, FP8 KV Cache, and low-bit optimizers should all serve the broader goal of making AReno runnable on more accessible hardware.

### 4. Domestic Compute Collaboration and Resource Constraints

The meeting discussed progress and constraints around domestic accelerator adaptation and external collaboration.

Current status:

- Partners such as the Modelers community also face compute scheduling and funding constraints, making large-scale domestic accelerator resources difficult to provide in the short term.
- Open-source small models from ModelBest may be a more realistic collaboration starting point. The community can first build examples with small models and available resources, then seek domestic accelerator support.
- If trial resources are used for community promotion, account limits, quotas, and stability need careful handling to avoid restrictions caused by large-scale parallel usage.
- Ascend, Hygon, and other domestic accelerator resources are subject to scheduling and rotation, so continuous fixed access is not guaranteed.
- Cambricon was also mentioned as a possible domestic compute direction for teaching and course projects, but further resource and engineering coordination is needed.

The meeting noted that compute remains the largest obstacle for domestic adaptation. If stable domestic accelerator resources are not available in the short term, the priority of related adaptation work should be adjusted according to resource availability.

### 5. Embodied Intelligence Collaboration and Demo Direction

The meeting reviewed progress on embodied intelligence.

Current plans:

- The embodied intelligence collaboration will start by supporting one algorithm and building a demonstrable demo.
- The initial priority is to complete an online RL demo in a simulation environment, rather than relying directly on real robots for data collection.
- Data collection may use a world-model-based workflow to reduce the cost of physical devices and real-time networking.
- There is already some VLA model adaptation groundwork, such as Pi0.5, but the full pipeline is still being built.
- If the embodied intelligence direction is used in courses or projects, the team can further explore support from domestic compute resources such as Cambricon.

The meeting considered embodied intelligence, Mac Mini, and DGX Spark as relatively high-priority directions. If the embodied intelligence demo can be completed, it will become an important external showcase for AReno's integrated training and inference capabilities.

### 6. MLX Backend and Mac Mini Direction

The meeting discussed follow-up work for the `MLX backend`.

Current progress and plans:

- `MLX backend` LoRA is under testing and will be included with the `0.0.8` release once validation is complete.
- Optimizer support for the `MLX backend` can continue next, including 4-bit and 8-bit AdamW.
- Mac Mini is a promising low-barrier device for local training and inference because of its unified memory.
- Future work will continue exploring what AReno can run on Mac Mini, how to make it more stable, and whether more LoRA and optimizer capabilities can be supported.

The meeting agreed that Mac Mini is meaningful for community accessibility and should be one of the key follow-up tracks for the `MLX backend`.

## Decisions

- `0.0.8` will be released as planned on September 4, 2026, after final validation of capabilities such as `MLX backend` LoRA.
- Future foundational capabilities should be organized by usage scenario rather than listed only as individual technical items.
- Mac Mini, consumer GPUs, DGX Spark, domestic accelerators, and embodied intelligence collaborations should each become trackable roadmap areas with priorities.
- Memory and performance optimization for lower-end compute should be prioritized, including prefix cache, FP8 KV Cache, and low-bit optimizers.
- The embodied intelligence direction should first complete an online RL demo in simulation before moving to real robots or more complex algorithms.
- Domestic accelerator adaptation should continue seeking resources, but its priority should be adjusted if stable accelerator access is not available in the short term.
- The `MLX backend` will continue to be a focus area, especially LoRA and optimizer support.

## Action Items

- Complete final validation for `0.0.8` and publish the release.
- Continue validating `MLX backend` LoRA and clarify its usability and limitations on Mac Mini.
- Define the next optimizer roadmap for the `MLX backend`, prioritizing evaluation of 4-bit and 8-bit AdamW.
- Reclassify the foundational capability roadmap by scenario: Mac Mini, consumer GPUs, DGX Spark, domestic accelerators, embodied intelligence, and related tracks.
- Continue improving MiniCPM-V-4.6 multimodal support and review related PRs.
- Follow up on trainer performance optimization PRs for GSPO, GRPO, and related trainers, including review, testing, and merge.
- Evaluate the effect of R3 rollout routing replay on MoE training consistency and add usage notes.
- Continue exploring version-aware prefix cache, FP8 KV Cache, context parallelism, and multimodal encoder feature cache.
- Follow up with the Modelers community, ModelBest small models, and domestic compute partners to identify short-term feasible collaboration paths.
- Have the embodied intelligence partner continue building the online RL demo pipeline in simulation.
- Continue coordinating with Cambricon and other domestic compute providers to evaluate course or project-based adaptation opportunities.
- Discuss the community meeting schedule in the group and decide whether the 13:00 time slot should be adjusted.

## Open Questions

- After splitting the foundational roadmap by scenario, how should the first batch of issues be defined for each track?
- How should Mac Mini, consumer GPUs, and DGX Spark be prioritized among low-barrier hardware targets?
- How should the benefits, accuracy impact, and implementation complexity of version-aware prefix cache and FP8 KV Cache be evaluated?
- For domestic accelerator adaptation, should the community first pursue Ascend, Hygon, Cambricon, or start from ModelBest small-model collaboration?
- Which algorithm and simulation environment should be selected for the first embodied intelligence demo?
- How should milestones be split for `MLX backend` LoRA, 4-bit AdamW, 8-bit AdamW, and related capabilities?
- Should the community meeting time be adjusted to improve attendance and discussion?

## Next Meeting

Date: TBD
Time: TBD
Suggested topics:

- Review feedback and bug fixes after the `0.0.8` release.
- Sync progress on `MLX backend` LoRA, optimizers, and Mac Mini validation.
- Follow up on scenario-based roadmap issues and priorities.
- Review progress on the embodied intelligence demo pipeline.
- Sync domestic compute and external collaboration progress.
- Confirm whether future community meetings should use a different time slot.
