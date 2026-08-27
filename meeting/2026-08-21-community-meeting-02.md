# AReno Community Meeting #2

Date: 2026-08-21
Time: 13:00-14:00 China Time (UTC+8)
Participants: Not counted
Format: Online community meeting

## Summary

The second AReno community meeting focused on recent release updates,
multimodal and local training capabilities, open source contributor
recruitment, external community collaboration, future technical directions, and
growth paths for new contributors.

The meeting covered capabilities that have been merged or are in progress after
`0.0.7`, including the experimental `MLX backend`, Gemma4 multimodal dynamic
understanding, MiniCPM-V-4.6 multimodal support, and Ling-3.0-tiny training
support. The discussion also focused on VLA/embodied AI training, quantized
training, LoRA, optimizer capabilities on Mac, domestic accelerator support,
and community resource support.

## Topics

### 1. Community Gifts and Contributor Recognition

- Sun-protective clothing for the community has been produced. Contributors
  with recent PR activity will be selected for distribution.
- Stickers and other materials are also being prepared for conferences and
  booth activities in September.
- Contributors who continue submitting PRs, adapting models, improving
  documentation, or writing technical articles may receive community gifts,
  promotion support, and conference participation opportunities.

### 2. Recent Release and Model Support Progress

The meeting reviewed recently completed and ongoing work:

- `0.0.7` has been released, and some updates may continue into `0.0.8`.
- Experimental `MLX backend` support enables large model training on Mac. The
  current validation mainly uses small models, with further testing planned on
  higher-end Mac devices.
- Multimodal training support is planned for the `MLX backend`.
- Gemma4 image, audio, and video dynamic understanding is supported, including
  Gemma4 E2B/E4B, MoE, and Gemma4 42B unified variants.
- MiniCPM-V-4.6 multimodal support has been contributed by a community member
  and merged.
- Ling-3.0-tiny support has progressed, and an article has been published about
  AReno training and support for Ling-3.0-tiny.
- Several bug fixes were also included alongside the main capability updates.

The meeting also noted that open source base models such as Tiny and Flash give
the community more room for experimentation. Compared with heavily post-trained
models, base models are better suited for trying different additional training
methods and observing training effects.

### 3. Open Source Promotion Plan and Contributor Development

The meeting discussed Open Source Promotion Plan recruitment and longer-term
community participation:

- Candidate selection for the Open Source Promotion Plan is mostly complete.
  Ziyi Wang will work on the multimodal direction and continue collaborating
  with the community.
- Students who were not formally selected for the program are still welcome to
  join the community and continue contributing.
- The program is only an entry point. After the three-month period, contributors
  can still participate in AReno model adaptation, capability development, and
  community building.
- The community will gradually clarify roles such as contributor, committer,
  and maintainer, while seeking more conference, promotion, and endorsement
  resources for sustained contributors.

### 4. External Collaboration, Events, and Compute Resources

The community has recently connected with foundations, university teachers, and
related communities in Chongqing and the Sichuan-Chongqing region. External
partners are interested in questions such as:

- Whether AReno can help students gain practical AI experience and support
  employment or internship opportunities.
- Whether AReno can lower the cost of local AI, small model training, and
  teaching scenarios.
- Whether universities, communities, and cloud vendors can jointly organize
  workshops, hackathons, or capability certification activities.

Discussion points:

- Chongqing and the broader Sichuan-Chongqing region may have venue, funding,
  prize, and event organization resources. Specific formats will depend on
  follow-up collaboration plans.
- If external events take shape, they will be shared with the AReno community,
  and contributors with AReno hands-on experience will be encouraged to join.
- Partners such as China Telecom Cloud may provide domestic accelerator
  resources. The community should pay attention to possible support for Ascend
  910/950, HPU/NPU, and other domestic hardware.
- High-end GPUs such as H200 are difficult for a broad community to access, so
  domestic accelerators and local device support will become increasingly
  important.
- OpenInfra and related events will take place in Shanghai from September 7 to
  September 9. Interested community members can contact community organizers to
  request tickets. Events such as the Bund Summit may also be worth applying
  for.

### 5. Future Direction: VLA and Embodied AI Training

VLA/embodied AI training was discussed as a higher-priority direction.

Background:

- The community may collaborate with university teachers on embodied AI
  research projects and applied scenarios.
- Embodied AI often involves relatively small models, so local single-machine or
  single-node training still has substantial room for exploration.
- Existing embodied AI training frameworks are often built on heavy upper-layer
  abstractions. AReno integrates training and inference, which may offer
  advantages in local training, logprob computation, and performance overhead.

Discussion points:

- The community can start by exploring adaptation for VLA models such as Pi0,
  together with simulation environments, multimodal training, and action
  sequence output.
- Embodied AI should be broken down as an independent topic or milestone rather
  than treated as adaptation for a single model.
- Model support, simulation environments, training workflows, evaluation, and a
  demonstrable demo should be split into multiple claimable issues.
- If a minimal demonstrable result can be completed, the community can later
  connect with universities, embodied AI startups, robotics companies, and
  related interview or promotion resources.
- The community can further identify members with embodied AI experience to
  join the effort.

### 6. LoRA, Multimodal Training, and Mac Optimization

The meeting discussed several concrete directions for multimodal model training
and low-resource training optimization:

- Some multimodal support still freezes components such as towers and
  projectors, which limits multimodal training. Unfreeze support should be
  added gradually.
- Gemma4 already supports partial unfreeze. Qwen3.5, Qwen3.5-MoE, MiniCPM, and
  other models still need follow-up support.
- Current Gemma4 work focuses on E2B and 42B unified variants, while Gemma4 MoE
  multimodal support still needs to be completed.
- CUDA-side LoRA support is already being advanced through an internal PR. It
  may first cover Qwen and Ling-3.0-tiny, with other models to follow.
- LoRA support on the `MLX backend` is high priority and is suitable for
  low-cost training experiments on Macs and other local devices.
- On Macs with 36 GB or 48 GB unified memory, LoRA may make training 3B, 4B, or
  slightly larger models more feasible, but this still needs empirical testing.
- Strategies such as optimizer state offload to disk, bucket-level computation,
  and double-buffered prefetch/onload/offload can significantly reduce memory
  usage, although at some speed cost.
- CUDA-side optimizations such as 8-bit Adam, FP32 master optimizer, and
  optimizer offload can be organized into a list and gradually migrated to the
  `MLX backend`.
- These optimizations are suitable for smaller issues, lowering the barrier for
  community contributors and providing material for technical blog posts.

### 7. How New Contributors Can Participate and Grow

A participant asked how to continue contributing and improving if they are not
yet familiar with the AReno framework, beyond training agents and recording
training data.

Shared understanding:

- AReno covers the full chain of training, inference, framework work,
  algorithms, and model adaptation. Contributors can choose directions based on
  their career interests.
- Contributors interested in training can start with issues around training
  pipelines, LoRA, optimizers, memory optimization, and model post-training.
- Contributors interested in inference can study inference optimizations from
  other communities and try adapting them to AReno.
- Contributors interested in embodied AI can follow future model, environment,
  and evaluation tasks in the VLA milestone.
- Open source contribution is not only about PR count. It is also a way to
  learn industrial project norms, understand community collaboration, and build
  habits around explainable code and engineering practice.
- The community encourages contributors to start with small tasks and gradually
  build demonstrable AI engineering experience through continued practice.

## Decisions

- VLA/embodied AI training has higher priority than quantized training and
  should be broken down as a separate milestone.
- The `MLX backend` remains a key direction for local training, especially for
  LoRA and optimizer capabilities.
- Multimodal training needs better unfreeze support for components such as
  towers and projectors.
- Students who were not selected for the Open Source Promotion Plan can still
  join the community and continue contributing.
- The community will continue providing gifts, conference participation,
  technical article promotion, and other support for strong contributors.

## Action Items

- Define the VLA/embodied AI training milestone and split model adaptation,
  simulation environments, training workflows, evaluation, and demos into
  claimable issues.
- Continue proof-of-concept work for VLA models such as Pi0 and align with
  university collaboration scenarios.
- Clarify the technical boundaries of quantized training, including the modules
  needed for QAT and quantized model training.
- Complete Gemma4 MoE multimodal support, as well as multimodal unfreeze support
  for Qwen3.5, Qwen3.5-MoE, MiniCPM, and related models.
- Follow up on the CUDA-side LoRA PR and plan LoRA support on the `MLX backend`.
- Organize the existing CUDA-side optimizer optimization list and split smaller
  tasks that can be migrated to the `MLX backend`.
- Continue investigating domestic accelerator adaptation needs, including
  Ascend 910/950, HPU/NPU, and related resource timelines.
- Follow up with universities, foundations, cloud vendors, and other external
  partners in the Sichuan-Chongqing region to evaluate workshops or hackathons.
- Review recent strong PR contributors and arrange distribution of
  sun-protective clothing, stickers, and other community gifts.
- Encourage community members to write technical blog posts about AReno
  practice, with community support for editing and distribution.

## Open Questions

- Which model, simulation environment, and evaluation task should be used for
  the first minimal demonstrable VLA milestone demo?
- Does embodied AI training need real robot devices, or should the first loop be
  completed in simulation?
- Which domestic hardware and software stacks should be prioritized for
  accelerator adaptation?
- How should LoRA, 8-bit Adam, optimizer offload, and other `MLX backend`
  capabilities be prioritized?
- How can community compute resources be opened to trusted contributors while
  maintaining isolation from internal company environments and data?
- How should new-contributor issues be labeled by difficulty and prerequisite
  knowledge so that they are suitable for students?

## Next Meeting

Date: 2026-09-04
Time: 13:00-14:00 China Time (UTC+8)
Suggested agenda:

- Follow up on the VLA/embodied AI milestone breakdown.
- Share progress on LoRA, the `MLX backend`, and multimodal unfreeze support.
- Review progress on domestic accelerator resources and external collaboration.
- Continue open Q&A for new contributors and help participants choose suitable
  contribution directions.

## Links

- Recording:
- Transcript:
- Slides:
- Related issues:
