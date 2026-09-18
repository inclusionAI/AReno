# AReno Community Meeting #4

Date: 2026-09-18
Time: Not recorded
Attendance: Not counted
Format: Online community meeting

## Meeting Summary

The fourth AReno community meeting reviewed recent progress on hardware and multimodal support, AReno Flow, local training on Mac, and community partnerships. The group also discussed a proof of concept for Recursive Self-Improvement (RSI).

Initial Huawei Ascend NPU adaptation is largely complete, with common small models running or under validation, while the latest multimodal work is nearing completion. For users without local GPUs, the team is developing AReno Flow to make cloud GPU training accessible through an Agent-based workflow. The meeting agreed to publish the next release after the relevant changes are merged and to launch community proposals for AReno RSI and Mac platform improvements.

## Agenda

### 1. Release and Foundational Capability Progress

The meeting reviewed recent development work:

- Initial Huawei Ascend NPU adaptation is largely complete. Common small models, including Qwen and Bailing models, are running or being validated.
- The current round of multimodal support is nearing completion. The next phase will be planned after the remaining changes are merged.
- Community-contributed MTP support for Bailing has been implemented and still needs final validation and merge.
- The next release is planned for next week, after the Ascend NPU, AReno Flow, MTP, and related changes are merged.

### 2. AReno Flow

The team introduced an early design for AReno Flow. It targets users who do not have local GPUs or only have consumer GPUs with limited memory, providing a lower-barrier training environment through cloud GPUs.

The proposed workflow includes:

- Users provide an API token for the cloud platform and start training through AReno without preparing their own GPU environment.
- The Agent in AReno Desktop estimates cost, recommends a suitable GPU, and launches the task based on the requested model and workload.
- Training status, metrics, and logs are returned to the local interface for monitoring.
- The team can prepare a reproducible demo and explore content and ecosystem collaboration with cloud platforms such as Modal.

### 3. Mac Platform and Local Small-Model Training

The meeting discussed the value of Mac Mini and other Apple Silicon devices for local small-model training, fine-tuning, and inference.

Key observations and directions:

- Many independent developers outside China use Mac Mini for small-model serving and post-training.
- Apple Silicon's unified memory architecture provides a relatively accessible environment for local training.
- A recipe for fine-tuning a personal model from chat history is currently being migrated and validated on Mac Mini.
- AReno should further improve stability, model coverage, and memory optimization on Mac.
- Low-bit optimizers such as 4-bit Adam can be explored to reduce memory consumption during local training.

### 4. Community Activities and External Collaboration

The meeting reviewed recent offline events and upcoming collaboration opportunities:

- Recent conference booths generated potential database ecosystem collaboration, university outreach, and feedback from developers in different industries.
- The team will visit Nanjing to refine a university-industry collaboration plan for incorporating AReno into intelligent computing courses.
- The community is evaluating technical events, panels, and conference talks in October and November.
- A community talk proposed for November 30 requires a topic summary and outline to secure the time slot.
- The team will participate in a public Hermes webinar next week to share related AReno practices.

### 5. AReno RSI Proof of Concept

The meeting discussed exploring Recursive Self-Improvement through AReno. The initial idea is for a model to explore tasks and its own capability boundaries, generate training data from failures or weak cases, and then improve through an iterative training loop.

The initial workflow is:

1. The model explores tasks and identifies failures or capability boundaries.
2. The model generates training data itself or with assistance from another model.
3. The generated data is used for post-training.
4. Evaluation, data generation, and training are repeated as a continuous loop.

The first proof of concept may focus on improving AReno itself. The immediate goal is to establish a reusable workflow and technical foundation rather than predetermine the final outcome. A short proposal will be shared in the community technical group to recruit interested contributors.

## Decisions

- Publish the next release next week after validating and merging the Ascend NPU, AReno Flow, Bailing MTP, and related changes.
- Continue improving AReno Flow and prepare a cloud-GPU demo and community collaboration materials.
- Start an AReno RSI proof of concept, initially focused on exploring and improving AReno itself.
- Treat Mac as an important platform for local small-model training and inference, with further work on support and memory optimization.
- Prepare separate proposals for AReno RSI and the Mac platform, then invite community contributors to participate.
- Advance university course collaboration and continue evaluating upcoming technical events and conference talks.

## Action Items

- Continue validating models such as Qwen and Gemma on Huawei Ascend NPU, then merge the changes after validation passes.
- Validate and merge the community-contributed Bailing MTP support.
- Merge AReno Flow and improve cost estimation, GPU recommendation, task execution, and log collection.
- Prepare a reproducible AReno Flow demo and explore joint outreach with the Modal community.
- Complete migration and validation of the personal chat-history fine-tuning recipe on Mac Mini.
- Define follow-up work for Mac, including model support, stability, and low-bit optimizers.
- Draft the AReno RSI proposal and implement a simple loop for capability-boundary exploration and training-data generation.
- Share the RSI and Mac proposals in the community technical group and recruit interested contributors.
- Refine the university course collaboration plan with faculty partners.
- Prepare the topic summary and outline for the proposed November 30 community talk.
- Prepare for next week's Hermes webinar.

## Open Questions

- What should be the first measurable task and evaluation criteria for AReno RSI?
- How should RSI-generated training data be filtered and validated to prevent quality degradation?
- Which cloud GPUs, models, and training scenarios should AReno Flow support first?
- Which models, optimizers, and performance capabilities should be prioritized on Mac?
- How can conference, university, and cloud-platform collaborations become sustained community contributions and reusable examples?

## Next Meeting

Date: TBD
Time: TBD
Suggested topics:

- Review the new release and the merge status of Ascend NPU, AReno Flow, and Bailing MTP support.
- Review the AReno RSI proposal and initial proof-of-concept results.
- Sync progress on the Mac roadmap and fine-tuning recipe validation.
- Follow up on university course collaboration, the Hermes webinar, and upcoming conference talks.
