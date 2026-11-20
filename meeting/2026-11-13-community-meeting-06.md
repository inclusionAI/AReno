# AReno Community Meeting #6

Date: 2026-11-13
Duration: Approximately 22 minutes
Format: Online community meeting

## Meeting Summary

The group reviewed the AReno 0.11.0 release, focusing on GPT-OSS-20B support, JEV decision model training, and Harbor agent workflows. The next steps are to prepare 0.12.0, document how to reproduce JEV training, and evaluate training on consumer GPUs, starting with an RTX 5090.

## Agenda

### 1. AReno 0.11.0 and JEV

- The release adds support for GPT-OSS-20B and extends LoRA support for Gemma 4.
- The team shared progress on a 9B JEV model and a JEV model based on Ling-3.0-tiny, along with experience using full-parameter training and LoRA on DGX Spark and Apple Silicon (MLX).
- The team reported that both models performed close to Open-Jev-9B on JevBench. Results on other benchmarks are still pending, and the team plans to continue improving the 9B model.
- Harbor agent workflows are now integrated with AReno. The team ran an initial validation using an SVG generation task.

### 2. Training on Consumer GPUs

- Establish a training baseline on an available RTX 5090, then test progressively smaller GPU memory budgets.
- The group discussed quantization, offloading, and streaming as possible optimizations. The next step is to define model sizes, memory targets, and acceptable training speeds for each stage.

### 3. RSI and Embodied AI

- Continue the Recursive Self-Improvement (RSI) experiments in game environments.
- A community member reported early results from using AReno for embodied AI post-training in simulation. The team plans to review the setup and comparative results next week and discuss how the work could connect to the existing RSI experiments.

### 4. Community Events and Content

- Continue preparing for the systems software conference talk in Beijing on November 21 and the research talk at the Hong Kong University of Science and Technology on November 30.
- An open-source campus event at Huazhong University of Science and Technology is tentatively scheduled for November 27. The proposed session includes a small hands-on JEV example for students.
- The community welcomes technical blog posts about AReno and can help with editing and promotion. A contributor plans to write about their speculative decoding integration once it is complete.

## Decisions

- Target the week after the meeting for the AReno 0.12.0 release.
- Start consumer GPU testing on the RTX 5090 and use the results to set milestones for GPUs with less memory.
- Continue the game RSI experiments and schedule a separate discussion on embodied AI post-training.

## Action Items

- Write a tutorial for reproducing JEV training from scratch, document practical training tips, and evaluate the models on additional benchmarks.
- Prepare a training example and guide for Mac mini. Update the project README and website homepage, and fix broken or incorrect links.
- Record model size, peak GPU memory use, and training speed for the RTX 5090 baseline.
- Follow up on the game RSI experiments and the discussion of embodied AI post-training in simulation.
- Prepare materials for the November conference and university talks, and confirm arrangements for the Huazhong University of Science and Technology event.

## Open Questions

- What model sizes and training speeds should be the first targets for consumer GPUs with less memory?
- How do the JEV models perform on benchmarks beyond JevBench?
- Which evaluation methods from embodied AI simulation tasks could be reused in RSI experiments?

## Next Meeting

Date: TBD. Suggested topics include the 0.12.0 release, JEV tutorials and evaluation, consumer GPU testing, and RSI experiments.
