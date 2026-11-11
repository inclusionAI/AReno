# AReno Community Meeting #5

Date: 2026-10-30
Time: Not recorded
Attendance: Not counted
Format: Online community meeting

## Meeting Summary

The fifth AReno community meeting reviewed the main additions in version 0.10.0, including Mac/MLX QLoRA, direct GGUF/Ollama export after CUDA training, Qwen3.5 and multimodal model support, JEV decision-model training, and Harbor integration with a SWE-bench-style agent-training example. The group also discussed speculative decoding, heterogeneous hardware, Agentic RL, and Recursive Self-Improvement (RSI).

The meeting agreed to continue improving JEV and RSI, with a dedicated RSI discussion planned for next week. Near-term community work will focus on technical conferences, university outreach, internal adoption examples, and a consistent cross-platform publishing workflow.

## Agenda

### 1. AReno 0.10.0 Progress

The release delivered the following capabilities:

- QLoRA support on the MLX backend for Apple Silicon, including validation of quantized training and adapter save-and-reload workflows.
- Direct export of GGUF and Ollama packages after CUDA training, without requiring an external conversion tool.
- Additional support for Qwen3.5, Bailing V3 MTP, and Phi-4 Multimodal. The multimodal path covers image and audio inputs.
- JEV, which trains a model-independent decision head on different base models and supports both CUDA and Mac/MLX workflows.
- Harbor integration connecting agents, execution environments, and training, together with a SWE-bench-style software-engineering example.

### 2. Speculative Decoding and Model Capability Roadmap

The group discussed MTP-based speculative decoding and the need to evaluate newer speculative-decoding techniques. The immediate priority is to stabilize the existing MTP path, followed by comparisons of training cost, inference gains, and adaptation complexity for other methods.

JEV will remain a major focus for the next phase. The project needs tutorials, examples, and blog posts covering decision-head training, saving, loading, and serving, together with compatibility validation across models, quantization options, and runtime backends.

### 3. Q4 Technical Roadmap

The team identified several priorities for the fourth quarter:

- Broaden heterogeneous hardware support, continue collaboration with domestic accelerator ecosystems, and publish a TPU usage guide.
- Improve memory use, training efficiency, and model coverage on developer-oriented devices such as Mac and DGX Spark.
- Strengthen Agentic RL and improve the connection between execution environments such as Harbor and AReno training.
- Continue integrating deployment ecosystems such as GGUF and Ollama so trained models can move into local inference more easily.
- Explore an RSI loop that updates model weights and provides reusable workflows and evaluation methods.

### 4. RSI Scenario Discussion

The meeting compared several possible RSI proof-of-concept scenarios:

- Kernel-writing agent: repeatedly generate, test, and optimize GPU kernels, using benchmark results or performance leaderboards as feedback. This scenario has a clear evaluation signal and relatively manageable resource requirements.
- Mathematics: build a loop around problem generation, solving, and training, though large-scale sampling can be expensive.
- Embodied AI: explore simulation and benchmark environments such as RoboDojo and RoboCasa, which may require less data, while verifying whether the process updates model weights rather than only optimizing high-level skills or prompts.

The team observed that many current RSI systems optimize programs, skills, or context without directly updating model weights. AReno could differentiate itself by connecting exploration, data generation, evaluation, and weight training in one loop. After an internal kernel-agent presentation next week, the team will hold a focused meeting to select the first scenario and define reusable framework requirements.

### 5. Community Activities and Publishing

Upcoming community work includes:

- Supporting booth activities and community outreach around the World Internet Conference in Wuzhen.
- Preparing an internal adoption example for publication after the website and discussion area are ready.
- Planning an open-source campus event at Huazhong University of Science and Technology in November, with the date still to be confirmed.
- Preparing a talk for the Global System Software Technology Conference in Beijing on November 21; the presentation materials are under review.
- Preparing a research talk at the Hong Kong University of Science and Technology on November 30.
- Standardizing the publishing workflow: submit each blog to GitHub as Markdown first, then adapt it for Medium, WeChat, and other channels.

## Decisions

- Build on the 0.10.0 release by adding documentation and reproducible examples for Mac/MLX QLoRA, GGUF/Ollama export, model support, JEV, and Harbor integration.
- Treat JEV and RSI as priorities for the next phase and prepare tutorials, blog posts, and community presentations.
- Hold a dedicated RSI discussion next week after the kernel-agent presentation and select the first proof-of-concept scenario.
- Continue Q4 work on heterogeneous hardware, developer-device optimization, Agentic RL, and inference/deployment ecosystem integration.
- Submit blog posts to GitHub in Markdown before republishing them on other channels.

## Action Items

- Document JEV training, checkpoint saving, loading, and serving, and validate CUDA, Mac/MLX, and quantized-training workflows.
- Continue validating MTP speculative decoding and compare the benefits and adaptation costs of other techniques.
- Improve the Harbor integration and SWE-bench-style agent-training example.
- Prepare a TPU training guide and continue domestic accelerator adaptation work.
- Improve training experience, resource use, and model coverage on Mac and DGX Spark.
- Organize the internal kernel-agent presentation and hold the RSI-focused meeting next week.
- Compare kernel, mathematics, and embodied-AI scenarios, then define the first RSI proof of concept and its evaluation metrics.
- Prepare materials for the November 21 and November 30 talks and follow up on the university-event date.
- Submit upcoming blog posts to GitHub before adapting them for Medium, WeChat, and other channels.

## Open Questions

- Should the first RSI proof of concept focus on kernels, mathematics, or embodied AI?
- How should an RSI loop be designed to update model weights reliably instead of only optimizing skills or prompts?
- What practical gains can newer speculative-decoding methods provide over the existing MTP path?
- How should model and feature priorities be balanced across Mac, DGX Spark, TPU, and domestic accelerators?
- How can upcoming conferences, university events, and internal outreach become reusable examples and sustained community contributions?

## Next Meeting

Date: TBD
Time: TBD
Suggested topics:

- Review progress on JEV tutorials, examples, and model compatibility validation.
- Select the first RSI proof-of-concept scenario, metrics, and reusable framework requirements.
- Sync progress on heterogeneous hardware, Mac/DGX Spark optimization, and speculative decoding.
- Follow up on November community events, talks, and publishing plans.
