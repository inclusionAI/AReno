# AReno 项目结构完整分析

> 生成时间: 2026-07-28
> 项目: areno v0.0.6 (inclusionAI/AReno)
> 分 支: issue_280_yixuan (基于 main)

---

## 一、项目概览

**AReno** (ASystem Reinforcement Learning Nano) 是一个自包含、全栈的 LLM 后训练工具包，用于在单节点上进行 RL、SFT/DPO 风格训练、模型服务和 Agentic RL。由蚂蚁集团 ASystem 团队发起。

- **名称**: areno (v0.0.6)
- **语言**: Python 3.10+
- **框架**: PyTorch 2.6+, CUDA
- **许可证**: Apache 2.0
- **仓库**: https://github.com/inclusionAI/AReno

---

## 二、顶层文件结构

```
AReno/
├── .github/                    # GitHub CI/CD 和 Issue 模板
│   ├── ISSUE_TEMPLATE/         # bug_report.yml, feature_request.yml, doc.yml, question.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/              # 11 个 CI workflow
│       ├── cpu_unit_tests.yml
│       ├── pre-commit.yml
│       ├── pr-style.yml
│       ├── publish-pypi.yml
│       ├── publish-docs.yml
│       ├── create-docker-image.yml
│       ├── publish-image-aliyun-acr.yml
│       ├── dashboard-bundle.yml
│       ├── push_admit_audit.yml
│       ├── modal_math_algorithms.yml
│       └── modal_train.yml
├── .idea/                      # PyCharm IDE 配置
├── .pre-commit-config.yaml     # pre-commit 钩子配置
├── .gitignore
├── .dockerignore
├── AGENTS.md                   # AI Agent 操作指南（本项目的核心行为规范）
├── CLAUDE.md                   # 指向 AGENTS.md 的软链接
├── CODEMAP.md                  # 代码地图 — 按任务导航到对应文件
├── CONTRIBUTING.md             # 贡献指南
├── RELEASE.md                  # 发布流程清单
├── README.md                   # 项目 README
├── LICENSE                     # Apache 2.0
├── Dockerfile                  # Docker 镜像构建文件
├── MANIFEST.in                 # setuptools 清单
├── pyproject.toml              # 项目元数据、依赖、构建配置
├── setup.py                    # CUDA 扩展编译 + 动态安装检查
├── agent.sh                    # 无需安装即可运行的 agent 入口脚本
├── areno/                      # 核心 Python 包
├── ci/                         # CI 辅助脚本（modal_gsm8k_gspo.py）
├── dashboard/                  # React 前端看板（pnpm + Vite）
├── docs/                       # Sphinx 文档 (.rst)
├── examples/                   # 完整示例（math, sft, agentic）
├── scripts/                    # 安装脚本 (install.sh)
├── skills/                     # AI agent 技能包（模型适配指导）
├── tests/                      # CPU 安全测试套件（28 个 *_cpu.py 文件）
└── 处理过程/                   # 本分析输出目录
```

---

## 三、核心包 `areno/` 详细结构

### 3.1 分层架构

```
areno/                          # 自顶向下四层设计
├── cli/                        # CLI 入口层
├── api/                        # SDK 与公共 API 层
├── engine/                     # 引擎运行时层
├── models/                     # 模型适配器层
├── agent/                      # Agent 循环与工具层
├── accel/                      # CUDA 加速层
├── experimental/               # 实验性功能
└── dashboard/                  # 看板服务端
```

### 3.2 `areno/cli/` — CLI 命令行入口

| 文件 | 职责 | 说明 |
|------|------|------|
| `main.py` | `ArenoCli` — Click 入口，延迟加载子命令 | 注册 6 个子命令：check, env, agent, dashboard, train, serve |
| `train.py` | `train_command` — 训练流程入口 | 解析算法、数据集、奖励函数、loss、trainer，调用 `fit` |
| `serve.py` | `serve_command` / `create_app` — 服务启动 | 构建 `ArenoEngine` + FastAPI OpenAI 兼容路由 |
| `agent.py` | `agent_command` — 本地运维助手 | 使用 OpenAI 兼容模型执行 AReno 命令 |
| `diagnostics.py` | 诊断命令 | `areno check` 检查 CUDA/PyTorch 状态, `areno env` 输出环境报告 |
| `auto_tune.py` | 自动调参 | 内存探测与参数自动调整 |
| `dashboard.py` | 看板命令 | 启动/停止 React 看板 |
| `dashboard_registry.py` | 看板注册表 | dashboard 功能注册 |
| `model_refs.py` | 模型引用处理 | 模型路径/ID 解析 |

### 3.3 `areno/api/` — SDK 公共 API 层

**这是用户直接使用的接口层。** 公共导出见 `areno/api/__init__.py`。

| 文件 | 职责 | 说明 |
|------|------|------|
| `trainer.py` | `Trainer` 类 — 高层 SDK 入口 | 持有 tokenizer、backend、context、metrics，提供 `init()`, `rollout_batch()`, `train()`, `close()` |
| `config.py` | `ArenoConfig` — 后端配置 dataclass | tp_size, dp_size, optimizer, runtime 等 |
| `trainer_config.py` | `TrainerConfig` / `PolicyTrainerConfig` / `DPOTrainerConfig` / `PPOTrainerConfig` | 各算法的训练超参数 |
| `algorithms.py` | 算法注册表 (`AlgorithmSpec`) | `register_algorithm`, `get_algorithm`, `list_algorithms` |
| `trainer_factory.py` | `build_trainer` — 工厂函数 | 根据 `TrainerConfig` 构建具体 trainer |
| `agentic.py` | `AgentBatch`, `AgentTrajectory`, `LossMaskPolicy`, `RolloutSession` | Agentic RL 专用数据结构和会话管理 |
| `data.py` | `PromptBatch`, `PromptItem` | 数据加载类型定义 |
| `data_utils.py` | 数据处理工具函数 | |
| `models.py` | `BackendType`, `SamplingParams`, `RolloutResult`, `RolloutSequence`, `TrainSequence` | 模型/采样/训练的数据模型 |
| `metrics.py` | `MetricsRecorder` | TensorBoard 指标记录 |
| `rewards.py` | `RewardEvent`, `RewardRecord` | 奖励函数类型定义 |
| `advantages.py` | 优势函数计算 | |
| `roles.py` | `ModelRole` | 模型角色枚举（actor, ref, reward, critic） |
| `tokenizer.py` | Tokenizer 加载与编码 | |
| `tool_call_parser.py` | 工具调用解析 | Agentic RL 中的 tool call 解析 |
| `openai_chat.py` | OpenAI 兼容对话代理 | |
| `context.py` | `Context` 适配器 | 后端通信上下文 |
| `defaults.py` | 默认值 | |
| `dashboard.py` | Dashboard API | |

**子目录:**

| 目录 | 职责 |
|------|------|
| `api/backend/` | 后端抽象 (`base.py` → `Backend` 协议) + 具体实现 (`areno/backend.py` → `ArenoBackend`) |
| `api/loss_fns/` | Loss 函数：`sft.py`, `dpo.py`, `gspo.py`, `grpo.py`, `ppo.py`, `layout.py` |
| `api/trainers/` | 具体 Trainer 实现：`policy_only.py` (GSPO/GRPO), `sft.py`, `dpo.py`, `ppo.py` |

### 3.4 `areno/engine/` — 引擎运行时层

**引擎核心 — 处理 TP/DP 并行、rollout、训练步骤、模型推理。**

| 文件 | 职责 | 说明 |
|------|------|------|
| `api.py` | `ArenoEngine` — 用户侧引擎协调器 | 分拆用户批次到 DP，通过 `TPCluster` 发送命令到 rank workers |
| `worker.py` | `ArenoWorker` — rank 侧 worker 主循环 | 接收 engine 命令，执行训练/推理，生命期管理 |
| `inference.py` | `InferenceManager` — 推理管理 | rank 侧 rollout/推理执行 |
| `training.py` | `TrainingManager` — 训练管理 | rank 侧训练步骤执行 |
| `config.py` | `EngineConfig`, `ModelConfig`, `OptimizerConfig`, `RuntimeConfig` | 引擎级配置 |
| `protocol.py` | `TPCluster`, `Op` 枚举, payload 类型 | 跨进程 TP 通信协议 |
| `modeling.py` | 模型构建与组装 | |
| `log.py` | 日志配置 | `configure_default_logging` |
| `roles.py` | 引擎侧 role 准备与清理 | |
| `训练执行链路`: | | |
| `Trainer → Backend → ArenoBackend → ArenoEngine → ArenoWorker` | | |

**子目录:**

| 目录 | 职责 |
|------|------|
| `engine/checkpoints/` | 模型检查点加载/保存 (`io.py`, `common.py`) |
| `engine/data/` | `batch.py`, `rollout_state.py`, `sampling.py`, `tokenizer.py` |
| `engine/layers/` | 网络层实现：`attention.py`, `mlp.py`, `linear.py`, `norm.py`, `rotary.py`, `vocab.py` |
| `engine/layers/attention_backend/` | 注意力后端：`common.py`, `infer.py`, `train.py` |
| `engine/optim/` | 优化器：`adamw_fp32_master.py`, `adamw_8bit.py` |
| `engine/parallel/` | TP 并行：`collectives.py` (all-reduce 等), `context.py` |
| `engine/runtime/` | 运行时工具：`common.py`, `decode_graph.py`, `logprobs.py`, `metadata.py`, `recompute.py`, `rollout.py`, `train_step.py` |

### 3.5 `areno/models/` — 模型适配器层

**插件式架构 — 每种模型家族一个子包，统一注册到 registry。**

| 文件 | 职责 | 说明 |
|------|------|------|
| `base.py` | `ModelAdapter` 抽象类 + `CausalLMOutput` | 定义 5 个核心方法：`match_hf_config`, `config_from_hf`, `build`, `load_weights`, `save_weights` |
| `registry.py` | 模型注册表 | `register_adapter`, `adapter_from_hf`, `config_from_hf`, `build_model`, `load_model_weights`, `save_model_weights` |
| `_shared/dynamo_wrappers.py` | torch.compile 包装器 | 共享工具 |

**支持的模型家族:**

| 目录 | 模型 | 文件 |
|------|------|------|
| `llama/` | LLaMA | `model.py`, `checkpoint.py` |
| `qwen3/` | Qwen3 | `model.py`, `checkpoint.py` |
| `qwen3_5/` | Qwen3.5 | `model.py`, `checkpoint.py` |
| `gemma4/` | Gemma4 | `model.py`, `checkpoint.py` |
| `bailing/` | Bailing (百灵) | `model.py`, `checkpoint.py` |
| `minicpmv46/` | MiniCPM-V 4.6 | `model.py`, `checkpoint.py` |

### 3.6 `areno/agent/` — Agent 循环与工具

| 文件 | 职责 | 说明 |
|------|------|------|
| `agent_loop.py` | 本地 Agent 循环 | `run_agentic_coding_loop`, `run_conversation_turns`, `run_single_task`, 系统提示词, 工具定义 |
| `tools.py` | `CodingWorkspace`, `run_tool` | 代码工具集（文件读取、搜索等） |
| `ops_knowledge.md` | 运维知识文档 | Agent 运维参考 |

### 3.7 `areno/accel/` — CUDA 加速层

**手写 CUDA kernel + Python 包装器，通过 `setup.py` 编译为 `areno.accel._areno_accel`。**

| 文件 | 职责 |
|------|------|
| `_extension.py` | Python 扩展入口 |
| `activations.py` | 激活函数包装 |
| `attention.py` | 注意力计算包装 |
| `conv.py` | 卷积包装 |
| `embedding.py` | 嵌入层包装 |
| `linear.py` | 线性层包装 |
| `moe.py` | MoE 包装 |
| `normalization.py` | 归一化包装 |
| `ops.py` | 算子入口 |
| `router.py` / `routing.py` | MoE 路由 |
| `topk.py` | TopK 选择 |
| `kernels/fused_moe.py` | 融合 MoE kernel |
| `kernels/group_rmsnorm.py` | 分组 RMSNorm kernel |
| `kernels/seg_la.py` | 分段线性注意力 |

**CUDA 源文件 (`csrc/`):**
`extension.cpp`, `activation.cu`, `attention.cu`, `conv.cu`, `embedding.cu`, `linear.cu`, `moe_align_kernel.cu`, `moe_permute.cu`, `normalization.cu`, `router.cu`, `topk.cu`

### 3.8 `areno/experimental/` — 实验性功能

| 文件 | 职责 |
|------|------|
| `__init__.py` | 实验性模块入口 |

### 3.9 `areno/dashboard/` — Dashboard 服务端

| 文件 | 职责 |
|------|------|
| `server.py` | Dashboard HTTP 服务器 |
| `agent_context.py` | Agent 上下文提供 |
| `agent_files.py` | Agent 文件服务 |
| `dist/index.html` | 预构建的 React 前端 |
| `dist/assets/index-CIfovKBG.css` | 前端样式 |
| `dist/assets/index-DF61sRZi.js` | 前端 JS bundle |

---

## 四、训练算法（支持的算法）

通过 `--algo` 标志或算法注册表选择：

| 算法 | 描述 | Trainer | Loss 函数 |
|------|------|---------|-----------|
| **SFT** | 监督微调 | `SFTTrainer` (`api/trainers/sft.py`) | `sft_loss_fn` |
| **DPO** | 直接偏好优化 | `DPOTrainer` (`api/trainers/dpo.py`) | `dpo_loss_fn` |
| **GSPO** | 分组标准策略优化 | `PolicyOnlyTrainer` | `gspo_loss_fn` |
| **GRPO** | 分组相对策略优化 | `PolicyOnlyTrainer` | `grpo_loss_fn` |
| **PPO** | 近端策略优化 | `PPOTrainer` (`api/trainers/ppo.py`) | `ppo_loss_fn` |

注册机制: `AlgorithmSpec` 包含 `loss_fn_factory` (可选的 clip_eps 绑定) 和 `trainer_factory`。

---

## 五、示例 `examples/`

| 目录 | 说明 | 关键文件 |
|------|------|----------|
| `examples/math/` | 数学 RLVR (GSM8K) | `dataset_loader.py`, `math_verify_reward.py` |
| `examples/sft/alpaca/` | SFT 示例 (Alpaca 格式) | `dataset_loader.py`, `README.md` |
| `examples/agentic/tictactoe/` | 井字棋 Agentic RL | `game.py`, `reward.py`, `run_agent.py`, `dataset_loader.py`, `web_ui.py`, `dataset_generator.py` |
| `examples/agentic/coding/` | 代码 Agent RL (SWE-bench 风格) | `run_agent.py`, `agent_loop.py`, `code_cli.py`, `coding_tools.py`, `reward.py`, `dataset_loader.py` |
| `examples/agentic/shopping/` | 购物 Agent RL | `game.py`, `reward.py`, `run_agent.py`, `dataset_loader.py`, `dataset_generator.py` |
| `examples/agentic/duelgrid/` | DuelGrid 游戏 Agent RL (带可视化) | `game.py`, `reward.py`, `run_agent.py`, `dataset_loader.py`, `web_ui.py`, `dataset_generator.py` |

---

## 六、测试 `tests/` (28 个 CPU 安全测试)

| 文件 | 测试范围 |
|------|----------|
| `helpers.py` | 测试辅助工具 |
| `test_algorithms_cpu.py` | 算法注册与 dispatch |
| `test_losses_rewards_cpu.py` | Loss 函数和奖励函数 |
| `test_more_losses_cpu.py` | 额外 loss 测试 |
| `test_trainer_api_cpu.py` | Trainer SDK API |
| `test_train_cli_config_cpu.py` | CLI 训练配置 |
| `test_config_data_cpu.py` | 配置和数据 |
| `test_trainer_dataset_utils_cpu.py` | 数据集工具 |
| `test_protocol_cpu.py` | 通信协议 |
| `test_logprobs_cpu.py` | Logprobs 计算 |
| `test_metrics_cpu.py` | 指标记录 |
| `test_sampling_cpu.py` | 采样 |
| `test_recompute_cpu.py` | 重计算 |
| `test_runtime_utils_cpu.py` | 运行时工具 |
| `test_registry_cpu.py` | 模型注册 |
| `test_registry_discovery_cpu.py` | 模型发现 |
| `test_inference_scheduler_cpu.py` | 推理调度 |
| `test_import_boundaries_cpu.py` | 导入边界 |
| `test_tokenizer_api_cpu.py` | Tokenizer API |
| `test_agentic_cpu.py` | Agentic RL 主流程 |
| `test_agentic_tictactoe_example_cpu.py` | 井字棋 Agent 示例 |
| `test_agentic_shopping_example_cpu.py` | 购物 Agent 示例 |
| `test_coding_agent_loop_cpu.py` | 代码 Agent 循环 |
| `test_sft_example_cpu.py` | SFT 示例 |
| `test_serve_cli_cpu.py` | 服务 CLI |
| `test_cli_diagnostics_cpu.py` | 诊断 CLI |
| `test_cli_model_refs_cpu.py` | 模型引用 CLI |
| `test_auto_tune_cpu.py` | 自动调参 |
| `test_setup_guardrails_cpu.py` | 安装护栏检查 |

---

## 七、文档 `docs/` (Sphinx RST)

| 文件/目录 | 内容 |
|-----------|------|
| `index.rst` | 文档首页 |
| `conf.py` | Sphinx 配置 |
| `requirements.txt` | 文档构建依赖 |
| `getting-started/welcome.rst` | 欢迎页 |
| `getting-started/installation.rst` | 安装指南 |
| `getting-started/quickstart.rst` | 快速入门 |
| `concepts/backend-topology.rst` | 后端拓扑架构 |
| `concepts/chat-templates.rst` | 对话模板 |
| `concepts/dataset-formats.rst` | 数据集格式 |
| `concepts/reward-functions.rst` | 奖励函数 |
| `concepts/training-loop.rst` | 训练循环 |
| `cli/training.rst` | CLI 训练说明 |
| `cli/inference.rst` | CLI 推理说明 |
| `cli/dataset_loaders.rst` | 数据集加载器 |
| `cli/agent.rst` | Agent CLI |
| `cli/diagnostics.rst` | 诊断 CLI |
| `cli/observability.rst` | 可观测性 |
| `cookbook/math-rlvr.rst` | 数学 RLVR 示例 |
| `cookbook/tictactoe-agentic-rl.rst` | 井字棋 Agentic RL 示例 |
| `cookbook/duelgrid-visual-agent.rst` | DuelGrid 可视化示例 |
| `models/supported.rst` | 支持的模型列表 |
| `sdk/trainer.rst` | Trainer SDK |
| `reference/cli.rst` | CLI 参考 |
| `reference/config.rst` | 配置参考 |
| `reference/dataset-loader-api.rst` | 数据集加载器 API |
| `reference/reward-function-api.rst` | 奖励函数 API |
| `reference/agentic-rollout-api.rst` | Agentic Rollout API |
| `reference/environment-variables.rst` | 环境变量参考 |
| `reference/experimental.rst` | 实验性功能参考 |
| `reference/index.rst` | 参考索引 |
| `troubleshooting/` | 故障排查 (9 个文件) |

---

## 八、CI/CD (`.github/workflows/`)

| Workflow | 触发条件 | 职责 |
|----------|----------|------|
| `cpu_unit_tests.yml` | PR/Push | CPU 单元测试 (`pytest tests/ -k cpu`) |
| `pre-commit.yml` | PR/Push | pre-commit 检查 |
| `pr-style.yml` | PR | PR 风格检查 |
| `publish-pypi.yml` | Tag push (`v*`) | PyPI 发布 |
| `publish-docs.yml` | 指定分支 | 文档发布 |
| `create-docker-image.yml` | Release | Docker 镜像构建 (ghcr.io) |
| `publish-image-aliyun-acr.yml` | Release | 阿里云 ACR 镜像推送 |
| `dashboard-bundle.yml` | PR | Dashboard 前端检查 |
| `modal_math_algorithms.yml` | Schedule/Manual | Modal 云 GPU 数学算法验证 |
| `modal_train.yml` | Schedule/Manual | Modal 云 GPU 训练验证 |
| `push_admit_audit.yml` | PR | 推送准入审计 |

---

## 九、依赖关系

### 核心依赖 (pyproject.toml)
```
torch >= 2.6
flash-linear-attention >= 0.2
safetensors >= 0.4
transformers >= 4.56
huggingface-hub >= 0.25
modelscope >= 1.20
datasets == 4.0.0
fastapi >= 0.110    (服务 API)
uvicorn >= 0.27     (ASGI 服务器)
click >= 8.1        (CLI)
rich >= 13          (终端美化)
tensorboard         (训练可视化)
pydantic >= 2       (数据校验)
openai              (OpenAI 兼容调用)
math-verify         (数学答案校验)
```

### 可选依赖
```
flash-attn >= 2.7   (FlashAttention 后端)
```

### Dashboard 前端依赖 (dashboard/package.json)
```
React 19, Vite 6, TypeScript 5.7, pnpm 11.9
lucide-react (图标), react-markdown (Markdown 渲染), remark-gfm (GFM 支持)
```

---

## 十、扩展点与注册机制

1. **算法扩展**: `register_algorithm(AlgorithmSpec(...))` 注册新算法 + loss + trainer
2. **模型适配器**: 继承 `ModelAdapter` 实现 5 个方法，调用 `register_adapter(adapter)`
3. **Loss 函数**: 在 `areno/api/loss_fns/` 添加新文件，返回 `Callable`
4. **奖励函数**: 独立 Python 文件，通过 `--reward-fn-path` 指定
5. **数据集加载器**: 独立 Python 文件，通过 `--dataset-loader-fn` 指定
6. **Agent 函数**: 独立 Python 文件，通过 `--agent-fn` 指定
7. **后端扩展**: 实现 `Backend` 协议 (`areno/api/backend/base.py`)

---

## 十一、关键设计原则

- **自包含 (self-contained)**: 不依赖外部训练框架或推理服务器
- **单节点优化**: 在单个 GPU 节点上提取最大性能
- **寄存器驱动**: 新能力通过注册表添加，不修改核心代码
- **CPU 测试安全**: tests/ 中所有测试均在 CPU 上运行 (`-k cpu`)
- **延迟加载**: CLI 子命令和 engine 模块在首次使用时才加载
- **TP/DP 透明**: Trainer 调用者无需关心内部 TP/DP 拆分
- **torch.compile 友好**: 为 prefill/decode/train 生成独立编译图