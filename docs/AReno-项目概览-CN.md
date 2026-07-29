# AReno 项目概览(中文)

> 本文档面向首次接手 AReno 的开发者,用中文梳理"它是什么、目录怎么分层、一次训练从 CLI 到 GPU 走了哪些模块、如何扩展"。具体的安装命令、硬性规则与逐文件规范以仓库内的 [`AGENTS.md`](../AGENTS.md)、[`CODEMAP.md`](../CODEMAP.md) 和 [`CONTRIBUTING.md`](../CONTRIBUTING.md) 为准;英文文档站点见 `docs/`。

---

## 1. 一句话定位

AReno(ASystem Reinforcement Learning Nano)是蚂蚁集团 inclusionAI ASystem 团队发起、社区维护的 **单节点、自包含 LLM 强化学习后训练工具包**。它把"训练框架 + 推理引擎 + 算子库"整合进同一个 Python 包,目标是在一台带 NVIDIA GPU 的 Linux 机器上,从基座模型一路走到"训练完成 + 对外服务",无需另起集群或拼装第三方后端。

核心理念:**registry-driven extension,not forking the core** —— 算法、模型适配器、奖励函数、算子都以注册方式扩展,尽量不动核心代码。

- 技术栈:Python 3.10+ / CUDA / PyTorch ≥ 2.6 / 可选 FlashAttention / flash-linear-attention / Transformers / safetensors
- 当前版本:`0.0.6`(见 `pyproject.toml`)
- 协议:Apache 2.0

---

## 2. 目录分层

AReno 的 Python 包 `areno/` 采用严格的 **cli → api → engine → accel** 自上而下分层,依赖只能向下流,不能反向。

```
areno/
├── cli/          # 命令行入口:check / env / agent / dashboard / train / serve
├── api/          # 公共 SDK:Trainer、算法注册表、loss_fns/、trainers/、backend/、agentic
├── engine/       # 引擎层:TP/DP worker 进程、rollout/train runtime、CUDA graph、checkpoint
│   ├── parallel/        # 张量并行 collectives / context
│   ├── runtime/         # rollout / train_step / decode_graph / recompute 等
│   ├── layers/          # attention / mlp / norm / rotary / vocab / linear
│   ├── optim/           # adamw_8bit、adamw_fp32_master
│   ├── checkpoints/     # HF 兼容的 checkpoint 读写
│   └── data/            # 采样参数、batch、rollout 状态
├── models/       # 各模型家族适配器(见下方列表)+ registry.py
├── accel/        # 融合 CUDA 算子:csrc/*.cu(编译为 _areno_accel)+ kernels/*.py
├── agent/        # areno agent 操作助手(OpenAI 兼容模型本地决策)
├── experimental/ # 实验性算法孵化区,稳定后迁入 api/
└── dashboard/    # 内置 React 仪表盘(dist 已打包)

examples/         # 可运行示例:math(RLVR)、sft、agentic(tictactoe/coding/duelgrid/shopping)
tests/            # CPU 测试套件(*_cpu.py,无 GPU 即可跑)
docs/             # Sphinx 文档:getting-started / concepts / cli / sdk / cookbook / reference
.agents/skills/   # 仓库本地 agent skills(11 个工作流,见下)
scripts/install.sh
```

**模型家族适配器**(在 `areno/models/__init__.py:register_models()` 中注册,通过 `registry.py` 按 HF config 匹配):
`llama`、`qwen3`、`qwen3`(MoE)、`qwen3_5`、`qwen3_5`(MoE)、`bailing`(MoE-linear-v2)、`gemma4`、`minicpmv46`、`olmo2`。

**算法**(在 `areno/api/algorithms.py` 注册):`sft`、`dpo`、`gspo`、`grpo`、`ppo`。

---

## 3. 顶层包的几个全局副作用

`areno/__init__.py` 在 import 时就拉起两个进程级开关,这点很重要 —— 任何动到它的修改都会影响所有使用者:

- **`CUDA_DEVICE_MAX_CONNECTIONS=1`**:单一 CUDA stream 连接,保证 NCCL 集合通信与计算按 AReno 的 TP/DP all-reduce + all-gather 假定顺序执行。
- **调大 TorchDynamo 缓存**(`cache_size_limit ≥ 64`、`accumulated ≥ 256`):训练 / prefill / decode / scoring 和多种 shape bucket 各自编译出独立图,缓存拉高避免跨 RL step 反复重编译驱逐图。
- 顶层导出采用 **`__getattr__` 懒加载**:`Trainer`、`ArenoEngine`、各类 Config、`SamplingParams` 等在被引用时才 import,保证 `import areno` 不触发 kernel-heavy 模块。

---

## 4. 一次训练的调用链(从 CLI 到 GPU)

理解这条链路是高效改动的关键。SDK 与 CLI 走的是同一条路。

```
areno train ...                          (areno/cli/main.py: ArenoCli 懒加载子命令)
  └─ train_command → run                 (areno/cli/train.py)
       │  • 解析选项 → TrainerConfig / DPOTrainerConfig / PPOTrainerConfig
       │  • get_algorithm(--algo) → AlgorithmSpec(trainer 类 + 默认 loss)
       │  • 构造 areno.api.Trainer,加载 dataset(可选 --dataset-loader-fn)
       └─ Trainer.init()                 (areno/api/trainer.py)
            • load_tokenizer + eos_token_ids
            • build Context(world_size, model_path, tokenizer, custom_config, eos)
            • get_backend_cls(BackendType.Areno) → backend.initialize(ctx)
  └─ 循环:rollout_session → rollout_batch → 评分 → train
```

引擎在 **Backend 边界** 之后才真正落到多进程 / GPU,其内部拓扑为:

```
Trainer                 (areno/api/trainer.py — 高层生命周期: init/rollout/train/close)
  → Backend             (areno/api/backend/base.py — 执行契约)
    → ArenoBackend      (areno/api/backend/areno/backend.py — 把公共 API 数据打包给引擎)
      → ArenoEngine     (areno/engine/api.py — 协调器,自己不持权重,负责拆 DP、合并结果)
        → ArenoWorker   (areno/engine/worker.py — 每个 TP/DP rank 一个进程)
            ├─ InferenceManager  (areno/engine/inference.py — rank 级 rollout)
            └─ TrainingManager   (areno/engine/training.py — rank 级训练,共享 runtime/ 辅助)
```

- `ArenoEngine` 通过 `TPCluster`(`areno/engine/protocol.py`)用 RPC 风格的 `Command` 协议向 worker 派发任务,把用户 batch 按 DP 切分,结果在 rank-0 合并回单进程视角。
- `areno/engine/layers/` 提供 attention / mlp / norm / rotary 等 `nn.Module`,可选 attention 后端(FlashAttention / 自研)由 `--attn-backend` 切换;`areno/engine/parallel/` 负责 TP/DP 集合通信与上下文。
- `areno/accel/` 是手写 CUDA 融合算子(`csrc/` 下 activation/attention/conv/embedding/linear/moe/normalization/router/topk),编译成扩展 `_areno_accel`,Python 侧通过 `accel/*.py` 调用。**修改 `.cu` 必须同步更新其 Python wrapper 并重建。**

### RL 循环的 SDK 写法

README 的 Quick Start 给出完整示例,核心五步:

1. `Trainer(...)` + `init()` —— 加载 tokenizer、启动 worker。
2. `async with trainer.rollout_session(sampling_params, ...)` 内调 `rollout_batch(prompts, n_samples, sampling_params)` 生成 on-policy 补全。
3. 用你自己的 `reward_fn(record, completions) -> list[float]` 打分,并 `to_advantages`(AReno 不替你算优势)。
4. 把 rollout 打包成 `TrainSequence`(`tokens` / `logprobs` / `advantages` / `prompt_mask` / `reward` / `eos_token_id`),调 `trainer.train(batch, loss_fn, mini_bs=...)` 跑一个优化器步。
5. 循环,最后 `close()`。

`train()` 返回标量指标 dict;若挂了 `MetricsRecorder`(传 `metrics_log_dir`)会自动把指标 + 每步耗时(train/rollout/e2e)写进 TensorBoard 和 dashboard。

---

## 5. 四种扩展点(注册,而非改工厂)

<span id="extension"></span>

| 扩展类型 | 在哪注册 | 关键约束 |
| --- | --- | --- |
| **算法** | `areno/api/algorithms.py`:`register_algorithm(AlgorithmSpec(...))` | `AlgorithmSpec` 含 `trainer_cls`(可懒加载)、`default_loss_fn`、`requires_rollout`、可选 `loss_fn_factory`。重复注册默认报错。新算法先进 `areno/experimental/`(`load_experimental_algorithms()` 按需加载),稳定后迁入 `api/`。 |
| **Loss** | `areno/api/loss_fns/` 下新增 `xxx_loss_fn` | 所有 loss 接收同样的 `data_pack` 字典,以便换算法无需改数据通路。GSPO/GRPO 的 `loss_fn_factory` 从 `TrainerConfig` 取 `clip_eps` 绑定。 |
| **模型家族** | `areno/models/<family>/` + `registry.py:register_adapter` | 实现 `ModelAdapter`(`match_hf_config` / `config_from_hf` / `build` / `load_model_weights` / `save_model_weights`),在 `models/__init__.py:register_models` 登记。**不得把 TransformerEngine / SGLang kernels / FLA 作为运行时依赖** —— 第三方代码只作张量语义参考。 |
| **奖励函数** | 一个 `--reward-fn-path` 指向的 Python 文件,暴露 `reward_fn(example, completions) -> list[float]` | 示例见 `examples/math/math_verify_reward.py`。 |

CLI 子命令本身**也是懒加载**的(`ArenoCli._COMMANDS` 里每个命令只在被选中时才 import 其模块),所以加新 CLI 命令只需在 `areno/cli/main.py` 的 `_COMMANDS` 表登记一个 entry。

---

## 6. Agentic RL(智能体强化学习)

AReno 的一等公民特性,核心文件 `areno/api/agentic.py`:

- 在 `rollout_session` 里可选拉起一个 **本地 OpenAI 兼容 HTTP 接口**(`ThreadingHTTPServer`,非流式 `/v1/chat/completions`),用户用普通 `openai` 客户端、带 `tools` / `tool_choice` 调用即可。
- 用户的 agent 函数(`--agent-fn` 指向,如 `examples/agentic/tictactoe/run_agent.py`)返回**显式的 `AgentTrajectory` / `AgentTrajectoryTurn`**(messages、tool calls、tool results、loss mask 全部显式),AReno 把这些 turn 转成与普通 trainer 相同的 token / logprob / loss-mask 行。
- `LossMaskPolicy`(`agentic.py`)控制哪些 span 参与策略 loss:默认 assistant 文本与 tool call 计入,**tool result 默认不计**(被 mask)。

示例目录:`~examples/agentic/` 下有 `tictactoe`(棋盘)、`coding`(SWE-bench 风格多轮软件工程,自带 `inspect_tree/read_file/rg/apply_patch/run_command/submit` 工具,纯本地任务)、`duelgrid`(浏览器游戏 UI + 多动作回合)、`shopping`。`areno/agent/` 则是 `areno agent` 操作助手(用 OpenAI 兼容模型在本地解读当前 checkout、产出/执行 AReno 命令)。

---

## 7. 常用开发命令

安装、训练、服务、CPU 测试的全量命令在 `AGENTS.md` Quick reference。这里补日常质量工具:

```bash
# 安装(需已有 Linux + NVIDIA GPU + CUDA + PyTorch ≥ 2.6 环境)
bash scripts/install.sh
# 或:pip install -e . --no-build-isolation
# 统一构建特定架构: TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=64 pip install -e . --no-build-isolation
# 仅做元数据/文档,跳过 CUDA 编译: ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation

# 机器就绪检查 / 环境报告
areno check              # OK/WARN/FAIL + 具体 next steps(CUDA、nvcc、CUDA_HOME、可选依赖、accel 扩展)
areno env --json

# 训练 / 服务
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py --algo gspo --tp-size 4
areno serve --model-path /path/to/model --port 8000

# 测试(CPU 套件,无 GPU 即可,是快速反馈环)
pytest tests/ -k cpu
pytest tests/test_algorithms_cpu.py::test_xxx       # 跑单条
# GPU 集成路径需实卡;在无 GPU 环境必须明确说明跳过

# Lint / 格式化 / 类型检查(ruff 配置在 pyproject.toml: E,F,W,I,UP;行宽 120;忽略 E501)
ruff check .
ruff format .
pyright                              # basic 模式,include areno/,排除 accel/csrc

# Pre-commit 门禁(ruff + 格式 + 空白 + 大文件 + 私钥 + conventional-commits)
pre-commit install --install-hooks
pre-commit run          # 当前 staged
pre-commit run -a       # 全量

# 改了 areno/accel/csrc 后,远程需重建(开发工作流见下)
pip install -e . --no-deps --no-build-isolation
```

**提交信息** 必须是 Conventional Commits,前缀限:`feat / fix / docs / style / refactor / perf / test / build / ci / chore / revert`。`commit-msg` hook 强制校验。

---

## 8. 远程 GPU 与 ModelScope 工作流

- **需要远程 GPU 主机时**:本地先在专用分支的 worktree 里改源码并 commit,再到远程 `git fetch` + `checkout`/`pull` 该分支;**不要**在远程直接改源码,也**不要**用拷贝未提交文件的方式隐式部署。仅当 `areno/accel` 改动时远程才需 `pip install -e . --no-deps --no-build-isolation` 重建,否则不要重装 AReno。
- **模型/数据集资产**:默认走 ModelScope。`areno train` / `areno serve` 用 `--model-hub modelscope`(仓库引用时);脚本侧用 `snapshot_download` / `MsDataset`。ModelScope 解析失败时**不要**静默回退到 HuggingFace,要报出缺失的 ModelScope 资产或改用用户提供的本地路径;确实要拉 HF 用 `--model-hub hf`。

---

## 9. 仓库本地 Skills(`.agents/skills/`)

AReno 自带 11 个面向 agent 的工作流 skill,各自指向可执行脚本与参考文档,按需加载只读匹配的那一个:

```
areno-run-training        areno-run-serving        areno-tune-capacity
areno-validate-correctness areno-debug-runtime     areno-profile-performance
areno-build-agentic-workflow areno-model-adaptation areno-add-algorithm
areno-develop-kernel      (areno agent 操作助手相关的 ops_knowledge.md 在 areno/agent/)
```

每个 skill 目录结构统一:`SKILL.md` + `references/` + `agents/` + `scripts/`。做对应任务时(如"加一个新算法""适配新模型""调试 runtime"),先读对应 `SKILL.md`。

---

## 10. 测试约定

- CPU 安全测试用 `*_cpu.py` 后缀放 `tests/`,覆盖新的算法 / loss / config 行为(默认"加一个 CPU 测试")。
- 需 GPU 的集成测试在无 GPU 时要**干净地 skip**,并在响应中明确说明跳过。
- **不得虚构**符号、报错、API 响应或栈;用了没读过的符号前先 `grep -r "name" areno/` 或读其定义,实在跳过就前缀 `# UNVERIFIED:`。`areno/` 顶层只从 `areno` 公共 SDK 引入(`from areno import Trainer`)。

---

## 11. 变更公共 API 前必读

公共 API(`areno/api/**` 的导出符号、Config dataclass、CLI 选项)是用户依赖的稳定面:

- **改 config dataclass**(`areno/api/trainer_config.py`、`areno/api/config.py`)、**加新依赖**(`pyproject.toml`)、**改 CLI 选项面**(`areno/cli/train.py` / `serve.py`)、**删除/重命名公共 API**、**跑 GPU 训练或服务** —— 这五类**先问**。
- 改动优先**加字段带默认值 / 先 deprecate 再删 / 避免类型变更**;deprecate 时发 `FutureWarning` 并给迁移指引 + 目标移除版本(模板见 `CONTRIBUTING.md`)。
- `experimental/` 下的代码无向后兼容承诺,可在版本间变更。

---

## 12. 进一步阅读

- [AGENTS.md](../AGENTS.md) —— 安装、命令、工作规则的权威来源(英文)。
- [CODEMAP.md](../CODEMAP.md) —— 按"任务类型 → 起手文件 → 邻近验证测试"的导航表。
- [CONTRIBUTING.md](../CONTRIBUTING.md) —— 提交流程、PR checklist、默认值与废弃规范。
- `docs/concepts/training-loop.rst`、`docs/concepts/backend-topology.rst` —— 训练循环与后端拓扑的精简概念图。
- `docs/cookbook/` —— math-rlvr、tictactoe-agentic-rl、duelgrid-visual-agent 三个可跑的端到端示例。