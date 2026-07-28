# AReno 项目启动文档

> 生成时间: 2026-07-28 | 项目: areno v0.0.6 | 分支: issue_280_yixuan

---

## 一、启动方式总览

AReno 提供 **5 种启动路径**，覆盖从安装到生产运行的完整生命周期：

| # | 入口 | 命令 / 代码 | 用途 |
|---|------|------------|------|
| 1 | **安装脚本** | `bash scripts/install.sh` | 一键安装并验证环境 |
| 2 | **CLI 训练** | `areno train --algo gspo --ckpt ...` | 命令行训练 |
| 3 | **CLI 服务** | `areno serve --model-path ... --port 8000` | OpenAI 兼容 API 服务 |
| 4 | **CLI Agent** | `areno agent "..."` 或 `./agent.sh "..."` | 自然语言驱动的运维助手 |
| 5 | **SDK 编程** | `Trainer(...)` → `init()` → `fit()` | Python SDK 编程式启动 |

此外还有 **诊断检查** (`areno check` / `areno env`)、**Dashboard** (`areno dashboard`) 等辅助命令。

---

## 二、方式 1：安装脚本启动 (`scripts/install.sh`)

### 2.1 完整安装流程

```bash
git clone https://github.com/inclusionAI/AReno.git
cd AReno
bash scripts/install.sh
```

### 2.2 安装脚本内部 7 步流程

```
+------------------+     +------------------+     +------------------+     +------------------+
| Step 1: 环境准备  | --> | Step 2: 检查     | --> | Step 3: 检查     | --> | Step 4: 检查     |
| Python 3.10+      |     | PyTorch >= 2.6   |     | setuptools+wheel |     | CUDA/nvcc/GPU   |
| 创建 .venv        |     | (CUDA版本)       |     |                  |     | 设置 CUDA_HOME   |
+------------------+     +------------------+     +------------------+     +------------------+
                                                                                   |
                                                                                   v
+------------------+     +------------------+     +------------------+
| Step 7: 验证     | <-- | Step 6: 编译     | <-- | Step 5: 安装依赖 |
| areno check      |     | AReno CUDA扩展   |     | psutil, ninja    |
|                  |     | pip install -e .  |     | flash-lin-attn  |
|                  |     | --no-build-isol  |     | flash-attn(可选) |
+------------------+     +------------------+     +------------------+
```

### 2.3 关键检查点

- **CUDA 工具链**: 需要 `nvcc` 编译器 + C++ 编译器 (g++)
- **GPU 架构**: 自动检测 `TORCH_CUDA_ARCH_LIST`（如 `8.0;9.0`）
- **FlashAttention**: 仅在 GPU Compute Capability >= 8.0 时安装
- **安装验证**: 最终执行 `areno check` 确认环境就绪

### 2.4 环境变量控制

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `ARENO_BUILD_EXT` | `0`=跳过 CUDA 扩展编译 | `1` |
| `TORCH_CUDA_ARCH_LIST` | 目标 GPU 架构列表 | 自动检测 |
| `MAX_JOBS` | 并行编译任务数 | 系统默认 |
| `ARENO_INSTALL_LOG` | 安装日志路径 | `~/.local/state/areno/install.log` |

---

## 三、方式 2：CLI 训练启动 (`areno train`)

### 3.1 命令行入口链路

```
用户执行:  areno train --algo gspo --ckpt Qwen/Qwen3-0.6B ...

    │
    v
┌──────────────────────────────────────────────────────┐
│ pyproject.toml: areno = "areno.cli.main:main"        │  ← setuptools console_scripts
└──────────────────┬───────────────────────────────────┘
                   │
                   v
┌──────────────────────────────────────────────────────┐
│ areno/cli/main.py                                    │
│   ├── ArenoCli(click.Group)                          │  ← 延迟加载子命令
│   │    ├── "train"  → areno.cli.train:train_command   │
│   │    ├── "serve"  → areno.cli.serve:serve_command   │
│   │    ├── "check"  → areno.cli.diagnostics           │
│   │    ├── "agent"  → areno.cli.agent                 │
│   │    └── "dashboard" → areno.cli.dashboard          │
│   └── main()                                          │
└──────────────────┬───────────────────────────────────┘
                   │ __import__("areno.cli.train")
                   v
┌──────────────────────────────────────────────────────┐
│ areno/cli/train.py: train_command(**options)          │
│   1. _trainer_config_from_options(**options)          │  ← 构造 TrainerConfig
│   2. [可选] smoke_infer / smoke_train                 │
│   3. [可选] tune_params (内存自动调参)                 │
│   4. _print_training_config_summary()                 │  ← 打印配置摘要
│   5. register_dashboard_job()                         │  ← 注册 dashboard 任务
│   6. run(trainer_config)                              │  ← 实际启动训练
└──────────────────┬───────────────────────────────────┘
                   │
                   v
┌──────────────────────────────────────────────────────┐
│ areno/cli/train.py: run(trainer_config)               │
│   1. resolve_model_refs_for_config()                  │  ← 解析模型引用（Modelscope/HF）
│   2. 加载 dataset (HF/Modelscope/本地文件)             │
│   3. 加载 reward_fn (从 --reward-fn-path)             │
│   4. 创建 areno.api.Trainer (→ ArenoBackend)         │  ← 创建训练器实例
│   5. build_trainer(...) → 具体 Trainer 类             │  ← 算法工厂 dispatch
│   6. trainer.fit()                                    │  ← 开始训练循环
└──────────────────────────────────────────────────────┘
```

### 3.2 训练器初始化 → `Trainer.fit()` 全链路

```
trainer.fit()
    │
    │  self.areno.init()                                ← PolicyOnlyTrainer._fit_initialized()
    v
┌──────────────────────────────────────────────────────────────────────┐
│ areno/api/trainer.py: Trainer.init()                                  │
│                                                                      │
│   1. self._tokenizer = load_tokenizer(real_path)                     │
│      → 加载 tokenizer（支持 Modelscope/HF 自动下载）                   │
│                                                                      │
│   2. self._ctx = Context(world_size, path, tokenizer, config, eos)   │
│      → 构建后端通信上下文                                              │
│                                                                      │
│   3. backend_cls = get_backend_cls(BackendType.Areno)                │
│      → 延迟导入 areno.api.backend.areno → ArenoBackend               │
│                                                                      │
│   4. self._backend = backend_cls()                                   │
│      self._backend.initialize(self._ctx)                             │
│                                                                      │
│      ┌───────────────────────────────────────────────────────────────┤
│      │ ArenoBackend.initialize(ctx)                                  │
│      │                                                               │
│      │   1. ctx.install_model_path()                                 │
│      │      → 解析远程 ref (Modelscope/HF) 到本地路径                   │
│      │                                                               │
│      │   2. ArenoEngine.from_pretrained(model_path, ...)             │
│      │      → 读取 config.json → ModelConfig                         │
│      │      → 构建 EngineConfig                                      │
│      │      → __init__: TPCluster(config, ArenoWorker)               │
│      │      → cluster.start() [启动多个 worker 子进程]                │
│      └───────────────────────────────────────────────────────────────┘
│                                                                      │
│   5. self._initialized = True                                        │
└──────────────────────────────────────────────────────────────────────┘

trainer.fit() 继续执行训练循环
    │
    v
┌──────────────────────────────────────────────────────────────────────┐
│ PolicyOnlyTrainer._fit_initialized()  [GSPO/GRPO 的训练循环]           │
│                                                                      │
│   for epoch in range(epochs):                                        │
│     for batch in dataset:                                            │
│                                                                      │
│       ┌─── Rollout ──────────────────────────────────────────────┐  │
│       │  1. begin_rollout_session()                               │  │
│       │     → ArenoEngine.begin_rollout_session()                 │  │
│       │     → TPCluster.call(Op.ROLLOUT_SESSION_BEGIN)            │  │
│       │     → 每个 worker: rollout_session_begin()                │  │
│       │         → _prepare_actor_onloaded()  [模型权重转移到 GPU]  │  │
│       │                                                           │  │
│       │  2. rollout_batch(prompts, n_samples, params)             │  │
│       │     → ArenoBackend.rollout_batch()                        │  │
│       │     → ArenoEngine.generate_rollout()                      │  │
│       │     → TPCluster.call(Op.ROLLOUT)                          │  │
│       │     → 每个 worker: run_rollout_command() → infer_rollout() │  │
│       │     → InferenceManager 执行 prefill + decode              │  │
│       │     → TP 间 all-reduce logits                             │  │
│       │     → 返回 completions + logprobs                         │  │
│       │                                                           │  │
│       │  3. end_rollout_session()                                 │  │
│       └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│       ┌─── Reward ───────────────────────────────────────────────┐  │
│       │  reward_fn(record, completions) → [score1, score2, ...]   │  │
│       │  advantages = group_normalize(rewards)                     │  │
│       │  → 构造 TrainSequence 列表                                  │  │
│       └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│       ┌─── Train ────────────────────────────────────────────────┐  │
│       │  trainer.train(batch, loss_fn, mini_bs)                   │  │
│       │  → ArenoBackend.train()                                   │  │
│       │  → _make_train_pack()  [打包 tensors]                     │  │
│       │  → ArenoEngine.step()                                     │  │
│       │  → TPCluster.call(Op.TRAIN)                               │  │
│       │  → 每个 worker: handle(Command) → self.training.train()   │  │
│       │  → TrainingManager.train(payload)                         │  │
│       │    → 按 microbatch 前向 → loss → 反向传播                   │  │
│       │    → TP 间 all-reduce grads                                │  │
│       │    → optimizer.step()                                      │  │
│       │  → 返回训练统计 TrainStats                                  │  │
│       └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│       trainer.finish_step()  [记录 metrics]                          │
│                                                                      │
│   trainer.close()                                                    │
│     → ArenoBackend.close() → ArenoEngine.close()                     │
│     → TPCluster.close() → 停止所有 worker 子进程                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 `ArenoEngine.__init__` → Worker 进程启动

```
ArenoEngine.__init__(config)
    │
    │  TPCluster(config, ArenoWorker)
    │  cluster.start()
    v
┌──────────────────────────────────────────────────────────────────────┐
│ TPCluster.start()                                                    │
│   for rank in range(world_size):                                     │
│     process = mp.Process(target=_worker_entry, args=(config, rank))  │
│     process.start()  ← 启动独立的子进程                               │
│                                                                      │
│ _worker_entry(config, rank):                                         │
│   1. torch.cuda.set_device(local_device)                             │
│   2. torch.distributed.init_process_group(...)  [TP 组初始化]         │
│   3. worker = ArenoWorker(config)                                    │
│      → build_model_on_device()  [根据 model_type 构建模型]            │
│      → load_model_weights()       [加载 HF checkpoint]               │
│      → build_optimizer()          [创建 AdamW 优化器]                 │
│      → InferenceManager(config)   [分配 KV cache, CUDA graphs]       │
│      → TrainingManager(config)    [准备训练前向图]                    │
│      → RoleManager(config)        [PPO 角色管理]                     │
│   4. worker.run_loop()  [主循环阻塞, 等待命令]                         │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 Worker 主循环

```
ArenoWorker.run_loop():
    while True:
        cmd = recv_command()  ← 从 IPC 管道接收命令

        match cmd.op:
            Op.ROLLOUT_SESSION_BEGIN → rollout_session_begin()
            Op.ROLLOUT              → run_rollout_command()
            Op.ROLLOUT_SESSION_END  → rollout_session_end()
            Op.TRAIN                → train()
            Op.SCORE_LOGPROBS       → score_logprobs()
            Op.SAVE_CHECKPOINT      → save_checkpoint()
            Op.EXIT                 → break
```

---

## 四、方式 3：CLI 服务启动 (`areno serve`)

### 4.1 服务启动链路

```
用户执行:  areno serve --model-path /path/to/model --tp-size 1 --port 8000

    │
    v
┌──────────────────────────────────────────────────────────────────────┐
│ areno/cli/serve.py: serve_command(...)                               │
│                                                                      │
│   1. 解析 model_path (Modelscope/HF 自动下载)                         │
│   2. resolve_model_ref() → 本地路径                                   │
│   3. app = create_app(model_path, tp_size, world_size, ...)          │
│      │                                                               │
│      │  ┌──────────────────────────────────────────────────────┐    │
│      │  │ create_app():                                         │    │
│      │  │                                                       │    │
│      │  │ a) load_tokenizer(model_path)                         │    │
│      │  │    → 加载 tokenizer + chat template 配置                │    │
│      │  │                                                       │    │
│      │  │ b) ArenoEngine.from_pretrained(                       │    │
│      │  │       model_path,                                     │    │
│      │  │       tp_size=...,                                   │    │
│      │  │       dp_size=world_size // tp_size,                  │    │
│      │  │       loss_fn=_serve_loss_fn,  ← 占位 loss (不可训练)  │    │
│      │  │       runtime_config=RuntimeConfig(                   │    │
│      │  │           eager_decode=...,                           │    │
│      │  │           attn_backend=...                            │    │
│      │  │       )                                               │    │
│      │  │    )                                                  │    │
│      │  │    → __init__ → TPCluster.start() → Workers 启动      │    │
│      │  │                                                       │    │
│      │  │ c) FastAPI app 构建:                                   │    │
│      │  │    - GET  /health           → 健康检查                  │    │
│      │  │    - GET  /v1/models        → 模型列表                  │    │
│      │  │    - POST /v1/chat/completions → OpenAI 兼容 API       │    │
│      │  │                                                       │    │
│      │  │ d) @app.on_event("startup"):                         │    │
│      │  │    await engine.begin_rollout_session_async()         │    │
│      │  │    → 准备持续 rollout 状态                              │    │
│      │  └──────────────────────────────────────────────────────┘    │
│                                                                      │
│   4. uvicorn.run(app, host=host, port=port)                         │
│      → 启动 ASGI 服务器，监听 HTTP 请求                                │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 服务请求处理流程

```
Client                          AReno Serve
  │                                  │
  │  POST /v1/chat/completions       │
  │  {model, messages, ...}          │
  │ ─────────────────────────────>   │
  │                                  │  chat_completions(request)
  │                                  │   1. _encode_messages() → token IDs
  │                                  │   2. 构造 SamplingParams
  │                                  │   3. 创建 PendingRequest + asyncio.Future
  │                                  │   4. engine.generate_rollout_async(...)
  │                                  │
  │                                  │   ┌─ 异步请求分发 ──────────────┐
  │                                  │   │ TPCluster.call_async(Op.ROLLOUT)│
  │                                  │   │ Worker: infer_rollout()     │
  │                                  │   │  → continuous batching      │
  │                                  │   │  → token 生成 → stream 返回  │
  │                                  │   └─────────────────────────────┘
  │                                  │
  │  SSE stream / JSON response      │
  │ <─────────────────────────────   │
  │                                  │
```

---

## 五、方式 4：CLI Agent 启动 (`areno agent`)

### 5.1 Agent 入口

```bash
# 方式 A: 安装后使用
areno agent "Give me a complete command to run the math demo..."

# 方式 B: 源码直接运行 (无需安装)
./agent.sh "Give me a complete command to run the math demo..."
```

### 5.2 `agent.sh` 内部

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
exec python -m areno.cli.main agent "$@"
```

直接设置 `PYTHONPATH` 指向仓库根目录，通过 `python -m areno.cli.main agent` 启动。

### 5.3 Agent 工作模式

- **配置模式** (`--set`): 设置 OpenAI 兼容 endpoint/model/api-key
- **交互模式**: 自然语言描述任务 → Agent 生成并执行 AReno 命令
- 内部使用 `areno/agent/agent_loop.py` 中的 `run_agentic_coding_loop`

---

## 六、方式 5：SDK 编程式启动

### 6.1 最小 SDK 启动流程

```python
import asyncio
from functools import partial
from datasets import load_dataset
from areno.api import (
    Areno, ArenoConfig, SamplingParams,
    Trainer, TrainSequence, gspo_loss_fn,
)
from examples.math.math_verify_reward import reward_fn

async def main():
    # 1. 创建 Trainer (不会初始化 workers)
    trainer = Trainer(
        world_size=1,
        model_path="Qwen/Qwen3-0.6B",
        backend_type=Areno,
        custom_config=ArenoConfig(tp_size=1),
    )

    # 2. 初始化 (加载 tokenizer, 启动 worker 进程)
    trainer.init()

    try:
        # 3. Rollout — 生成 on-policy 样本
        async with trainer.rollout_session(sampling_params=sampling, proxy=False):
            rollout = trainer.rollout_batch([prompt], n_samples=8, sampling_params=sampling)[0]

        # 4. Reward & Advantages
        completions = [trainer.get_tokenizer().decode(seq.resp_tokens) for seq in rollout.sequences]
        rewards = reward_fn(row, completions)
        advantages = to_advantages(rewards)

        # 5. 构造 TrainSequence
        batch = [TrainSequence(
            prompt_mask=..., tokens=..., logprobs=..., advantages=..., reward=...,
            eos_token_id=...,
        ) for seq, reward, advantage in zip(...)]

        # 6. Train — 一步优化
        stats = trainer.train(batch, partial(gspo_loss_fn, clip_eps=3.0e-4), mini_bs=4)

    finally:
        # 7. 清理
        trainer.close()

asyncio.run(main())
```

### 6.2 SDK 启动控制流

```
sdTrainer.__init__()  ← 构造，不启动
    │
    │  Trainer(world_size, model_path, backend_type, custom_config)
    │  → 存储参数, 创建 MetricsRecorder(可选)
    │  → _initialized = False
    │
    v
Trainer.init()
    │
    │  load_tokenizer(real_path)
    │  Context(...)
    │  get_backend_cls(BackendType.Areno)  ← 延迟导入 ArenoBackend
    │  backend.initialize(ctx)
    │    → 解析 model_path
    │    → ArenoEngine.from_pretrained(...)
    │       → 读取 config.json → ModelConfig → EngineConfig
    │       → TPCluster(config, ArenoWorker).start()  ← Workers 启动
    │    → _prefer_repo_areno()  [优先使用源码内的 areno 模块]
    │  _initialized = True
    │
    v
[用户代码控制训练循环]
    │
    │   begin_rollout_session()  → rollout_batch()  → end_rollout_session()
    │   train(batch, loss_fn, mini_bs)
    │   finish_step()
    │   save_checkpoint(path)
    │
    v
Trainer.close()
    → backend.close()
    → engine.close()
    → TPCluster.close()  ← Worker 进程退出
```

---

## 七、辅助启动命令

### 7.1 `areno check` — 环境诊断

```bash
areno check
```
检查项目: CUDA 可用性、PyTorch 版本、nvcc 路径、`areno_accel` 扩展、FlashAttention 等。

### 7.2 `areno env --json` — 环境报告

```bash
areno env --json
```
输出完整的 AReno/Python/平台/PyTorch/GPU/nvcc/依赖环境报告。

### 7.3 `areno dashboard` — Web Dashboard

```bash
areno dashboard       # 启动 React Dashboard
```
Dashboard 前端 (React 19 + Vite 6) 预构建在 `areno/dashboard/dist/` 中，通过 `dashboard/server.py` 提供 HTTP 服务。

---

## 八、启动过程关键决策点

| 决策点 | 控制变量 / 参数 | 影响 |
|--------|----------------|------|
| CUDA 扩展是否编译 | `ARENO_BUILD_EXT` (默认 `1`) | `0`=跳过 CUDA 编译(仅用于元数据安装) |
| 目标 GPU 架构 | `TORCH_CUDA_ARCH_LIST` | 仅编译指定架构的 kernel |
| 模型来源 | `--model-hub` (modelscope/hf) | 决定从哪个平台下载模型 |
| TP 并行度 | `--tp-size` | 张量并行分片数 |
| DP 并行度 | `world_size / tp_size` | 数据并行组数 |
| 注意力后端 | `--attn-backend` (flash/native) | flash-attn 或 areno 原生注意力 |
| Worker 内存探针 | `--tune-params --mem-frac 0.9` | 自动调整 batch/并发参数 |
| 解密模式 | `--eager-decode` | 不使用 CUDA graph 加速解码 |

---

## 九、启动文件索引

| 文件路径 | 角色 |
|----------|------|
| `pyproject.toml` | 声明 console_scripts 入口: `areno = areno.cli.main:main` |
| `setup.py` | CUDA 扩展动态编译、平台检查、PyTorch 版本检查 |
| `scripts/install.sh` | 一键安装脚本 (7 步流程) |
| `agent.sh` | 无需安装的 agent 入口 |
| `areno/cli/main.py` | CLI 顶层 dispatcher (`ArenoCli`) |
| `areno/cli/train.py` | `train_command` → `run()` → `trainer.fit()` |
| `areno/cli/serve.py` | `serve_command` → `create_app()` → `uvicorn.run()` |
| `areno/cli/diagnostics.py` | `check_command`, `env_command` |
| `areno/cli/agent.py` | `agent_command` |
| `areno/cli/dashboard.py` | `dashboard_command` |
| `areno/api/trainer.py` | SDK `Trainer` 类 — `init()` / `rollout_batch()` / `train()` / `close()` |
| `areno/api/trainer_factory.py` | `build_trainer()` — 算法 → 具体 Trainer 类 dispatch |
| `areno/api/backend/base.py` | `Backend` 抽象协议 + 延迟注册表 |
| `areno/api/backend/areno/backend.py` | `ArenoBackend` — 连接 Trainer 与 ArenoEngine |
| `areno/api/trainers/policy_only.py` | `PolicyOnlyTrainer.fit()` — GSPO/GRPO 训练循环 |
| `areno/api/trainers/sft.py` | `SFTTrainer.fit()` |
| `areno/api/trainers/dpo.py` | `DPOTrainer.fit()` |
| `areno/api/trainers/ppo.py` | `PPOTrainer.fit()` |
| `areno/api/algorithms.py` | 算法注册表 `AlgorithmSpec` |
| `areno/engine/api.py` | `ArenoEngine` — 用户侧引擎协调器 |
| `areno/engine/worker.py` | `ArenoWorker` — rank 侧 worker 主循环 |
| `areno/engine/protocol.py` | `TPCluster` — 跨进程 TP 通信协议 |
| `areno/engine/inference.py` | `InferenceManager` — rank 侧推理执行 |
| `areno/engine/training.py` | `TrainingManager` — rank 侧训练执行 |
| `areno/models/registry.py` | 模型适配器发现与 dispatch |

---

## 十、完整启动流程图

```
                            ┌─────────────────────┐
                            │  areno 命令/脚本入口  │
                            └─────────┬───────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           v                          v                          v
   ┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐
   │ install.sh   │         │ areno train      │       │ areno serve      │
   │ (7步安装)     │         │ areno.cli.train   │       │ areno.cli.serve  │
   └──────┬───────┘         └────────┬─────────┘       └────────┬─────────┘
          │                          │                          │
          v                          v                          v
   ┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐
   │ setup.py     │         │ Trainer.__init__ │       │ create_app()     │
   │ (CUDA编译)    │         │ Trainer.init()   │       │   tokenizer      │
   └──────────────┘         └────────┬─────────┘       │   engine         │
                                     │                 │   FastAPI        │
                                     v                 └────────┬─────────┘
                            ┌──────────────────┐                │
                            │ ArenoBackend    │                │
                            │ .initialize()   │                │
                            └────────┬────────┘                │
                                     │                          │
                                     v                          v
                            ┌──────────────────────────────────────┐
                            │        ArenoEngine.__init__()         │
                            │    ┌──────────────────────────────┐   │
                            │    │  TPCluster(config, Worker)    │   │
                            │    │     .start()                  │   │
                            │    │       → 启动 world_size 个    │   │
                            │    │         子进程                  │   │
                            │    │       → 每个进程:              │   │
                            │    │         - init_process_group  │   │
                            │    │         - ArenoWorker(config) │   │
                            │    │         - worker.run_loop()   │   │
                            │    └──────────────────────────────┘   │
                            └──────────────────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          v                     v
                   ┌────────────┐      ┌──────────────┐
                   │ 训练模式    │      │ 服务模式       │
                   │            │      │               │
                   │ rollout →  │      │ POST /v1/chat │
                   │  reward →  │      │  /completions │
                   │  train     │      │  → generate   │
                   │            │      │  → stream/JSON│
                   └────────────┘      └──────────────┘
```

`ArenoEngine.__init__` 在 **训练** 和 **服务** 两种模式中都是核心——它统一负责 worker 进程的启动和管理。训练模式在此基础上增加 reward 评分和梯度更新；服务模式在此基础上增加 HTTP API 和 continuous batching。