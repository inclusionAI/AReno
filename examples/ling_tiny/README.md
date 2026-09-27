# 用 AReno 后训 Ling-3.0-tiny

本文回答两个问题：AReno 目前能对 `inclusionAI/Ling-3.0-tiny` 做哪些后训，以及每种怎么跑。
可执行的命令都在同目录的 `train.sh` 里：

```bash
bash examples/ling_tiny/train.sh <recipe> [额外的 areno 参数...]
# recipe: lora-sft | full-sft | dpo | gspo | classify | serve
```

## 1. 结论：支持矩阵

Ling-3.0-tiny 的 `model_type` 是 `bailing_hybrid`，由 AReno 的 Bailing-MoE V3 适配器
（`areno/models/bailing_v3/`）加载。加载、推理、serve 和训练都走这一个适配器。

| 后训方式 | 状态 | 依据 |
| --- | --- | --- |
| GSPO / GRPO（RLVR、agentic RL） | ✅ 仓库里有跑通的例子 | `examples/agentic/tictactoe`（DGX Spark 指南）、`examples/agentic/bash_game`（有训练前后的对比结果）、`examples/agentic/terminal_hacking` |
| LoRA SFT | ✅ 代码支持，有使用记录 | 适配器有 Bailing V3 专用的 LoRA 注入；Draco、Ling-flash-Fin、OPC 几个 SFT 例子在 `feat/ling-sft-example` 分支上，**尚未合入 main** |
| 全参 SFT | ⚠️ 代码路径通用，没有 Ling 的例子 | 未在 Ling 上验证 |
| DPO（LoRA，基座兼做 reference） | ⚠️ 代码路径通用，没有 Ling 的例子 | 未在 Ling 上验证 |
| PPO（带 critic） | ⚠️ 代码路径通用 | critic 从 Ling checkpoint 构建，value head 零初始化；未在 Ling 上验证 |
| 分类 / 打分（`classify`，JevForge 式） | ⚠️ 本次提交新增 | Bailing V3 的 forward 已支持 `defer_lm_head`；未在 GPU 上跑过 |
| serve（OpenAI 兼容接口，可挂 LoRA） | ✅ | `areno serve --lora-adapter-path` |
| MLX（Apple Silicon） | ❓ | 取决于已安装的 `mlx-lm` 是否支持 `bailing_hybrid`；未验证 |

一句话：**RL 和 LoRA SFT 可以直接用。全参 SFT、DPO、PPO、classify 的代码路径是通的，
但还没有在 Ling-3.0-tiny 上实际跑过**，第一次跑建议先用 `--max-steps 2` 做冒烟测试。

## 2. 模型结构对训练的影响

下面的信息来自 ModelScope 上 `inclusionai/ling-3.0-tiny` 的 `config.json`：

| 字段 | 值 | 对训练的影响 |
| --- | --- | --- |
| `num_hidden_layers` / `hidden_size` | 24 / 1536 | |
| `layer_group_size` | 4 | 每组 4 层中有 1 层是 softmax 注意力（MLA：`q_lora_rank=256`、`kv_lora_rank=512`），其余 3 层是 KDA 线性注意力 |
| `num_experts` / `num_experts_per_tok` / `num_shared_experts` | 128 / 8 / 1 | 稀疏 MoE。第 0 层是 dense MLP（`first_k_dense_replace=1`）。绝大多数参数在路由专家里 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 16 | `--tp-size` 只能取 1、2、4、8、16 |
| `vocab_size` / `max_position_embeddings` | 157184 / 131072 | |
| `no_kda_lora` | true | 满足 AReno native LoRA 对 Bailing V3 的前提条件 |
| `moe_router_bias_update_rate` | 未设置，默认 0 | 训练时不更新路由 expert bias，所以可以用 LoRA |

实际运行时的几点要求：

- **依赖**：`bailing_v3` 在导入时就会 `import fla`，所以必须装 `flash-linear-attention`。
  `--attn-backend flash`（默认）还需要 `flash-attn`；没有的话加 `--attn-backend native`，速度会慢一些。
- **显存**：单卡（例如 DGX Spark 的 GB10，128 GB 统一内存）上，现有例子都用 LoRA、`--adam-4bit` 和
  activation checkpointing。全参训练要为 128 个专家都保存优化器状态，所以 `full-sft` 配方默认开了
  `--optimizer-state-offload cpu`。
- **MoE 的 RL**：rollout 类算法默认开启 rollout routing replay（R3），训练时会复用采样阶段的专家路由。

## 3. 环境

```bash
pip install psutil flash-linear-attention
pip install flash-attn                      # 可选，只有 --attn-backend flash 需要
pip install -e . --no-build-isolation
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

模型默认从 ModelScope 下载（`--model-hub modelscope`）。`MODEL` 也可以指向本地目录。

## 4. 数据格式

| 算法 | 每行的字段 | 说明 |
| --- | --- | --- |
| SFT | `prompt`、`response` | prompt 会自动套 Ling 的 chat template（`add_generation_prompt=True`）；如果 prompt 已经是 chat 格式，就原样使用。loss 只算在 response 上 |
| SFT（多轮 / 工具调用） | `tokens`、`prompt_mask`，可选 `loss_mask` | 已经编码好的行，适合自己渲染多轮对话或工具调用轨迹的情况 |
| DPO | `prompt`、`chosen`、`rejected` | `--mini-bs` 必须是偶数，保证 chosen/rejected 成对 |
| GSPO / GRPO / PPO | 由 `--dataset-loader-fn` 生成 prompt 行 | 再配 `--reward-fn-path`（`reward_fn`）；多轮 agent 任务还要配 `--agent-fn` |
| classify | JevForge records | 见 `examples/classify/jev/README.md` |

原始数据字段不一样时，写一个 `load_training_dataset(dataset_path, *, default_loader, **_)`，
用 `LOADER=...` 传进去即可，可以参考 `examples/sft/alpaca/dataset_loader.py`。

## 5. 配方

所有配方都认这些环境变量：`MODEL`、`MODEL_HUB`、`TP_SIZE`、`WORLD_SIZE`、`SAVE_PATH`、
`SAVE_INTERVAL`、`LOADER`。另外还有 `BATCH_SIZE`、`MINI_BS`、`EPOCHS`、`LR`、`MIN_LR`、
`MAX_PROMPT_TOKENS`、`MAX_NEW_TOKENS`、`LORA_RANK` 等。命令行后面追加的参数会覆盖默认值。
所有配方都带 `--disable-thinking`，服务端也带，这样训练和推理的 prompt 格式一致。

### 5.1 LoRA SFT（推荐的起点）

```bash
DATASET=data/train.jsonl SAVE_PATH=outputs/ling-lora SAVE_INTERVAL=20 \
  bash examples/ling_tiny/train.sh lora-sft --epochs 5
```

- 默认参数：rank 16、`--adam-4bit`、lr `1e-4 → 1e-5`，和分支上 Draco / Flash-Fin 例子的设置一致。
- **SFT trainer 只在 `--save-interval` 的整数倍步保存，结束时不会再存一次**，所以
  `SAVE_INTERVAL` 要小于总步数（总步数 = 行数 / `BATCH_SIZE` × epochs）。
- LoRA 默认的目标模块是 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。按代码读下来：
  - 这组默认值覆盖 KDA 层的注意力，以及所有 MLP（路由专家逐个挂 LoRA，共享专家也挂）；
  - **MLA 那几层 softmax 注意力的投影叫 `q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/`dense`，默认不在里面**。
    如果也想训这几层，追加
    `--lora-target-modules q_proj,k_proj,v_proj,o_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,dense,gate_proj,up_proj,down_proj`。
    这一点是读代码推断的，没有在真机上验证。
- 知识注入比风格迁移需要更多轮次。效果不够时，先加 epochs 或把 rank 提到 32，再考虑加数据。

### 5.2 全参 SFT（未在 Ling 上验证）

```bash
DATASET=data/train.jsonl TP_SIZE=4 WORLD_SIZE=8 bash examples/ling_tiny/train.sh full-sft
```

多卡时专家会按 TP rank 切分，每个 rank 持有 `128 / tp_size` 个专家。显存不够时按顺序尝试：
减小 `MINI_BS`，设 `OPTIMIZER_OFFLOAD=disk` 并追加 `--optimizer-state-offload-dir /nvme/path`，加大 `TP_SIZE`。

### 5.3 DPO（未在 Ling 上验证）

```bash
DATASET=data/prefs.jsonl bash examples/ling_tiny/train.sh dpo
```

配方用的是 LoRA 加 `--reference-mode reuse_actor_base`：关掉 LoRA 的冻结基座直接充当 reference 模型，
不用再加载第二份权重。想用独立的 reference，就去掉这个参数，并传 `--ref-ckpt`。

### 5.4 GSPO / GRPO（RLVR、agentic）

```bash
DATASET=/data/train.jsonl LOADER=my/dataset_loader.py REWARD=my/reward.py AGENT=my/run_agent.py \
  bash examples/ling_tiny/train.sh gspo
```

- 仓库里跑通过的完整例子：`examples/agentic/tictactoe/DGX_SPARK_GUIDE_CN.md`、
  `examples/agentic/bash_game/README.md`（Ling-3.0-tiny 上 solved_frac 从 0.44 提到 0.67）。
- RL 的 lr 要比 SFT 小很多。现有例子用的是 `1e-8` 到 `1e-5`，配方默认 `1e-6`。
- rollout 显存主要由 `--max-running-prompts` 和 `--max-new-tokens` 决定。单卡紧张时加 `--drop-rollout-state`。
- 想用 GRPO，改成 `--algo grpo` 即可（追加的参数会覆盖默认值）。

### 5.5 分类 / 打分（classify，未在 Ling 上验证）

```bash
RECORDS=/data/jevforge/web_full bash examples/ling_tiny/train.sh classify --max-steps 600
```

做法是在最后一个 token 上接一个打分头，同一道题的所有候选放进一个 softmax，损失是 CE + Brier。
只支持全参训练，不支持 LoRA。jev-forge 用 `trust_remote_code=False` 加载模型，所以除非你装的
transformers 自带 `bailing_hybrid`，否则 `export_jevforge.py` 导不出 Ling 的 checkpoint。
这种情况下可以直接读 AReno 存下的 HF 目录加 `score_head.safetensors`。

## 6. 服务与评测

```bash
# LoRA：基座 + adapter
ADAPTER=outputs/ling-lora/step_000100 bash examples/ling_tiny/train.sh serve
# 全参：直接指向 checkpoint 目录
MODEL=outputs/ling-tiny-full-sft/step_000100 bash examples/ling_tiny/train.sh serve
```

服务端提供 OpenAI 兼容的 `/v1/chat/completions`。评测时请用训练时的格式发请求：同一个 system prompt、
同样的 thinking 开关。

## 7. 注意事项

- 首次运行时先加 `--max-steps 2` 做冒烟测试，确认加载、训练、保存都正常，再正式开训。
- `--disable-thinking` 在训练和推理两边要一致。Ling 的 chat template 对这个开关到底有什么影响，没有核实过；
  两边保持一致就不会出错。
- 超过 `--max-prompt-tokens` / `--max-new-tokens` 的行会被丢掉，日志里会打印丢了多少。
- 需要 GPU 的训练、服务和评测，都在有 CUDA 的机器上跑。本文的命令只做过参数拼装的干跑，
  **没有在 GPU 上执行过**。
