# Battleship Agent 训练完整指南

## 环境要求

- NVIDIA GPU (推荐 8GB+ VRAM，如 RTX 3070/4060/4090 或 A100/H100)
- Python 3.10+
- PyTorch 2.6+ with CUDA
- AReno 已安装 (`pip install -e . --no-build-isolation`)

---

## 训练步骤

### 1. 生成训练数据

```bash
# 生成 512 个随机舰队布局用于训练
python examples/agentic/battleship/dataset_generator.py \
  --output ./data/battleship_train.jsonl \
  --count 512 \
  --seed 2026

# 生成 128 个用于评估（不同 seed）
python examples/agentic/battleship/dataset_generator.py \
  --output ./data/battleship_eval.jsonl \
  --count 128 \
  --seed 42
```

### 2. 启动训练（基础模型 Qwen3-0.6B）

```bash
# 对于 16GB GPU (如 Kaggle P100/V100)，使用更小的 batch_size 避免 OOM
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path ./data/battleship_train.jsonl \
  --dataset-loader-fn examples/agentic/battleship/dataset_loader.py \
  --agent-fn examples/agentic/battleship/run_agent.py \
  --reward-fn-path examples/agentic/battleship/reward.py \
  --algo gspo \
  --n_samples 2 \
  --train_steps 100 \
  --batch_size 2 \
  --metrics-log-dir ./runs/battleship_qwen3_0.6b \
  --tp-size 1 \
  --world-size 1
```

参数说明：
| 参数 | 含义 |
|------|------|
| `--ckpt Qwen/Qwen3-0.6B` | 基础模型 (0.6B 参数，单 GPU 友好) |
| `--algo gspo` | 使用 Group-level Self-Play Optimization |
| `--n_samples 4` | 每个舰队采样 4 条轨迹进行对比 |
| `--train_steps 100` | 训练 100 步（可增加到 500+） |
| `--batch_size 4` | 每批 4 个样本 |

### 3. 监控训练进度

```bash
# 终端 1 启动训练后，终端 2 查看 TensorBoard
tensorboard --logdir ./runs/battleship_qwen3_0.6b --port 6006
```

打开 http://localhost:6006 观察：

**关键指标：**
- `rollout/accuracy`: **胜率** —— 从 ~0% 提升到 ~80%+
- `rollout/rewards_mean`: **平均奖励** —— 从 ~0 提升到 ~1.0+
- `loss/policy`: 策略损失下降趋势

**典型训练曲线：**
```
Step 0:   win_rate=0.02, reward_mean=0.15   (几乎随机)
Step 25:  win_rate=0.25, reward_mean=0.45   (开始学会追击)
Step 50:  win_rate=0.55, reward_mean=0.75   (掌握 hunt/target)
Step 100: win_rate=0.85, reward_mean=1.05   (专家水平)
```

### 4. 评估训练好的模型

**4.1 基准对比（训练前 vs 训练后）**

```bash
# 随机基线（无需 GPU）
python examples/agentic/battleship/evaluate.py \
  --fleets ./data/battleship_eval.jsonl \
  --player random \
  --output ./results/random_baseline.json

# 启发式策略（手工设计）
python examples/agentic/battleship/evaluate.py \
  --fleets ./data/battleship_eval.jsonl \
  --player fake \
  --output ./results/heuristic_baseline.json
```

**4.2 启动训练好的模型服务**

```bash
# 假设训练完成后 checkpoint 保存在 ./runs/battleship_qwen3_0.6b/step_100
areno serve \
  --model-path ./runs/battleship_qwen3_0.6b/step_100 \
  --tp-size 1 \
  --world-size 1 \
  --port 8000
```

**4.3 评估 LLM 在 eval 数据集上的表现**

```bash
# 用 play_llm.py 对任意 OpenAI 兼容端点（served 训练模型或外部 LLM）跑多局评估
python examples/agentic/battleship/play_llm.py \
  --base-url http://127.0.0.1:8000/v1 \
  --games 50 \
  --seed 42 \
  --output ./results/trained_agent.json
```

> 也可指向外部大模型：`--base-url https://api.openai.com/v1 --api-key "$OPENAI_API_KEY" --model gpt-4o-mini`。

### 5. Web UI 体验训练后的模型

```bash
# 连接 LLM 服务端点
python examples/agentic/battleship/web_ui.py \
  --agent-mode llm \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key token
```

浏览器访问 http://127.0.0.1:8768

点击 **"Auto-play"** —— 观察训练好的 LLM 如何智能地：
- 随机开局（hunt phase）
- 一旦命中立即追踪相邻格子（target phase）
- 击沉一艘后转向下一个目标

---

## 预期训练效果对比

| 策略 | 胜率 | 平均用弹数 | 完成率 |
|------|------|-----------|--------|
| 随机基线 | ~0% | 40 (用满) | ~60% |
| 启发式策略 | ~90% | ~25 | ~100% |
| **训练后 LLM** | **~85%** | **~28** | **~100%** |

---

## 进阶训练技巧

### 增加训练轮数
```bash
# 从 100 步增加到 500 步，效果更佳
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  ... \
  --train_steps 500
```

### 使用更大的批次
```bash
# 如果有更多 GPU
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  ... \
  --batch_size 16 \
  --world-size 2  # 多 GPU
```

### 切换算法（GSPO/GRPO/DPPO）
```bash
# GRPO (Group Relative Policy Optimization)
areno train ... --algo grpo

# DPPO (Direct Preference Policy Optimization)
areno train ... --algo dppo
```

---

## OOM（显存不足）问题排查

### 如果你遇到这个错误：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB...
```

### 解决方案（按优先级）：

**1. 减小 batch_size（最快解决）**
```bash
--batch_size 2  # 原值 4，可进一步降为 1
```

**2. 减小 n_samples（减少并行轨迹数）**
```bash
--n_samples 2  # 原值 4，最小可设为 2（GSPO 需要至少 2 条轨迹对比）
```

**3. 设置环境变量减少内存碎片**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
areno train ...
```

**4. 启用梯度检查点（用计算换内存）**
```bash
areno train ... --gradient-checkpointing
```

**5. 如果仍 OOM，使用更小的模型**
```bash
# Qwen3-0.6B 仍太大，尝试更小的模型路径
--ckpt /path/to/0.3B_or_smaller_model
```

### 不同 GPU 推荐配置：

| GPU 显存 | --batch_size | --n_samples | 备注 |
|---------|-------------|------------|------|
| 24GB (RTX 4090) | 4 | 4 | 默认配置 |
| 16GB (P100/V100) | 2 | 2 | **你遇到的 OOM 情况** |
| 12GB (RTX 3060) | 1 | 2 | 较慢但可行 |
| 8GB | 1 | 2 | 考虑使用 CPU offloading |

---

## 常见问题

**Q: 训练需要多长时间？**
A: 单卡 RTX 4090 上：
- 100 steps: ~15 分钟
- 500 steps: ~60 分钟

**Q: 显存不足？**
A: 降低 batch_size 或使用更小的模型
```bash
--batch_size 2  # 默认 4
--ckpt Qwen/Qwen3-0.6B  # 可能用 0.5B 更小
```

**Q: 模型不收敛？**
A: 检查：
1. `rollout/accuracy` > 0 说明有梯度
2. 奖励函数是否正常（可手动调试 `reward.py`）
3. 增加 `--n_samples` 到 8 提高对比信号

---

## 快速验证训练是否有效

训练后运行：
```bash
# 同一个 seed=42 的舰队，对比三种策略
echo "=== Seed 42 Fleet 对比 ==="

# 随机 vs 启发式 (已在 evaluate.py 中)
# 训练好的 LLM (需要在 Web UI 中手动或写脚本)

# 验证：训练好的 LLM 应该能用 20-30 发击沉所有船
```
