# 在 Kaggle 上运行 Maze Agentic RL Demo

## 可用 GPU 资源

Kaggle 提供以下 GPU，均可运行此 demo：

| GPU | 显存 | 推荐模型 | TP size |
|-----|------|---------|---------|
| T4 ×2 | 16GB ×2 | Qwen3-0.6B | 1 (单卡) |
| P100 | 16GB | Qwen3-0.6B | 1 |
| TPU v5e-8 | — | 暂不支持（AReno 仅支持 CUDA） |

**推荐**：使用 T4 ×2 + Qwen3-0.6B，单卡训练 (tp-size=1)。

---

## Kaggle Notebook 设置

### 1. 创建 Notebook

- 打开 https://www.kaggle.com/code
- 新建 Notebook
- **Settings → Accelerator → GPU T4 ×2**

### 2. 安装环境

在第一个 cell 中运行：

```python
# 安装 AReno 及依赖（Kaggle 自带 PyTorch，只需补充缺失依赖）
!pip install -q psutil flash-linear-attention openai --no-deps
!pip install -q math-verify addict

# 克隆你的 fork 并切换到 maze 分支
!git clone -b feat/maze-agentic-rl https://github.com/sliverdancer/AReno.git /kaggle/working/AReno
%cd /kaggle/working/AReno

# 安装 AReno（跳过 CUDA 编译，Kaggle 自带 PyTorch）
!ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation

# 验证安装
!python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"
!areno --version
```

### 3. 生成迷宫数据集

```python
# 生成 2048 个 7×7 迷宫，vision_radius=1 (3×3 视野)
!python examples/agentic/maze/dataset_generator.py \
  --output /kaggle/working/mazes.jsonl \
  --count 2048 \
  --seed 2026 \
  --width 7 \
  --height 7 \
  --vision-radius 1
```

### 4. 运行训练

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /kaggle/working/mazes.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 64 \
  --tp-size 1 \
  --world-size 1 \
  --max-steps 100
```

### 5. 运行 CPU 测试

```bash
# 全量 CPU 测试（不需要 GPU）
pytest tests/test_agentic_maze_example_cpu.py -v
```

### 6. 通过 ngrok 暴露 Dashboard

Kaggle Notebook 没有公网端口，使用 ngrok 隧道在外部浏览器访问 AReno Dashboard。

前置条件：在 Kaggle Notebook 的 **Add-ons → Secrets** 中添加一个名为 `ngrok_key` 的 secret，值为你的 ngrok authtoken（从 https://dashboard.ngrok.com/get-started/your-authtoken 获取）。

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
ngrok_key = user_secrets.get_secret("ngrok_key")

!pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token(ngrok_key)
public_url = ngrok.connect(8000)
print(public_url)

!areno dashboard --start --host 0.0.0.0 --port 8000
```

运行后终端会输出一个 `https://xxxx.ngrok-free.app` 地址，在任意浏览器打开即可查看训练指标、奖励曲线和轨迹详情。

---

## 参数调优建议

### T4 (16GB) 配置

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /kaggle/working/mazes.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 64 \
  --tp-size 1 \
  --world-size 1 \
  --max-steps 200 \
  --max-prompt-tokens 256
```

### P100 (16GB) 配置

与 T4 相同的配置。P100 不支持 FlashAttention，AReno 会自动回退到 native attention。

### 更大模型 (Qwen3-1.7B)

```bash
# 仅在 T4×2 上尝试，使用 tp-size=2 分布到两张卡
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /kaggle/working/mazes.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 4 \
  --max-new-tokens 64 \
  --tp-size 2 \
  --world-size 2 \
  --max-steps 100
```

---

## 注意事项

1. **Kaggle 会话限制**：T4×2 每次最多 12 小时，P100 也是 12 小时。
2. **模型缓存**：Qwen 模型通过 ModelScope 下载，首次运行需要 ~1-2GB 下载。
   - 设置 `--model-hub huggingface` 可改用 HuggingFace 下载。
3. **TPU 不支持**：AReno 依赖 CUDA，TPU v5e-8 无法运行。
4. **存储**：Kaggle 工作目录有 20GB 限制，生成的迷宫 JSONL 很小（~几 MB）。
5. **保存检查点**：训练输出默认在工作目录，Kaggle 退出后会被保留。