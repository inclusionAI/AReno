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
# 安装缺失依赖（Kaggle 自带 PyTorch / Transformers）
!pip install -q psutil flash-linear-attention openai --no-deps
!pip install -q math-verify addict
```

在第二个 cell 中运行：

```python
# 克隆你的 fork 并切换到 maze 分支
!git clone -b feat/maze-agentic-rl https://github.com/sliverdancer/AReno.git /kaggle/working/AReno
%cd /kaggle/working/AReno

# 安装 AReno（跳过 CUDA 编译，Kaggle 自带 PyTorch）
!ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation
```

在第三个 cell 中运行：

```python
# 修复 PATH（Kaggle 的 pip install 可能不把 areno 放到默认 PATH）
import sysconfig, os
os.environ["PATH"] = sysconfig.get_path("scripts") + ":" + os.environ["PATH"]

# 验证安装
!python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"
!areno --version
```

> **如果 `areno --version` 仍然报 `command not found`**，所有后续 `areno` 命令
> 都改用 `python -m areno.cli.main` 替代。例如：
> `!python -m areno.cli.main --version`

### 3. 生成迷宫数据集

```python
# 生成 2048 个迷宫（7×7 为主，自动混入少量 9×9 增加多样性），vision_radius=1 (3×3 视野)
!python examples/agentic/maze/dataset_generator.py \
  --output /kaggle/working/mazes.jsonl \
  --count 2048 \
  --seed 2026 \
  --width 7 \
  --height 7 \
  --vision-radius 1
```

### 4. 运行 CPU 测试（验证代码完整性）

```python
!pip install -q pytest
!python -m pytest tests/test_agentic_maze_example_cpu.py -v
```

### 5. 运行训练

```python
!areno train \
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

> 如果 `areno` 不在 PATH，改用 `!python -m areno.cli.main train ...`

### 6. 通过 ngrok 暴露 Dashboard

Kaggle Notebook 没有公网端口，使用 ngrok 隧道在外部浏览器访问 AReno Dashboard。

前置条件：在 Kaggle Notebook 的 **Add-ons → Secrets** 中添加一个名为 `ngrok_key` 的 secret，值为你的 ngrok authtoken（从 https://dashboard.ngrok.com/get-started/your-authtoken 获取）。

```python
# Step 1: 后台启动 Dashboard（必须先启动，再连 ngrok）
!nohup areno dashboard --start --host 0.0.0.0 --port 8000 > /tmp/dashboard.log 2>&1 &
# 如果 areno 不在 PATH，改用：
# !nohup python -m areno.cli.main dashboard --start --host 0.0.0.0 --port 8000 > /tmp/dashboard.log 2>&1 &
```

```python
# Step 2: 等待 Dashboard 启动
import time
time.sleep(3)

# Step 3: 安装 pyngrok 并建立隧道
!pip install -q pyngrok

from kaggle_secrets import UserSecretsClient
from pyngrok import ngrok

user_secrets = UserSecretsClient()
ngrok_key = user_secrets.get_secret("ngrok_key")
ngrok.set_auth_token(ngrok_key)
public_url = ngrok.connect(8000)
print("Dashboard URL:", public_url)
```

打开输出的 `https://xxxx.ngrok-free.app` 地址即可在外部浏览器查看训练指标、奖励曲线和轨迹详情。

---

## 参数调优建议

### T4 (16GB) 配置

```python
!areno train \
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

```python
# 仅在 T4×2 上尝试，使用 tp-size=2 分布到两张卡
!areno train \
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
6. **更新代码**：如果已经克隆过仓库，重新拉取最新代码：
   ```python
   %cd /kaggle/working/AReno
   !git pull https://github.com/sliverdancer/AReno.git feat/maze-agentic-rl
   ```