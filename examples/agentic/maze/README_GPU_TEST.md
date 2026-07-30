# 阿里云 A10 GPU 测试与对比实验完整指南

> 基于实际踩坑修正，包含环境搭建、训练、中断保存、恢复、对比实验全流程。
> 适用于 2×A10 (24GB×2) 阿里云 GPU 机器，Docker 环境运行。

---

## Phase 1: 环境搭建（~30 分钟）

### 1.1 SSH 登录

```bash
ssh root@<IP地址>
# 输入密码
```

### 1.2 拉取镜像并启动容器

```bash
docker pull crpi-3kcpcptydwdbnr4i.cn-hangzhou.personal.cr.aliyuncs.com/areno/areno:v0.0.6

docker run -d --name areno --gpus all --network host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  crpi-3kcpcptydwdbnr4i.cn-hangzhou.personal.cr.aliyuncs.com/areno/areno:v0.0.6 \
  sleep infinity

docker exec -it areno bash
```

### 1.3 安装 PyTorch（容器内执行）

> **坑1**：Docker 镜像不自带 PyTorch，必须手动安装。
> **坑2**：默认 `pip install torch` 装的是 CUDA 13.0 版本，和系统 CUDA 12.8 冲突。
> **坑3**：需要 `--break-system-packages`（Ubuntu 限制）。
> **坑4**：用 `python3` 不是 `python`。

```bash
# 装 PyTorch 2.6.0（CUDA 12.4，兼容系统 CUDA 12.8）
pip3 install --break-system-packages "torch==2.6.0" -i https://mirrors.cloud.aliyuncs.com/pypi/simple/

# 验证 CUDA 版本匹配
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
# 期望输出：torch 2.6.0 cuda 12.4 True
```

### 1.4 安装 AReno 及全部依赖

> **坑5**：`pip install -e .` 时 `rich` 包和 Debian 冲突，导致依赖没装全。
> **坑6**：`flash-linear-attention` 用 `--no-deps` 装会缺模块。
> **坑7**：`datasets`、`modelscope`、`transformers` 等需要单独装。

```bash
# clone 代码
cd /root
git clone -b feat/maze-agentic-rl https://github.com/sliverdancer/AReno.git
cd AReno

# 装全部依赖（一条命令，用 --ignore-installed 绕过 Debian rich 冲突）
pip3 install --break-system-packages --ignore-installed \
  transformers safetensors huggingface-hub modelscope datasets \
  numpy pydantic click rich tqdm tensorboard fastapi uvicorn \
  prompt-toolkit psutil addict math-verify openai httpx anyio \
  jiter sniffio pytest \
  -i https://mirrors.cloud.aliyuncs.com/pypi/simple/

# 装 flash-linear-attention（不要用 --no-deps）
pip3 install --break-system-packages flash-linear-attention \
  -i https://mirrors.cloud.aliyuncs.com/pypi/simple/

# 编译安装 AReno（A10 = Ampere = compute capability 8.0）
TORCH_CUDA_ARCH_LIST="8.0" pip3 install --break-system-packages -e . --no-build-isolation
```

### 1.5 验证环境

> **坑8**：Triton 3.2.0 和 FLA 不兼容，但不能升级（PyTorch 2.6.0 锁定 triton==3.2.0）。
> **解决**：用 `--attn-backend native` 绕过 FLA/Triton。

```bash
# 验证 torch
python3 -c "import torch; print('GPU:', torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"
# 期望：GPU: 2 x NVIDIA A10

# 验证 AReno CUDA 扩展
python3 -c "from areno.accel._extension import extension; print('CUDA extension OK')"

# 验证 areno CLI
python3 -m areno.cli.main --version
```

### 1.6 生成数据 + CPU 测试

```bash
# 5×5 迷宫
python3 examples/agentic/maze/dataset_generator.py \
  --output /root/mazes_5x5.jsonl \
  --count 512 --seed 2026 --width 5 --height 5 --vision-radius 1

# CPU 测试（期望 12 passed）
python3 -m pytest tests/test_agentic_maze_example_cpu.py -v
```

---

## Phase 2: 启动 Dashboard（后台运行）

> **坑9**：不用单独开终端2，用 `nohup &` 在同一终端后台启动即可。

```bash
nohup python3 -m areno.cli.main dashboard --start --host 0.0.0.0 --port 8000 \
  > /tmp/dashboard.log 2>&1 &
```

浏览器打开：**http://<机器公网IP>:8000**

---

## Phase 3: 实验A — 禁思考 + BFS shaping

> **坑10**：`--batch-size 2 --n-samples 8`（16个轨迹）会 OOM。
> **坑11**：不加 `--attn-backend native` 时 FLA/Triton 崩溃。
> **解决**：降 batch、加省显存参数、用 native attention。

```bash
cd /root/AReno

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m areno.cli.main train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /root/mazes_5x5.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 4 \
  --max-new-tokens 128 \
  --disable-thinking \
  --attn-backend native \
  --tp-size 2 \
  --world-size 2 \
  --max-steps 500 \
  --save-interval 100 \
  --mini-bs 1 \
  --adam-8bit \
  --activation-checkpointing \
  --save-path /root/ckpt_expA
```

**Dashboard 观察重点**：
- `rollout/rewards_mean`：从 -0.5 开始，有 BFS shaping 应出现波动上升
- `rollout/rewards_max`：出现接近 0 或正值 = 有 sample 到达终点
- smooth 调到 0.8 看趋势

---

## Phase 4: 中断时保存 checkpoint

时间不够时 `Ctrl+C` 停训练：

```bash
# 确认 checkpoint 已保存
ls -lh /root/ckpt_expA/

# 退出容器
exit

# 宿主机拷出
docker cp areno:/root/ckpt_expA /root/ckpt_expA
```

本地 Mac 终端下载：

```bash
scp -r root@<IP地址>:/root/ckpt_expA ~/Desktop/ckpt_expA
```

---

## Phase 5: 明天恢复训练

### 5.1 重建环境（同 Phase 1）

```bash
ssh root@<新IP地址>
# docker run + docker exec（同 Phase 1.2）
# pip3 install torch==2.6.0（同 Phase 1.3）
# pip3 install 全部依赖（同 Phase 1.4）
# git clone + pip install -e .（同 Phase 1.4）
```

### 5.2 上传 checkpoint

本地 Mac 终端：

```bash
scp -r ~/Desktop/ckpt_expA root@<新IP地址>:/root/ckpt_expA
```

服务器上拷入容器：

```bash
docker cp /root/ckpt_expA areno:/root/ckpt_expA
docker exec -it areno bash
ls -lh /root/ckpt_expA/
```

### 5.3 重新生成数据

```bash
cd /root/AReno
python3 examples/agentic/maze/dataset_generator.py \
  --output /root/mazes_5x5.jsonl \
  --count 512 --seed 2026 --width 5 --height 5 --vision-radius 1
```

### 5.4 从 checkpoint 继续训练

```bash
# 启动 Dashboard
nohup python3 -m areno.cli.main dashboard --start --host 0.0.0.0 --port 8000 \
  > /tmp/dashboard.log 2>&1 &

# 从 checkpoint 继续（--ckpt 指向保存的目录）
cd /root/AReno

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m areno.cli.main train \
  --ckpt /root/ckpt_expA \
  --dataset-path /root/mazes_5x5.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 4 \
  --max-new-tokens 128 \
  --disable-thinking \
  --attn-backend native \
  --tp-size 2 \
  --world-size 2 \
  --max-steps 500 \
  --save-interval 100 \
  --mini-bs 1 \
  --adam-8bit \
  --activation-checkpointing \
  --save-path /root/ckpt_expA_resumed
```

> AReno 无显式 resume，step counter 从 0 重置，但模型权重保留之前的训练成果。

---

## Phase 6: 实验B — 思考模式对比

实验A（今天或明天）跑完后，从原始模型跑对比：

```bash
cd /root/AReno

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m areno.cli.main train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /root/mazes_5x5.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 4 \
  --max-new-tokens 256 \
  --attn-backend native \
  --tp-size 2 \
  --world-size 2 \
  --max-steps 500 \
  --save-interval 100 \
  --mini-bs 1 \
  --adam-8bit \
  --activation-checkpointing \
  --save-path /root/ckpt_expB
```

区别：不禁思考、`--max-new-tokens 256`，从原始 Qwen3-0.6B 开始。

---

## Phase 7: 对比分析

在 Dashboard 对比两轮训练：

| 指标 | 实验A（禁思考 128） | 实验B（思考 256） |
|------|-----------|-----------|
| `rewards_mean` 起始 | ~-0.5 | ~-0.5 |
| `rewards_mean` 结束 | 期望>-0.3 | 期望>-0.3 |
| `rewards_max` | 是否>0 | 是否>0 |
| `response_len_mean` | 是否下降 | 是否下降 |
| 每步耗时 | 较快 | 较慢 |

**判断标准**：
- rewards_mean 上升到 -0.3 以上 → BFS reward shaping 生效
- rewards_max 出现正值 → agent 到达过终点
- 实验A vs B 哪个先上升 → 哪个更有效

---

## 踩坑速查表

| 现象 | 原因 | 解决 |
|------|------|------|
| `externally-managed-environment` | Ubuntu PEP 668 | 加 `--break-system-packages` |
| `python: command not found` | 无 python 软链接 | 用 `python3` |
| `areno: command not found` | pip scripts 不在 PATH | 用 `python3 -m areno.cli.main` |
| `No module named 'torch'` | Docker 镜像无 PyTorch | `pip3 install torch==2.6.0` |
| CUDA version mismatch (12.8 vs 13.0) | 装了新 torch | 指定 `torch==2.6.0`（CUDA 12.4） |
| `No module named 'fla.ops'` | flash-linear-attention 用了 --no-deps | 去掉 `--no-deps` 重装 |
| `No module named 'datasets'` | rich 冲突导致依赖没装全 | 用 `--ignore-installed` 装全部依赖 |
| `No module named 'modelscope'` | 同上 | 同上 |
| `worker exited during ROLLOUT_SESSION_END` | Triton 3.2 和 FLA 不兼容 | 加 `--attn-backend native` |
| Triton 版本冲突 | PyTorch 2.6.0 锁定 triton==3.2.0 | 不要升级 triton，用 native attention |
| CUDA OOM (训练阶段) | batch×samples 太大 | `--batch-size 1 --n-samples 4 --mini-bs 1 --adam-8bit --activation-checkpointing` |
| 显存碎片 | PyTorch 内存分配 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Docker 容器不存在 | 新 session 或容器被删 | 重新 `docker run` |
| `rich` 包冲突 | Debian 管理的包 | `pip3 install --ignore-installed rich` |
| `/workspace: No such file` | Docker 镜像无此目录 | 用 `/root` 替代所有路径 |