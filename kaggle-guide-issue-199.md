# Kaggle 运行指南 — AReno issue #199 PR #311

> 目标：在 Kaggle GPU 环境跑通 AReno 的 CPU 测试 + demo 脚本，验证 PR #311（configurable trainable-turn selection）

---

## 0. 前提

你需要：
- 一个 Kaggle 账号
- fork 了的 `sliverdancer/AReno` 仓库（GitHub 上已有）
- Kaggle Notebook 开启 GPU 加速器

## 1. 创建 Kaggle Notebook

1. 打开 https://www.kaggle.com/code 点 **New Notebook**
2. 右侧面板 **Settings** → **Accelerator** 选 **GPU T4 x2**（或 P100）
3. **Internet** 开关打开（安装依赖和 clone 仓库要用）
4. **Persistence** 选 **Files only**（保存输出文件）

## 2. 安装依赖

在第一个 cell 里跑：

```python
# 查看环境和 GPU
!nvidia-smi
!python --version
```

Kaggle 自带 Python 3.10+ 和 PyTorch，但版本可能不够。先确认：

```python
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
```

AReno 要求 `torch>=2.6`。如果 Kaggle 自带的 torch 低于 2.6，需要升级（跳过这步如果已经满足）：

```python
!pip install -q "torch>=2.6" --index-url https://download.pytorch.org/whl/cu121
```

## 3. Clone 仓库并切换到 PR 分支

```python
%cd /kaggle/working
!rm -rf AReno
!git clone https://github.com/sliverdancer/AReno.git
%cd AReno
!git checkout feat/configurable-trainable-turns
!git log --oneline -3
```

确认输出里能看到 `1cf414a feat(agentic): configurable trainable-turn selection and tool-call arg masking`。

## 4. 安装 AReno

```python
%cd /kaggle/working/AReno

# psutil 是 --no-build-isolation 的前置依赖
!pip install -q psutil

# flash-linear-attention
!pip install -q flash-linear-attention

# 跳过 flash-attn（Kaggle 环境编译可能受限，用 native attention 即可）
# 如果需要 flash-attn：
# !pip install -q flash-attn --no-build-isolation

# 安装 AReno 本体（跳过 CUDA 扩展编译，只做 metadata 安装）
# 这样 demo 脚本和测试能跑，不需要完整编译 fused kernel
!ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation
```

如果你的 GPU 架构支持（T4 = SM 7.5，P100 = SM 6.0），可以尝试完整编译：

```python
# 完整编译（耗时较长，5-15 分钟）
!TORCH_CUDA_ARCH_LIST="7.5" MAX_JOBS=4 pip install -e . --no-build-isolation
```

如果完整编译报错，退回 `ARENO_BUILD_EXT=0` 那条命令即可——CPU 测试和 demo 不需要 fused kernel。

## 5. 跑 CPU 测试

```python
!cd /kaggle/working/AReno && python -m pytest tests/test_agentic_cpu.py -q
```

预期输出：

```
................................................................          [100%]
64 passed in 2.80s
```

如果 pytest 没装：

```python
!pip install -q pytest
```

## 6. 跑新增功能验证

```python
!cd /kaggle/working/AReno && python -c "
from areno.api.agentic import LossMaskPolicy, ResponseSpan, LossSelectionMode
from areno.api.trainer_config import TrainerConfig

p = LossMaskPolicy()
print('1. 默认模式:', p.trainable_turns, '| arg屏蔽:', p.mask_tool_call_args, '| dead flag已移除:', not hasattr(p, 'final_assistant_text'))

try:
    TrainerConfig(algo='gspo', ckpt='x', dataset_path='y', trainable_turns='bogus')
except ValueError as e:
    print('2. 非法模式被拒:', e)

span = ResponseSpan(kind='assistant_text', length=3)
print('3. ResponseSpan:', span)
print('   LossSelectionMode:', LossSelectionMode.__args__)
print('全部验证通过')
"
```

## 7. 跑 demo 脚本

```python
!cd /kaggle/working/AReno && python examples/agentic/trainable_turns_demo.py
```

这个脚本用一个假 tokenizer 构造多轮 tool-call trajectory，打印三种模式下的逐 token loss_mask 和 trainable_tokens 数量，并演示非法输入被拒绝。

## 8. 跑一个最小的 agentic 训练

用小模型跑 2 步训练验证端到端。

几个注意点：
- gsm8k 原始行格式是 `question`/`answer`，AReno 要求 `prompt` 字段，必须加 `--dataset-loader-fn examples/math/dataset_loader.py`
- 单张 T4（15GB）跑 Qwen3-0.6B 的 GSPO 训练会 OOM（rollout KV cache + 反向传播 + 优化器同时驻留），需要双卡 + 省显存参数：`--tp-size 2 --world-size 2`、`--drop-rollout-state`、`--eager-decode`、缩短序列长度

```python
!cd /kaggle/working/AReno && PYTORCH_ALLOC_CONF=expandable_segments:True areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo \
  --tp-size 2 \
  --world-size 2 \
  --batch-size 2 \
  --n-samples 2 \
  --max-steps 2 \
  --max-prompt-tokens 384 \
  --max-new-tokens 128 \
  --mini-bs 1 \
  --score-micro-bs 1 \
  --gradient-accumulation-steps 4 \
  --activation-checkpointing \
  --drop-rollout-state \
  --eager-decode \
  --trainable-turns final_answer
```

跑 2 个训练步，用的是 `final_answer` 模式（只训最终答案 span）。日志里看 `trainable_tokens` 和 `masked_response_tokens` 的数值。

对比 `all_assistant` vs `final_answer`（把下面的 `--trainable-turns` 换成默认就是 `all_assistant`，省略即可）：

```python
# all_assistant（默认，所有 assistant span 入 loss）
!cd /kaggle/working/AReno && PYTORCH_ALLOC_CONF=expandable_segments:True areno train \
  --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo --tp-size 2 --world-size 2 \
  --batch-size 2 --n-samples 2 --max-steps 2 \
  --max-prompt-tokens 384 --max-new-tokens 128 \
  --mini-bs 1 --score-micro-bs 1 \
  --gradient-accumulation-steps 4 \
  --activation-checkpointing --drop-rollout-state --eager-decode

# final_answer
!cd /kaggle/working/AReno && PYTORCH_ALLOC_CONF=expandable_segments:True areno train \
  --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo --tp-size 2 --world-size 2 \
  --batch-size 2 --n-samples 2 --max-steps 2 \
  --max-prompt-tokens 384 --max-new-tokens 128 \
  --mini-bs 1 --score-micro-bs 1 \
  --gradient-accumulation-steps 4 \
  --activation-checkpointing --drop-rollout-state --eager-decode \
  --trainable-turns final_answer
```

对比两次日志的 `trainable_tokens`——`final_answer` 应该明显更少。

## 9. 保存结果

Kaggle Notebook 的 `/kaggle/working/` 目录内容会保存为 output。如果你想保存测试输出：

```python
!cd /kaggle/working/AReno && python -m pytest tests/test_agentic_cpu.py -q > /kaggle/working/test-results.txt 2>&1
!cd /kaggle/working/AReno && python examples/agentic/trainable_turns_demo.py > /kaggle/working/demo-output.txt 2>&1
```

提交 Notebook 后这些文件可以在 output 里下载。

## 10. 常见问题

**Q: `pip install -e .` 报 CUDA 编译错误**
A: 用 `ARENO_BUILD_EXT=0 pip install -e . --no-build-isolation` 跳过 CUDA 扩展。CPU 测试和 demo 不需要。

**Q: `git clone` 报网络错误**
A: 确认 Kaggle Notebook 的 Internet 开关打开了。有时需要等几秒重试。

**Q: `torch.cuda.is_available()` 返回 False**
A: 检查 Notebook Settings 里 Accelerator 是否选了 GPU。

**Q: 模型下载慢**
A: Kaggle 的网络在国内有时候到 HuggingFace 较慢。可以加 `--model-hub modelscope` 用 ModelScope 下载：
```bash
areno train --model-hub modelscope ...
```