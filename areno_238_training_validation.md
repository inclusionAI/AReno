# AReno #238 训练验证全过程

## 目标

实现 Issue #238：当 AReno 训练出现异常数值（NaN/Inf）时，生成可读报告，帮助用户快速定位问题根因。

## 实现过程

### 第一阶段：方案设计

- 分析 100 个 Issue，选定 #238（priority/important-soon）
- 设计两层检测策略：loss 快速检测（每步零开销）+ 深度检测（每 100 步或 loss 异常时遍历参数）
- 复用现有基础设施（`_param_grad` / `_grad_norm` 等），不另起炉灶

### 第二阶段：代码实现

| 文件 | 内容 |
|------|------|
| `non_finite.py` | 检测器核心模块（检测 + 报告格式化 + JSON 输出） |
| `training.py` | Actor 训练注入检测 |
| `roles.py` | Critic 训练注入检测 |
| `__init__.py` | 导出 `NonFiniteReport` |
| `test_non_finite_report.py` | 单元测试 |

### 第三阶段：环境准备

- Kaggle Notebook，Tesla T4 x2（TP=2）
- 模型：Qwen/Qwen3-0.6B（HuggingFace 缓存）
- 数据集：AI-MO/NuminaMath-CoT
- 算法：GSPO

## 遇到的问题及解决

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 1 | `--model-hub modelscope` 下载极慢（~700KB/s） | ModelScope 在 Kaggle 网络差 | 改用 `--model-hub hf` 利用 HuggingFace 缓存 |
| 2 | `--model-hub huggingface` 报错 | 合法值只有 `hf` 或 `modelscope` | 改为 `--model-hub hf` |
| 3 | Tesla T4 不支持 flash-attn | cc 7.5 不满足要求 | 自动回退 native attention |
| 4 | Tesla T4 不支持 BF16 | 硬件限制 | 自动回退 eager model execution |
| 5 | async SDK reward_fn 参数不匹配 | SDK 接口与文档不一致 | 改用 `areno train` CLI 命令 |
| 6 | `_param.data[0, 0] = NaN` 越界 | 1D 参数（如 bias）不能用二维索引 | 改为 `.flatten()[0]` |
| 7 | 两个 rank 都注入 NaN | TP=2 时 rank 0 和 rank 1 都执行注入 | 加 `if _ctx.rank == 0` 只在 rank 0 注入 |
| 8 | 优化器没有 `param_groups` | AReno 用自定义 `AdamWFP32Master` | 添加 `_safe_optimizer_state` 兼容函数 |
| 9 | 报告输出所有 28 层刷屏 | NaN 全量传播后事件 200+ 个 | 截断显示前 5 个 + 汇总统计 |
| 10 | `_merge_metrics` 崩溃 | `to_dict` 包含字符串 `"actor"` 无法转 float | `to_dict` 只输出纯数值字段 |
| 11 | `__init__.py` 缺少 `NonFiniteReport` 导入 | 漏加 | 补充导入 |
| 12 | `to_json_dict` 访问 `self.gpu_memory` | 字段实际叫 `gpu_memory_gb` | 修正属性名 |
| 13 | JSON 文件无法解析 | NaN 不是合法 JSON 值 | 添加 `_sanitize` 递归替换 NaN/Inf → null |
| 14 | `to_json_file` 中 `json` 未定义 | 缺 import | 在方法内加 `import json` |
| 15 | JSON 报告看不出是异常 | 缺少明确标识 | 添加 `alert`/`alert_type`/`severity` 字段 |

## 最终成功验证

**Step 0-4**：正常训练，reward_mean 0~0.25，grad_norm 0~2.8

**Step 5**（注入 NaN 后首次检测）：

终端报告：

```text
========================================================
 WARNING Non-Finite Value Training Report
========================================================

LOCATION
 Step: 5 | Phase: actor
 Last checkpoint: N/A

ANOMALIES DETECTED
 [GRAD] embed_tokens.weight.grad
 -> 475,136 NaN (0.61%)
 -> grad_norm = nan
 [PARAM] layers.0.input_layernorm.weight
 -> 1 NaN (0.10%)
 -> max=1.0469e+00 min=1.2158e-01
 [GRAD] layers.0.input_layernorm.weight.grad
 -> 1024 NaN (100.00%)
 -> grad_norm = nan
 [GRAD] layers.0.self_attn.qkv_proj.weight.grad
 -> 2097152 NaN (100.00%)
 -> grad_norm = nan
 [GRAD] layers.0.self_attn.o_proj.weight.grad
 -> 1048576 NaN (100.00%)
 -> grad_norm = nan

CONTEXT
 Loss: nan
 LR: 1.00e-06
 Global grad_norm: nan
 GPU memory: 4.07 GB
 ... and 223 more events (showing first 5)
 SUMMARY: 227 gradient + 1 parameter events
 Total NaN: 298,532,865  Total Inf: 0
 Affected layers: 31 (layers.21, layers.2, layers.27...)

LIKELY CAUSES
 1. [MID] Single-layer anomaly -> layers.0

SUGGESTED FIXES
 1. Check input data range/distribution for that layer
 2. Consider adding LayerNorm or reducing init variance

JSON REPORT: non_finite_reports/step_5_actor.json
========================================================
