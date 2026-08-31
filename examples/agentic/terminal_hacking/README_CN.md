# 复古终端入侵 Agentic RL Demo

这是一个原创的复古未来主义单词推理环境。它保留双列内存转储、同位字符匹配、有限尝试次数和括号奖励，但不包含原作 Logo、原文或贴图素材。

## 玩法

- 每题包含 16–20 个等长候选词、3 个括号探针和至少 18 行内存转储。
- 错误猜测消耗一次机会，并返回 `likeness`：与密码在相同位置上相同的字母数。
- RLVR 是单步 contextual bandit：输入一个 compact state，只输出一次 `submit_candidates` 候选集合，训练 rollout 内不执行游戏 action。
- WebUI 保留可玩 loop：候选筛选调用与 RLVR prompt 完全一致，从返回集合随机猜词，并在最后一次机会仍有歧义时使用免费 probe。
- 密码只存在于 `source_record`；prompt 仅包含可观察猜测和 likeness。
- reward 完全由算法计算：多报一个不一致 candidate 会被重罚，漏报一致 candidate 只按比例轻罚。

## 生成 8192 条不同数据

```bash
python examples/agentic/terminal_hacking/dataset_generator.py \
  --output /tmp/terminal_hacking.jsonl \
  --count 8192 \
  --workers 20 \
  --seed 2026
```

生成器使用唯一子种子，并对完整谜题指纹去重；发生重复时会重新生成，而不是只修改文件 ID。

## 训练

```bash
areno train \
  --ckpt inclusionai/ling-3.0-tiny \
  --dataset-path /tmp/terminal_hacking.jsonl \
  --dataset-loader-fn examples/agentic/terminal_hacking/dataset_loader.py \
  --reward-fn-path examples/agentic/terminal_hacking/reward.py \
  --agent-fn examples/agentic/terminal_hacking/run_agent.py \
  --algo gspo \
  --batch-size 4 \
  --n-samples 8 \
  --mini-bs 1 \
  --max-running-prompts 32 \
  --max-context-len 8192 \
  --max-prompt-tokens 4096 \
  --max-new-tokens 1024
```

RLVR 与 WebUI 的 candidate-select 调用复用 `game.py` 中完全相同的 candidate-filter system prompt、静态 memory-dump prompt、compact 当前 state 和唯一 `submit_candidates` schema。RLVR 每个样本只生成一次 completion，不运行 episode。WebUI 不展示候选集合，而是随机转成一个可见猜词；probe 仍由 WebUI 游戏 controller 处理。

## WebUI

```bash
python examples/agentic/terminal_hacking/web_ui.py --port 8771
```

连接 OpenAI 兼容推理服务：

```bash
python examples/agentic/terminal_hacking/web_ui.py \
  --port 8771 \
  --base-url http://127.0.0.1:8000/v1 https://api.example.com/v1 \
  --api-key EMPTY "$SECOND_API_KEY" \
  --model policy second-model
```

三个列表按位置配对；只传一个 API key 或 model 时会广播到全部 base URL。打开 `http://127.0.0.1:8771` 后可从模型选择器切换推理目标。页面还支持中英文说明、绿色/琥珀色荧光主题和无需推理服务的 `ALGO` 模式。
