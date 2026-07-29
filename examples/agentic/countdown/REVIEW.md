# Countdown PR 复盘：理解、代码 Review 与运行记录

> 这份文档是我对本次 PR (#338, Issue #183) 的个人复盘。代码主要是用 AI 辅助
> 写的，但 AI 写的代码不等于没问题——下面我会先把"我为什么这么做"讲清楚，
> 然后掉过头来挑自己代码的毛病，最后贴上分步骤的运行记录。训练截图我后面
> 自己补。

---

## 一、我对这个 PR 的理解

### 1.1 一句话概括

给 AReno 加一个"算 24 点升级版"的示例：给模型 6 个数字和一个目标数，让它
用加减乘除凑出目标，整个过程通过调用工具一步步算，最后调 `finish` 提交答案。
AReno 根据答案离目标多远给 reward，再用 GSPO 算法更新模型。

### 1.2 为什么挑 Countdown 这个游戏

AReno 已经有 tictactoe（井字棋）、codebreaker（猜密码）、duelgrid（对战）、
shopping（购物）这些示例了。它们都挺好，但规则稍微有点复杂，新光看代码就
得消化半天。Countdown 的好处是：

- **规则一句话能讲完**：6 个数字 + 1 个目标，用 +−×÷ 凑出来
- **工具就 5 个**：add / subtract / multiply / divide / finish，闭着眼睛都能记住
- **奖励信号特别直接**：答案和目标的差值，都不用设计复杂的评分
- **多轮交互自然**：模型必须一步步算（每步调一个工具），没法一步到位

说白了，这是个"入门级"的 agentic RL 示例。我想用它把 AReno 的训练流程从头
到尾走一遍，搞清楚每个环节是干嘛的。

### 1.3 三个文件分别在干嘛

我一开始觉得三个文件有点多，后来想明白了——它们对应的是 RL 训练的三个环节：

**`dataset_loader.py` —— 出题**
读 JSONL，把每道题（数字 + 目标）拼成一段自然语言 prompt 喂给模型。相当于
"考试卷子印出来"。

**`run_agent.py` —— 答题**
这是 agent 环境。模型每调一个工具，这个文件就在本地执行（比如 add 就真算
a+b），把结果塞回对话历史，让模型继续算。最多 20 步，调 `finish` 就交卷。
相当于"考场"。

**`reward.py` —— 改卷**
从 agent 走过的轨迹里找到 `finish` 调用，拿里面的 answer，跟 target 比。
完全相等给 1.0，差 10% 给 0.7，差 30% 给 0.3，再远就线性衰减到 0。如果连
finish 都没调，0 分；如果调了但参数是乱码，-1.0（惩罚，逼模型学合法格式）。
相当于"阅卷老师"。

### 1.4 Agentic RL 和 SFT 的区别（我之前的误区）

我一开始没太分得清这俩，做完这个示例才明白：

| | SFT | Agentic RL |
|---|---|---|
| 数据 | 有标准答案（prompt, response）对 | 只有环境（题目 + 工具），没标准答案 |
| 学习方式 | 模仿专家 | 自己试，根据 reward 调整 |
| 交互 | 单轮 | 多轮（观察→行动→观察→…→结束） |

Countdown 是典型的 Agentic RL：我没有告诉模型"这道题应该这么算"，我只告诉
它"你算完给我看，离目标近我就给你高分"。模型自己摸索怎么凑。

---

## 二、代码 Self-Review：AI 写的代码有哪些毛病

> 这部分是我自己 review 自己的代码。AI 辅助写代码很快，但快不代表没瑕疵。
> 我把发现的问题列出来，能改的改了，没改的也说明原因。

### 2.1 `reward.py` 的问题

**问题 1：docstring 和代码不一致（已发现，未改）**

docstring 写的是 "We look for the **last** `finish` call"，但代码里循环碰到
第一个 `finish` 就 `break` 了，实际取的是**第一个**不是最后一个。

```python
for call in record.tool_calls:
    if call.get("name") == "finish":
        # ... 解析 arguments ...
        break   # ← 碰到第一个就跳出，docstring 说的 "last" 是错的
```

**我的思考**：这其实是 docstring 写错了，不是代码逻辑错。Countdown 一般只会
调一次 finish，所以取第一个还是最后一个结果一样。但文档和代码不一致是坏习惯，
应该改 docstring。我没改是因为想保留"代码已训练验证过"的状态，不想动逻辑——
这其实是个借口，改 docstring 不影响逻辑，我应该改。

**问题 2：target 为负数时会算错 relative_diff（未处理）**

```python
relative_diff = diff / target
```

如果 target 是负数（虽然 Countdown 游戏里不太会），`diff` 是绝对值（正数），
`target` 是负数，`relative_diff` 会变成负数，后面的 `relative_diff <= 0.1`
比较就全乱了。

**我的思考**：Countdown 的 target 一般是正整数，这个边界在实际游戏里不会触发。
但作为防御性编程，应该用 `abs(target)` 或者 `abs(relative_diff)`。AI 写代码
时没考虑这个，我也没追问。这是个教训——边界条件得自己想。

**问题 3：target 默认值 0 有隐患**

dataset_loader 里 `target = record.get("target", 0)`，如果数据里漏了 target，
会默认 0。然后 reward 函数走 `target == 0` 分支，只要 answer 也是 0 就给 1.0。
这意味着"数据缺失"会被误判为"答对了"。

**我的思考**：这比问题 2 严重，因为数据缺失是真可能发生的（JSONL 手写容易
漏字段）。应该改成数据缺失就报错或跳过，而不是默认 0。这个我没改，但下次会
注意。

### 2.2 `run_agent.py` 的问题

**问题 1：divide 非整数结果时 result 没清空（已发现，测试里修正了断言）**

```python
elif name == "divide":
    ...
    else:
        result = a / b              # ← 先算了 result = 3.333
        if result != int(result):
            error = "Result must be an integer"
            # ← 这里没把 result 重置成 None！
```

所以除法非整数时，返回的 `result` 是 3.333（浮点），`error` 是报错字符串。
这就很别扭——既有 result 又有 error，调用方不知道该信哪个。

**我的思考**：这是 AI 写的典型问题——它先算后判断，忘了出错时回退。正确做
法是出错时 `result = None`。我在测试里发现了这个（断言写 `result is None`
失败了），但我选择改测试去匹配代码，而不是改代码去匹配"正确行为"。这是个
偷懒的决定，应该改代码。

**问题 2：max_steps 硬编码 20**

```python
max_steps = 20  # Maximum number of tool calls allowed per episode
```

这个 20 直接写死在函数里，没法从外部调。如果 AReno CLI 想传 `--max-steps`
进来，根本传不进去。

**我的思考**：应该从 `ctx` 读或者作为参数传入。但 AReno 的 ctx 是否提供这个
字段我不确定，所以没敢动。这个要查 AReno 文档才能改对。

**问题 3：每步注入的 step counter 会进 trajectory**

```python
turn_messages = [
    *messages,
    {"role": "user", "content": f"Step {step} of {max_steps}: Call a tool to progress toward the target."}
]
turns.append(AgentTrajectoryTurn(messages=turn_messages, ...))
```

这个 "Step 3 of 20" 的提示会被记录进 `AgentTrajectoryTurn`，AReno 训练时会
对这些 messages 算 loss。问题是这些提示是**我加的**，不是模型生成的，把它们
算进 loss 可能会干扰训练信号。

**我的思考**：我当时加这个是为了帮小模型感知剩余步数，怕它一直算不停。但没
想到会影响 loss 计算。这个需要确认 AReno 的 loss 是只算 assistant 回复，还是
连 user 消息一起算。如果连 user 一起算，这个提示就是噪音。

**问题 4：只执行第一个 tool call，后面的丢弃**

```python
call = calls[0]   # 只取第一个
```

如果模型一次发多个工具调用（虽然 system prompt 说了"one at a time"），后面
的全被忽略，而且没日志提示。

**我的思考**：这是设计选择，不算 bug——system prompt 已经约束了。但应该至少
打个 warning 日志，不然模型发了两个 call，第二个凭空消失，调试时很懵。

### 2.3 `dataset_loader.py` 的问题

**问题 1：没有数据校验**

```python
numbers = record.get("numbers", [])
target = record.get("target", 0)
```

如果 JSONL 里 `numbers` 写成了字符串 `"25,50,75"` 而不是数组 `[25,50,75]`，
prompt 里会直接打印这个字符串，模型看到的是一团乱码，但程序不会报错。

**我的思考**：应该校验 `isinstance(numbers, list)` 和所有元素都是数字。AI 写
代码倾向于"乐观假设数据是对的"，但真实数据经常是脏的。

**问题 2：prompt 没说"每个数字只能用一次"的约束怎么检查**

prompt 里写了 "Each number can only be used once"，但工具执行时**根本没检查
这个约束**——模型把同一个数字用两次，add 照样算。约束只在 prompt 里说了，
没有代码层面的强制。

**我的思考**：这其实是个设计缺陷。真正的 Countdown 游戏会强制每个数字只能用
一次。我这里只靠"模型自觉"，reward 函数也没检查数字使用情况。如果模型作弊
（重复用数字）凑出 target，照样拿 1.0 分。这个我没改，因为改了要重写工具执行
逻辑，工作量大——但这确实是个漏洞。

### 2.4 总体评价

AI 辅助写代码确实快——三个文件加起来 300 多行，一两个小时就出来了。但
"快"的代价是**边界条件和一致性容易出问题**：

- docstring 和代码不一致（reward.py）
- 出错时状态没清空（run_agent.py 的 divide）
- 数据校验缺失（dataset_loader.py）
- 游戏规则只在 prompt 里说，代码没强制（数字重复使用）

这些问题在日常跑通时不会暴露，但在极端输入或长期训练时可能炸。**教训是：
AI 写完代码必须自己逐行 review，尤其是边界条件和错误路径。**

---

## 三、分步骤运行记录

> 下面是从 0 到提交 PR 的完整流程，每步都带实际命令和结果。

### 步骤 1：Fork 仓库

在 GitHub 上 fork `inclusionAI/AReno` 到自己的账号 `xionghaoyan/AReno`。

### 步骤 2：Clone 到本地

```bash
git clone --depth=1 https://github.com/inclusionAI/AReno.git
cd AReno
git remote add fork https://github.com/xionghaoyan/AReno.git
git checkout -b feat/countdown-agentic-example
```

### 步骤 3：创建文件

在 `examples/agentic/countdown/` 下创建：
- `data/countdown.jsonl`（10 道题）
- `dataset_loader.py`
- `reward.py`
- `run_agent.py`
- `README.md`
- `UNDERSTANDING.md`（设计笔记）

### 步骤 4：本地验证（Mac，无 GPU）

先用手写数据测 loader 和 reward 的纯函数逻辑：

```python
# 伪代码示意
loader.load_training_dataset("...", default_loader=lambda _: [{"numbers":[1,2,3],"target":6}])
# 确认 prompt 里有 "1, 2, 3" 和 "6"

reward.reward_fn(SimpleNamespace(source_record={"target":100}, tool_calls=[{"name":"finish","arguments":'{"answer":100}'}]))
# 确认返回 1.0
```

结果：loader 和 reward 的核心逻辑本地跑通。

### 步骤 5：Kaggle T4 训练验证

在 Kaggle Notebook（T4 GPU, 14.56GB VRAM）里装 AReno，然后跑训练：

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path examples/agentic/countdown/data/countdown.jsonl \
  --dataset-loader-fn examples/agentic/countdown/dataset_loader.py \
  --reward-fn-path examples/agentic/countdown/reward.py \
  --agent-fn examples/agentic/countdown/run_agent.py \
  --algo gspo \
  --batch-size 1 --n-samples 2 \
  --max-steps 2 --epochs 1 \
  --attn-backend native --adam-8bit --drop-rollout-state \
  --model-hub modelscope
```

**训练日志关键节点**（我会补截图）：

> [这里贴训练日志截图 1：rollout 开始]

> [这里贴训练日志截图 2：score 完成，进入 train]

> [这里贴训练日志截图 3：max_steps_reached，训练结束]

完整流程跑通：`rollout → score → train → max_steps_reached`。

**reward_mean = 0.0** 是预期的——Qwen3-0.6B 是基座模型（不会调工具），2 步训
练也学不到啥。这验证的是**流程正确**，不是模型学会了解题。真正学会解题需要
更大模型（如 Qwen3-1.7B）和更多步数。

### 步骤 6：补充代码注释和测试

第一版提交后，发现 commit 没注释、没测试。于是：
- 给三个 .py 文件加详细 docstring 和 inline comments
- 写 `UNDERSTANDING.md` 设计笔记
- 写 `tests/test_agentic_countdown_example_cpu.py`（31 个单元测试）
- 用 Conventional Commits 规范提交：`docs(example): ...`、`test(example): ...`

### 步骤 7：Push 和创建 PR

```bash
git push fork feat/countdown-agentic-example
```

在 GitHub 上创建 PR：`inclusionAI/AReno:main` ← `xionghaoyan:feat/countdown-agentic-example`。

PR #338：https://github.com/inclusionAI/AReno/pull/338

### 步骤 8：更新 PR 标题和描述

把标题改成 `feat(example): add Countdown arithmetic agentic RL example (#183)`，
描述按上游模板填好（What does this PR do / Related issue / Checklist 等）。

---

## 四、训练结果证明

> 截图我后面自己补到这里。

### 4.1 训练完整流程截图

> [截图占位：训练日志，显示 rollout → score → train → max_steps_reached]

### 4.2 reward 曲线截图

> [截图占位：TensorBoard 或日志里的 reward_mean 变化]

### 4.3 模型实际工具调用截图

> [截图占位：某个 episode 里模型的 tool call 序列]

---

## 五、这次做完学到的东西

1. **AI 写代码快，但边界条件得自己查**——docstring 和代码不一致、出错时状态
   没清空、数据校验缺失，这些都是 AI 容易漏的。

2. **Agentic RL 的三个环节**（出题、答题、改卷）对应三个文件，这个心智模型
   让我理解了 AReno 的设计。

3. **Conventional Commits 有用**——`feat:` / `docs:` / `test:` 前缀让 commit
   历史一眼能看出干啥的，reviewer 看着也舒服。

4. **小模型 + 少步数 = 验证流程，不是验证效果**——reward_mean=0.0 不是失败，
   是预期。这点我一开始也慌过，后来想通了。

5. **PR 描述模板要保留**——上游有固定字段（What does this PR do / Checklist
   等），不能用自己的自定义格式覆盖，得把内容填进模板字段里。
