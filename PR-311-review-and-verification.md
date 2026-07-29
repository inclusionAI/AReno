# PR #311 自理解、Code Review 与运行验证

> PR: https://github.com/inclusionAI/AReno/pull/311
> Issue: https://github.com/inclusionAI/AReno/issues/199
> 日期: 2026-07-28 ~ 29

---

## 一、这个 PR 在做什么

### 解决的问题

AReno 做 agentic RL 训练时，把一条多轮 trajectory 里每轮 assistant 的输出都拿来算 loss。但对很多任务来说，中间那些"我想想""让我调用一个工具"的 token 没什么训练价值，真正该学的是最后的那个答案。之前没有开关来控制"训哪些 turn"。

代码里有个 `final_assistant_text` 字段，看着像是为此设计的，但全仓 grep 下来没人读它——是个 dead flag。

### PR 加了什么

三个模式（通过 `--trainable-turns` 控制）：
- `all_assistant`：全部训（默认，跟改之前一样）
- `last_assistant`：只训最后一轮
- `final_answer`：只训最后一个工具调用之后的文本回答

还有一个 `--mask-tool-call-args` 开关：屏蔽工具调用里的 JSON 参数 token，只保留工具名可训。这个定位成研究 ablation——业界做工具使用的训练（ToolFormer、Gorilla 等）都是整段训的，没人单独屏蔽参数部分。

### 我觉得有意思的地方

这个需求看似简单（加个开关选训哪些 token），但真到代码里就碰到一个核心矛盾：判断"这是不是最后一个 span"需要看到整条 trajectory，而现有的 mask 生成函数是逐 span 调用的，每次只看一个 span。你现在不知道后面还有没有别的 span。

解决方式是在 trajectory 组装期先把每轮的 kind 和长度存下来（`response_spans` 清单），等整条 trajectory 拼完了，在一个统一入口（`_train_rows_from_samples`）里做一次后处理重写。这样不管数据从 HTTP proxy 进来还是从 trajectory 对象进来，都走同一个入口，逻辑只写一遍。

另一个设计取舍是 mask 的组合顺序。代码里已经有一套 mask 逻辑在屏蔽工具返回的 result 区，如果新功能从零重建 mask，就会把已有屏蔽覆盖掉。所以必须是"在已有 mask 上叠加修改"，不能重建。这个顺序如果搞反了，测试不一定能暴露——因为默认模式下两者结果一样，只有同时开了 arg 屏蔽和 result 屏蔽才能看出差别。

### B 档接口预留

PR 只做静态规则（A 档），不做基于 reward 的动态判断（B 档）。但 `response_spans` 这个清单本身就是 B 档的基础——未来如果想根据每条 trajectory 的 reward 分数决定训不训某个 span，有了 span 清单就能定位，不用重新读代码追数据流。PR 没实现 B 档逻辑，但保证字段形状不排斥将来接入。

---

## 二、Code Review 要点

代码主要改在 `agentic.py`（核心逻辑）、`trainer_config.py`（配置）、`policy_only.py`（映射）、`cli/train.py`（CLI 选项），加上测试和文档，一共 10 个文件 +781 行。

### 做得好的地方

**改动范围克制。** 没有动 trainer 或 rollout 引擎，核心逻辑全加在 `agentic.py` 的数据路径里。chokepoint 选在 `_train_rows_from_samples` 的循环顶部，两条数据路径都经过这里，不用改两处。

**向后兼容。** 默认 `all_assistant` + 不屏蔽参数，跟改之前的行为完全一致。测试里有专门的 parity case 钉这个。

**防御性检查。** `_apply_trainable_turn_mode` 开头检查 span 长度之和是否等于 response token 数，不等就跳过不改——避免数据异常时错位。

### 我觉得可以改进的地方

**arg 屏蔽的 brace-matching 不处理字符串内花括号。** 如果工具调用的 JSON 参数里有字符串值包含 `{` 或 `}`，depth 计数会错位。不过实际工具调用的参数一般是简单 JSON，而且代码里明确标了"approximate"，算是个已知的局限。

**校验只查"有调用没结果"，不查"有结果没调用"。** orphan tool result 被容忍了。这是因为现有测试 fixture 就有这种形态（tool 消息不带结构化 tool_calls 字段），如果严格检查会破坏现有测试。是个有意的取舍，但文档里应该写明白。

**回写 `loss_mask_row` 的 O(n) 遍历。** 注释里提到过，每 sample 一次，实际可接受，但如果以后 trajectory 变得很长可以注意。

---

## 三、遇到的问题与解决

### 1. 计划文档把一个非 bug 当成了 bug

原 issue 计划写 `_tool_call_loss_mask` 的两个 marker "相同，疑似 bug"。我实际读代码发现是 `("<|tool_response>", "<|im_end>")` 两个不同的 sentinel——一个是工具返回区的起始标记，一个是消息结束标记。取最早出现的那个来屏蔽其后的内容，是有意的启发式。

如果照着计划去"修"这个"bug"，会引入错误改动。这件事让我意识到：**不能盲信设计文档，得到代码里验证才能下结论**。

### 2. 逻辑落点跟计划写的不一样

计划说三种模式都落在 `_response_loss_mask_for_span`。追了数据流才发现这个函数只看单个 span，判不了"是不是最后一个"。而且 `_append_sample_response` 把 `response_kind` 折叠成只剩末轮的 kind，前面几轮的信息组装后就丢了。

最后改成在组装期捕获 `response_spans` 清单，在 chokepoint 做后处理。这个改动让计划的落点描述失效了，但实际更合理——计划写的时候可能没追到 `response_kind` 被折叠这个细节。

### 3. 校验逻辑第一版写错了

第一版遍历 `sample.messages` 检查 `tool_calls` 字段。跑测试发现有 fixture 传了 tool 消息但不含结构化 `tool_calls`，被误判成非法。改成基于 trace 事件配对，并容忍 orphan tool result。

**教训是写校验前先搞清楚现有数据到底长什么样**，不然很容易把合法数据判成非法。

### 4. 本机环境跑不了测试

开发机是 macOS，只有 Python 3.9（项目要 3.10+），没装 torch 和 pytest。试了几次 `pip install torch` 都卡在版本不兼容。后来用 `uv` 拉了个 Python 3.12 venv，配合清华镜像源装 torch + pytest 跑通了。折腾环境花了不短的时间。

**下次做这类任务前应该先确认运行环境**，而不是代码写完了才发现跑不了。

### 5. PR 描述超长被拒

GitHub 报 "Body is too long (maximum is 65536 characters)"。排查发现复制时连带了 IDE 注入的隐藏标签，实际正文才两千多字符。清理后就好了。

---

## 四、运行验证

### 环境

在 Kaggle GPU T4 x2 上跑的。用的是 `uv` 创建的 Python 3.12 venv，通过 `--dataset-loader-fn examples/math/dataset_loader.py` 处理 gsm8k 的数据格式转换。

### 测试结果

```
pytest tests/test_agentic_cpu.py -q
→ 64 passed in 7.58s
```

21 个新增测试覆盖：三种模式的逐 token mask、arg 屏蔽开/关、非法输入拒绝、边界情况（空响应/bare trailing tool call 零信号/无 tool result 退化）、metrics 数值、两条代码路径（proxy + trajectory）。

### demo 脚本输出

```
Trajectory: assistant_text(2) | tool_call(2) | assistant_text(2)
all_assistant    loss_mask=[True, True, True, True, True, True]     trainable_tokens=6
last_assistant   loss_mask=[False, False, False, False, True, True] trainable_tokens=2
final_answer     loss_mask=[False, False, False, False, True, True] trainable_tokens=2
validation rejected: agentic trajectory has a tool call without a matching tool result
```

三种模式在同一条 trajectory 上的效果一目了然：`all_assistant` 全训 6 个 token，`last_assistant` 和 `final_answer` 只训最后 2 个。非法输入被正确拒绝。

### GPU 训练

用 Qwen3-0.6B + GSPO + gsm8k 在双 T4 上跑了 2 步，分别用 `all_assistant` 和 `final_answer`。config summary 正确显示了 `trainable_turns` 差异，两次都正常到 `max_steps_reached`。

需要说明的是，这两次走的是标准 GSPO 路径（没加 `--agent-fn`），trainable-turn 的 mask 重写只在 agentic 路径触发。所以 GPU 训练验证的是"CLI 选项到 config 的传递没崩、新字段不影响现有训练"，mask 逻辑本身的正确性由 CPU 测试覆盖。

### 改动统计

10 个文件，+781/-5，无新依赖。

---

## 五、Kaggle 运行截图

### 截图 1：CPU 测试 + demo 脚本

![Kaggle CPU 测试与 demo](screenshots/kaggle-cpu-test-demo.png)

Kaggle Notebook 里的两段输出。上面是 64 个测试全过，下面是 demo 脚本打印三种模式的逐 token loss_mask 和 trainable_tokens，以及非法输入被校验拒绝。

### 截图 2：GPU 训练日志

![Kaggle GSPO 训练日志](screenshots/kaggle-training-log.png)

双 T4 上跑 Qwen3-0.6B + GSPO 的训练日志。config summary 里 `trainable_turns` 正确显示了两者的差异（`all_assistant` vs `final_answer`），各 2 步训练正常完成。

---

## 六、一点反思

这个 PR 让我最深的一点体会是：**写代码之前的花在"读代码、追数据流"上的时间不亏**。计划文档里两处跟实际代码对不上的地方（marker 误判 bug、落点错误），如果没提前发现，后面返工的成本远大于多花十几分钟读代码。

另一点是测试驱动的设计思路。先写测试断言"期望的 mask 长什么样"，再写实现让它通过——这样写出来的逻辑更不容易跑偏，而且改完之后跑一遍就知道有没有破坏现有功能。64 个测试全过，比口头说"应该没问题"有说服力得多。

至于 Phase 6 的研究化扩展（ablation 实验），需要用 `--agent-fn` 跑 agentic 训练来对比三种模式下 `trainable_tokens` 的真实差异。这次没做，留到有更多 GPU 时间的时候补。不过 mask 统计脚本已经有了，跑出数字只是时间和算力的问题。