# Payment JSON v1：固定训练与验证数据

原始来源是 `mlx-finetune-microsystem` 提交
`280a5d921b8679dffd1b1941ce6d6632bfe9d84c` 中的
`data/sft_payment_train.jsonl`，共 1,500 条。
本版复用全部请求及标准答案，不包含增强、反思样本，不使用原始 `solution` 推理文本。

| 文件 | 样本数 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 1,400 | SFT 参数更新 |
| `valid.jsonl` | 100 | 验证与后续训练配置选择；不是最终评测集 |
| `manifest.json` | — | 来源、内容指纹、制作代码指纹、划分规则及统计 |

Notebook 直接加载这些文件；启动时不生成、重划分或静默过滤数据。
本次只整理训练来源，尚未发布 recipe 的独立评测集。

## 样本格式

每行是一个 JSON 对象，字段如下：

- `case_id`、`source_line`：原始样本 ID 和来源文件的一基行号。
- `bucket`：原生成器的场景标签，代表生成意图，不代表重新验证后的边界分类。
- `request_fingerprint`：排除 `order.order_id` 后的请求 SHA-256。
- `input`、`ground_truth`：保留原始结构化请求与标准答案，方便追溯和评估。
- `prompt`：完整支付规则加紧凑请求 JSON。规则文本来自 `../system_prompt.txt`。
- `response`：紧凑的标准答案 JSON 字符串，不含 `<think>`、解释或 Markdown。

`prompt`/`response` 对应现有 AReno SFT 的数据字段要求。
模型专属 chat template、thinking 开关以及实际 tokenizer 长度预算尚待后续运行配置验证。
推理时只传 `prompt`；`response` 和 `ground_truth` 都是标签，不可拼入推理输入。
后续基线和微调后的推理必须使用相同的规则文本及请求序列化方式。

## 整理规则与已执行的数据检查

1. 对全部 1,500 条请求调用迁入的 `PaymentCalculationService` 重新计算，
   与来源 `ground_truth` 比较规范化 JSON，差异为 0。
   这是对来源答案的一致性检查，不是对业务实现独立正确性的证明。
2. 请求去重忽略订单 ID，但保留渠道、优惠 ID，以及所有数组顺序。
   不处理渠道或优惠 ID 重命名后等价的请求，不把此检查称为全部语义去重。
3. 用原始 `eval_testset_500.jsonl` 的 500 条请求排除重叠，未将其加入训练或验证。
   训练来源内部重复为 0，与该评测文件重叠为 0，未删除任何样本。
4. 固定 seed=42，按场景比例以最大余数法分配 100 个验证名额；
   各场景内按 `SHA256("42:" + request_fingerprint)` 排序选择，再按相同键排列产物。
   这是重新固定的划分，不沿用历史 `mlx_sft` 或增强版的验证集。
5. 不做 tokenizer 长度过滤，不截断请求或答案。长度兼容性尚未验证；后续应显式
   检查并选择可容纳该数据的配置。若必须修改数据，应发布新的数据版本和指纹。
6. 已检查输出标签、文件指纹、训练/验证请求互斥及全部来源行覆盖；在临时目录
   重新执行离线整理，两个 JSONL 和 manifest 均与发布产物逐字节一致。

| 原场景标签 | train | valid |
| --- | ---: | ---: |
| SPLIT_PAYMENT | 357 | 25 |
| RECOMMEND_SWITCH | 264 | 19 |
| EXCLUSIVE_VS_STACK | 213 | 15 |
| ROUND_HALF_UP_BOUNDARY | 203 | 15 |
| SINGLE_CHANNEL_NORMAL | 148 | 11 |
| TIE_BREAKER | 140 | 10 |
| INSUFFICIENT_FUNDS | 45 | 3 |
| CHANNEL_NOT_FOUND | 30 | 2 |

## 维护者离线复现

在 AReno 仓库根目录执行，`PAYMENT_SOURCE_REPO` 指向上述版本的原始仓库。
输出目录必须没有同名产物；脚本不会覆盖已有文件。只需要 Python 标准库，
不需要加载模型或安装 tokenizer。这不是 Notebook 用户的执行步骤。

```bash
python3 -m recipes.ling-tiny-fine-tuning.prepare_data \
  --source "$PAYMENT_SOURCE_REPO/data/sft_payment_train.jsonl" \
  --evaluation "$PAYMENT_SOURCE_REPO/data/eval_testset_500.jsonl" \
  --source-revision 280a5d921b8679dffd1b1941ce6d6632bfe9d84c \
  --output-dir ling-tiny-fine-tuning-json-v1-rebuild
```

来源和产物的实际文件 SHA-256 记录在 `manifest.json`，可用于核对复现结果。
