# AReno 训练数据流逻辑图(中文)

> 以一次 `areno train --algo gspo ...` 为例,追踪"启动 → 生成 rollout → 打分拼 batch → 反向更新 → 下一步"的完整数据流向,以及每一步数据**流到了哪个节点(rank/进程)**。箭头标注的都是真实代码里传递的字段,已对照 `areno/cli/train.py`、`areno/api/trainer.py`、`areno/api/backend/areno/backend.py`、`areno/engine/api.py`、`areno/engine/worker.py` 核实。

---

## 0. 一张总图(CLI → 进程 → GPU rank)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  用户终端                                                                  │
│  areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \          │
│    --reward-fn-path examples/math/math_verify_reward.py \                │
│    --algo gspo --tp-size 4                                                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ Click 参数解析
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CLI 进程(单进程, areno/cli/train.py)                                     │
│  train_command()                                                            │
│   ├─ resolve_model_refs_for_config()  从 ModelScope 拉模型ckpt路径         │
│   ├─ get_algorithm("gspo") → AlgorithmSpec(PolicyOnlyTrainer, gspo_loss)   │
│   └─ run()                                                                  │
│        └─ 构造 areno.api.Trainer(world_size, model_path, ArenoConfig)       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ Trainer.init()
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SDK 进程(areno/api/trainer.py, 仍是单进程)                                │
│  Trainer                                                                     │
│   ├─ load_tokenizer(model_path)                                             │
│   ├─ Context(world_size, model_path, tokenizer, ArenoConfig, eos_ids)       │
│   ├─ get_backend_cls(Areno) → ArenoBackend                                  │
│   └─ backend.initialize(ctx) ────────────┐                                  │
└─────────────────────────────────────────────┼─────────────────────────────┘
                                              │ ArenoBackend.initialize
                                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ArenoBackend (areno/api/backend/areno/backend.py, 仍在 SDK 进程)           │
│   ├─ world = dp * tp 必须整除; world=4, tp=4 → dp=1                       │
│   └─ ArenoEngine.from_pretrained(model_path, tp=4, dp=1, devices=0..3)     │
│        └─ TPCluster(config, ArenoWorker)  【spawn 4 个 worker 进程】        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ fork 4 个 TP rank 进程(每个占 1 张 GPU)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Worker 进程 ×4 (areno/engine/worker.py, 这是真正的 GPU 进程)               │
│   TP rank 0/1/2/3,各加载一份模型分片(shard)                                 │
│   每个 worker 持有:                                                          │
│     InferenceManager  (areno/engine/inference.py — rollout)                │
│     TrainingManager   (areno/engine/training.py — 训练)                    │
│   TP rank 间通过 NCCL 集合通信(all_reduce / all_gather)同步                │
└──────────────────────────────────────────────────────────────────────────┘
```

**关键边界**:SDK 进程不含模型权重、不跑 GPU 计算;它只是通过 `TPCluster` 的 RPC(`cluster.call(Op.X, Payload)`)向 4 个 worker 进程派发命令、回收结果。`model_path` 只是路径字符串传下去,**权重实际在 worker 进程里按 TP 分片加载**。

---

## 1. 阶段一:Init(模型加载)

```
SDK进程  ArenoBackend.initialize
   │  ArenoEngine.from_pretrained(model_path, tp=4, dp=1)
   │  ──────────────────────────────────────────────▶ TPCluster
   │                              spawn + Op.INIT 派发给每个 rank
   ▼
每个 Worker(rank 0..3):
   ├─ adapter_from_hf(model_path)  按 HF config 选 Qwen3Adapter
   ├─ config_from_hf() → ModelConfig → build_model(config)
   │      └─ 模型 nn.Module(attention/mlp/norm/rotary 来自 engine/layers/)
   ├─ load_model_weights(model, config, model_path)
   │      └─ 按 TP 切分权重分片,各 rank 只拿自己那一份
   └─ 训练侧:OptimizerConfig → adamw(adamw_8bit / adamw_fp32_master)
```
数据落地:**权重分片在 4 个 GPU 进程**,SDK 进程只持有 `model_path` 字符串。

---

## 2. 阶段二:Rollout 生成(on-policy 采样)

以一条 prompt 为例,跟踪 token 的去向:

```
①  SDK进程  trainer.rollout_session(sampling_params)   (async with)
       └─ backend.begin_rollout_session() → engine.begin_rollout_session()
            └─ 向 4 个 rank 派发 Op.BEGIN_ROLLOUT_SESSION

②  SDK进程  trainer.rollout_batch(["Solve: ..."], n_samples=8, sampling_params)
       ├─ tokenizer 编码 → prompt_tokens: list[list[int]]   (1 行 prompt)
       └─ backend.rollout_batch(ctx, prompt_tokens, 8, sampling_params)

③  ArenoBackend.rollout_batch
       ├─ flat_prompts = 每条 prompt 复制 n_samples=8 次 → 8 行 token-id
       ├─ _rollout_options(): SamplingParams(temperature, top_p, top_k, stop/suppress_ids)
       └─ engine.generate_rollout(flat_prompts[8行], max_new_tokens, max_running_prompts, ...)

④  ArenoEngine.generate_rollout   (这里是切 DP 的地方)
       ├─ dp_size=1,所以 8 行全部留给 DP-rank0
       ├─ split_list_by_dp(chunk, dp_size)   → prompts_by_dp
       ├─ cluster.call(Op.INFER_ROLLOUT, RolloutPayload(
       │        prompts_by_dp, prompt_indices_by_dp,
       │        max_new_tokens, eos_token_id, sampling_params,
       │        max_running_seqs, max_cache_len, num_blocks, block_size))
       │  ──────────────────────────────────────────────▶ 4 个 rank
       │
       ▼  每个 Worker(rank 0..3):
            InferenceManager.infer_rollout(payload)
            ├─ 初始化 paged-KV cache(num_blocks × block_size)
            ├─ Prefill:8 行 prompt 同时喂入,各 TP rank 各算自己分片
            │     └─ attention 在 TP 上切 head,forward 后 TP all_reduce 聚合
            ├─ Decode:逐 token 生成,采样用 sampling_params(温度/top_k/top_p)
            │     └─ 可选 areno/accel 自研算子 或 flash-attn 后端
            └─ 停止条件:max_new_tokens / 遇 eos_token_id / stop_token_ids
            每个 TP rank 都生成"完整"结果(TP 内部已 all_reduce),但只有 rank0 返回
       ◀──────────────────────────────────────────────  results[rank]

⑤  ArenoEngine 合并
       ├─ dp_rank0_results(results, tp_size=4, dp_size=1)   丢弃 TP 冗余
       └─ _merge_dp_rollouts_in_input_order(...)            按原始输入顺序还原

⑥  ArenoBackend.rollout_batch  组装回 SDK 类型
       └─ 包装成 RolloutResult[
              RolloutSequence(resp_tokens, resp_logprobs)  ×8
            ]   每个 prompt 一组 n_samples 条

⑦  SDK进程  trainer.rollout_batch 返回 list[RolloutResult]   (1 个 prompt → 1 个 result)
```

**数据落地**:rollout 的 `response_ids` + `logprobs` 现在回到 **SDK 进程**的 Python 内存(已 `.tolist()` 成 list)。GPU 一侧的 KV cache 被释放(`end_rollout_session`),不再保留。

---

## 3. 阶段三:打分 + 拼 TrainSequence(纯 CPU,在 SDK 进程)

AReno 在这里**不替你算优势**——这是用户/CLI 脚本的职责:

```
SDK进程(你的训练脚本 / CLI run 内)
   ├─ completions = [tokenizer.decode(seq.resp_tokens) for seq in rollout.sequences]   8 条文本
   ├─ rewards = reward_fn(record, completions)   ← --reward-fn-path 指向的用户函数,返回 list[float] 8 个
   ├─ advantages = (rewards - mean) / std        ← 用户自行归一化(GSPO 也常用排序统计)
   └─ for seq, r, adv in zip(rollout.sequences, rewards, advantages):
        batch.append(TrainSequence(
            tokens       = prompt_tokens + seq.resp_tokens,        拼接
            logprobs     = [0.0]*prompt_len + seq.resp_logprobs,   采集时的 logp
            advantages   = [0.0]*prompt_len + [adv]*resp_len,      prompt 段置 0
            prompt_mask  = [True]*prompt_len + [False]*resp_len,
            reward       = r,
            eos_token_id = tokenizer.eos_token_id,
        ))
   → 得到 list[TrainSequence]
```

`reward_fn` 是普通 Python 文件(`reward_fn(example, completions) -> list[float]`),通过 `--reward-fn-path` 注入,典型的如 `examples/math/math_verify_reward.py`(math-verify 校验 `\boxed{}` 答案)。

---

## 4. 阶段四:Train(反向传播一步)

```
①  SDK进程  trainer.train(batch=list[TrainSequence], loss_fn=bind(gspo_loss), mini_bs=4)
       └─ backend.train(ctx, batch, loss_fn, mini_bs=4)

②  ArenoBackend.train
       ├─ 切 mini_bs=4:每 4 条一个 micro-batch → 2 个 pack
       ├─ _make_train_pack(seqs)   把 TrainSequence 右填充成 (B, max_len) 张量:
       │      {
       │        input_ids  int64,  prompt+response, 末尾用 eos 填充
       │        labels      ← input_ids 的 clone(下一 token 目标)
       │        lengths     int32 (B,) 真实长度
       │        prompt_mask bool,  prompt 段 True
       │        loss_mask   bool,  可选(agentic 时由 LossMaskPolicy 决定 tool result 段 False)
       │        logprobs    float, 采集时策略 logp(对齐 input_ids)
       │        advantages  float, 每 token 优势(prompt 段 0)
       │        returns / values / ref_logprobs  ← 仅当至少一条序列带该字段才分配(PPO 用)
       │      }
       ├─ 给每个 pack 盖戳: pack["_loss_fn"] = loss_fn   ← 用户 loss 随包送进 worker
       └─ engine.step(packs, gradient_accumulation_steps)

③  ArenoEngine.step   (这里按 DP 切包)
       ├─ 每个 pack: split_data_pack_by_dp(pack, dp_size=1)  → 一份(因为 dp=1)
       ├─ to_cpu(data_packs_by_dp)  + 共享内存拷贝(避免大张量 pickle 跨 IPC)
       ├─ cluster.call(Op.TRAIN, TrainPayload(data_packs_by_dp, grad_accum))  ──▶ 4 个 rank
       │
       ▼  每个 Worker(rank 0..3):
            TrainingManager.train(payload)
            for pack in packs_by_dp:
              ├─ 前向: model(input_ids)  各 TP rank 算自己分片 → TP all_reduce
              ├─ _external_loss_dispatcher(pack, train_logprobs)
              │      └─ pack["_loss_fn"](pack, train_logprobs)
              │           = gspo_loss_fn(data_pack, train_logprobs)
              │             ├─ ratio = exp(train_logp - rollout_logp)   (用 pack["logprobs"])
              │             ├─ 序列级 clip(clip_eps 来自 bind_gspo_loss)
              │             └─ advantage 加权 → loss
              ├─ 反向: loss.backward()
              │      └─ DP 组内 all_reduce 梯度(若 dp>1)  +  TP 组内 all_reduce 梯度
              ├─ optimizer.step()  (adamw_8bit / adamw_fp32_master)
              └─ 返回 TrainStats(loss + ratio_mean / logp_diff 等指标)
       ◀────────────────────────────────────────────────  results[rank]

④  ArenoEngine.step 合并
       ├─ dp_rank0_results 去掉 TP 冗余
       └─ merge_train_stats(...)  对 loss / 各指标 SUM 或 MEAN

⑤  ArenoBackend.train 汇总指标
       ├─ loss = 平均各 microbatch 的 loss
       ├─ ratio_mean / logp_diff 等"rollout-policy 指标"取首个 microbatch(见 _is_rollout_policy_metric)
       ├─ 其余指标取所有 microbatch 均值
       └─ 附 step_rollout_time_s / step_train_time_s / step_e2e_time_s

⑥  SDK进程  trainer.train 返回 dict[str, float]
       ├─ 若挂了 MetricsRecorder → 写 TensorBoard + dashboard state
       └─ trainer.finish_step()
```

**数据落地**:权重更新发生在 **4 个 GPU 进程**内,各 rank 的分片同步前进。SDK 进程只拿到指标 dict。

---

## 5. 阶段五:循环 & 结束

```
   ┌────────────────────────循环(直到 epochs / max_steps)────────────────────────┐
   │  对每个 prompt batch:                                                          │
   │    rollout_session → rollout_batch → reward_fn → to_advantages                │
   │      → train(TrainSequence..., loss_fn)                                       │
   │      → 新权重在 4 个 GPU 进程内原地更新                                          │
   │  下一轮 rollout 自动用新权重采样新 completions                                  │
   └─────────────────────────────────────────────────────────────────────────────┘

结束后:
   trainer.save_checkpoint(path)
      └─ engine.save_checkpoint(path) → 各 rank 把分片权重写回 HF 兼容格式
   trainer.close()
      └─ TPCluster 关闭 4 个 worker 进程,释放 GPU
```

---

## 6. 各节点"持有 / 处理什么"速查表

| 节点 | 持有模型权重? | 在这一步拿到什么数据 | 产出什么 |
| --- | --- | --- | --- |
| **CLI / SDK 进程**(单进程) | ✗(只有 `model_path` 字符串) | dataset 行、prompt 文本、rollout 后的 `response_ids`/`logprobs`(CPU list)、rewards、advantages | `TrainSequence` 批、指标 dict |
| **ArenoBackend**(SDK 进程内对象) | ✗ | `TrainSequence` 列表、`SamplingParams` | `RolloutPayload` / `TrainPayload`(发给 cluster)、回传的 `RolloutResult` / 指标 |
| **ArenoEngine**(SDK 进程内对象) | ✗(把活外包给 worker) | prompt token 行、data pack 张量 | 按 DP 切分、调 `cluster.call`、合并 rank 结果 |
| **TPCluster**(SDK 进程内) | ✗ | Op + Payload | 把命令 RPC 给各 rank 进程、收集 `results[rank]` |
| **Worker 进程 ×4**(每个 1 GPU) | ✓(TP 分片) | `RolloutPayload` / `TrainPayload` 中属于本 rank 的那份 | rollout 的 response+logprobs(经 TP all_reduce);`TrainStats`;权重原地更新 |

---

## 7. 关键控制开关(影响数据怎么走)

- **`--tp-size` / `--world-size`**:`world = dp * tp` 必须整除。上例 `world=4, tp=4` → `dp=1`,所有 prompt 留在单个 DP 组、4 路张量并行。
- **`--algo`**:决定 `AlgorithmSpec` 选哪个 trainer + 哪个 loss factory。`gspo`/`grpo` 用 `PolicyOnlyTrainer`(`requires_rollout=True`);`sft`/`dpo` 是离线算法(`requires_rollout=False`),数据来自 `--dataset-loader-fn` 直接拼 `TrainSequence`,不走 rollout 阶段二。
- **`--reward-fn-path`**:阶段三打分函数,纯 CPU,在 SDK 进程跑。
- **`--agent-fn`**(Agentic RL):阶段二三被替换——`rollout_session` 起本地 OpenAI 兼容 HTTP 端点,agent 函数返回 `AgentTrajectory`(显式 messages/tool_calls/tool_results),AReno 按默认 `LossMaskPolicy`(assistant 计入,tool result 默认 mask)转成 `TrainSequence` 的 `loss_mask`,之后阶段四完全相同。
- **`--tune-params`**:正式跑前先 `probe_rollout_cache`(分配 KV + 捕获 decode graph 但不生成),用来回填 `--max-running-prompts`/`--batch-size`/`--mini-bs` 的保守值。
- **可选 attention 后端 `--attn-backend`**:决定阶段二/四的 attention 走 flash-attn 还是 areno 自研(`engine/layers/attention_backend/` + `accel/csrc/attention.cu`)。