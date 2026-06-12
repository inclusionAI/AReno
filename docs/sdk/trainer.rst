Trainer SDK reference
=====================

The SDK is for custom training loops and algorithm experiments. Use it when the
CLI is too high-level and you want to control rollout, reward calculation,
advantage construction, loss selection, role scoring, or checkpoint cadence
directly from Python.

.. py:class:: areno.Trainer(world_size, model_path, backend_type=None, custom_config=None, metrics_log_dir=None)

   Main entry point for local Areno training workflows.

   ``Trainer`` initializes tokenizer and backend workers, generates rollout
   batches, runs policy training steps, manages PPO/DPO auxiliary roles, scores
   logprobs/values/rewards, and saves Hugging Face-compatible checkpoints.

   It provides methods to:

   * create a local tensor-parallel Areno backend
   * load prompt batches from dataset-like objects
   * generate text rollouts from string prompts or token ids
   * run agentic rollouts through a local OpenAI-compatible proxy
   * train policy batches with caller-provided loss functions
   * prepare reference, reward, and critic roles for PPO/DPO workflows
   * score logprobs, values, and rewards through backend-owned roles
   * save Hugging Face-compatible checkpoints

   Direct rollout calls must run inside ``async with
   trainer.rollout_session(...)``. The session is the lifecycle boundary for
   rollout state, actor onload/offload, and optional agentic proxy serving.

   .. rubric:: Typical flow

   .. code-block:: python

      import asyncio
      import areno
      from areno import Trainer

      async def main():
          # Near-instant: constructs the Python wrapper only.
          trainer = Trainer(
              world_size=1,
              model_path="Qwen/Qwen3.5-4B",
              backend_type=areno.Areno,
              custom_config=areno.ArenoConfig(tp_size=1),
          )

          # Takes a moment: loads tokenizer, starts workers, loads checkpoint.
          trainer.init()

          # Rollout calls must run inside an explicit rollout session. The
          # session owns actor onload/offload and rollout-state cleanup.
          sampling = areno.SamplingParams(max_new_tokens=128)
          async with trainer.rollout_session(sampling_params=sampling, proxy=False):
              rollout = trainer.rollout_batch(["Solve 12 * 13."], n_samples=1, sampling_params=sampling)

          # Runs one backend optimizer step.
          stats = trainer.train(batch_data, loss_fn, mini_bs=1)

          # Release metric writers and local resources.
          trainer.close()

      asyncio.run(main())

   .. note::

      ``Trainer(...)`` does not load the model. ``init()`` is the expensive
      boundary because it initializes workers and model weights. Rollout,
      scoring, and training calls then reuse the initialized backend.

   .. code-block:: python

      import areno
      from areno import Trainer

      trainer = Trainer(
          world_size=1,
          model_path="Qwen/Qwen3.5-4B",
          backend_type=areno.Areno,
          custom_config=areno.ArenoConfig(tp_size=1),
      )
      trainer.init()

   :param int world_size: Total number of devices or local worker ranks.
   :param str model_path: Local checkpoint path or Hugging Face repo ID.
   :param backend_type: Backend selector. Defaults to Areno when omitted.
   :param custom_config: Backend-specific configuration, such as
      ``areno.ArenoConfig(tp_size=1)``.
   :param str | None metrics_log_dir: Optional TensorBoard metrics directory.

   .. py:method:: init()

      Load the tokenizer, create the backend context, and initialize backend
      workers.

      .. code-block:: python

         trainer.init()

      :returns: ``None``

      .. important::

         Call ``init()`` exactly once before rollout, scoring, training, or
         checkpoint saving.

   .. py:method:: get_tokenizer()

      Return the initialized tokenizer.

      .. code-block:: python

         tokenizer = trainer.get_tokenizer()
         ids = tokenizer.encode("Hello")

      :returns: tokenizer object from the selected model path.

   .. py:method:: load_prompt_batches(dataset, *, batch_size, max_prompt_tokens, prompt_key="prompt", solutions_key="solutions")

      Yield tokenized prompt batches from a dataset-like object.

      The dataset must already expose the normalized prompt schema. If your raw
      dataset has different field names, normalize it before calling this
      method or use the CLI ``--dataset-loader-fn`` path.

      :param dataset: Object supporting ``len(dataset)`` and row indexing.
      :param int batch_size: Number of accepted rows per prompt batch.
      :param int max_prompt_tokens: Skip rows whose tokenized prompt is longer
         than this limit.
      :param str prompt_key: Field containing the prompt text.
      :param str solutions_key: Optional field containing reference answers.
      :returns: iterable of ``PromptBatch``.

      .. code-block:: python

         for prompt_batch in trainer.load_prompt_batches(
             dataset,
             batch_size=8,
             max_prompt_tokens=1024,
         ):
             prompts = [item.prompt for item in prompt_batch.items]

   .. py:method:: rollout_batch(prompts, n_samples, sampling_params)

      Generate completions from text prompts.

      Must be called inside ``async with trainer.rollout_session(...,
      proxy=False)`` for direct prompt rollouts. The explicit session defines
      the rollout lifecycle and prevents accidental consecutive rollouts from
      leaving stale rollout state.

      :param list[str] prompts: Prompt strings.
      :param int n_samples: Number of completions per prompt.
      :param SamplingParams sampling_params: Generation controls.
      :returns: ``list[RolloutResult]``

      This method tokenizes prompts with ``encode_generation_prompt`` and then
      delegates to :meth:`rollout_token_batch`.

      .. code-block:: python

         from areno import SamplingParams

         sampling = SamplingParams(max_new_tokens=128, temperature=1.0)
         async with trainer.rollout_session(sampling_params=sampling, proxy=False):
             rollouts = trainer.rollout_batch(
                 ["Solve 12 * 13."],
                 n_samples=4,
                 sampling_params=sampling,
             )

   .. py:method:: rollout_token_batch(prompt_tokens, n_samples, sampling_params)

      Generate completions from pre-tokenized prompts.

      Must be called inside an explicit rollout session, same as
      :meth:`rollout_batch`.

      :param list[list[int]] prompt_tokens: Prompt token ids.
      :param int n_samples: Number of completions per prompt.
      :param SamplingParams sampling_params: Generation controls.
      :returns: ``list[RolloutResult]``

      Use this method when your loop already tokenized prompts while building a
      dataset batch.

      .. code-block:: python

         tokenizer = trainer.get_tokenizer()
         prompt_tokens = [tokenizer.encode("Solve 12 * 13.")]
         sampling = SamplingParams(max_new_tokens=128, temperature=1.0)
         async with trainer.rollout_session(sampling_params=sampling, proxy=False):
             rollouts = trainer.rollout_token_batch(
                 prompt_tokens,
                 n_samples=4,
                 sampling_params=sampling,
             )

   .. py:method:: rollout_session(*, sampling_params, loss_mask_policy=None, max_running_prompts=None, timeout_s=300.0, proxy=True)

      Create an async rollout session.

      The session is the required lifecycle boundary for rollout. On enter, it
      prepares actor rollout state. On exit, it finalizes rollout-only state
      and prepares the backend for scoring or training. For direct prompt
      rollout, pass ``proxy=False``. For agentic rollout, keep the default
      ``proxy=True`` so the session starts a local OpenAI-compatible proxy.

      In proxy mode, agent code calls ``ctx.get_base_url()`` with a standard
      OpenAI client, and Areno collects assistant text, assistant tool calls,
      token ids, rollout logprobs, reward records, and loss masks. Assistant
      text and assistant tool-call spans are trainable by default; tool-result
      spans are masked unless enabled through ``LossMaskPolicy``.

      :param SamplingParams sampling_params: Default generation controls.
      :param LossMaskPolicy | None loss_mask_policy: Optional span-level loss
         mask policy.
      :param int | None max_running_prompts: Global concurrent prompt budget.
      :param float timeout_s: Proxy request and collection timeout.
      :param bool proxy: Whether to start the local OpenAI-compatible proxy.
      :returns: async ``RolloutSession`` context manager.

      .. code-block:: python

         async with trainer.rollout_session(
             sampling_params=SamplingParams(max_new_tokens=32, temperature=0.7),
             max_running_prompts=64,
         ) as ctx:
             print(ctx.get_base_url())

   .. py:method:: train(batch_data, loss_fn, mini_bs=8, gradient_accumulation_steps=None)

      Run one backend policy training step with a caller-provided loss
      function.

      :param list[TrainSequence] batch_data: Token, mask, logprob, reward, and
         advantage rows.
      :param Callable loss_fn: Loss function called by the backend.
      :param int mini_bs: Backend training microbatch size.
      :param int | None gradient_accumulation_steps: Optimizer step interval in
         microbatches.
      :returns: ``dict[str, float]`` with scalar training metrics.

      ``loss_fn`` receives the backend data pack and current logprobs. Built-in
      loss functions live under ``areno.loss_fns``.

      .. code-block:: python

         from functools import partial
         from areno.loss_fns import gspo_loss_fn

         stats = trainer.train(batch, partial(gspo_loss_fn, clip_eps=3.0e-4), mini_bs=4)

   .. py:method:: ensure_roles(roles)

      Prepare backend-owned auxiliary model roles for algorithms like PPO and
      DPO.

      :param dict[str, ModelRole] roles: Role name to model role configuration.
      :returns: ``None``

      .. code-block:: python

         from areno import ModelRole

         trainer.ensure_roles({
             "ref": ModelRole(name="ref", path="/path/to/reference", trainable=False),
             "critic": ModelRole(name="critic", path="/path/to/critic", trainable=True, optimizer_lr=1e-5),
         })

   .. py:method:: score_logprobs(role, token_rows)

      Score fixed token sequences with a backend-owned model role.

      :param str role: Role name, such as ``ref`` or ``actor``.
      :param list[list[int]] token_rows: Token rows to score.
      :returns: ``list[list[float]]``

      .. code-block:: python

         ref_logprobs = trainer.score_logprobs("ref", token_rows)

   .. py:method:: score_values(role, token_rows)

      Score per-token critic values with a backend-owned model role.

      :param str role: Role name, such as ``critic``.
      :param list[list[int]] token_rows: Token rows to score.
      :returns: ``list[list[float]]``

      .. code-block:: python

         values = trainer.score_values("critic", token_rows)

   .. py:method:: score_rewards(role, token_rows)

      Score sequence rewards with a backend-owned reward model role.

      :param str role: Role name, such as ``reward``.
      :param list[list[int]] token_rows: Token rows to score.
      :returns: ``list[float]``

      .. code-block:: python

         rewards = trainer.score_rewards("reward", token_rows)

   .. py:method:: train_values(role, batch_data, mini_bs, gradient_accumulation_steps=None, *, cliprange_value=0.5, value_loss_coef=0.5)

      Train a backend-owned critic or value role.

      :param str role: Role name, such as ``critic``.
      :param list[TrainSequence] batch_data: Training rows.
      :param int mini_bs: Critic training microbatch size.
      :param int | None gradient_accumulation_steps: Optimizer step interval in
         microbatches.
      :param float cliprange_value: PPO value-function clipping range.
      :param float value_loss_coef: Value loss coefficient.
      :returns: ``dict[str, float]``

      .. code-block:: python

         critic_stats = trainer.train_values("critic", batch_data, mini_bs=4)

   .. py:method:: save_checkpoint(path)

      Save a Hugging Face-compatible checkpoint when supported by the backend.

      :param str path: Output directory.
      :returns: saved checkpoint path as ``str``.

      .. code-block:: python

         saved_path = trainer.save_checkpoint("/tmp/areno-step-10")

   .. py:method:: close()

      Release local resources such as metric writers.

      :returns: ``None``

Data classes
------------

.. py:class:: areno.SamplingParams(greedy=False, top_p=1.0, top_k=-1, max_new_tokens=16, temperature=1.0, stop=None, stop_token_ids=None, ignore_eos=False, skip_special_tokens=True, max_prompt_len=None)

   Generation controls used by rollout APIs.

   :param bool greedy: Force greedy decoding. Overrides temperature in the
      backend.
   :param float top_p: Nucleus sampling threshold.
   :param int top_k: Top-k sampling threshold. ``-1`` disables top-k filtering.
   :param int max_new_tokens: Maximum number of generated response tokens.
   :param float temperature: Sampling temperature.
   :param list[str] | None stop: Stop strings.
   :param list[int] | None stop_token_ids: Stop token ids.
   :param bool ignore_eos: Continue generation without EOS stopping.
   :param bool skip_special_tokens: Decode helper preference for completions.
   :param int | None max_prompt_len: Optional prompt length cap.

.. py:class:: areno.TrainSequence(prompt_mask=None, tokens=None, logprobs=None, advantages=None, returns=None, values=None, ref_logprobs=None, reward=0.0, eos_token_id=0)

   One rollout sequence converted into a policy-gradient training sample.

   :param list[bool] prompt_mask: ``True`` for prompt or padded positions;
      losses train on response positions.
   :param list[int] tokens: Prompt and response token ids.
   :param list[float] logprobs: Rollout-policy logprobs aligned with tokens.
   :param list[float] advantages: Per-token advantages.
   :param list[float] returns: Optional value targets for PPO.
   :param list[float] values: Optional old value predictions for PPO.
   :param list[float] ref_logprobs: Optional reference logprobs for KL.
   :param float reward: Sequence-level reward.
   :param int eos_token_id: EOS id used for padding backend packs.

.. py:class:: areno.ModelRole(name, path, trainable, optimizer_lr=None)

   Auxiliary model role owned by the backend.

   :param str name: Role name, for example ``ref``, ``reward``, or ``critic``.
   :param str path: Checkpoint path or Hugging Face repo ID.
   :param bool trainable: Whether the role has an optimizer.
   :param float | None optimizer_lr: Optimizer LR for trainable roles.

.. py:class:: areno.ArenoConfig(model_path=None, tp_size=1, dp_size=None, devices=None, dummy_load=False, optimizer=None, runtime=None, max_running_prompts=64, decode_progress_interval_s=10.0)

   Backend configuration for the local Areno engine.

   :param str | None model_path: Optional backend model path override.
   :param int tp_size: Tensor-parallel size.
   :param int | None dp_size: Data-parallel size. Defaults to
      ``world_size // tp_size``.
   :param list[int] | None devices: Device ids for worker ranks.
   :param bool dummy_load: Build model without loading checkpoint weights.
   :param dict | None optimizer: Advanced optimizer config passed to the
      engine.
   :param dict | None runtime: Advanced runtime config passed to the engine.
   :param int max_running_prompts: Concurrent rollout prompt limit.
   :param float decode_progress_interval_s: Worker decode progress log
      interval. Logs report per-DP scheduled decode throughput for the current
      window and include ``cuda_graph=True`` when CUDA graph replay was used in
      that window.

.. py:class:: areno.api.agentic.AgentBatch(records, prompts, input_tokens, n_samples)

   Prompt batch expanded into one item per prompt/sample pair for agent
   execution.

   :param list[dict] records: Source dataset records.
   :param list[str] prompts: Prompt strings.
   :param list[list[int]] input_tokens: Prompt token ids.
   :param int n_samples: Samples per prompt.

.. py:class:: areno.api.agentic.RewardRecord(...)

   Unified reward input for agentic rollouts.

   Reward functions receive one ``RewardRecord`` per completed trajectory. For
   multi-turn agents, the record represents one prompt/sample pair, not one HTTP
   request.

   ``completion`` contains concatenated assistant response spans for backwards
   compatibility. ``final_answer`` contains the last assistant response.
   ``messages`` is the full OpenAI-style message list, including tool-result
   messages. ``rendered_completion`` is the same trajectory rendered through the
   tokenizer chat template when available. ``tool_calls`` and ``tool_results``
   expose parsed tool calls and environment observations. ``tokens``,
   ``logprobs``, and ``loss_mask`` describe the model-generated response spans.

   Tool-result/context spans are included in train rows so logprob scoring sees
   the same context as rollout, but they are masked from policy loss by default.

.. py:class:: areno.api.agentic.LossMaskPolicy(assistant_text=True, assistant_tool_calls=True, tool_results=False, final_assistant_text=True, system_prompt=False, user_prompt=False)

   Span-level policy-loss controls for agentic trajectories.

   :param bool assistant_text: Train assistant text spans.
   :param bool assistant_tool_calls: Train assistant tool-call spans.
   :param bool tool_results: Train tool-result spans. Defaults to ``False``.
   :param bool final_assistant_text: Reserved for final-response text spans.
   :param bool system_prompt: Reserved for system prompt spans.
   :param bool user_prompt: Reserved for user prompt spans.

One GSPO-style rollout/train step
---------------------------------

.. code-block:: python

   import asyncio
   from functools import partial

   from datasets import load_dataset

   import areno
   from areno import SamplingParams, TrainSequence, Trainer
   from areno.loss_fns import gspo_loss_fn


   def normalize_rewards(rewards):
       mean = sum(rewards) / len(rewards)
       var = sum((reward - mean) ** 2 for reward in rewards) / max(len(rewards), 1)
       std = max(var ** 0.5, 1e-6)
       return [(reward - mean) / std for reward in rewards]


   async def main():
       trainer = Trainer(
           world_size=1,
           model_path="Qwen/Qwen3.5-4B",
           backend_type=areno.Areno,
           custom_config=areno.ArenoConfig(tp_size=1),
       )
       trainer.init()

       row = load_dataset("gsm8k", "main", split="train[0:1]")[0]
       target = str(row["answer"]).rsplit("####", 1)[-1].strip()
       prompt = (
           "Solve the problem and put the final answer in \\boxed{}.\n\n"
           f"Problem: {row['question']}\nSolution:"
       )
       prompt_tokens = trainer.get_tokenizer().encode(prompt)
       sampling = SamplingParams(max_new_tokens=128, temperature=1.0)

       async with trainer.rollout_session(sampling_params=sampling, proxy=False):
           rollout = trainer.rollout_token_batch([prompt_tokens], n_samples=4, sampling_params=sampling)[0]

       completions = [trainer.get_tokenizer().decode(seq.resp_tokens) for seq in rollout.sequences]
       rewards = [1.0 if target in completion else 0.0 for completion in completions]
       advantages = normalize_rewards(rewards)

       batch = []
       for seq, reward, advantage in zip(rollout.sequences, rewards, advantages, strict=True):
           response_len = len(seq.resp_tokens)
           batch.append(
               TrainSequence(
                   prompt_mask=[True] * len(prompt_tokens) + [False] * response_len,
                   tokens=prompt_tokens + seq.resp_tokens,
                   logprobs=[0.0] * len(prompt_tokens) + seq.resp_logprobs,
                   advantages=[0.0] * len(prompt_tokens) + [advantage] * response_len,
                   reward=reward,
                   eos_token_id=trainer.get_tokenizer().eos_token_id,
               )
           )
       )

       stats = trainer.train(batch, partial(gspo_loss_fn, clip_eps=3.0e-4), mini_bs=4)
       print(stats)
       trainer.close()

   asyncio.run(main())

Agentic rollout with tools
--------------------------

This example shows the SDK pieces used by ``--agent-fn``. The agent calls a
local OpenAI-compatible proxy with Chat Completions ``tools``. Areno parses
model-native tool-call output for supported model families and returns an
``AgentTrainBatch`` that can be passed to the same trainers/loss functions as
regular rollouts.

.. code-block:: python

   import asyncio

   import areno
   from areno import SamplingParams, Trainer
   from areno.api.agentic import AgentBatch, AgentTrajectory, AgentTrajectoryTurn
   from openai import AsyncOpenAI


   tools = [
       {
           "type": "function",
           "function": {
               "name": "choose_move",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                   },
                   "required": ["direction"],
               },
           },
       }
   ]


   async def run_agent(ctx, batch):
       client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, max_retries=0)

       async def run_one(item):
           messages = [
               {"role": "system", "content": "Call choose_move with the selected direction."},
               {"role": "user", "content": item.prompt},
           ]
           tool_choice = {"type": "function", "function": {"name": "choose_move"}}
           response = await client.chat.completions.create(
               model="policy",
               messages=messages,
               tools=tools,
               tool_choice=tool_choice,
               max_tokens=16,
               temperature=0.7,
           )
           return AgentTrajectoryTurn(
               item=item,
               messages=messages,
               response=response,
               tools=tools,
               tool_choice=tool_choice,
           )

       try:
           return AgentTrajectory(
               turns=list(await asyncio.gather(*(run_one(item) for item in batch.iter_samples())))
           )
       finally:
           await client.close()


   def reward_fn(record):
       if not record.tool_calls:
           return -1.0
       return 1.0


   async def collect_agentic_batch(trainer, prompt_batch):
       agent_batch = AgentBatch.from_prompt_batch(prompt_batch, n_samples=4)
       async with trainer.rollout_session(
           sampling_params=SamplingParams(max_new_tokens=16, temperature=0.7),
           max_running_prompts=len(agent_batch),
       ) as ctx:
           return await run_agent(ctx, agent_batch)


   trainer = Trainer(
       world_size=1,
       model_path="Qwen/Qwen3-0.6B",
       backend_type=areno.Areno,
       custom_config=areno.ArenoConfig(tp_size=1),
   )
   trainer.init()

   # In a full loop, load a PromptBatch with trainer.load_prompt_batches(...).
   # Then call: agent_train_batch = asyncio.run(collect_agentic_batch(trainer, prompt_batch))
