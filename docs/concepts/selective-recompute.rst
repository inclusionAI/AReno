Selective activation recompute for Ling v3
========================================

CUDA Ling v3 supports selecting decoder recompute by module and zero-based
layer index, independently from SFT vocabulary projection/logprobs recompute.
Routing and its counters execute once, outside the expert recompute boundary.

CLI
---

For example, checkpoint only the routed/shared experts in layers 8 through 15,
and retain chunked output scoring without checkpointing that scoring::

    areno train --algo sft --backend cuda --ckpt /path/to/Ling-3.0-tiny \
      --dataset-path /path/to/train.jsonl \
      --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
      --recompute-modules routed_experts,shared_experts \
      --recompute-layers 8..15 --no-sft-output-recompute

Module names are ``softmax_attention``, ``kda_attention``, ``dense_mlp``,
``routed_experts``, and ``shared_experts``. ``all`` and ``none`` are accepted.
``--recompute-layers 0,4..7`` selects layer 0 and layers 4, 5, 6, 7.
The effective decoder selection is the intersection of the decoder global
switch, the module selection, and the layer selection. Unknown modules,
negative indices, reversed ranges, and indices outside the loaded model fail
before training. Explicit module/layer selection on unsupported adapters fails.

``--no-activation-checkpointing`` disables Ling decoder recompute, including
the tokenwise mini-sequence recompute schedule. It does not change the output
setting. ``--sft-output-recompute`` and ``--no-sft-output-recompute`` explicitly
enable/disable the independent SFT output checkpoint. Both explicit settings
retain chunked scoring and exactly the same targets and loss mask.

SDK
---

``TrainerConfig`` (and ``SFTTrainerConfig``) accepts the same selectors::

    config = TrainerConfig(
        algo="sft", backend="cuda", ckpt=model_path, dataset_path=data_path,
        recompute_modules=("routed_experts", "shared_experts"),
        recompute_layers="8..15",
        sft_output_recompute=False,
    )

Selectors may be comma-separated strings or SDK sequences. An empty sequence
selects nothing. Omitted selectors preserve the existing decoder schedule.
``sft_output_recompute=None`` is the compatibility default: it preserves the
existing output path, including whether that path was chunked. Explicit
``True`` or ``False`` requests chunked hidden-state scoring with the chosen
checkpoint setting. This preserves callers whose previous TP/MST layout used
full model logits. Chunked scoring uses the existing chunk size, which is
256 supervised positions by default.

These new controls currently require CUDA. Explicit SFT output control requires
``algo="sft"``. Lower-level ``CudaConfig.runtime`` and ``RuntimeConfig`` accept
the same fields; the latter resolves module/layer selection against the actual
``EngineConfig.model``.

Resolved training plans
-----------------------

On its first real training batch, each Ling worker writes
``recompute-plan.<pid>.rank<global_rank>.json`` to the configured metrics
directory. ``TrainerConfig.metrics_log_dir`` selects that directory; lower-level
runtime callers can set ``recompute_plan_dir``. If it is disabled, the resolved
schedule summary is still logged, without a JSON file.

The file records the request and the actual per-layer modules with strategies
``direct``, ``checkpoint``, or ``mst``. It also records the effective output
path, checkpoint flag and chunk size, MST chunk size, source commit, model path,
dtype, Torch/CUDA versions, actual Torch/LoRA seeds, TP/DP/SP topology, and first
packed-batch token counts and fingerprint. Absent model modules are not listed.
The stdout summary is derived from this same resolved plan.

For TP=1 without sequence parallelism, Ling dense and expert blocks use the
same compute tiles with and without activation recomputation. The compute
tile size is the internal MST chunk size, or 256 when MST is disabled.
Direct execution keeps tile activations; MST checkpoints each tile and
replays only that tile during backward. Without MST, an enabled decoder
checkpoint wraps the entire tiled module. Routing remains outside all tiles.
Disabling activation checkpointing disables replay, while retaining the
common compute tiling.

This aligns native GEMM shapes and parameter-gradient accumulation across
recompute selections. Low-precision parameter partials accumulate in FP32
before conversion back to storage dtype, without copying the base weights.
The reference is direct execution with the same compute tiling: it does not
promise bitwise equality with the previous unchunked GEMM or with arbitrary
different compute tile sizes. Supervision, loss masks and normalization are
unchanged. Plans retain the actual ``direct``/``checkpoint``/``mst`` modes
and record the compute schedule in ``decoder.compute_tiling``. The
independent output scoring path and its chunk bound are unchanged.
