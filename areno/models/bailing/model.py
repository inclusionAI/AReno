"""BailingMoE-Linear-V2 causal-LM adapter.

Targets the BailingMoeV2_5ForCausalLM HF family (``model_type ==
"bailing_moe_linear"``). The architecture interleaves two attention flavours
and a sparse mixture-of-experts MLP:
    * Attention layers alternate between ``BailingSoftmaxAttention`` (a
      classic GQA / optional MLA-style low-rank KV pathway running on flash
      attention) and ``BailingLinearAttention`` (a chunked lightning-attention
      / seg_la recurrent linear attention with ALiBi-style slope biases). The
      pattern is governed by ``layer_group_size``: every group_size-th layer
      is softmax, all others are linear; the trailing layers always run
      softmax to anchor positions.
    * The MLP is either a dense SwiGLU (``BailingDenseMLP``) for the leading
      ``first_k_dense_replace`` layers, or a sparse ``BailingSparseMoeBlock``
      for the remainder.
    * The MoE router (``BailingGate``) uses sigmoid scoring with a learnable
      per-expert bias and SGLang-style biased grouped top-k routing: experts
      are partitioned into ``n_group`` groups, top ``topk_group`` groups win,
      then top ``num_experts_per_tok`` experts inside the winning groups
      score the token. The bias is updated online during training to balance
      load.
    * Experts are stored in ``BailingGroupedExperts`` as fused 3D weights
      (``[local_experts, out, in]``) for ``areno_grouped_linear`` /
      ``areno_moe_topk_permute`` / fused-MoE inference kernels. Expert
      parallelism is collapsed into the TP group (each rank owns
      ``num_experts / world_size`` experts) and routing uses
      ``all_reduce`` to sum back the per-rank contributions.
    * An optional ``shared_experts`` dense MLP runs unconditionally on every
      token and is added to the routed output.
    * Inference uses a separate ``_forward_fused_moe`` path that stacks the
      per-expert gate/up/down weights into contiguous w1/w2 buffers and runs
      ``areno_fused_experts``; training keeps the permute/unpermute path
      so autograd can flow through.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from areno.accel import areno_grouped_linear, areno_grouped_topk_router, areno_linear, areno_moe_topk_permute, areno_moe_unpermute, areno_sigmoid, areno_silu
from fla.ops.lightning_attn import chunk_lightning_attn
from torch import nn

from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend, build_infer_attention_backend
from areno.engine.layers.attention_backend.train import build_train_attention_backend
from areno.models.bailing.checkpoint import CHECKPOINT_SPEC
from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.parallel.collectives import (
    all_reduce,
    gather_from_sequence_parallel_region,
    is_sequence_parallel_active,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    sequence_parallel_region,
)
from areno.engine.parallel.context import get_tp_context
from areno.engine.layers.linear import ColumnParallelLinear, RowParallelLinear, _shard_range, mark_tensor_parallel_parameter
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.layers.norm import GroupRMSNormSigmoidGate, RMSNorm
from areno.engine.layers.rotary import PartialRotaryEmbedding
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.accel.ops import (
    FusedMoeConfig,
    SegLaMeta,
    areno_silu_and_mul,
    areno_fused_experts,
    log_once,
    seg_la_fwd,
)


class BailingDenseMLP(nn.Module):
    """Plain SwiGLU MLP used for shared experts and for the first
    ``first_k_dense_replace`` decoder layers (before MoE kicks in)."""

    def __init__(self, config: ModelConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Fused SiLU(gate) * up kernel — same as areno_silu_and_mul but reused
        # under torch._dynamo.disable so eager and compiled paths agree.
        hidden = _areno_silu_pair_no_compile(gate, up)
        return self.down_proj(hidden)


class BailingGate(nn.Module):
    """Sigmoid-scored biased grouped top-k MoE router.

    Produces, for every token, ``top_k`` expert indices and renormalized
    routing weights. The bias is added to the per-expert logit *before* the
    group/top-k pruning step (the "noaux_tc" variant of grouped routing);
    during training the bias is updated to push load back towards balance.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.score_function != "sigmoid":
            raise ValueError(f"BailingGate only supports sigmoid scoring, got {config.score_function!r}")
        if not config.moe_router_enable_expert_bias:
            raise ValueError("BailingGate requires expert bias for biased grouped topk")
        if not config.norm_topk_prob:
            raise ValueError("BailingGate requires norm_topk_prob=True")
        self.top_k = config.num_experts_per_tok
        self.num_experts = int(config.num_experts or 0)
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.router_dtype = config.moe_router_dtype
        # Router projection: hidden_size -> num_experts logits. Replicated
        # across TP ranks (sequence-parallel on the input side) so every rank
        # sees identical routing decisions and accumulates the gradient via
        # all-reduce.
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, dtype=self.router_dtype))
        mark_tensor_parallel_parameter(self.weight, False, sequence_parallel=True, tp_grad_allreduce=True)
        self.bias_update_rate = config.moe_router_bias_update_rate
        self.expert_parallel_size = 1
        # Per-step token-per-expert counter used to drive bias updates at the
        # end of an optimizer step. ``expert_bias`` is the slowly-updated
        # canonical value, ``local_expert_bias`` is the snapshot used by the
        # kernel each forward pass.
        self.register_buffer("local_tokens_per_expert", torch.zeros(self.num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("expert_bias", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("local_expert_bias", torch.zeros(self.num_experts), persistent=False)

    @torch._dynamo.disable
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = hidden_states.view(-1, hidden_states.shape[-1])
        # Promote to router_dtype (typically fp32) for numerical stability of
        # the sigmoid/top-k selection.
        logits = _areno_linear_no_compile(x.to(dtype=self.weight.dtype), self.weight)
        topk_idx, topk_weight = self._forward_grouped_topk(logits)
        if torch.is_grad_enabled():
            # Only track load when training; eval calls keep counters cold.
            _accumulate_tokens_per_expert(self.local_tokens_per_expert, topk_idx, self.num_experts)
        return topk_idx, topk_weight.float(), logits

    @torch._dynamo.disable
    def _forward_grouped_topk(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_once("areno_grouped_topk_router", "using ARENO grouped router topk kernel")
        return _areno_grouped_topk_router_no_compile(
            logits,
            self.local_expert_bias.to(device=logits.device, dtype=torch.float32),
            self.top_k,
            self.n_group,
            self.topk_group,
        )

    @torch.no_grad()
    def finalize_expert_bias(self, tp_group, dp_group) -> None:
        """Apply the per-step bias update and reset the token counter.

        Called once per training step from the trainer. The token counts are
        summed across data-parallel replicas, then nudged towards balance
        using a signed step (``sign(mean - tokens) * rate``); finally the
        zero-mean projection keeps the bias from drifting unboundedly.
        """
        del tp_group
        tokens_per_expert = self.local_tokens_per_expert
        if dist.is_available() and dist.is_initialized():
            if dp_group is not None:
                dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM, group=dp_group)
        if self.bias_update_rate != 0.0:
            mean_tokens = tokens_per_expert.mean(dim=-1, keepdim=True)
            offset = mean_tokens - tokens_per_expert
            # Under-loaded experts get a positive nudge, over-loaded a negative
            # one. Re-center to mean zero to avoid drift.
            self.expert_bias.add_(torch.sign(offset) * self.bias_update_rate)
            self.expert_bias.sub_(self.expert_bias.mean())
        self.local_expert_bias.copy_(self.expert_bias)
        tokens_per_expert.zero_()


@torch._dynamo.disable
def _accumulate_tokens_per_expert(tokens_per_expert: torch.Tensor, topk_idx: torch.Tensor, num_experts: int) -> None:
    """Histogram top-k expert assignments into ``tokens_per_expert`` (in place)."""
    with torch.no_grad():
        tokens_per_expert.add_(
            torch.bincount(topk_idx.reshape(-1), minlength=num_experts).to(
                device=tokens_per_expert.device,
                dtype=tokens_per_expert.dtype,
            )
        )


class BailingSparseMoeBlock(nn.Module):
    """Sparse MoE block: router -> permute -> grouped experts -> unpermute,
    plus an optional dense shared-expert pathway.

    Training and inference take different code paths: training keeps the
    autograd-friendly permute/unpermute split, inference uses the fused
    ``areno_fused_experts`` kernel over stacked w1/w2 weight tiles.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts = int(config.num_experts or 0)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = BailingGate(config)
        self.experts = BailingGroupedExperts(config)
        # Shared experts always run (no routing decision) — their intermediate
        # size scales with ``num_shared_experts``. Output is added to the
        # routed result post-reduce.
        self.shared_experts = (
            BailingDenseMLP(config, config.moe_intermediate_size * config.num_shared_experts)
            if config.num_shared_experts is not None
            else None
        )
        # Inference-only fused weight buffers, populated by
        # ``prepare_infer_weights``. w1 stacks the (gate, up) projections per
        # expert; w2 holds the per-expert down projection.
        self.register_buffer("_infer_gate_weight", torch.empty(0), persistent=False)
        self.register_buffer("_infer_up_weight", torch.empty(0), persistent=False)
        self.register_buffer("_infer_down_weight", torch.empty(0), persistent=False)
        self.register_buffer("_infer_w1_weight", torch.empty(0), persistent=False)
        self.register_buffer("_infer_w2_weight", torch.empty(0), persistent=False)
        self._infer_weights_ready = False
        self._fused_moe_config = (
            FusedMoeConfig(
                num_experts=self.experts.local_num_experts,
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.moe_intermediate_size,
                top_k=self.num_experts_per_tok,
                routed_scaling_factor=self.config.routed_scaling_factor,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # In SP mode we need the full (un-scattered) hidden states for routing
        # since the router weight isn't sharded along the sequence dim.
        moe_sequence_parallel = is_sequence_parallel_active()
        if moe_sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        identity = hidden_states
        bsz, seqlen, hidden = hidden_states.shape
        with sequence_parallel_region(False):
            topk_idx, topk_weight, _ = self.gate(hidden_states)
            flat = hidden_states.view(-1, hidden)
            if self.training:
                # Permute/unpermute path is autograd-friendly.
                out = self.experts(flat, topk_idx, topk_weight).view(bsz, seqlen, hidden)
                if self.shared_experts is not None:
                    out = out + self.shared_experts(identity)
                return reduce_scatter_to_sequence_parallel_region(out) if moe_sequence_parallel else out
            # Inference: fused-MoE kernel over the stacked w1/w2 weights.
            out = self._forward_fused_moe(flat, topk_idx, topk_weight)
        out = out.view(bsz, seqlen, hidden)
        if self.shared_experts is not None:
            out = out + self.shared_experts(identity)
        return reduce_scatter_to_sequence_parallel_region(out) if moe_sequence_parallel else out

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        """Stack per-expert weights into contiguous fused tiles for inference.

        ``_infer_w1_weight`` concatenates (gate, up) per expert so the fused
        kernel does one matmul per expert. ``_infer_w2_weight`` mirrors the
        down projection. Buffers are reused across calls if the shape/device
        already match to avoid reallocating on every weight refresh.
        """
        gate_weights, up_weights, down_weights = self.experts.expert_weights()
        self._infer_gate_weight = self._updated_infer_weight(
            self._infer_gate_weight,
            torch.stack(gate_weights, dim=0).to(dtype=self.config.dtype).contiguous(),
        )
        self._infer_up_weight = self._updated_infer_weight(
            self._infer_up_weight,
            torch.stack(up_weights, dim=0).to(dtype=self.config.dtype).contiguous(),
        )
        self._infer_down_weight = self._updated_infer_weight(
            self._infer_down_weight,
            torch.stack(down_weights, dim=0).to(dtype=self.config.dtype).contiguous(),
        )
        # w1 = [gate || up] along the intermediate dim so SiLU(gate) * up can
        # be folded into a single fused kernel call.
        self._infer_w1_weight = self._updated_infer_weight(
            self._infer_w1_weight,
            torch.cat((self._infer_gate_weight, self._infer_up_weight), dim=1).contiguous(),
        )
        self._infer_w2_weight = self._updated_infer_weight(self._infer_w2_weight, self._infer_down_weight.contiguous())
        self._infer_weights_ready = True

    @torch.no_grad()
    def _updated_infer_weight(self, current: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Reuse the existing storage when possible — important when the trainer
        # keeps swapping weights in and out (e.g. for evaluation cycles).
        if current.shape == value.shape and current.device == value.device and current.dtype == value.dtype:
            current.copy_(value)
            return current
        return value

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        """Drop the fused inference tiles and reclaim memory."""
        device = self._infer_gate_weight.device
        dtype = self._infer_gate_weight.dtype
        self._infer_gate_weight = torch.empty(0, device=device, dtype=dtype)
        self._infer_up_weight = torch.empty(0, device=device, dtype=dtype)
        self._infer_down_weight = torch.empty(0, device=device, dtype=dtype)
        self._infer_w1_weight = torch.empty(0, device=device, dtype=dtype)
        self._infer_w2_weight = torch.empty(0, device=device, dtype=dtype)
        self._infer_weights_ready = False

    def _forward_fused_moe(self, flat: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        if self._infer_w1_weight.numel() == 0:
            raise RuntimeError("fused MoE inference weights are not prepared")
        log_once("areno_fused_moe", "using areno fused MoE expert kernel")
        # Drop routes pointing at experts owned by other ranks (their weight
        # is zero, so they contribute nothing locally) and remap global expert
        # ids into the local 0..local_num_experts-1 range.
        local_idx, local_weight = self.experts.local_routes(topk_idx, topk_weight)
        out = _areno_fused_experts_no_compile(
            flat.contiguous(),
            self._infer_w1_weight,
            self._infer_w2_weight,
            local_weight.float(),
            local_idx.int(),
            self._fused_moe_config,
        )
        # Sum each rank's owned-expert contribution back together.
        return all_reduce(out)


class BailingGroupedExperts(nn.Module):
    """Bank of MoE expert FFNs stored as a single fused 3D weight tensor.

    Expert parallelism is piggy-backed on the TP group: each rank owns
    ``local_num_experts = num_experts / world_size`` consecutive experts.
    The grouped-linear kernel takes ``tokens_per_expert`` and runs one fused
    GEMM per expert without materialising per-expert slices.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        ctx = get_tp_context()
        self.config = config
        self.num_experts = int(config.num_experts or 0)
        if self.num_experts % ctx.world_size != 0:
            raise ValueError(f"num_experts={self.num_experts} must be divisible by TP/EP size={ctx.world_size}")
        # Contiguous slab of experts owned by this rank.
        self.local_num_experts = self.num_experts // ctx.world_size
        self.local_expert_start = ctx.rank * self.local_num_experts
        self.local_expert_end = self.local_expert_start + self.local_num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        # fc1 = [gate || up] fused into ``2 * intermediate_size`` rows so
        # SiLU(gate) * up can collapse into a single kernel; fc2 is the down
        # projection back to hidden size.
        self.linear_fc1 = _build_grouped_linear(
            self.local_num_experts,
            self.hidden_size,
            2 * self.intermediate_size,
            dtype=config.dtype,
        )
        self.linear_fc2 = _build_grouped_linear(
            self.local_num_experts,
            self.intermediate_size,
            self.hidden_size,
            dtype=config.dtype,
        )
        # Expert weights are sharded by EP (collapsed into TP); flag them as
        # not-TP/not-SP so the standard TP collectives leave them alone.
        for param in self.parameters():
            mark_tensor_parallel_parameter(param, False, sequence_parallel=False)

    def forward(self, flat: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        return self._forward_fused_permute(flat, topk_idx, topk_weight)

    def _forward_fused_permute(self, flat: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        log_once("areno_moe_permute", "using fused MoE permute/unpermute kernels")
        # Permute: group tokens by destination expert, keeping only routes for
        # locally-owned experts. ``tokens_per_expert`` is a 1D count used to
        # offset the grouped GEMM.
        x, sorted_route_weight, sorted_token_idx, tokens_per_expert = _areno_moe_topk_permute_no_compile(
            flat,
            topk_idx,
            topk_weight.float(),
            self.local_expert_start,
            self.local_num_experts,
        )
        if x.shape[0] == 0:
            # No tokens routed to this rank — still need to all_reduce to keep
            # collective sync with peers.
            return all_reduce(flat.new_zeros(flat.shape))
        hidden, _ = _grouped_linear_forward(self.linear_fc1, x.contiguous(), tokens_per_expert)
        # Apply routing weight before fc2 so it stays inside the fp32 reduction.
        hidden = (
            _areno_silu_and_mul_no_compile(hidden)
            * sorted_route_weight.unsqueeze(-1).to(dtype=hidden.dtype)
        ).contiguous()
        expert_out, _ = _grouped_linear_forward(self.linear_fc2, hidden, tokens_per_expert)
        # Unpermute back to original (batch, seq) order, then scale and reduce.
        out = _areno_moe_unpermute_no_compile(expert_out, sorted_token_idx, merging_probs=None, restore_shape=flat.shape)
        return all_reduce(out * self.config.routed_scaling_factor)

    def local_routes(self, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask routes that miss locally-owned experts and remap ids to 0-based local."""
        local_mask = (topk_idx >= self.local_expert_start) & (topk_idx < self.local_expert_end)
        local_idx = (topk_idx - self.local_expert_start).clamp(0, self.local_num_experts - 1)
        return local_idx, topk_weight * local_mask.to(dtype=topk_weight.dtype)

    @torch.no_grad()
    def copy_expert(self, expert_id: int, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, rank: int, world_size: int) -> None:
        """Copy one expert's HF weights into the appropriate local slot."""
        del rank, world_size
        if expert_id < self.local_expert_start or expert_id >= self.local_expert_end:
            return
        local_expert_id = expert_id - self.local_expert_start
        fc1_weight = _grouped_weight(self.linear_fc1, local_expert_id)
        # Concatenate gate+up along the output dim to match fc1's fused layout.
        fc1_weight.copy_(torch.cat((gate, up), dim=0).to(dtype=fc1_weight.dtype))
        fc2_weight = _grouped_weight(self.linear_fc2, local_expert_id)
        fc2_weight.copy_(down.to(dtype=fc2_weight.dtype))

    @torch.no_grad()
    def expert_weights(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Return per-local-expert (gate, up, down) views for fused-MoE prep."""
        gate_weights = []
        up_weights = []
        down_weights = []
        for expert_id in range(self.local_num_experts):
            fc1_weight = _grouped_weight(self.linear_fc1, expert_id).detach()
            # Split the fused gate||up tile back into the two halves.
            gate, up = fc1_weight.chunk(2, dim=0)
            gate_weights.append(gate)
            up_weights.append(up)
            down_weights.append(_grouped_weight(self.linear_fc2, expert_id).detach())
        return gate_weights, up_weights, down_weights

    @torch.no_grad()
    def full_expert_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Gather all experts onto DP rank 0 for checkpoint saving (None elsewhere)."""
        gate_weights, up_weights, down_weights = self.expert_weights()
        gate = _gather_expert_parallel_tensor(torch.stack(gate_weights, dim=0))
        up = _gather_expert_parallel_tensor(torch.stack(up_weights, dim=0))
        down = _gather_expert_parallel_tensor(torch.stack(down_weights, dim=0))
        if gate is None or up is None or down is None:
            return None
        return gate, up, down

    @torch.no_grad()
    def offload_to_cpu(self) -> None:
        """Move all expert params to CPU (used during inference-only phases)."""
        for param in self.parameters():
            param.data = param.data.to(device="cpu")

    @torch.no_grad()
    def onload_to_device(self, device: torch.device) -> None:
        """Restore params to the target device for training."""
        for param in self.parameters():
            if param.device != device:
                param.data = param.data.to(device=device)


def _build_grouped_linear(num_gemms: int, in_features: int, out_features: int, *, dtype: torch.dtype) -> nn.Module:
    return ArenoGroupedLinear(num_gemms, in_features, out_features, dtype=dtype)


class ArenoGroupedLinear(nn.Module):
    """Fused per-expert linear: ``(num_gemms, out, in)`` weight + per-expert
    token counts driving ``areno_grouped_linear``."""

    def __init__(self, num_gemms: int, in_features: int, out_features: int, *, dtype: torch.dtype):
        super().__init__()
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_gemms, out_features, in_features, dtype=dtype))

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor | Sequence[int]) -> torch.Tensor:
        if isinstance(tokens_per_expert, torch.Tensor):
            if tokens_per_expert.numel() != self.num_gemms:
                raise ValueError(f"expected {self.num_gemms} expert token counts, got {tokens_per_expert.numel()}")
            if x.shape[0] == 0:
                # Avoid invoking the kernel on an empty batch — return a fresh
                # empty tensor with the right out feature dim.
                return x.new_empty((0, self.out_features))
            return areno_grouped_linear(x.contiguous(), self.weight, tokens_per_expert)
        if len(tokens_per_expert) != self.num_gemms:
            raise ValueError(f"expected {self.num_gemms} expert token counts, got {len(tokens_per_expert)}")
        offset = sum(tokens_per_expert)
        if offset != x.shape[0]:
            raise ValueError(f"tokens_per_expert sums to {offset}, but input has {x.shape[0]} rows")
        if offset == 0:
            # Avoid invoking the kernel on an empty batch — return a fresh
            # empty tensor with the right out feature dim.
            return x.new_empty((0, self.out_features))
        return areno_grouped_linear(x.contiguous(), self.weight, tokens_per_expert)


def _grouped_linear_forward(module: nn.Module, x: torch.Tensor, tokens_per_expert: torch.Tensor | Sequence[int]) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = module(x, tokens_per_expert)
    if isinstance(out, tuple):
        return out
    return out, None


@torch._dynamo.disable
def _areno_silu_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_silu(x)


@torch._dynamo.disable
def _areno_sigmoid_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_sigmoid(x)


@torch._dynamo.disable
def _areno_grouped_topk_router_no_compile(
    logits: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
    num_groups: int,
    topk_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return areno_grouped_topk_router(logits, expert_bias, top_k, num_groups, topk_group)


@torch._dynamo.disable
def _areno_linear_no_compile(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return areno_linear(x, weight, None)


@torch._dynamo.disable
def _areno_silu_pair_no_compile(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return areno_silu_and_mul(torch.cat((gate, up), dim=-1))


@torch._dynamo.disable
def _areno_silu_and_mul_no_compile(x: torch.Tensor) -> torch.Tensor:
    return areno_silu_and_mul(x)


@torch._dynamo.disable
def _areno_moe_topk_permute_no_compile(
    flat: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    local_expert_start: int,
    local_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return areno_moe_topk_permute(flat, topk_idx, topk_weight, local_expert_start, local_num_experts)


@torch._dynamo.disable
def _areno_moe_unpermute_no_compile(
    expert_out: torch.Tensor,
    sorted_token_idx: torch.Tensor,
    *,
    merging_probs: None,
    restore_shape: tuple[int, int],
) -> torch.Tensor:
    del merging_probs
    return areno_moe_unpermute(expert_out, sorted_token_idx, restore_shape)


def _gather_expert_parallel_tensor(tensor: torch.Tensor) -> torch.Tensor | None:
    ctx = get_tp_context()
    if ctx.dp_rank != 0:
        return None
    local = tensor.detach().contiguous()
    if ctx.world_size == 1:
        return local.cpu()
    if ctx.rank == 0:
        chunks = [torch.empty_like(local) for _ in range(ctx.world_size)]
        dist.gather(local, gather_list=chunks, dst=ctx.dp_rank * ctx.world_size, group=ctx.group)
        return torch.cat(chunks, dim=0).cpu()
    dist.gather(local, dst=ctx.dp_rank * ctx.world_size, group=ctx.group)
    return None


def _grouped_weight(module: nn.Module, expert_id: int) -> torch.Tensor:
    weight = getattr(module, f"weight{expert_id}", None)
    if weight is not None:
        return weight
    weights = getattr(module, "weight", None)
    if isinstance(weights, torch.Tensor) and weights.dim() == 3:
        return weights[expert_id]
    if isinstance(weights, (list, tuple, nn.ParameterList)):
        return weights[expert_id]
    raise AttributeError(f"cannot find grouped expert weight for expert {expert_id}")


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


class BailingSoftmaxAttention(nn.Module):
    """Softmax attention layer with either GQA (fused QKV projection) or
    MLA-style low-rank KV pathway.

    When ``kv_lora_rank`` is set the layer follows the DeepSeek-V2-style MLA
    factorisation: Q is computed normally, K/V come from a low-rank
    compressed KV projection, RoPE is applied to a dedicated ``qk_rope_head_dim``
    slice while the rest of QK uses non-rope ``qk_nope_head_dim`` channels.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        # Head-dim split: rope vs non-rope channels on Q/K, plus separate V dim.
        self.qk_nope_head_dim = config.qk_nope_head_dim or config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim or int(config.head_dim * config.partial_rotary_factor)
        self.v_head_dim = config.v_head_dim or config.head_dim
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        # Per-rank head counts (TP sharded).
        self.local_heads = self.num_heads // ctx.world_size
        self.local_kv_heads = self.num_kv_heads // ctx.world_size
        self.kv_lora_rank = config.kv_lora_rank
        self.query_key_value = None
        self.q_proj = None
        self.kv_a_proj_with_mqa = None
        self.kv_a_layernorm = None
        self.kv_b_proj = None
        if self.kv_lora_rank is None:
            # Standard GQA: one fused QKV projection.
            self.num_qkv_heads = self.num_heads + 2 * self.num_kv_heads
            self.local_qkv_heads = self.local_heads + 2 * self.local_kv_heads
            self.query_key_value = ColumnParallelLinear(config.hidden_size, self.num_qkv_heads * config.head_dim, bias=config.qkv_bias)
            self.query_layernorm = RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else None
            self.key_layernorm = RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else None
        else:
            # MLA: Q has full per-head projection, KV goes through a shared
            # low-rank bottleneck (``kv_a_proj_with_mqa``) plus a per-head
            # decompression (``kv_b_proj``). The ``kv_a`` output also carries
            # the rope channels in its tail (``qk_rope_head_dim`` cols).
            self.q_proj = ColumnParallelLinear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
            mark_tensor_parallel_parameter(self.kv_a_proj_with_mqa.weight, False, sequence_parallel=True, tp_grad_allreduce=True)
            self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
            self.kv_b_proj = ColumnParallelLinear(config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
            self.query_layernorm = None
            self.key_layernorm = None
        self.dense = RowParallelLinear(self.num_heads * self.v_head_dim, config.hidden_size, bias=config.use_bias)
        # Rotary embedding applied only on the qk_rope_head_dim slice.
        self.rope = PartialRotaryEmbedding(
            self.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            1.0,
            is_neox_style=False,
        )
        self.train_backend = build_train_attention_backend()
        self.infer_backend: FlashAttnInferBackend | None = None
        # KV cache slots populated by the runtime at engine setup.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, train_meta: TrainMeta | None, infer_meta: InferMeta | None) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        q, k, v = self._project(hidden_states, position_ids)
        if infer_meta is not None:
            # Lazily build the inference backend so weight-only training paths
            # don't pay the cost.
            if self.infer_backend is None:
                self.infer_backend = build_infer_attention_backend()
            out = self.infer_backend(q, k, v, self.k_cache, self.v_cache, infer_meta)
        else:
            out = self.train_backend(q, k, v, train_meta)
        return self.dense(out.contiguous().view(bsz, seqlen, self.local_heads * self.v_head_dim))

    def _project(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = hidden_states.shape
        if self.kv_lora_rank is None:
            # Standard GQA path: split fused QKV, optional per-head norm, rope
            # on the trailing qk_rope_head_dim channels only.
            assert self.query_key_value is not None
            qkv = self.query_key_value(hidden_states).view(bsz, seqlen, self.local_heads + 2 * self.local_kv_heads, self.head_dim)
            q, k, v = qkv.split([self.local_heads, self.local_kv_heads, self.local_kv_heads], dim=-2)
            if self.query_layernorm is not None:
                q = self.query_layernorm(q)
                k = self.key_layernorm(k)
            q_rope, k_rope = self.rope(q[..., -self.qk_rope_head_dim :], k[..., -self.qk_rope_head_dim :], position_ids)
            return torch.cat((q[..., : self.qk_nope_head_dim], q_rope), dim=-1), torch.cat((k[..., : self.qk_nope_head_dim], k_rope), dim=-1), v

        # MLA path: split Q into nope/rope chunks, decompress KV from the
        # shared low-rank rep, then re-attach a broadcast rope channel.
        assert self.q_proj is not None and self.kv_a_proj_with_mqa is not None and self.kv_a_layernorm is not None and self.kv_b_proj is not None
        q = self.q_proj(hidden_states).view(bsz, seqlen, self.local_heads, self.head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_rope = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, seqlen, self.local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q_rope, k_rope = self.rope(q_rope, k_rope.unsqueeze(2), position_ids)
        # Single rope channel is broadcast over all heads (MQA-style).
        k_rope = k_rope.expand(-1, -1, self.local_heads, -1)
        return torch.cat((q_nope, q_rope), dim=-1), torch.cat((k_nope, k_rope), dim=-1), v

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache

    def clear_kv_cache(self) -> None:
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.infer_backend = None

    def reset_kv_cache(self) -> None:
        return None


class BailingLinearAttention(nn.Module):
    """Linear-attention layer using chunked lightning-attn (training) or
    ``seg_la`` recurrent kernels (inference).

    Each layer carries a per-head ALiBi-style decay slope plus an optional
    SiLU on QKV. A sigmoid-gated RMSNorm on the output (``g_norm`` /
    ``g_proj``) acts as the layer-output gating mechanism, mirroring the
    Lightning-Attention / Minimax linear attention paper recipe.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        ctx = get_tp_context()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.local_heads = self.num_heads // ctx.world_size
        # Linear attention treats Q/K/V symmetrically (same head count).
        self.num_kv_heads = self.num_heads
        self.num_qkv_heads = 3 * self.num_heads
        self.local_qkv_heads = 3 * self.local_heads
        self.head_dim = config.head_dim
        self.num_layers = config.num_hidden_layers
        self.scaling = self.head_dim**-0.5
        self.linear_scale = config.linear_scale
        self.linear_silu = config.linear_silu
        # Fused QKV projection. ``g_proj`` produces the output gate.
        self.query_key_value = ColumnParallelLinear(config.hidden_size, self.num_qkv_heads * self.head_dim, bias=config.qkv_bias)
        self.dense = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)
        self.g_proj = ColumnParallelLinear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.group_norm_size = config.group_norm_size
        if self.group_norm_size > 1:
            # Grouped RMSNorm + sigmoid gate fused into one kernel — used when
            # the model exposes a coarser head-group granularity.
            self.g_norm = GroupRMSNormSigmoidGate(
                self.num_heads * self.head_dim,
                self.group_norm_size,
                ctx.world_size,
                config.rms_norm_eps,
            )
        else:
            # Plain per-rank RMSNorm; the sigmoid gate is applied outside.
            self.g_norm = RMSNorm(
                self.local_heads * self.head_dim,
                config.rms_norm_eps,
                tensor_model_parallel=True,
                sequence_parallel=False,
                tp_grad_allreduce=False,
            )
        self.query_layernorm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.key_layernorm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.rope = PartialRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.partial_rotary_factor,
            is_neox_style=True,
        )
        if config.linear_backend != "seg_la":
            raise ValueError(f"BailingMoeV2_5ForCausalLM expects linear_backend='seg_la', got {config.linear_backend!r}")
        # Per-head decay slope (ALiBi-like, head- and layer-dependent) used as
        # the gating factor in the recurrent state update.
        self.register_buffer("slope", _build_slope_tensor(self.local_heads, config.num_attention_heads, layer_idx, config.num_hidden_layers, ctx.rank, ctx.world_size), persistent=False)
        # Recurrent state cache populated at engine setup; shape is
        # ``[num_slots, local_heads, head_dim, head_dim]``.
        self.state_cache = torch.tensor([])

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, train_meta: TrainMeta | None, infer_meta: InferMeta | None) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        if self.linear_silu:
            qkv = _areno_silu_no_compile(qkv)
        qkv = qkv.view(bsz, seqlen, 3 * self.local_heads, self.head_dim)
        q, k, v = qkv.split([self.local_heads, self.local_heads, self.local_heads], dim=-2)
        if self.query_layernorm is not None:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        q, k = self.rope(q, k, position_ids)
        if self.linear_scale:
            q = q * self.scaling
        if infer_meta is not None:
            out = self._forward_infer(q, k, v, infer_meta)
        else:
            out = self._forward_train(q, k, v, train_meta)
        out = out.to(hidden_states.dtype).reshape(bsz, seqlen, -1)
        gate = self.g_proj(hidden_states)
        if self.group_norm_size > 1:
            # Fused norm+sigmoid-gate kernel.
            out = self.g_norm(out, gate)
        else:
            out = self.g_norm(out) * _areno_sigmoid_no_compile(gate)
        return self.dense(out.to(hidden_states.dtype))

    def _forward_train(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, train_meta: TrainMeta | None) -> torch.Tensor:
        # Packed sequences need ``cu_seqlens`` so the chunked kernel can
        # respect document boundaries; otherwise fall back to dense path.
        if train_meta is None or not train_meta.packed or train_meta.cu_seqlens is None:
            return self._forward_full(q, k, v)
        cu_seqlens = train_meta.cu_seqlens.to(device=q.device, dtype=torch.int32)
        return self._forward_lightning(q, k, v, cu_seqlens=cu_seqlens)

    def _forward_full(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self._forward_lightning(q, k, v, cu_seqlens=None)

    def _forward_lightning(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
        log_once("chunk_lightning_attn", "using chunk lightning attention training kernel")
        k = k.to(dtype=q.dtype)
        v = v.to(dtype=q.dtype)
        # g_gamma = -slope feeds the gated decay; chunked impl runs an
        # efficient block-recurrent pass without materialising the full
        # attention matrix.
        out, _ = chunk_lightning_attn(
            q,
            k,
            v,
            g_gamma=-self.slope,
            layer_idx=self.layer_idx,
            num_layers=self.num_layers,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            head_first=False,
        )
        return out

    def _forward_infer(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, infer_meta: InferMeta) -> torch.Tensor:
        if infer_meta.block_table is None:
            raise RuntimeError("linear attention inference requires block_table")
        if self.state_cache.numel() == 0:
            raise RuntimeError("linear attention inference requires recurrent state cache")
        # Each request maps to one state-cache slot (first column of its block table).
        slots = infer_meta.block_table[:, 0].long()
        if infer_meta.mode == "decode":
            return self._forward_decode(q, k, v, slots)
        if infer_meta.mode == "prefill":
            if infer_meta.cu_seqlens is None:
                raise RuntimeError("linear attention prefill requires cu_seqlens")
            return self._forward_prefill(q, k, v, slots, infer_meta.cu_seqlens)
        raise ValueError(f"unsupported inference mode: {infer_meta.mode}")

    def _forward_decode(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        log_once("seg_la_decode", "using seg_la linear attention decode kernel")
        # Decode runs one token per request; flatten to (batch, heads, dim)
        # and supply unit-stride offsets and "scale=1" gating per request.
        q_flat = q.reshape(-1, self.local_heads, self.head_dim).contiguous()
        k_flat = k.reshape(-1, self.local_heads, self.head_dim).contiguous()
        v_flat = v.reshape(-1, self.local_heads, self.head_dim).contiguous()
        batch = q_flat.shape[0]
        q_offsets = torch.arange(batch + 1, device=q.device, dtype=torch.int32)
        # s_scales=True -> apply the previously-stored state for decode.
        s_scales = torch.ones(batch, device=q.device, dtype=torch.bool)
        out = self._forward_seg_la(q_flat, k_flat, v_flat, self.state_cache, slots.to(torch.int32), q_offsets, s_scales)
        return out.view_as(q)

    def _forward_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        q_flat = q.reshape(-1, self.local_heads, self.head_dim)
        k_flat = k.reshape(-1, self.local_heads, self.head_dim)
        v_flat = v.reshape(-1, self.local_heads, self.head_dim)
        log_once("seg_la_prefill", "using seg_la linear attention prefill kernel")
        batch = slots.numel()
        # Prefill starts from zero state, so s_scales=False (no prior state).
        s_scales = torch.zeros(batch, device=q.device, dtype=torch.bool)
        return self._forward_seg_la(
            q_flat.contiguous(),
            k_flat.contiguous(),
            v_flat.contiguous(),
            self.state_cache,
            slots.to(torch.int32),
            cu_seqlens.to(device=q.device, dtype=torch.int32),
            s_scales,
        ).view_as(q)

    def _forward_seg_la(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        q_offsets: torch.Tensor,
        s_scales: torch.Tensor,
    ) -> torch.Tensor:
        q_lengths = q_offsets.diff()
        meta = SegLaMeta(
            batch_size=int(slots.numel()),
            max_q_length=0,
            q_offsets=q_offsets,
            s_offsets=slots,
            q_lengths=q_lengths,
            s_scales=s_scales,
            mask=None,
        )
        return _seg_la_fwd_no_compile(
            q=q,
            k=k,
            v=v,
            s=state,
            decay_scales=self.slope.to(device=q.device, dtype=torch.float32),
            meta=meta,
        )

    def set_state_cache(self, state_cache: torch.Tensor) -> None:
        self.state_cache = state_cache

    def clear_kv_cache(self) -> None:
        self.state_cache = torch.tensor([])

    @torch.no_grad()
    def reset_kv_cache(self) -> None:
        # Zero the recurrent state but keep the buffer (re-allocation is
        # expensive on hot reload).
        if self.state_cache.numel() > 0:
            self.state_cache.zero_()


@torch._dynamo.disable
def _seg_la_fwd_no_compile(**kwargs) -> torch.Tensor:
    return seg_la_fwd(**kwargs)


@torch._dynamo.disable
def _areno_fused_experts_no_compile(*args, **kwargs) -> torch.Tensor:
    return areno_fused_experts(*args, **kwargs)


class BailingDecoderLayer(nn.Module):
    """One Bailing decoder layer: pre-norm attention (softmax or linear) +
    pre-norm MLP (dense or MoE).

    ``_is_softmax_layer`` decides which attention flavour this layer runs:
    every ``layer_group_size``-th layer (and the trailing tail) is softmax,
    the rest are linear. ``first_k_dense_replace`` controls how many leading
    layers use the dense SwiGLU MLP before MoE kicks in.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_layer_type = "attention" if _is_softmax_layer(config, layer_idx) else "linear_attention"
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = (
            BailingSoftmaxAttention(config, layer_idx)
            if self.attention_layer_type == "attention"
            else BailingLinearAttention(config, layer_idx)
        )
        # Dense MLP for the warmup layers, sparse MoE for the rest.
        self.mlp = (
            BailingSparseMoeBlock(config)
            if config.num_experts is not None and layer_idx >= config.first_k_dense_replace
            else BailingDenseMLP(config, config.intermediate_size)
        )

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, train_meta: TrainMeta | None, infer_meta: InferMeta | None) -> torch.Tensor:
        # Standard pre-norm residual: norm -> sublayer -> add.
        residual = hidden_states
        hidden_states = residual + self.attention(self.input_layernorm(hidden_states), position_ids, train_meta, infer_meta)
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class BailingMoeLinearV2ForCausalLM(nn.Module):
    """Top-level Bailing-MoE-Linear-V2 causal LM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([BailingDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None, train_meta: TrainMeta | None = None, infer_meta: InferMeta | None = None) -> CausalLMOutput:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.word_embeddings(input_ids)
        use_sequence_parallel = bool(train_meta is not None and train_meta.sequence_parallel)
        if use_sequence_parallel:
            # SP shards the sequence dim before entering the layer stack.
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        with sequence_parallel_region(use_sequence_parallel):
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_ids, train_meta, infer_meta)
            hidden_states = self.norm(hidden_states)
            return CausalLMOutput(logits_shard=self.lm_head(hidden_states), hidden_states=hidden_states)

    def set_kv_caches(self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Bind per-softmax-layer KV caches and pre-allocate per-linear-layer
        recurrent state cache.

        ``kv_caches`` lists KV pairs *only* for softmax layers (in order of
        appearance); linear layers get a fresh zero state of shape
        ``[num_slots, heads, head_dim, head_dim]`` sized from the first KV
        cache.
        """
        softmax_idx = 0
        for layer in self.layers:
            if isinstance(layer.attention, BailingSoftmaxAttention):
                layer.attention.set_kv_cache(*kv_caches[softmax_idx])
                softmax_idx += 1
            elif isinstance(layer.attention, BailingLinearAttention):
                num_slots = kv_caches[0][0].shape[0] if kv_caches else 1
                device = kv_caches[0][0].device if kv_caches else next(self.parameters()).device
                state = torch.zeros(
                    num_slots,
                    layer.attention.local_heads,
                    layer.attention.head_dim,
                    layer.attention.head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                layer.attention.set_state_cache(state)

    @torch.no_grad()
    def prepare_infer_weights(self) -> None:
        """Stack per-expert weights for fused-MoE inference on every MoE block."""
        for layer in self.layers:
            if isinstance(layer.mlp, BailingSparseMoeBlock):
                layer.mlp.prepare_infer_weights()

    @torch.no_grad()
    def clear_infer_weights(self) -> None:
        """Drop fused-MoE inference tiles to reclaim memory before training."""
        for layer in self.layers:
            if isinstance(layer.mlp, BailingSparseMoeBlock):
                layer.mlp.clear_infer_weights()

    @torch.no_grad()
    def offload_train_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer.mlp, BailingSparseMoeBlock):
                layer.mlp.experts.offload_to_cpu()

    @torch.no_grad()
    def onload_train_weights(self, device: torch.device) -> None:
        for layer in self.layers:
            if isinstance(layer.mlp, BailingSparseMoeBlock):
                layer.mlp.experts.onload_to_device(device)

    @torch.no_grad()
    def finalize_router_expert_bias(self, tp_group, dp_group) -> None:
        """Apply the per-step router-bias update on every MoE layer."""
        for layer in self.layers:
            if isinstance(layer.mlp, BailingSparseMoeBlock):
                layer.mlp.gate.finalize_expert_bias(tp_group, dp_group)

    def allocate_kv_caches(self, num_blocks: int, block_size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate paged KV caches — only for softmax-attention layers."""
        caches = []
        for layer in self.layers:
            if not isinstance(layer.attention, BailingSoftmaxAttention):
                continue
            k_cache = torch.empty(
                num_blocks,
                block_size,
                layer.attention.local_kv_heads,
                layer.attention.head_dim,
                device=device,
                dtype=self.config.dtype,
            )
            v_cache = torch.empty(
                num_blocks,
                block_size,
                layer.attention.local_kv_heads,
                layer.attention.head_dim,
                device=device,
                dtype=self.config.dtype,
            )
            caches.append((k_cache, v_cache))
        return caches

    def clear_kv_caches(self) -> None:
        for layer in self.layers:
            layer.attention.clear_kv_cache()

    @torch.no_grad()
    def reset_kv_caches(self) -> None:
        for layer in self.layers:
            layer.attention.reset_kv_cache()

    @torch.no_grad()
    def offload_kv_caches(self) -> None:
        for layer in self.layers:
            attn = layer.attention
            if isinstance(attn, BailingSoftmaxAttention):
                if attn.k_cache.numel() > 0:
                    attn.k_cache = attn.k_cache.to(device="cpu")
                if attn.v_cache.numel() > 0:
                    attn.v_cache = attn.v_cache.to(device="cpu")
                attn.infer_backend = None
            elif isinstance(attn, BailingLinearAttention):
                if attn.state_cache.numel() > 0:
                    attn.state_cache = attn.state_cache.to(device="cpu")

    @torch.no_grad()
    def onload_kv_caches(self, device: torch.device) -> bool:
        found = False
        for layer in self.layers:
            attn = layer.attention
            if isinstance(attn, BailingSoftmaxAttention):
                if attn.k_cache.numel() > 0:
                    found = True
                    if attn.k_cache.device != device:
                        attn.k_cache = attn.k_cache.to(device=device)
                if attn.v_cache.numel() > 0:
                    found = True
                    if attn.v_cache.device != device:
                        attn.v_cache = attn.v_cache.to(device=device)
            elif isinstance(attn, BailingLinearAttention):
                if attn.state_cache.numel() > 0:
                    found = True
                    if attn.state_cache.device != device:
                        attn.state_cache = attn.state_cache.to(device=device)
        return found


class BailingMoeLinearV2Adapter(ModelAdapter):
    """Model adapter binding HF Bailing-MoE-Linear-V2 checkpoints to the areno runtime."""

    name = "bailing_moe_linear_v2"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        architectures = hf_config.get("architectures") or []
        model_type = str(hf_config.get("model_type", "")).lower()
        return "BailingMoeV2_5ForCausalLM" in architectures or model_type == "bailing_moe_linear"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        """Build a ``ModelConfig`` from a Bailing HF config dict.

        Bailing uses several alternate key spellings (SGLang lineage); we
        accept either ``num_experts``/``n_routed_experts``, ``n_group``/
        ``moe_router_num_groups``, ``score_function``/``scoring_func`` etc.
        and validate that the routing setup matches what ``BailingGate``
        actually implements (sigmoid scoring, expert bias enabled, top-k
        renormalization on).
        """
        dtype = _parse_dtype(hf_config.get("torch_dtype") or hf_config.get("dtype"))
        num_heads = int(hf_config["num_attention_heads"])
        head_dim = int(hf_config.get("head_dim", hf_config["hidden_size"] // num_heads))
        rotary_dim = int(hf_config.get("rotary_dim", hf_config.get("qk_rope_head_dim", head_dim)))
        num_experts = hf_config.get("num_experts", hf_config.get("n_routed_experts"))
        moe_intermediate_size = int(hf_config.get("moe_intermediate_size", hf_config.get("intermediate_size", 0)))
        num_experts_per_tok = int(hf_config.get("num_experts_per_tok", hf_config.get("moe_router_topk", 1)))
        n_group = int(hf_config.get("n_group", hf_config.get("moe_router_num_groups", 1)))
        topk_group = int(hf_config.get("topk_group", hf_config.get("moe_router_group_topk", 1)))
        routed_scaling_factor = float(hf_config.get("routed_scaling_factor", hf_config.get("moe_router_topk_scaling_factor", 1.0)))
        shared_size = hf_config.get("moe_shared_expert_intermediate_size")
        num_shared_experts = hf_config.get("num_shared_experts")
        if shared_size is not None and num_shared_experts is None:
            # Derive shared-expert count from total intermediate size when only
            # the aggregate is given.
            num_shared_experts = max(1, int(shared_size) // max(1, moe_intermediate_size))
        linear_backend = str(hf_config.get("linear_backend", "seg_la")).lower()
        score_function = str(
            hf_config.get("score_function", hf_config.get("scoring_func", hf_config.get("moe_router_score_function", "sigmoid")))
        ).lower()
        topk_method = str(hf_config.get("topk_method", "noaux_tc")).lower()
        moe_router_enable_expert_bias = _parse_bool(hf_config.get("moe_router_enable_expert_bias"), True)
        norm_topk_prob = _parse_bool(hf_config.get("norm_topk_prob"), True)
        moe_router_dtype = _parse_dtype(hf_config.get("moe_router_dtype") or "fp32")
        if score_function != "sigmoid":
            raise ValueError(f"BailingMoeLinearV2 only supports score_function='sigmoid', got {score_function!r}")
        if not moe_router_enable_expert_bias:
            raise ValueError("BailingMoeLinearV2 requires moe_router_enable_expert_bias=True to match SGLang biased grouped topk")
        if not norm_topk_prob:
            raise ValueError("BailingMoeLinearV2 requires norm_topk_prob=True to match SGLang TopK renormalize")
        if topk_method not in {"noaux_tc", "group_limited_greedy"}:
            raise ValueError(f"unsupported BailingMoeLinearV2 topk_method={topk_method!r}")
        return ModelConfig(
            model_type=self.name,
            vocab_size=int(hf_config["vocab_size"]),
            hidden_size=int(hf_config["hidden_size"]),
            intermediate_size=int(hf_config["intermediate_size"]),
            num_hidden_layers=int(hf_config["num_hidden_layers"]),
            num_attention_heads=num_heads,
            num_key_value_heads=int(hf_config.get("num_key_value_heads", num_heads)),
            head_dim=head_dim,
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf_config.get("rope_theta", hf_config.get("rotary_base", 10000.0))),
            max_position_embeddings=int(hf_config.get("max_position_embeddings", 4096)),
            tie_word_embeddings=_parse_bool(hf_config.get("tie_word_embeddings"), False),
            qkv_bias=_parse_bool(hf_config.get("use_qkv_bias", hf_config.get("qkv_bias")), False),
            qk_norm=_parse_bool(hf_config.get("use_qk_norm"), True),
            dtype=dtype,
            hidden_act=str(hf_config.get("hidden_act", "silu")),
            use_bias=_parse_bool(hf_config.get("use_bias"), False),
            layer_group_size=int(hf_config.get("layer_group_size", 1)),
            partial_rotary_factor=float(hf_config.get("partial_rotary_factor", rotary_dim / head_dim)),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            first_k_dense_replace=int(hf_config.get("first_k_dense_replace", 0)),
            moe_intermediate_size=moe_intermediate_size,
            num_shared_experts=num_shared_experts,
            moe_router_enable_expert_bias=moe_router_enable_expert_bias,
            norm_topk_prob=norm_topk_prob,
            moe_router_dtype=moe_router_dtype,
            score_function=score_function,
            topk_method=topk_method,
            group_norm_size=int(hf_config.get("group_norm_size", hf_config.get("linear_attn_norm_group_size", hf_config.get("head_dim", 128)))),
            num_nextn_predict_layers=int(hf_config.get("num_nextn_predict_layers", 0)),
            mtp_loss_scaling_factor=float(hf_config.get("mtp_loss_scaling_factor", 0.0)),
            qk_nope_head_dim=int(hf_config.get("qk_nope_head_dim", head_dim)),
            qk_rope_head_dim=int(hf_config.get("qk_rope_head_dim", rotary_dim)),
            v_head_dim=int(hf_config.get("v_head_dim", head_dim)),
            kv_lora_rank=hf_config.get("kv_lora_rank"),
            linear_backend=linear_backend,
            linear_scale=linear_backend == "minimax",
            linear_silu=_parse_bool(hf_config.get("use_linear_silu", hf_config.get("linear_silu")), False),
            sequence_parallel=_parse_bool(hf_config.get("sequence_parallel"), True),
            moe_router_bias_update_rate=float(hf_config.get("moe_router_bias_update_rate", 0.0)),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        return BailingMoeLinearV2ForCausalLM(config)

    @torch.no_grad()
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        if not isinstance(model, BailingMoeLinearV2ForCausalLM):
            raise TypeError(f"BailingMoeLinearV2Adapter cannot load weights into {type(model)!r}")
        load_checkpoint_weights(model, model_path, CHECKPOINT_SPEC)

    @torch.no_grad()
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        if not isinstance(model, BailingMoeLinearV2ForCausalLM):
            raise TypeError(f"BailingMoeLinearV2Adapter cannot save weights from {type(model)!r}")
        return save_checkpoint_weights(model, output_path, source_path, CHECKPOINT_SPEC)


def _is_softmax_layer(config: ModelConfig, layer_idx: int) -> bool:
    """Return True iff this layer should run softmax attention.

    Bailing groups layers into chunks of ``layer_group_size``: the last layer
    of each group is softmax, the rest are linear. Any trailing layers that
    don't fit a full group (when num_hidden_layers isn't divisible by
    layer_group_size) are also forced to softmax so attention always anchors
    the sequence end.
    """
    return (layer_idx + 1) % config.layer_group_size == 0 or layer_idx >= config.num_hidden_layers // config.layer_group_size * config.layer_group_size


def _build_slope_tensor(
    local_heads: int,
    total_heads: int,
    layer_idx: int,
    num_hidden_layers: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Build the ALiBi-style decay slopes for one rank's local heads.

    The slope table follows the standard ALiBi recipe (geometric sequence
    starting at ``2^(-2^(-(log2(n)-3)))``), with a small per-layer linear
    decay so deeper layers attenuate slightly more slowly. The slice for
    this rank is the standard TP shard of the full per-head table.
    """
    def get_slopes(n: int) -> list[float]:
        def get_slopes_power_of_2(m: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(m) - 3)))
            return [start * start**i for i in range(m)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        # Non-power-of-two head counts: take the next-smaller power-of-two
        # slopes and interleave the doubled sequence to fill the rest.
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]

    if total_heads != local_heads * world_size:
        raise ValueError(f"total_heads={total_heads} must equal local_heads * world_size={local_heads * world_size}")
    start, end = _shard_range(total_heads, rank, world_size)
    # Layer scaling: deeper layers get a slightly smaller multiplier; the
    # +1e-5 floor keeps the zero-layer single-layer case finite.
    layer_scale = 1 + 1e-5 if num_hidden_layers <= 1 else 1 - layer_idx / (num_hidden_layers - 1) + 1e-5
    return torch.tensor(get_slopes(total_heads)[start:end], dtype=torch.float32) * layer_scale
