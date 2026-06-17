#include <torch/extension.h>

void areno_silu_and_mul_cuda(torch::Tensor out, torch::Tensor input);
void areno_gelu_tanh_and_mul_cuda(torch::Tensor out, torch::Tensor input);
torch::Tensor areno_silu_cuda(torch::Tensor input);
torch::Tensor areno_sigmoid_cuda(torch::Tensor input);
torch::Tensor areno_softplus_cuda(torch::Tensor input);
void areno_d_silu_and_mul_cuda(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input);
void areno_d_gelu_tanh_and_mul_cuda(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input);
torch::Tensor areno_d_silu_cuda(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor areno_d_sigmoid_cuda(torch::Tensor grad_output, torch::Tensor output);
torch::Tensor areno_d_softplus_cuda(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor areno_linear_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, bool use_bias);
std::vector<torch::Tensor> areno_linear_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    bool use_bias);
torch::Tensor areno_causal_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t query_start,
    int64_t window_left,
    double softmax_scale);
std::vector<torch::Tensor> areno_causal_attention_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    int64_t query_start,
    int64_t window_left,
    double softmax_scale);
torch::Tensor areno_varlen_causal_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens,
    int64_t window_left,
    double softmax_scale);
std::vector<torch::Tensor> areno_varlen_causal_attention_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor cu_seqlens,
    int64_t window_left,
    double softmax_scale);
torch::Tensor areno_paged_causal_attention_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_update,
    torch::Tensor v_update,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor cache_seqlens,
    int64_t window_left,
    int64_t num_splits,
    double softmax_scale);
torch::Tensor areno_grouped_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<int64_t> tokens_per_expert);
torch::Tensor areno_grouped_linear_forward_counts_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor tokens_per_expert);
std::vector<torch::Tensor> areno_grouped_linear_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<int64_t> tokens_per_expert);
std::vector<torch::Tensor> areno_grouped_linear_backward_counts_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor tokens_per_expert);
std::vector<torch::Tensor> areno_depthwise_causal_conv1d_silu_forward_cuda(torch::Tensor input, torch::Tensor weight);
std::vector<torch::Tensor> areno_depthwise_causal_conv1d_silu_decode_cuda(
    torch::Tensor current,
    torch::Tensor history,
    torch::Tensor weight);
std::vector<torch::Tensor> areno_depthwise_causal_conv1d_silu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor preact);
std::vector<torch::Tensor> areno_packed_depthwise_causal_conv1d_silu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor cu_seqlens);
std::vector<torch::Tensor> areno_packed_depthwise_causal_conv1d_silu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor cu_seqlens,
    torch::Tensor preact);
std::vector<torch::Tensor> areno_grouped_topk_router_cuda(
    torch::Tensor logits,
    torch::Tensor expert_bias,
    int64_t top_k,
    int64_t num_groups,
    int64_t topk_group);
std::vector<torch::Tensor> areno_topk_softmax_forward_cuda(torch::Tensor logits, int64_t top_k, bool renormalize);
torch::Tensor areno_topk_softmax_backward_cuda(
    torch::Tensor grad_topk_weight,
    torch::Tensor logits,
    torch::Tensor topk_idx,
    bool renormalize);
torch::Tensor areno_vocab_embedding_forward_cuda(torch::Tensor input_ids, torch::Tensor weight, int64_t vocab_start, int64_t vocab_end);
torch::Tensor areno_vocab_embedding_backward_cuda(torch::Tensor grad_output, torch::Tensor input_ids, torch::Tensor weight, int64_t vocab_start, int64_t vocab_end);
std::vector<torch::Tensor> areno_moe_permute_forward_cuda(torch::Tensor input, torch::Tensor probs, torch::Tensor routing_map, int64_t num_out_tokens);
torch::Tensor areno_moe_unpermute_forward_cuda(torch::Tensor input, torch::Tensor token_index, int64_t tokens, int64_t hidden);
torch::Tensor areno_moe_gather_by_token_index_cuda(torch::Tensor input, torch::Tensor token_index);
std::vector<torch::Tensor> areno_moe_topk_permute_forward_cuda(
    torch::Tensor input,
    torch::Tensor topk_idx,
    torch::Tensor topk_weight,
    int64_t local_expert_start,
    int64_t local_num_experts);
torch::Tensor areno_moe_topk_weight_backward_cuda(
    torch::Tensor grad_route_weight,
    torch::Tensor token_index,
    torch::Tensor topk_position,
    int64_t tokens,
    int64_t top_k);
std::vector<torch::Tensor> areno_rmsnorm_forward_cuda(torch::Tensor input, torch::Tensor weight, double eps);
std::vector<torch::Tensor> areno_rmsnorm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_rms);
std::vector<torch::Tensor> areno_optional_scale_rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    bool use_scale);
std::vector<torch::Tensor> areno_rmsnorm_silu_gate_forward_cuda(
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight,
    double eps);
std::vector<torch::Tensor> areno_optional_scale_rmsnorm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_rms,
    bool use_scale);
std::vector<torch::Tensor> areno_rmsnorm_silu_gate_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor inv_rms);
void areno_moe_align_cuda(
    torch::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor cumsum_buffer,
    bool pad_sorted_token_ids);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("areno_silu_and_mul", &areno_silu_and_mul_cuda, "ARENO SiLU and multiply");
  m.def("areno_gelu_tanh_and_mul", &areno_gelu_tanh_and_mul_cuda, "ARENO tanh GELU and multiply");
  m.def("areno_silu", &areno_silu_cuda, "ARENO SiLU");
  m.def("areno_sigmoid", &areno_sigmoid_cuda, "ARENO sigmoid");
  m.def("areno_softplus", &areno_softplus_cuda, "ARENO softplus");
  m.def("areno_d_silu_and_mul", &areno_d_silu_and_mul_cuda, "ARENO SiLU and multiply backward");
  m.def("areno_d_gelu_tanh_and_mul", &areno_d_gelu_tanh_and_mul_cuda, "ARENO tanh GELU and multiply backward");
  m.def("areno_d_silu", &areno_d_silu_cuda, "ARENO SiLU backward");
  m.def("areno_d_sigmoid", &areno_d_sigmoid_cuda, "ARENO sigmoid backward");
  m.def("areno_d_softplus", &areno_d_softplus_cuda, "ARENO softplus backward");
  m.def("areno_linear_forward", &areno_linear_forward_cuda, "ARENO linear forward");
  m.def("areno_linear_backward", &areno_linear_backward_cuda, "ARENO linear backward");
  m.def("areno_causal_attention_forward", &areno_causal_attention_forward_cuda, "ARENO causal attention forward");
  m.def("areno_causal_attention_backward", &areno_causal_attention_backward_cuda, "ARENO causal attention backward");
  m.def("areno_varlen_causal_attention_forward", &areno_varlen_causal_attention_forward_cuda, "ARENO varlen causal attention forward");
  m.def("areno_varlen_causal_attention_backward", &areno_varlen_causal_attention_backward_cuda, "ARENO varlen causal attention backward");
  m.def("areno_paged_causal_attention_decode_forward", &areno_paged_causal_attention_decode_forward_cuda, "ARENO paged causal attention decode forward");
  m.def("areno_grouped_linear_forward", &areno_grouped_linear_forward_cuda, "ARENO grouped linear forward");
  m.def("areno_grouped_linear_forward_counts", &areno_grouped_linear_forward_counts_cuda, "ARENO grouped linear forward with GPU counts");
  m.def("areno_grouped_linear_backward", &areno_grouped_linear_backward_cuda, "ARENO grouped linear backward");
  m.def("areno_grouped_linear_backward_counts", &areno_grouped_linear_backward_counts_cuda, "ARENO grouped linear backward with GPU counts");
  m.def("areno_depthwise_causal_conv1d_silu_forward", &areno_depthwise_causal_conv1d_silu_forward_cuda, "ARENO depthwise causal conv1d SiLU forward");
  m.def("areno_depthwise_causal_conv1d_silu_decode", &areno_depthwise_causal_conv1d_silu_decode_cuda, "ARENO depthwise causal conv1d SiLU decode");
  m.def("areno_depthwise_causal_conv1d_silu_backward", &areno_depthwise_causal_conv1d_silu_backward_cuda, "ARENO depthwise causal conv1d SiLU backward");
  m.def("areno_packed_depthwise_causal_conv1d_silu_forward", &areno_packed_depthwise_causal_conv1d_silu_forward_cuda, "ARENO packed depthwise causal conv1d SiLU forward");
  m.def("areno_packed_depthwise_causal_conv1d_silu_backward", &areno_packed_depthwise_causal_conv1d_silu_backward_cuda, "ARENO packed depthwise causal conv1d SiLU backward");
  m.def("areno_grouped_topk_router", &areno_grouped_topk_router_cuda, "ARENO grouped top-k router");
  m.def("areno_topk_softmax_forward", &areno_topk_softmax_forward_cuda, "ARENO softmax top-k router forward");
  m.def("areno_topk_softmax_backward", &areno_topk_softmax_backward_cuda, "ARENO softmax top-k router backward");
  m.def("areno_vocab_embedding_forward", &areno_vocab_embedding_forward_cuda, "ARENO vocab embedding forward");
  m.def("areno_vocab_embedding_backward", &areno_vocab_embedding_backward_cuda, "ARENO vocab embedding backward");
  m.def("areno_moe_permute_forward", &areno_moe_permute_forward_cuda, "ARENO MoE permute forward");
  m.def("areno_moe_unpermute_forward", &areno_moe_unpermute_forward_cuda, "ARENO MoE unpermute forward");
  m.def("areno_moe_gather_by_token_index", &areno_moe_gather_by_token_index_cuda, "ARENO MoE gather by token index");
  m.def("areno_moe_topk_permute_forward", &areno_moe_topk_permute_forward_cuda, "ARENO MoE top-k permute forward");
  m.def("areno_moe_topk_weight_backward", &areno_moe_topk_weight_backward_cuda, "ARENO MoE top-k route weight backward");
  m.def("areno_rmsnorm_forward", &areno_rmsnorm_forward_cuda, "ARENO RMSNorm forward");
  m.def("areno_rmsnorm_backward", &areno_rmsnorm_backward_cuda, "ARENO RMSNorm backward");
  m.def("areno_optional_scale_rmsnorm_forward", &areno_optional_scale_rmsnorm_forward_cuda, "ARENO optional-scale RMSNorm forward");
  m.def("areno_optional_scale_rmsnorm_backward", &areno_optional_scale_rmsnorm_backward_cuda, "ARENO optional-scale RMSNorm backward");
  m.def("areno_rmsnorm_silu_gate_forward", &areno_rmsnorm_silu_gate_forward_cuda, "ARENO RMSNorm SiLU gate forward");
  m.def("areno_rmsnorm_silu_gate_backward", &areno_rmsnorm_silu_gate_backward_cuda, "ARENO RMSNorm SiLU gate backward");
  m.def("areno_moe_align", &areno_moe_align_cuda, "ARENO MoE align block size");
}
