#pragma once
#include <cstdint>

namespace areno_npu {
enum RoutingOp : uint32_t { TopKForward, TopKBackward, GroupedRouter };
void launch_routing(uint32_t blocks, void* stream, uint32_t storage, RoutingOp op,
    const void* logits, const float* bias, int64_t* indices, float* weights, void* grad_logits,
    int64_t tokens, int experts, int top_k, bool renormalize, int groups = 1, int topk_group = 1);
} // namespace areno_npu
