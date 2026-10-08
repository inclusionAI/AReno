#pragma once
#include <ATen/ATen.h>

namespace areno_npu {
// Shared by the public alignment entry and fused expert evaluation. Counts
// and offsets stay on the device; capacity is a shape-only upper bound.
void align(const at::Tensor& ids, int64_t experts, int64_t block_size, at::Tensor routed,
    at::Tensor block_ids, at::Tensor total, at::Tensor scratch, bool initialize);
} // namespace areno_npu
