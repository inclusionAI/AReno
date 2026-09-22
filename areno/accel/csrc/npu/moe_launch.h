#pragma once
#include <cstdint>

namespace areno_npu {
constexpr int64_t kRouteTile = 256;
constexpr int64_t kExpertTile = 256;
enum RouteKind : uint32_t { TopKRoutes, DenseRoutes, AlignRoutes };

void launch_route_count(uint32_t blocks, void* stream, RouteKind kind, uint32_t id_storage,
    const void* keys, const float* weights, int32_t* partial, int32_t* counts,
    int64_t routes, int64_t columns, int64_t expert_start, int64_t experts);
void launch_route_offsets(void* stream, const int32_t* counts, int64_t* offsets, int64_t experts,
    int64_t block_size, int32_t* block_experts, int32_t* total, int32_t* scratch, int64_t block_capacity = 0);
void launch_route_prefix(uint32_t blocks, void* stream, int32_t* partial, const int64_t* offsets,
    int64_t route_tiles, int64_t experts);
void launch_route_metadata(uint32_t blocks, void* stream, RouteKind kind, uint32_t id_storage,
    const void* keys, const float* weights, const int32_t* partial, float* route_weight,
    int64_t* token_index, int32_t* position, int32_t* aligned_routes,
    int64_t routes, int64_t columns, int64_t expert_start, int64_t experts, int64_t capacity);
void launch_route_weight_backward(uint32_t blocks, void* stream, const float* grad,
    const int64_t* tokens, const int32_t* positions, float* output, int64_t rows, int64_t top_k);
void launch_route_fill(uint32_t blocks, void* stream, int32_t* output, int64_t elements, int32_t value);
} // namespace areno_npu
