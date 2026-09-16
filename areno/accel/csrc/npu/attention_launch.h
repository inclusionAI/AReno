#pragma once
#include <cstdint>

namespace areno_npu {
constexpr int64_t kAttentionTile = 512;
enum AttentionLayout : uint32_t { DenseAttention, PackedAttention, PagedAttention };
struct AttentionShape {
    int64_t rows, q_heads, kv_heads, dim, q_len, k_len, sequences;
    int64_t query_start, window_left, block_size, max_blocks, splits;
};

void launch_attention(uint32_t blocks, void* stream, uint32_t storage, AttentionLayout layout, bool backward,
    const void* q, const void* k, const void* v, const void* grad, const void* saved,
    const int32_t* boundaries, const int32_t* table, const int32_t* lengths,
    void* output, float* dk, float* dv, float* split_stats, float* split_acc, AttentionShape shape, float scale);
void launch_attention_cache_update(uint32_t blocks, void* stream, uint32_t storage,
    const void* k, const void* v, void* k_cache, void* v_cache, const int32_t* table, const int32_t* lengths,
    int64_t batch, AttentionShape shape);
void launch_attention_split_reduce(uint32_t blocks, void* stream, uint32_t storage,
    const float* stats, const float* acc, void* output, AttentionShape shape);
} // namespace areno_npu
