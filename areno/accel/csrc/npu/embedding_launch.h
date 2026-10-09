#pragma once
#include <cstdint>

namespace areno_npu {
constexpr uint32_t kEmbeddingTile = 1024;
void launch_embedding(uint32_t blocks, void* stream, uint32_t storage, bool backward,
                       const int64_t* ids, const void* input, void* output,
                       int64_t tokens, int64_t hidden, int64_t vocab_start, int64_t vocab_end);
} // namespace areno_npu
