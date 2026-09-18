#pragma once

// Device files supply their compiler's function qualifier before including
// this header. Selection order is ordinary C++, shared with the CPU tests.
#ifndef ARENO_ROUTING_INLINE
#define ARENO_ROUTING_INLINE inline
#endif

namespace areno_accel {
namespace routing {
constexpr int kMaxExperts = 512;
constexpr int kMaxGroups = 64;
constexpr int kMaxTopK = 16;

ARENO_ROUTING_INLINE void insert_topk(float value, int index, float* values, int* indices, int k) {
  for (int pos = 0; pos < k; ++pos) {
    if (value > values[pos] || (value == values[pos] && index < indices[pos])) {
      for (int move = k - 1; move > pos; --move) {
        values[move] = values[move - 1];
        indices[move] = indices[move - 1];
      }
      values[pos] = value;
      indices[pos] = index;
      break;
    }
  }
}
} // namespace routing
} // namespace areno_accel
