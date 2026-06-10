#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace areno_accel {

template <typename scalar_t>
__device__ inline void atomic_add(scalar_t* address, scalar_t value) {
  atomicAdd(address, value);
}

template <>
__device__ inline void atomic_add<c10::Half>(c10::Half* address, c10::Half value) {
  atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(value));
}

template <>
__device__ inline void atomic_add<c10::BFloat16>(c10::BFloat16* address, c10::BFloat16 value) {
  atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), static_cast<__nv_bfloat16>(value));
}

}  // namespace areno_accel
