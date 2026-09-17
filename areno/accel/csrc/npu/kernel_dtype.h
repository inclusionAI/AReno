#pragma once
#include <cstdint>
#include "kernel_operator.h"

namespace areno_npu {

// CANN derives host stub specializations from device object symbols. The host
// compiler does not know the device's half/__bf16 types, so only integer dtype
// IDs may cross that boundary. Resolve the actual type inside each kernel.
// Keep the IDs consistent with the public launch APIs: FP32=0, FP16=1, BF16=2.
template <uint32_t Storage> struct KernelDtype;
template <> struct KernelDtype<0> { using type = float; };
template <> struct KernelDtype<1> { using type = half; };
template <> struct KernelDtype<2> { using type = bfloat16_t; };

template <typename T> struct KernelDtypeId;
template <> struct KernelDtypeId<float> { static constexpr uint32_t value = 0; };
template <> struct KernelDtypeId<half> { static constexpr uint32_t value = 1; };
template <> struct KernelDtypeId<bfloat16_t> { static constexpr uint32_t value = 2; };

} // namespace areno_npu
