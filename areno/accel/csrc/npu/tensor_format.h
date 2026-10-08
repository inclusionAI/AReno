#pragma once

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

namespace areno_npu {

inline bool is_base_format(const at::Tensor& tensor) {
    // FormatHelper is internal to TorchNPU and is not exported by its wheels.
    // Use the public query while preserving its base-format classification.
    const auto format = at_npu::native::get_npu_format(tensor);
    return format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
           format == ACL_FORMAT_NHWC || format == ACL_FORMAT_NCDHW;
}

} // namespace areno_npu
