#include <torch/csrc/utils/pybind.h>
#include "moe_permute_common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allocate_topk", [](const at::Tensor& input, const at::Tensor& counts) {
        auto buffers = areno_accel::moe::allocate_topk(input, counts);
        auto result = buffers.result();
        result.push_back(buffers.offsets);
        return result;
    });
}
