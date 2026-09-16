#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <algorithm>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "conv_launch.h"

namespace areno_npu {
namespace {
void check_tensor(const at::Tensor& tensor, const at::Tensor& input, at::ScalarType dtype) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == input.device(),
                "conv tensors must be on the same Ascend NPU device");
    TORCH_CHECK(tensor.scalar_type() == dtype, "conv tensor dtype mismatch");
    TORCH_CHECK(tensor.is_contiguous(), "conv native tensors must be contiguous");
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(tensor), "conv requires base NPU storage format");
}

void check_input(const at::Tensor& input, const at::Tensor& weight, bool decode = false) {
    check_tensor(input, input, input.scalar_type());
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "conv supports FP32, FP16 or BF16 input");
    check_tensor(weight, input, at::kFloat);
    TORCH_CHECK(input.dim() == (decode ? 2 : 3), "conv input rank mismatch");
    TORCH_CHECK(weight.dim() == 3 && weight.size(1) == 1 && weight.size(2) > 0,
                "conv weight must have shape (channels, 1, positive kernel size)");
    TORCH_CHECK(input.size(-1) == weight.size(0), "conv channel mismatch");
    TORCH_CHECK(static_cast<uint64_t>(weight.size(2) - 1) <= std::numeric_limits<uint32_t>::max() / sizeof(float),
                "conv kernel stride exceeds the Ascend DMA limit");
}

void check_segments(const at::Tensor& input, const at::Tensor& cu) {
    check_tensor(cu, input, at::kInt);
    TORCH_CHECK(input.size(0) == 1, "packed conv input must have shape (1, tokens, channels)");
    TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2, "packed conv cu_seqlens must be 1D with at least two entries");
    TORCH_CHECK(input.size(1) <= std::numeric_limits<int32_t>::max(), "packed conv token count exceeds int32");
    // Like CUDA, callers supply nondecreasing offsets beginning at zero and
    // ending at the token count. Read them on device, without a host sync.
}

void run(ConvOp op, const at::Tensor& input, const at::Tensor& weight, const at::Tensor& grad,
         at::Tensor& preact, at::Tensor& output, const at::Tensor& history, const at::Tensor& cu) {
    if (output.numel() == 0) return;
    int64_t channels = input.size(-1), batch = input.size(0), seqlen = op == ConvDecode ? 1 : input.size(1);
    int64_t tiles = (channels - 1) / kConvTile + 1;
    int64_t tasks = op == ConvWeightGrad ? weight.size(2) * tiles : batch * seqlen * tiles;
    uint32_t storage = input.scalar_type() == at::kFloat ? 0 : input.scalar_type() == at::kHalf ? 1 : 2;
    auto stream = c10_npu::getCurrentNPUStream(input.device().index()).stream(true);
    launch_conv(static_cast<uint32_t>(std::min<int64_t>(tasks, 32)), stream, storage, op, cu.defined(),
        input.const_data_ptr(), weight.const_data_ptr<float>(), grad.defined() ? grad.const_data_ptr() : nullptr,
        preact.data_ptr<float>(), output.data_ptr(), history.defined() ? history.const_data_ptr() : nullptr,
        cu.defined() ? cu.const_data_ptr<int32_t>() : nullptr, batch, seqlen, channels, weight.size(2),
        cu.defined() ? cu.numel() - 1 : batch);
}

std::vector<at::Tensor> forward(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& cu = {}) {
    check_input(input, weight);
    if (cu.defined()) check_segments(input, cu);
    const c10_npu::NPUGuard guard(input.device());
    auto output = at::empty(input.sizes(), input.options());
    auto preact = at::empty(input.sizes(), input.options().dtype(at::kFloat));
    run(ConvForward, input, weight, {}, preact, output, {}, cu);
    return {output, preact};
}

std::vector<at::Tensor> backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
                                at::Tensor preact, const at::Tensor& cu = {}) {
    check_input(input, weight);
    if (cu.defined()) check_segments(input, cu);
    check_tensor(grad, input, input.scalar_type());
    check_tensor(preact, input, at::kFloat);
    TORCH_CHECK(grad.sizes() == input.sizes() && preact.sizes() == input.sizes(), "conv gradient or preact shape mismatch");
    const c10_npu::NPUGuard guard(input.device());
    auto dx = at::empty(input.sizes(), input.options());
    auto dw = at::empty(weight.sizes(), weight.options());
    run(ConvInputGrad, input, weight, grad, preact, dx, {}, cu);
    run(ConvWeightGrad, input, weight, grad, preact, dw, {}, cu);
    return {dx, dw};
}

std::vector<at::Tensor> decode(const at::Tensor& input, const at::Tensor& history, const at::Tensor& weight) {
    check_input(input, weight, true);
    check_tensor(history, input, input.scalar_type());
    TORCH_CHECK(history.dim() == 3 && history.size(0) == input.size(0) && history.size(1) == input.size(1)
        && history.size(2) == weight.size(2) - 1, "conv history shape mismatch");
    const c10_npu::NPUGuard guard(input.device());
    auto output = at::empty(input.sizes(), input.options());
    auto preact = at::empty(input.sizes(), input.options().dtype(at::kFloat));
    run(ConvDecode, input, weight, {}, preact, output, history, {});
    return {output, preact};
}
} // namespace
} // namespace areno_npu

void register_conv(pybind11::module_& m) {
    m.def("areno_depthwise_causal_conv1d_silu_forward", [](const at::Tensor& x, const at::Tensor& w) {
        return areno_npu::forward(x, w);
    });
    m.def("areno_depthwise_causal_conv1d_silu_backward", [](const at::Tensor& g, const at::Tensor& x,
        const at::Tensor& w, const at::Tensor& p) { return areno_npu::backward(g, x, w, p); });
    m.def("areno_packed_depthwise_causal_conv1d_silu_forward", [](const at::Tensor& x, const at::Tensor& w,
        const at::Tensor& cu) { return areno_npu::forward(x, w, cu); });
    m.def("areno_packed_depthwise_causal_conv1d_silu_backward", [](const at::Tensor& g, const at::Tensor& x,
        const at::Tensor& w, const at::Tensor& cu, const at::Tensor& p) { return areno_npu::backward(g, x, w, p, cu); });
    m.def("areno_depthwise_causal_conv1d_silu_decode", &areno_npu::decode);
}
