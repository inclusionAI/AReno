#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <algorithm>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "attention_launch.h"

namespace areno_npu {
namespace {
void check_tensor(const at::Tensor& tensor, const at::Tensor& q, at::ScalarType dtype) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == q.device(),
                "attention tensors must be on the same Ascend NPU device");
    TORCH_CHECK(tensor.scalar_type() == dtype, "attention tensor dtype mismatch");
    TORCH_CHECK(tensor.is_contiguous(), "attention native tensors must be contiguous");
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(tensor), "attention requires base NPU storage format");
}
void check_qkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "attention supports FP32, FP16 or BF16");
    for (const auto& tensor : {q, k, v}) check_tensor(tensor, q, q.scalar_type());
    TORCH_CHECK(k.sizes() == v.sizes(), "attention key/value shape mismatch");
}
AttentionShape dense_shape(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    int64_t start, int64_t window) {
    check_qkv(q, k, v);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.size(0) == k.size(0) && q.size(1) == k.size(1)
        && q.size(3) == k.size(3) && q.size(1) > 0 && q.size(3) > 0, "attention dense shape mismatch");
    TORCH_CHECK(start >= 0 && q.size(2) <= k.size(2) && start <= k.size(2) - q.size(2),
                "attention query positions exceed key length");
    return {q.numel() / q.size(3), q.size(1), k.size(1), q.size(3), q.size(2), k.size(2), 0,
            start, window, 0, 0, 1};
}
AttentionShape packed_shape(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& boundaries, int64_t window) {
    check_qkv(q, k, v);
    check_tensor(boundaries, q, at::kInt);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && q.size(0) == k.size(0) && q.size(2) == k.size(2)
        && q.size(1) > 0 && k.size(1) > 0 && q.size(1) % k.size(1) == 0 && q.size(2) > 0,
        "attention packed shape/head mismatch");
    TORCH_CHECK(boundaries.dim() == 1 && boundaries.numel() >= 2 && q.size(0) <= std::numeric_limits<int32_t>::max(),
                "attention requires int32 packed boundaries spanning all tokens");
    return {q.numel() / q.size(2), q.size(1), k.size(1), q.size(2), q.size(0), k.size(0), boundaries.numel() - 1,
            0, window, 0, 0, 1};
}
uint32_t storage(const at::Tensor& q) { return q.scalar_type() == at::kFloat ? 0 : q.scalar_type() == at::kHalf ? 1 : 2; }
void* stream(const at::Tensor& q) { return c10_npu::getCurrentNPUStream(q.device().index()).stream(true); }
uint32_t blocks(AttentionShape s, bool split = false) {
    int64_t tiles = (s.dim - 1) / kAttentionTile + 1;
    // Only the small launch grid needs clipping; the device walks all tasks.
    if (s.rows >= 32 || tiles >= 32 || (split && s.splits >= 32)) return 32;
    return static_cast<uint32_t>(std::min<int64_t>(s.rows * tiles * (split ? s.splits : 1), 32));
}
at::Tensor forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& boundaries, AttentionShape s, AttentionLayout layout, double scale) {
    const c10_npu::NPUGuard guard(q.device());
    auto output = at::empty(q.sizes(), q.options());
    if (s.rows) launch_attention(blocks(s), stream(q), storage(q), layout, false,
        q.const_data_ptr(), k.const_data_ptr(), v.const_data_ptr(), nullptr, nullptr,
        boundaries.defined() ? boundaries.const_data_ptr<int32_t>() : nullptr, nullptr, nullptr,
        output.data_ptr(), nullptr, nullptr, nullptr, nullptr, s, static_cast<float>(scale));
    return output;
}
std::vector<at::Tensor> backward(const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& saved, const at::Tensor& boundaries,
    AttentionShape s, AttentionLayout layout, double scale) {
    check_tensor(grad, q, q.scalar_type());
    check_tensor(saved, q, q.scalar_type());
    TORCH_CHECK(grad.sizes() == q.sizes() && saved.sizes() == q.sizes(), "attention grad/out shape mismatch");
    const c10_npu::NPUGuard guard(q.device());
    auto options = q.options().dtype(at::kFloat);
    auto dq = at::empty(q.sizes(), options), dk = at::zeros(k.sizes(), options), dv = at::zeros(v.sizes(), options);
    if (s.rows) launch_attention(blocks(s), stream(q), storage(q), layout, true,
        q.const_data_ptr(), k.const_data_ptr(), v.const_data_ptr(), grad.const_data_ptr(), saved.const_data_ptr(),
        boundaries.defined() ? boundaries.const_data_ptr<int32_t>() : nullptr, nullptr, nullptr,
        dq.data_ptr(), dk.data_ptr<float>(), dv.data_ptr<float>(), nullptr, nullptr, s, static_cast<float>(scale));
    // CUDA also accumulates and returns native gradients in FP32. The shared
    // Python autograd wrapper casts each result to its input's storage dtype.
    return {dq, dk, dv};
}
at::Tensor paged(const at::Tensor& q, const at::Tensor& ku, const at::Tensor& vu, at::Tensor kc, at::Tensor vc,
    const at::Tensor& table, const at::Tensor& lengths, int64_t window, int64_t splits, double scale) {
    check_qkv(q, kc, vc);
    for (const auto& tensor : {ku, vu}) check_tensor(tensor, q, q.scalar_type());
    for (const auto& tensor : {table, lengths}) check_tensor(tensor, q, at::kInt);
    TORCH_CHECK(q.dim() == 3 && ku.dim() == 3 && vu.sizes() == ku.sizes() && kc.dim() == 4
        && table.dim() == 2 && lengths.dim() == 1, "attention paged tensor rank/shape mismatch");
    TORCH_CHECK(q.size(0) == ku.size(0) && q.size(0) == table.size(0) && q.size(0) == lengths.size(0)
        && q.size(2) == ku.size(2) && q.size(2) == kc.size(3) && ku.size(1) == kc.size(2)
        && q.size(1) > 0 && kc.size(2) > 0 && q.size(1) % kc.size(2) == 0 && q.size(2) > 0
        && kc.size(1) > 0 && (q.size(0) == 0 || (kc.size(0) > 0 && table.size(1) > 0)),
        "attention paged shape/head mismatch");
    TORCH_CHECK(splits >= 1 && splits <= std::numeric_limits<int32_t>::max(), "attention num_splits must be positive int32");
    // Values of lengths/table are device metadata, as in CUDA. Callers must
    // provide valid pages and distinct cache slots for simultaneous updates.
    AttentionShape s{q.numel() / q.size(2), q.size(1), kc.size(2), q.size(2), 1, 0, 0,
                     0, window, kc.size(1), table.size(1), splits};
    const c10_npu::NPUGuard guard(q.device());
    auto output = at::empty(q.sizes(), q.options());
    if (!s.rows) return output;
    auto options = q.options().dtype(at::kFloat);
    auto stats = at::empty({s.rows, splits, 2}, options);
    auto acc = at::empty({s.rows, splits, s.dim}, options);
    launch_attention_cache_update(blocks(s), stream(q), storage(q), ku.const_data_ptr(), vu.const_data_ptr(),
        kc.data_ptr(), vc.data_ptr(), table.const_data_ptr<int32_t>(), lengths.const_data_ptr<int32_t>(), q.size(0), s);
    launch_attention(blocks(s, true), stream(q), storage(q), PagedAttention, false,
        q.const_data_ptr(), kc.const_data_ptr(), vc.const_data_ptr(), nullptr, nullptr, nullptr,
        table.const_data_ptr<int32_t>(), lengths.const_data_ptr<int32_t>(), output.data_ptr(), nullptr, nullptr,
        stats.data_ptr<float>(), acc.data_ptr<float>(), s, static_cast<float>(scale));
    launch_attention_split_reduce(blocks(s), stream(q), storage(q), stats.const_data_ptr<float>(),
        acc.const_data_ptr<float>(), output.data_ptr(), s);
    return output;
}
} // namespace
} // namespace areno_npu

void register_attention(pybind11::module_& m) {
    using namespace areno_npu;
    m.def("areno_causal_attention_forward", [](const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
        int64_t start, int64_t window, double scale) {
        return forward(q, k, v, {}, dense_shape(q, k, v, start, window), DenseAttention, scale);
    });
    m.def("areno_causal_attention_backward", [](const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k,
        const at::Tensor& v, const at::Tensor& saved, int64_t start, int64_t window, double scale) {
        return backward(grad, q, k, v, saved, {}, dense_shape(q, k, v, start, window), DenseAttention, scale);
    });
    m.def("areno_varlen_causal_attention_forward", [](const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
        const at::Tensor& cu, int64_t window, double scale) {
        return forward(q, k, v, cu, packed_shape(q, k, v, cu, window), PackedAttention, scale);
    });
    m.def("areno_varlen_causal_attention_backward", [](const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k,
        const at::Tensor& v, const at::Tensor& saved, const at::Tensor& cu, int64_t window, double scale) {
        return backward(grad, q, k, v, saved, cu, packed_shape(q, k, v, cu, window), PackedAttention, scale);
    });
    m.def("areno_paged_causal_attention_decode_forward", &paged);
}
