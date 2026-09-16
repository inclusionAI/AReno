#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <algorithm>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "../moe_permute_common.h"
#include "embedding_launch.h"
#include "moe_launch.h"

namespace areno_npu {
namespace {
constexpr int64_t kMaxRoutes = std::numeric_limits<int32_t>::max();
uint32_t blocks(int64_t tasks) { return static_cast<uint32_t>(std::min<int64_t>(tasks, 32)); }
int64_t route_tiles(int64_t routes) { return (routes + kRouteTile - 1) / kRouteTile; }
int64_t expert_tiles(int64_t experts) { return (experts + kExpertTile - 1) / kExpertTile; }
void* stream(const at::Tensor& tensor) { return c10_npu::getCurrentNPUStream(tensor.device().index()).stream(true); }

void check_tensor(const at::Tensor& tensor, const at::Tensor& input, at::ScalarType dtype) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == input.device(),
                "MoE tensors must be on the same Ascend NPU device");
    TORCH_CHECK(tensor.scalar_type() == dtype, "MoE tensor dtype mismatch");
    TORCH_CHECK(tensor.is_contiguous(), "MoE native tensors must be contiguous");
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(tensor), "MoE requires base NPU storage format");
}

void check_input(const at::Tensor& input) {
    check_tensor(input, input, input.scalar_type());
    TORCH_CHECK(input.dim() == 2, "MoE input must be 2D");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "MoE supports FP32, FP16 or BF16 input");
}

struct Plan { at::Tensor partial, counts; };
Plan count_routes(const at::Tensor& keys, const at::Tensor& weights, RouteKind kind,
    int64_t columns, int64_t start, int64_t experts, uint32_t storage = 0) {
    TORCH_CHECK(keys.numel() <= kMaxRoutes, "MoE route count exceeds int32");
    TORCH_CHECK(experts >= 0 && experts <= kMaxRoutes && start <= std::numeric_limits<int64_t>::max() - experts,
                "MoE invalid expert range");
    auto options = keys.options().dtype(at::kInt);
    Plan plan{at::empty({route_tiles(keys.numel()), experts}, options), at::zeros({experts}, options)};
    int64_t tasks = route_tiles(keys.numel()) * expert_tiles(experts);
    if (tasks > 0) launch_route_count(blocks(tasks), stream(keys), kind, storage, keys.const_data_ptr(),
        weights.defined() ? weights.const_data_ptr<float>() : nullptr, plan.partial.data_ptr<int32_t>(),
        plan.counts.data_ptr<int32_t>(), keys.numel(), columns, start, experts);
    return plan;
}

void plan_prefix(Plan& plan, const at::Tensor& offsets) {
    if (plan.partial.numel() == 0) return;
    launch_route_prefix(blocks(expert_tiles(plan.counts.numel())), stream(offsets), plan.partial.data_ptr<int32_t>(),
        offsets.const_data_ptr<int64_t>(), plan.partial.size(0), plan.counts.numel());
}

void transfer(const at::Tensor& input, const at::Tensor& ids, at::Tensor& output, bool scatter) {
    if (ids.numel() == 0 || input.size(1) == 0 || output.numel() == 0) return;
    int64_t hidden = input.size(1), tasks = ids.numel() * ((hidden - 1) / kEmbeddingTile + 1);
    uint32_t storage = input.scalar_type() == at::kFloat ? 0 : input.scalar_type() == at::kHalf ? 1 : 2;
    // Embedding's row gather/scatter is the same operation. Reuse its exact
    // DMA transfers and storage-dtype atomics rather than another copy kernel.
    launch_embedding(blocks(tasks), stream(input), storage, scatter, ids.const_data_ptr<int64_t>(),
        input.const_data_ptr(), output.data_ptr(), ids.numel(), hidden, 0, scatter ? output.size(0) : input.size(0));
}

at::Tensor unpermute(const at::Tensor& input, const at::Tensor& ids, int64_t tokens, int64_t hidden) {
    check_input(input);
    check_tensor(ids, input, at::kLong);
    TORCH_CHECK(ids.dim() == 1 && ids.numel() == input.size(0) && hidden == input.size(1) && tokens >= 0,
                "MoE unpermute shape mismatch");
    const c10_npu::NPUGuard guard(input.device());
    auto output = at::zeros({tokens, hidden}, input.options());
    transfer(input, ids, output, true);
    return output;
}

at::Tensor gather(const at::Tensor& input, const at::Tensor& ids) {
    check_input(input);
    check_tensor(ids, input, at::kLong);
    TORCH_CHECK(ids.dim() == 1, "MoE token_index must be 1D");
    const c10_npu::NPUGuard guard(input.device());
    auto output = at::empty({ids.numel(), input.size(1)}, input.options());
    transfer(input, ids, output, false);
    return output;
}

std::vector<at::Tensor> permute(const at::Tensor& input, const at::Tensor& probs,
    const at::Tensor& map, int64_t rows) {
    check_input(input);
    check_tensor(probs, input, at::kFloat);
    check_tensor(map, input, at::kBool);
    TORCH_CHECK(probs.dim() == 2 && probs.sizes() == map.sizes() && probs.size(0) == input.size(0), "MoE routing map shape mismatch");
    TORCH_CHECK(rows >= 0 && rows <= map.numel(), "MoE invalid num_out_tokens");
    const c10_npu::NPUGuard guard(input.device());
    auto plan = count_routes(map, probs, DenseRoutes, probs.size(1), 0, probs.size(1));
    auto offsets = at::empty({probs.size(1) + 1}, input.options().dtype(at::kLong));
    launch_route_offsets(stream(input), plan.counts.const_data_ptr<int32_t>(), offsets.data_ptr<int64_t>(),
        probs.size(1), 1, nullptr, nullptr, nullptr);
    plan_prefix(plan, offsets);
    auto output = at::empty({rows, input.size(1)}, input.options());
    auto weight = at::empty({rows}, input.options().dtype(at::kFloat));
    auto ids = at::empty({rows}, input.options().dtype(at::kLong));
    int64_t tasks = route_tiles(map.numel()) * expert_tiles(probs.size(1));
    if (rows && tasks) launch_route_metadata(blocks(tasks), stream(input), DenseRoutes, 0, map.const_data_ptr(),
        probs.const_data_ptr<float>(), plan.partial.const_data_ptr<int32_t>(), weight.data_ptr<float>(),
        ids.data_ptr<int64_t>(), nullptr, nullptr, map.numel(), probs.size(1), 0, probs.size(1), rows);
    transfer(input, ids, output, false);
    return {output, weight, ids};
}

std::vector<at::Tensor> topk_permute(const at::Tensor& input, const at::Tensor& indices,
    const at::Tensor& weight, int64_t start, int64_t experts) {
    check_input(input);
    check_tensor(indices, input, at::kLong);
    check_tensor(weight, input, at::kFloat);
    TORCH_CHECK(indices.dim() == 2 && indices.sizes() == weight.sizes() && indices.size(0) == input.size(0),
                "MoE top-k shape mismatch");
    TORCH_CHECK(start >= 0, "MoE local expert start must be non-negative");
    const c10_npu::NPUGuard guard(input.device());
    auto plan = count_routes(indices, weight, TopKRoutes, indices.size(1), start, experts);
    auto buffers = areno_accel::moe::allocate_topk(input, plan.counts);
    if (buffers.output.size(0) > 0) {
        plan_prefix(plan, buffers.offsets);
        int64_t tasks = route_tiles(indices.numel()) * expert_tiles(experts);
        launch_route_metadata(blocks(tasks), stream(input), TopKRoutes, 0, indices.const_data_ptr(),
            weight.const_data_ptr<float>(), plan.partial.const_data_ptr<int32_t>(), buffers.weight.data_ptr<float>(),
            buffers.token_index.data_ptr<int64_t>(), buffers.position.data_ptr<int32_t>(), nullptr,
            indices.numel(), indices.size(1), start, experts, buffers.output.size(0));
        transfer(input, buffers.token_index, buffers.output, false);
    }
    return buffers.result();
}

at::Tensor weight_backward(const at::Tensor& grad, const at::Tensor& ids, const at::Tensor& positions,
    int64_t tokens, int64_t top_k) {
    check_tensor(grad, grad, at::kFloat);
    check_tensor(ids, grad, at::kLong);
    check_tensor(positions, grad, at::kInt);
    TORCH_CHECK(grad.dim() == 1 && ids.sizes() == grad.sizes() && positions.sizes() == grad.sizes()
        && tokens >= 0 && top_k >= 0 && (grad.numel() == 0 || (tokens > 0 && top_k > 0)), "MoE weight gradient shape mismatch");
    const c10_npu::NPUGuard guard(grad.device());
    auto output = at::zeros({tokens, top_k}, grad.options());
    if (grad.numel() > 0) launch_route_weight_backward(blocks(route_tiles(grad.numel())), stream(grad),
        grad.const_data_ptr<float>(), ids.const_data_ptr<int64_t>(), positions.const_data_ptr<int32_t>(),
        output.data_ptr<float>(), grad.numel(), top_k);
    return output;
}

void align(const at::Tensor& ids, int64_t experts, int64_t block_size, at::Tensor routed,
    at::Tensor block_ids, at::Tensor total, at::Tensor scratch, bool initialize) {
    check_tensor(ids, ids, ids.scalar_type());
    uint32_t storage;
    switch (ids.scalar_type()) {
        case at::kLong: storage = 0; break;
        case at::kInt: storage = 1; break;
        case at::kShort: storage = 2; break;
        case at::kChar: storage = 3; break;
        case at::kByte: storage = 4; break;
        default: TORCH_CHECK(false, "MoE alignment ids must be integral");
    }
    for (const auto& tensor : {routed, block_ids, total, scratch}) check_tensor(tensor, ids, at::kInt);
    TORCH_CHECK(experts > 0 && experts <= kMaxRoutes && block_size > 0 && block_size <= kMaxRoutes
        && ids.numel() <= kMaxRoutes && (block_size == 1 || experts <= (kMaxRoutes - ids.numel()) / (block_size - 1)),
        "MoE alignment dimensions exceed int32");
    TORCH_CHECK(routed.dim() == 1 && routed.numel() >= ids.numel() && routed.numel() <= kMaxRoutes
        && block_ids.dim() == 1 && block_ids.numel() >= (routed.numel() + block_size - 1) / block_size
        && total.numel() == 1 && scratch.dim() == 1 && scratch.numel() >= experts + 1, "MoE alignment buffer shape mismatch");
    const c10_npu::NPUGuard guard(ids.device());
    auto plan = count_routes(ids, {}, AlignRoutes, 0, -1, experts, storage);
    auto offsets = at::empty({experts + 1}, ids.options().dtype(at::kLong));
    if (initialize && routed.numel() > 0) launch_route_fill(blocks(route_tiles(routed.numel())), stream(ids),
        routed.data_ptr<int32_t>(), routed.numel(), static_cast<int32_t>(ids.numel()));
    launch_route_offsets(stream(ids), plan.counts.const_data_ptr<int32_t>(), offsets.data_ptr<int64_t>(), experts,
        block_size, block_ids.data_ptr<int32_t>(), total.data_ptr<int32_t>(), scratch.data_ptr<int32_t>(), block_ids.numel());
    plan_prefix(plan, offsets);
    int64_t tasks = route_tiles(ids.numel()) * expert_tiles(experts);
    if (tasks) launch_route_metadata(blocks(tasks), stream(ids), AlignRoutes, storage, ids.const_data_ptr(), nullptr,
        plan.partial.const_data_ptr<int32_t>(), nullptr, nullptr, nullptr, routed.data_ptr<int32_t>(),
        ids.numel(), 0, -1, experts, routed.numel());
}
} // namespace
} // namespace areno_npu

void register_moe(pybind11::module_& m) {
    m.def("areno_moe_permute_forward", &areno_npu::permute);
    m.def("areno_moe_unpermute_forward", &areno_npu::unpermute);
    m.def("areno_moe_gather_by_token_index", &areno_npu::gather);
    m.def("areno_moe_topk_permute_forward", &areno_npu::topk_permute);
    m.def("areno_moe_topk_weight_backward", &areno_npu::weight_backward);
    m.def("areno_moe_align", &areno_npu::align);
}
