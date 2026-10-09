#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <algorithm>
#include <limits>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tensor_format.h"
#include "activation_launch.h"
#include "embedding_launch.h"
#include "fused_experts_launch.h"
#include "moe.h"

namespace areno_npu {
namespace {
int64_t tiles(int64_t size, int64_t tile) { return (size + tile - 1) / tile; }
uint32_t blocks(int64_t tasks, uint32_t cores) { return static_cast<uint32_t>(std::min<int64_t>(tasks, cores)); }

void check_tensor(const at::Tensor& tensor, const at::Tensor& input) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == input.device(),
                "fused_experts tensors must be on the same Ascend NPU device");
    TORCH_CHECK(is_base_format(tensor), "fused_experts requires base NPU storage format");
}

struct Platform { uint32_t cube, vector, workspace; };
const Platform& platform() {
    // Queried after NPUGuard initializes the current device. The extension is
    // compiled for one exact SoC; no route data or device tensor is read here.
    static const Platform value = [] {
        const char* soc = aclrtGetSocName();
        TORCH_CHECK(soc, "fused_experts could not query the Ascend SoC");
        auto* info = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
        TORCH_CHECK(info, "fused_experts could not query the CANN platform");
        Platform result{info->GetCoreNumAic(), info->GetCoreNumAiv(), info->GetLibApiWorkSpaceSize()};
        TORCH_CHECK(result.cube > 0 && result.vector > 0, "fused_experts requires Ascend Cube and Vector cores");
        return result;
    }();
    return value;
}

at::Tensor fused_experts(const at::Tensor& input, const at::Tensor& w1, const at::Tensor& w2,
    const at::Tensor& route_weight, const at::Tensor& route_ids, const pybind11::object& config,
    const std::string& activation) {
    for (const auto& tensor : {input, w1, w2, route_weight, route_ids}) check_tensor(tensor, input);
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "fused_experts requires FP16/BF16 hidden_states");
    TORCH_CHECK(w1.scalar_type() == input.scalar_type() && w2.scalar_type() == input.scalar_type(),
                "fused_experts expert weights must match input dtype");
    TORCH_CHECK(route_weight.scalar_type() == at::kFloat || route_weight.scalar_type() == at::kHalf
        || route_weight.scalar_type() == at::kBFloat16, "fused_experts routing weights must be FP32/FP16/BF16");
    TORCH_CHECK(at::isIntegralType(route_ids.scalar_type(), false), "fused_experts routing ids must be integral");
    TORCH_CHECK(input.dim() == 2 && w1.dim() == 3 && w2.dim() == 3 && route_ids.dim() == 2,
                "fused_experts expects input [T,H], w1 [E,2F,H], w2 [E,H,F], ids [T,K]");
    int64_t tokens = input.size(0), hidden = input.size(1), experts = w1.size(0), intermediate = w2.size(2);
    int64_t top_k = route_ids.size(1), routes = route_ids.numel();
    TORCH_CHECK(experts > 0 && hidden > 0 && intermediate > 0 && top_k > 0
        && w1.size(1) == 2 * intermediate && w1.size(2) == hidden && w2.size(0) == experts && w2.size(1) == hidden
        && route_ids.size(0) == tokens && route_ids.sizes() == route_weight.sizes(), "fused_experts shape mismatch");
    TORCH_CHECK(config.attr("num_experts").cast<int64_t>() == experts, "fused_experts config expert count mismatch");
    float scale = config.attr("routed_scaling_factor").cast<float>();
    Activation gate;
    if (activation == "silu") gate = SiluMul;
    else if (activation == "gelu_tanh") gate = GeluTanhMul;
    else throw pybind11::value_error("unsupported fused MoE activation " + activation);
    const c10_npu::NPUGuard guard(input.device());
    auto output = at::empty({tokens, hidden}, input.options());
    if (tokens == 0) return output;

    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(hidden <= limit - kExpertN && 2 * intermediate <= limit - kExpertN && experts < limit / kExpertM
        && routes <= limit - (experts + 1) * (kExpertM - 1) - (kExpertM - 1), "fused_experts dimensions exceed int32");
    const auto& hardware = platform();
    auto stream = c10_npu::getCurrentNPUStream(input.device().index()).stream(true);
    uint32_t storage = input.scalar_type() == at::kHalf ? 1 : 2;
    auto x = input.contiguous(), gate_weight = w1.contiguous(), down_weight = w2.contiguous();
    auto weights = route_weight.to(at::kFloat).contiguous(), ids = route_ids.to(at::kInt).contiguous();
    int64_t capacity = tiles(routes + (experts + 1) * (kExpertM - 1), kExpertM) * kExpertM;
    auto integer = input.options().dtype(at::kInt);
    auto aligned = at::empty({capacity}, integer), block_ids = at::empty({capacity / kExpertM}, integer);
    auto total = at::empty({1}, integer), scratch = at::empty({experts + 2}, integer);
    align(ids, experts + 1, kExpertM, aligned, block_ids, total, scratch, true);
    auto token_ids = at::empty({capacity}, input.options().dtype(at::kLong));
    launch_expert_tokens(blocks(tiles(capacity, kExpertVectorTile), hardware.vector), stream,
        aligned.const_data_ptr<int32_t>(), block_ids.const_data_ptr<int32_t>(), total.const_data_ptr<int32_t>(),
        token_ids.data_ptr<int64_t>(), capacity, routes, top_k);
    auto packed = at::empty({capacity, hidden}, input.options());
    launch_embedding(blocks(capacity * tiles(hidden, kEmbeddingTile), hardware.vector), stream, storage, false,
        token_ids.const_data_ptr<int64_t>(), x.const_data_ptr(), packed.data_ptr(), capacity, hidden, 0, tokens);

    // Both projections share FP32 accumulation storage and storage-dtype
    // cache. The activation and packed input have independent lifetimes.
    int64_t width = std::max(2 * intermediate, hidden);
    int64_t gate_stride = tiles(2 * intermediate, kExpertN) * kExpertN, down_stride = tiles(hidden, kExpertN) * kExpertN;
    auto accumulator = at::empty({capacity * std::max(gate_stride, down_stride)}, input.options().dtype(at::kFloat));
    auto cache = at::empty({capacity * width}, input.options());
    auto activated = at::empty({capacity, intermediate}, input.options());
    auto workspace = at::empty({std::max<int64_t>(hardware.workspace, 1)}, input.options().dtype(at::kByte));
    auto matmul = [&](const at::Tensor& a, const at::Tensor& b, int64_t n, int64_t k, int64_t stride) {
        auto result = accumulator.narrow(0, 0, capacity * stride);
        result.zero_(); // Skipped -1 blocks and unused capacity remain zero.
        // TorchNPU may queue ATen zero_ on its host dispatcher. Flush that
        // queue before launching directly onto the stream; this is not a
        // device synchronization or a route-count readback.
        auto launch_stream = c10_npu::getCurrentNPUStream(input.device().index()).stream(true);
        launch_expert_matmul(blocks(capacity / kExpertM * tiles(n, kExpertN), hardware.cube), launch_stream, storage,
            a.const_data_ptr(), b.const_data_ptr(), result.data_ptr<float>(), block_ids.const_data_ptr<int32_t>(),
            total.const_data_ptr<int32_t>(), capacity, n, k, stride, workspace.data_ptr());
    };
    matmul(packed, gate_weight, 2 * intermediate, hidden, gate_stride);
    launch_expert_cast(blocks(capacity * tiles(2 * intermediate, kExpertVectorTile), hardware.vector), stream,
        storage, accumulator.const_data_ptr<float>(), cache.data_ptr(), capacity, 2 * intermediate, gate_stride);
    launch_activation(blocks(capacity * tiles(intermediate, kActivationTile), hardware.vector), stream, storage,
        gate, activated.data_ptr(), cache.const_data_ptr(), nullptr, capacity, intermediate);
    matmul(activated, down_weight, hidden, intermediate, down_stride);
    launch_expert_weight_scatter(blocks(capacity * tiles(hidden, kExpertVectorTile), hardware.vector), stream,
        storage, accumulator.const_data_ptr<float>(), weights.const_data_ptr<float>(), aligned.const_data_ptr<int32_t>(),
        block_ids.const_data_ptr<int32_t>(), total.const_data_ptr<int32_t>(), cache.data_ptr(), capacity, routes, hidden, down_stride);
    launch_expert_reduce(blocks(tokens * tiles(hidden, kExpertVectorTile), hardware.vector), stream, storage,
        cache.const_data_ptr(), output.data_ptr(), tokens, hidden, top_k, scale);
    return output;
}
} // namespace
} // namespace areno_npu

void register_fused_experts(pybind11::module_& m) {
    m.def("areno_fused_experts", &areno_npu::fused_experts, pybind11::arg("hidden_states"), pybind11::arg("w1"),
        pybind11::arg("w2"), pybind11::arg("topk_weights"), pybind11::arg("topk_ids"), pybind11::arg("config"),
        pybind11::kw_only(), pybind11::arg("activation") = "silu");
}
