#include <ATen/ATen.h>
#include <limits>
#include <torch/csrc/utils/pybind.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "optimizer_launch.h"

namespace areno_npu {
namespace {
void check_tensor(const at::Tensor& x, const at::Tensor& model) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.device() == model.device(),
                "AdamW tensors must be on the same Ascend NPU device");
    TORCH_CHECK(x.is_contiguous(), "AdamW tensors must be contiguous");
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(x), "AdamW requires base NPU storage format");
}

void check_model(const at::Tensor& model, const at::Tensor& grad) {
    check_tensor(model, model);
    check_tensor(grad, model);
    TORCH_CHECK((model.scalar_type() == at::kFloat || model.scalar_type() == at::kBFloat16) &&
                (grad.scalar_type() == at::kFloat || grad.scalar_type() == at::kBFloat16),
                "AdamW model and gradient must be BF16 or FP32");
    TORCH_CHECK(model.numel() == grad.numel(), "AdamW model and gradient lengths must match");
}

void fp32_step(at::Tensor model, const at::Tensor& grad, at::Tensor moment, at::Tensor variance,
                at::Tensor low, at::Tensor carries, int64_t offset, bool master,
                double beta1, double beta2, double lr, double decay, double eps,
                double step_size, double bias_sqrt) {
    check_model(model, grad);
    check_tensor(moment, model);
    check_tensor(variance, model);
    TORCH_CHECK(moment.scalar_type() == at::kFloat && variance.scalar_type() == at::kFloat,
                "AdamW moments must be FP32");
    TORCH_CHECK(moment.numel() == variance.numel(), "AdamW moment lengths must match");
    TORCH_CHECK(offset >= 0 && offset <= moment.numel() && model.numel() <= moment.numel() - offset,
                "AdamW state slice is out of bounds");
    if (master) {
        check_tensor(low, model);
        check_tensor(carries, model);
        TORCH_CHECK(low.scalar_type() == at::kUInt16 && carries.scalar_type() == at::kByte,
                    "compact master metadata must use uint16 low bits and uint8 packed carries");
        TORCH_CHECK(low.numel() == moment.numel() && carries.numel() >= (low.numel() + 7) / 8,
                    "compact master metadata lengths do not match state");
    } else TORCH_CHECK(model.numel() == moment.numel(), "AdamW FP32 state length must match model");
    if (model.numel() == 0) return;
    const c10_npu::NPUGuard guard(model.device());
    auto stream = c10_npu::getCurrentNPUStream(model.device().index()).stream(true);
    const bool compact = master && model.scalar_type() == at::kBFloat16;
    // Tile boundaries are in bucket coordinates, so no two blocks modify a
    // shared carry byte even when the model slice starts inside that byte.
    const int64_t tiles = (offset % kAdamTile + model.numel() + kAdamTile - 1) / kAdamTile;
    launch_adamw_fp32(static_cast<uint32_t>(std::min<int64_t>(tiles, 32)), stream,
        model.scalar_type() == at::kBFloat16, grad.scalar_type() == at::kBFloat16, compact,
        model.data_ptr(), grad.const_data_ptr(), compact ? low.data_ptr<uint16_t>() : nullptr,
        compact ? carries.data_ptr<uint8_t>() : nullptr, moment.data_ptr<float>(), variance.data_ptr<float>(),
        model.numel(), offset, beta1, beta2, lr, decay, eps, step_size, bias_sqrt);
}

void check_slice(int64_t offset, int64_t size, int64_t capacity) {
    TORCH_CHECK(offset >= 0 && offset <= capacity && size <= capacity - offset,
                "quantized AdamW state slice is out of bounds");
}

void quantized_step(at::Tensor model, const at::Tensor& grad, at::Tensor moment, at::Tensor moment_scale,
    at::Tensor variance, at::Tensor variance_scale, const at::Tensor& signed_map, const at::Tensor& unsigned_map,
    int64_t moment_offset, int64_t moment_scale_offset, int64_t variance_offset, int64_t variance_scale_offset,
    int64_t block_size, bool four_bit, double beta1, double beta2, double lr, double decay,
    double eps, double step_size, double bias_sqrt) {
    check_model(model, grad);
    for (const auto& tensor : {moment, moment_scale, variance, variance_scale}) check_tensor(tensor, model);
    TORCH_CHECK(moment.scalar_type() == at::kByte && variance.scalar_type() == at::kByte,
                "quantized AdamW codes must use uint8");
    TORCH_CHECK(moment_scale.scalar_type() == at::kFloat && variance_scale.scalar_type() == at::kFloat,
                "quantized AdamW scales must be FP32");
    if (four_bit) {
        TORCH_CHECK(block_size >= 32 && block_size <= 1024 && (block_size & (block_size - 1)) == 0,
                    "AdamW4bit block size must be a power of two in [32, 1024]");
    } else {
        TORCH_CHECK(block_size >= 1 && block_size <= 4096, "AdamW8bit block size must be in [1, 4096]");
        for (const auto& map : {signed_map, unsigned_map}) {
            check_tensor(map, model);
            TORCH_CHECK(map.scalar_type() == at::kFloat && map.numel() == 256,
                        "AdamW8bit codebooks must have 256 FP32 entries");
        }
    }
    const int64_t n = model.numel(), codes = four_bit ? (n + 1) / 2 : n;
    const int64_t count = (n + block_size - 1) / block_size;
    check_slice(moment_offset, codes, moment.numel());
    check_slice(variance_offset, codes, variance.numel());
    check_slice(moment_scale_offset, count, moment_scale.numel());
    check_slice(variance_scale_offset, count, variance_scale.numel());
    if (!four_bit) {
        TORCH_CHECK(moment.numel() == n && variance.numel() == n && moment_scale.numel() == count &&
                    variance_scale.numel() == count, "AdamW8bit state lengths must match quantization blocks");
    }
    if (n == 0) return;
    const c10_npu::NPUGuard guard(model.device());
    auto stream = c10_npu::getCurrentNPUStream(model.device().index()).stream(true);
    launch_adamw_quantized(static_cast<uint32_t>(std::min<int64_t>(count, 32)), stream,
        model.scalar_type() == at::kBFloat16, grad.scalar_type() == at::kBFloat16, four_bit,
        model.data_ptr(), grad.const_data_ptr(), moment.data_ptr<uint8_t>(), moment_scale.data_ptr<float>(),
        variance.data_ptr<uint8_t>(), variance_scale.data_ptr<float>(),
        four_bit ? nullptr : signed_map.const_data_ptr<float>(), four_bit ? nullptr : unsigned_map.const_data_ptr<float>(),
        n, moment_offset, moment_scale_offset, variance_offset, variance_scale_offset,
        block_size, beta1, beta2, lr, decay, eps, step_size, bias_sqrt);
}

void check_factors(const at::Tensor& grad, const at::Tensor& factors, const at::Tensor& invalid,
                    int64_t start, int64_t rows, int64_t columns) {
    check_tensor(grad, grad);
    check_tensor(factors, grad);
    check_tensor(invalid, grad);
    TORCH_CHECK(grad.scalar_type() == at::kFloat || grad.scalar_type() == at::kBFloat16,
                "AdamW4bit factored gradient must be BF16 or FP32");
    TORCH_CHECK(factors.scalar_type() == at::kFloat, "AdamW4bit factors must be FP32");
    TORCH_CHECK(invalid.scalar_type() == at::kInt && invalid.numel() == 1,
                "AdamW4bit invalid flag must be one int32 value");
    TORCH_CHECK(rows > 0 && columns > 0 && rows <= std::numeric_limits<int64_t>::max() / columns &&
                rows <= std::numeric_limits<int64_t>::max() - columns, "AdamW4bit factored shape is invalid");
    TORCH_CHECK(factors.numel() == rows + columns, "AdamW4bit factor count must match matrix shape");
    check_slice(start, grad.numel(), rows * columns);
}

void factored_stats(const at::Tensor& grad, at::Tensor sums, at::Tensor invalid,
                     int64_t start, int64_t rows, int64_t columns) {
    check_factors(grad, sums, invalid, start, rows, columns);
    if (grad.numel() == 0) return;
    const c10_npu::NPUGuard guard(grad.device());
    auto stream = c10_npu::getCurrentNPUStream(grad.device().index()).stream(true);
    const int64_t column_tiles = (columns - 1) / kAdamTile + 1;
    const int64_t end = start + grad.numel() - 1;
    const int64_t first = start / columns * column_tiles + start % columns / kAdamTile;
    const int64_t last = end / columns * column_tiles + end % columns / kAdamTile;
    launch_adamw_factored_stats(static_cast<uint32_t>(std::min<int64_t>(last - first + 1, 32)), stream,
        grad.scalar_type() == at::kBFloat16, grad.const_data_ptr(), sums.data_ptr<float>(), invalid.data_ptr<int32_t>(),
        grad.numel(), start, rows, columns);
}

void factored_step(at::Tensor model, const at::Tensor& grad, at::Tensor moment, at::Tensor moment_scale,
    const at::Tensor& factors, const at::Tensor& row_mean, const at::Tensor& invalid,
    int64_t moment_offset, int64_t moment_scale_offset, int64_t start, int64_t block_size,
    int64_t rows, int64_t columns, double beta1, double lr, double decay, double eps, double step, double bias) {
    check_model(model, grad);
    check_factors(grad, factors, invalid, start, rows, columns);
    for (const auto& tensor : {moment, moment_scale, row_mean}) check_tensor(tensor, model);
    TORCH_CHECK(moment.scalar_type() == at::kByte, "AdamW4bit packed momentum must use uint8");
    TORCH_CHECK(moment_scale.scalar_type() == at::kFloat && row_mean.scalar_type() == at::kFloat && row_mean.numel() == 1,
                "AdamW4bit scales and single row mean must be FP32");
    TORCH_CHECK(block_size >= 32 && block_size <= 1024 && (block_size & (block_size - 1)) == 0,
                "AdamW4bit block size must be a power of two in [32, 1024]");
    const int64_t count = (model.numel() + block_size - 1) / block_size;
    check_slice(moment_offset, (model.numel() + 1) / 2, moment.numel());
    check_slice(moment_scale_offset, count, moment_scale.numel());
    if (model.numel() == 0) return;
    const c10_npu::NPUGuard guard(model.device());
    auto stream = c10_npu::getCurrentNPUStream(model.device().index()).stream(true);
    launch_adamw_factored_step(static_cast<uint32_t>(std::min<int64_t>(count, 32)), stream,
        model.scalar_type() == at::kBFloat16, grad.scalar_type() == at::kBFloat16,
        model.data_ptr(), grad.const_data_ptr(), moment.data_ptr<uint8_t>(), moment_scale.data_ptr<float>(),
        factors.const_data_ptr<float>(), row_mean.const_data_ptr<float>(), invalid.const_data_ptr<int32_t>(),
        model.numel(), moment_offset, moment_scale_offset, start, rows, columns, block_size, beta1, lr, decay, eps, step, bias);
}
} // namespace
} // namespace areno_npu

void register_optimizer(pybind11::module_& m) {
    using namespace areno_npu;
    m.def("areno_adamw_4bit_factored_stats", &factored_stats);
    m.def("areno_adamw_4bit_factored_step", &factored_step);
    m.def("areno_adamw_fp32_state_step", [](at::Tensor model, const at::Tensor& grad,
        at::Tensor moment, at::Tensor variance, double beta1, double beta2, double lr, double decay,
        double eps, double step_size, double bias_sqrt) {
        fp32_step(model, grad, moment, variance, {}, {}, 0, false,
                  beta1, beta2, lr, decay, eps, step_size, bias_sqrt);
    });
    m.def("areno_adamw_fp32_master_step", [](at::Tensor model, at::Tensor low, at::Tensor carries,
        const at::Tensor& grad, at::Tensor moment, at::Tensor variance, int64_t offset,
        double beta1, double beta2, double lr, double decay, double eps, double step_size, double bias_sqrt) {
        fp32_step(model, grad, moment, variance, low, carries, offset, true,
                  beta1, beta2, lr, decay, eps, step_size, bias_sqrt);
    });
    m.def("areno_adamw_8bit_step", [](at::Tensor model, const at::Tensor& grad,
        at::Tensor moment, at::Tensor moment_scale, at::Tensor variance, at::Tensor variance_scale,
        const at::Tensor& signed_map, const at::Tensor& unsigned_map, int64_t block_size,
        double beta1, double beta2, double lr, double decay, double eps, double step_size, double bias_sqrt) {
        quantized_step(model, grad, moment, moment_scale, variance, variance_scale, signed_map, unsigned_map,
            0, 0, 0, 0, block_size, false, beta1, beta2, lr, decay, eps, step_size, bias_sqrt);
    });
    m.def("areno_adamw_4bit_step", [](at::Tensor model, const at::Tensor& grad,
        at::Tensor moment, at::Tensor moment_scale, at::Tensor variance, at::Tensor variance_scale,
        int64_t moment_offset, int64_t moment_scale_offset, int64_t variance_offset, int64_t variance_scale_offset,
        int64_t block_size, double beta1, double beta2, double lr, double decay,
        double eps, double step_size, double bias_sqrt) {
        quantized_step(model, grad, moment, moment_scale, variance, variance_scale, {}, {},
            moment_offset, moment_scale_offset, variance_offset, variance_scale_offset, block_size, true,
            beta1, beta2, lr, decay, eps, step_size, bias_sqrt);
    });
}
