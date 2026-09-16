#include "native.h"
#include "quantized_optimizer_params.h"
#include <limits>

namespace {
using namespace areno_hpu;

template<class F> void for_each_quantized(F&& fn) {
    for (int bits : {4, 8}) for (auto grad : {"f32", "bf16"}) for (auto model : {"f32", "bf16"}) {
        fn("areno_adamw" + std::to_string(bits) + "_g" + grad + "_" + model, bits);
    }
}

Tensor slice(Tensor tensor, int64_t offset, int64_t count) {
    TORCH_CHECK(offset >= 0 && offset <= tensor.numel() && count <= tensor.numel() - offset,
                "quantized AdamW state slice is out of bounds");
    return tensor.view({-1}).narrow(0, offset, count);
}

void step(int bits, Tensor model, Tensor grad, Tensor mq, Tensor ms, Tensor vq, Tensor vs,
          Tensor signed_book, Tensor unsigned_book, int64_t block, double beta1, double beta2, double lr,
          double decay, double eps, double step_size, double correction) {
    TORCH_CHECK(block >= 1 && block <= 4096, "AdamW quantization block must be between 1 and 4096");
    TORCH_CHECK(bits != 4 || (block >= 32 && block <= 1024 && (block & (block-1)) == 0),
                "AdamW4 block size must be a power of two between 32 and 1024");
    for (const auto& tensor : {model, grad}) {
        TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kBFloat16,
                    "quantized AdamW model and gradient must be float32 or bfloat16");
        check_tensor(tensor, model, tensor.scalar_type());
    }
    for (auto tensor : {mq, vq}) check_tensor(tensor, model, at::kByte);
    for (auto tensor : {ms, vs}) check_tensor(tensor, model, at::kFloat);
    auto count = model.numel(), codes = bits == 4 ? (count + 1) / 2 : count, blocks = (count + block - 1) / block;
    TORCH_CHECK(count <= std::numeric_limits<int>::max(), "quantized AdamW shard exceeds TPC indexing");
    TORCH_CHECK(grad.numel() == count && mq.numel() == codes && vq.numel() == codes
                && ms.numel() == blocks && vs.numel() == blocks, "quantized AdamW state size mismatch");
    if (bits == 8) for (auto book : {signed_book, unsigned_book}) {
        check_tensor(book, model, at::kFloat);
        TORCH_CHECK(book.numel() == 256, "AdamW8 codebooks must have 256 values");
    }
    if (count == 0) return;
    at::Stack args{model.view({-1}), grad.view({-1}), mq.view({-1}), ms.view({-1}), vq.view({-1}), vs.view({-1})};
    if (bits == 8) { args.emplace_back(signed_book.view({-1})); args.emplace_back(unsigned_book.view({-1})); }
    for (c10::IValue value : {c10::IValue(block), c10::IValue(beta1), c10::IValue(beta2), c10::IValue(lr),
                            c10::IValue(decay), c10::IValue(eps), c10::IValue(step_size), c10::IValue(correction)}) {
        args.emplace_back(value);
    }
    auto result = call("areno_adamw" + std::to_string(bits) + "_g" + dtype_suffix(grad.scalar_type()), model.scalar_type(), args);
    model.copy_(result[0].view(model.sizes()));
    mq.copy_(result[1].view(mq.sizes()));
    ms.copy_(result[2].view(ms.sizes()));
    vq.copy_(result[3].view(vq.sizes()));
    vs.copy_(result[4].view(vs.sizes()));
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_quantized([&](const std::string& name, int bits) {
        std::string signature = "(Tensor model, Tensor grad, Tensor mq, Tensor ms, Tensor vq, Tensor vs, ";
        if (bits == 8) signature += "Tensor signed_book, Tensor unsigned_book, ";
        signature += "int block_size, float beta1, float beta2, float lr, float decay, float eps, float step_size, float correction) -> (Tensor, Tensor, Tensor, Tensor, Tensor)";
        m.def((name + signature).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::" + name, name,
            [](const at::Stack& args) {
                Metadata result;
                for (int index : {0, 2, 3, 4, 5}) {
                    auto tensor = args[index].toTensor();
                    result.push_back({tensor.scalar_type(), tensor.sizes().vec()});
                }
                return result;
            }, [bits](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                int first = bits == 8 ? 8 : 6;
                size = sizeof(QuantizedOptimizerParams);
                return std::make_shared<QuantizedOptimizerParams>(QuantizedOptimizerParams{
                    static_cast<int>(args[0].toTensor().numel()), static_cast<int>(args[first].toInt()),
                    static_cast<float>(args[first+1].toDouble()), static_cast<float>(args[first+2].toDouble()),
                    static_cast<float>(args[first+3].toDouble()), static_cast<float>(args[first+4].toDouble()),
                    static_cast<float>(args[first+5].toDouble()), static_cast<float>(args[first+6].toDouble()),
                    static_cast<float>(args[first+7].toDouble())});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_quantized([&](const std::string& name, int) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_quantized([&](const std::string& name, int) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}
void bind_quantized_optimizer(pybind11::module_& m) {
    m.def("areno_adamw_8bit_step", [](Tensor model, Tensor grad, Tensor mq, Tensor ms, Tensor vq, Tensor vs,
        Tensor signed_book, Tensor unsigned_book, int64_t block, double beta1, double beta2, double lr,
        double decay, double eps, double step_size, double correction) {
        step(8, model, grad, mq, ms, vq, vs, signed_book, unsigned_book, block, beta1, beta2, lr, decay, eps, step_size, correction);
    });
    m.def("areno_adamw_4bit_step", [](Tensor model, Tensor grad, Tensor mq, Tensor ms, Tensor vq, Tensor vs,
        int64_t mq_offset, int64_t ms_offset, int64_t vq_offset, int64_t vs_offset, int64_t block,
        double beta1, double beta2, double lr, double decay, double eps, double step_size, double correction) {
        TORCH_CHECK(block >= 32 && block <= 1024 && (block & (block-1)) == 0, "invalid AdamW4 block size");
        auto codes = (model.numel()+1)/2, blocks = (model.numel()+block-1)/block;
        step(4, model, grad, slice(mq, mq_offset, codes), slice(ms, ms_offset, blocks), slice(vq, vq_offset, codes),
            slice(vs, vs_offset, blocks), {}, {}, block, beta1, beta2, lr, decay, eps, step_size, correction);
    });
}
