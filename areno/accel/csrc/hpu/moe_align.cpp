#include "native.h"
#include "moe_align_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
void align(Tensor ids,int64_t slots,int64_t block,Tensor sorted,Tensor experts,Tensor total,Tensor scratch,bool pad) {
    TORCH_CHECK(at::isIntegralType(ids.scalar_type(),false),"MoE align ids must use integral storage");
    check_tensor(ids,ids,ids.scalar_type());
    for (auto tensor : {sorted,experts,total,scratch}) check_tensor(tensor,ids,at::kInt);
    const int64_t limit=std::numeric_limits<int>::max();
    TORCH_CHECK(slots > 0 && slots <= limit && block > 0 && block <= limit && ids.numel() <= limit
        && (block == 1 || slots <= (limit-ids.numel())/(block-1)),"MoE alignment exceeds TPC indexing");
    TORCH_CHECK(sorted.dim() == 1 && experts.dim() == 1 && scratch.dim() == 1 && total.numel() == 1
        && sorted.numel() >= ids.numel()+slots*(block-1) && experts.numel() >= (sorted.numel()+block-1)/block
        && scratch.numel() >= slots+1 && sorted.numel() <= limit && experts.numel() <= limit && scratch.numel() <= limit,
        "MoE alignment buffers must cover the padded route capacity");
    if (ids.numel() == 0) { if (pad) sorted.zero_(); total.zero_(); scratch.zero_(); return; }
    auto result=call("areno_moe_align",at::kFloat,{ids.to(at::kLong).view({-1}),sorted,experts,scratch,slots,block,pad});
    sorted.copy_(result[0]); experts.copy_(result[1]); total.copy_(result[2].view(total.sizes())); scratch.copy_(result[3]);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("areno_moe_align_f32(Tensor ids, Tensor sorted, Tensor experts, Tensor scratch, int slots, int block, bool pad) -> (Tensor, Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::areno_moe_align_f32","areno_moe_align_f32",
        [](const at::Stack& args) { return Metadata{{at::kInt,args[1].toTensor().sizes().vec()},{at::kInt,args[2].toTensor().sizes().vec()},
            {at::kInt,{1}},{at::kInt,args[3].toTensor().sizes().vec()}}; },
        [](const at::Stack& args,size_t& size)->std::shared_ptr<void> {
            size=sizeof(MoeAlignParams);
            return std::make_shared<MoeAlignParams>(MoeAlignParams{static_cast<int>(args[0].toTensor().numel()),static_cast<int>(args[4].toInt()),
                static_cast<int>(args[5].toInt()),static_cast<int>(args[1].toTensor().numel()),static_cast<int>(args[2].toTensor().numel()),
                static_cast<int>(args[3].toTensor().numel()),args[6].toBool()});
        });
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) { m.impl("areno_moe_align_f32",torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>()); }
TORCH_LIBRARY_IMPL(custom_op,Meta,m) { m.impl("areno_moe_align_f32",torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>()); }
void bind_moe_align(pybind11::module_& m) { m.def("areno_moe_align",&align); }
