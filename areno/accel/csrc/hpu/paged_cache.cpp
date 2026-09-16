#include "native.h"
#include "paged_cache_params.h"
#include <limits>

namespace {
using namespace areno_hpu;
template<class F> void for_each_cache(F&& fn) {
    fn("areno_cache_owners_f32", true);
    for (auto dtype : {"f32", "bf16", "f16"}) fn(std::string("areno_cache_update_")+dtype, false);
}
Tensor paged_decode(Tensor q, Tensor ku, Tensor vu, Tensor kc, Tensor vc, Tensor table, Tensor lengths,
                    int64_t window, int64_t splits, double scale) {
    for (auto tensor : {q, ku, vu, kc, vc}) check_tensor(tensor, q, q.scalar_type());
    dtype_suffix(q.scalar_type());
    for (auto tensor : {table, lengths}) check_tensor(tensor, q, at::kInt);
    TORCH_CHECK(q.dim() == 3 && ku.dim() == 3 && vu.sizes() == ku.sizes() && kc.dim() == 4 && vc.sizes() == kc.sizes(),
                "paged attention q/update/cache shape mismatch");
    TORCH_CHECK(table.dim() == 2 && lengths.dim() == 1 && q.size(0) == table.size(0)
        && q.size(0) == lengths.size(0) && q.size(0) == ku.size(0), "paged attention batch mismatch");
    TORCH_CHECK(q.size(2) > 0 && q.size(2) == kc.size(3) && q.size(2) == ku.size(2)
        && q.size(1) > 0 && kc.size(2) > 0 && q.size(1)%kc.size(2) == 0 && ku.size(1) == kc.size(2), "paged attention head mismatch");
    const auto limit = std::numeric_limits<int>::max();
    TORCH_CHECK(kc.size(0) > 0 && kc.size(1) > 0 && table.size(1) > 0 && splits >= 1 && window >= -1 && window <= limit,
                "invalid paged attention cache or window configuration");
    TORCH_CHECK(kc.numel()/kc.size(3) <= limit && q.numel()/q.size(2) <= limit && kc.size(3) <= limit && table.numel() <= limit,
                "paged attention shape exceeds TPC indexing");
    if (q.size(0) == 0) return at::empty_like(q);
    auto slots = kc.size(0)*kc.size(1), heads = kc.size(2), hidden = kc.size(3), block = kc.size(1);
    auto owners = call("areno_cache_owners", at::kFloat, {table, lengths, slots, block, heads, hidden})[0];
    auto caches = call("areno_cache_update", q.scalar_type(),
        {kc.view({-1, hidden}), vc.view({-1, hidden}), ku.view({-1, hidden}), vu.view({-1, hidden}), owners, block, heads});
    // num_splits remains a tuning hint. This first native implementation walks
    // the complete window per query; it does not parallelize across splits yet.
    auto output = call("areno_attention_paged", q.scalar_type(),
        {q.view({-1, hidden}), caches[0], caches[1], table, lengths,
         q.size(1), heads, 1, slots, 0, window, 0, scale, block, table.size(1)})[0];
    kc.copy_(caches[0].view(kc.sizes()));
    vc.copy_(caches[1].view(vc.sizes()));
    return output.view(q.sizes());
}
} // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    for_each_cache([&](const std::string& name, bool owners) {
        m.def((name + (owners ? "(Tensor table, Tensor lengths, int slots, int block, int heads, int hidden) -> Tensor"
            : "(Tensor kc, Tensor vc, Tensor ku, Tensor vu, Tensor owners, int block, int heads) -> (Tensor, Tensor)")).c_str());
        habana::custom_op::registerUserCustomOp("custom_op::"+name, name,
            [owners](const at::Stack& args) {
                if (owners) return Metadata{{at::kInt, {args[2].toInt()}}};
                auto cache = args[0].toTensor();
                return Metadata{{cache.scalar_type(), cache.sizes().vec()}, {cache.scalar_type(), cache.sizes().vec()}};
            }, [owners](const at::Stack& args, size_t& size) -> std::shared_ptr<void> {
                size = sizeof(PagedCacheParams);
                if (owners) return std::make_shared<PagedCacheParams>(PagedCacheParams{
                    static_cast<int>(args[0].toTensor().size(0)), static_cast<int>(args[2].toInt()),
                    static_cast<int>(args[3].toInt()), static_cast<int>(args[0].toTensor().size(1)),
                    static_cast<int>(args[4].toInt()), static_cast<int>(args[5].toInt())});
                return std::make_shared<PagedCacheParams>(PagedCacheParams{
                    static_cast<int>(args[2].toTensor().size(0)/args[6].toInt()), static_cast<int>(args[4].toTensor().numel()),
                    static_cast<int>(args[5].toInt()), 0, static_cast<int>(args[6].toInt()), static_cast<int>(args[0].toTensor().size(1))});
            });
    });
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    for_each_cache([&](const std::string& name, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::execute_boxed>());
    });
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    for_each_cache([&](const std::string& name, bool) {
        m.impl(name.c_str(), torch::CppFunction::makeFromBoxedFunction<&areno_hpu::meta_boxed>());
    });
}
void bind_paged_cache(pybind11::module_& m) {
    m.def("areno_paged_causal_attention_decode_forward", &paged_decode);
}
