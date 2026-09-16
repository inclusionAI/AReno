#include "tensor_io.h"
#include "index_io.h"

void main(tensor ids, tensor input, tensor output, int tokens, int hidden, int vocab_start, int vocab_end) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int token = start[1]; token < end[1]; ++token) {
        int5 id_coords = {token, 0, 0, 0, 0};
        int id = load_index64(id_coords, ids);
        int valid = id >= vocab_start && id < vocab_end;
        for (int block = start[0]; block < end[0]; ++block) {
            int5 local = {block * 128, id - vocab_start, 0, 0, 0};
            int5 token_coords = {block * 128, token, 0, 0, 0};
#if ARENO_DIRECTION == 0
            float128 result;
            result.v1 = 0; result.v2 = 0;
            if (valid) result = load_f32(local, input);
            store_f32(token_coords, output, result);
#else
            if (!valid) continue;
            // GC zero-initializes the output before these atomic stores.
#if ARENO_DTYPE == 1
            v_bf16_st_tnsr_rmw(local, output, v_bf16_ld_tnsr_b(token_coords, input), RMW_SET | RMW_OP_ADD | RMW_DT_BF16);
#elif ARENO_DTYPE == 2
            v_f16_st_tnsr_rmw(local, output, v_f16_ld_tnsr_b(token_coords, input), RMW_SET | RMW_OP_ADD | RMW_DT_FP16);
#else
            float128 grad = load_fp32_pair(token_coords, input);
            v_f32_st_tnsr_rmw(local, output, grad.v1, RMW_SET | RMW_OP_ADD | RMW_DT_FP32);
            local[0] += 64;
            v_f32_st_tnsr_rmw(local, output, grad.v2, RMW_SET | RMW_OP_ADD | RMW_DT_FP32);
#endif
#endif
        }
    }
}
