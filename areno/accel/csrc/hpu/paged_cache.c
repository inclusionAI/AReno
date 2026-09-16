#include "tensor_io.h"

#if ARENO_KIND == 0
void main(tensor table, tensor lengths, tensor owners,
          int batch, int slots, int block_size, int max_blocks, int kv_heads, int hidden) {
    const int5 start = get_index_space_offset(), end = start+get_index_space_size();
    for (int b=start[0]; b<end[0]; ++b) {
        int5 c = {b,0,0,0,0};
        int position = s_i32_ld_g(gen_addr(c, lengths));
        c[0] = position/block_size; c[1] = b;
        int page = s_i32_ld_g(gen_addr(c, table));
        c[0] = page*block_size+position%block_size; c[1] = 0;
        s_i32_st_g(gen_addr(c, owners), b);
    }
}
#else
void main(tensor k_cache, tensor v_cache, tensor k_update, tensor v_update, tensor owners,
          tensor next_k, tensor next_v,
          int batch, int slots, int block_size, int max_blocks, int kv_heads, int hidden) {
    const int5 start = get_index_space_offset(), end = start+get_index_space_size();
    for (int row=start[1]; row<end[1]; ++row) {
        int5 c = {row/kv_heads,0,0,0,0};
        int owner = s_i32_ld_g(gen_addr(c, owners));
        for (int block=start[0]; block<end[0]; ++block) {
            int5 target = {block*128,row,0,0,0};
            if (owner >= 0) {
                int5 source = {block*128,owner*kv_heads+row%kv_heads,0,0,0};
                copy_storage(source, k_update, target, next_k);
                copy_storage(source, v_update, target, next_v);
            } else {
                copy_storage(target, k_cache, target, next_k);
                copy_storage(target, v_cache, target, next_v);
            }
        }
    }
}
#endif
