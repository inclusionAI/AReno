#include "index_io.h"
static inline int5 coord(int index) { int5 c={index,0,0,0,0}; return c; }

void main(tensor ids,tensor old_sorted,tensor old_experts,tensor old_scratch,
          tensor sorted,tensor experts,tensor total,tensor scratch,
          int routes,int expert_slots,int block,int capacity,int block_capacity,int scratch_size,int pad) {
    for (int i=0; i<capacity; ++i) s_i32_st_g(gen_addr(coord(i),sorted),pad ? routes : s_i32_ld_g(gen_addr(coord(i),old_sorted)));
    for (int i=0; i<block_capacity; ++i) s_i32_st_g(gen_addr(coord(i),experts),s_i32_ld_g(gen_addr(coord(i),old_experts)));
    for (int i=0; i<scratch_size; ++i) s_i32_st_g(gen_addr(coord(i),scratch),s_i32_ld_g(gen_addr(coord(i),old_scratch)));
    int offset=0;
    // Slot zero is the -1 expert sentinel, matching CUDA's route planner.
    for (int slot=0; slot<expert_slots; ++slot) {
        int count=0;
        for (int route=0; route<routes; ++route) if (load_signed_index64(coord(route),ids) == slot-1) {
            s_i32_st_g(gen_addr(coord(offset+count),sorted),route);
            ++count;
        }
        s_i32_st_g(gen_addr(coord(slot),scratch),offset+count);
        int next=offset+(count/block+(count%block != 0))*block;
        for (int b=offset/block; b<next/block; ++b) s_i32_st_g(gen_addr(coord(b),experts),slot-1);
        offset=next;
    }
    s_i32_st_g(gen_addr(coord(expert_slots),scratch),offset);
    s_i32_st_g(gen_addr(coord(0),total),offset);
}
