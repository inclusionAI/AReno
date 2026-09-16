#include "tensor_io.h"
#include "index_io.h"

static inline int5 coords(int row,int column) { int5 c={column,row,0,0,0}; return c; }
static inline int5 vector_index(int index) { return coords(0,index); }

#if ARENO_KIND < 2
void main(tensor routes,tensor weights,tensor counts,int tokens,int hidden,int experts,int top_k,int local_start,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for (int expert=start[0]; expert<end[0]; ++expert) {
        int count=0;
        for (int token=0; token<tokens; ++token) {
#if ARENO_KIND == 0
            count+=s_u8_ld_g(gen_addr(coords(token,expert),routes)) != 0;
#else
            for (int p=0; p<top_k; ++p) if (load_index64(coords(token,p),routes) == expert+local_start
                && load_scalar_f32(coords(token,p),weights) != 0) ++count;
#endif
        }
        store_index64(vector_index(expert),counts,count);
    }
}
#elif ARENO_KIND < 4
void main(tensor input,tensor routes,tensor weights,tensor counts,
          tensor output,tensor route_weight,tensor token_index,
#if ARENO_KIND == 3
          tensor topk_position,
#endif
          int tokens,int hidden,int experts,int top_k,int local_start,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for (int expert=start[1]; expert<end[1]; ++expert) for (int block=start[0]; block<end[0]; ++block) {
        int row=0;
        for (int previous=0; previous<expert; ++previous) row+=load_index64(vector_index(previous),counts);
        for (int token=0; token<tokens; ++token) {
#if ARENO_KIND == 2
            if (s_u8_ld_g(gen_addr(coords(token,expert),routes)) == 0) continue;
            int position=expert;
#else
            for (int position=0; position<top_k; ++position) {
                if (load_index64(coords(token,position),routes) != expert+local_start
                    || load_scalar_f32(coords(token,position),weights) == 0) continue;
#endif
                copy_storage(coords(token,block*128),input,coords(row,block*128),output);
                if (block == 0) {
                    s_f32_st_g(gen_addr(vector_index(row),route_weight),load_scalar_f32(coords(token,position),weights));
                    store_index64(vector_index(row),token_index,token);
#if ARENO_KIND == 3
                    s_i32_st_g(gen_addr(vector_index(row),topk_position),position);
#endif
                }
                ++row;
#if ARENO_KIND == 3
            }
#endif
        }
    }
}
#else
void main(tensor grad,tensor token_index,tensor topk_position,tensor output,
          int tokens,int hidden,int experts,int top_k,int local_start,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for (int row=start[0]; row<end[0]; ++row) {
        int token=load_index64(vector_index(row),token_index);
        int pos=s_i32_ld_g(gen_addr(vector_index(row),topk_position));
        s_f32_st_g(gen_addr(coords(token,pos),output),load_scalar_f32(vector_index(row),grad));
    }
}
#endif
