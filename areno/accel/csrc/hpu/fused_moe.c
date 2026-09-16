#include "tensor_io.h"
#include "index_io.h"
static inline int5 coords(int row,int col) { int5 c={col,row,0,0,0}; return c; }
#if ARENO_KIND == 0
void main(tensor down,tensor weights,tensor ids,tensor positions,tensor output,
          int rows,int tokens,int hidden,int top_k,float scale) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for (int row=begin[1]; row<end[1]; ++row) for (int block=begin[0]; block<end[0]; ++block) {
        int token=load_index64(coords(0,row),ids);
        int position=s_i32_ld_g(gen_addr(coords(0,row),positions));
        float weight=load_scalar_f32(coords(0,row),weights);
        float128 value=load_fp32_pair(coords(row,block*128),down);
        value.v1*=weight; value.v2*=weight;
        // Match CUDA: round each weighted down projection before top-k reduction.
        store_f32(coords(token*top_k+position,block*128),output,value);
    }
}
#else
void main(tensor input,tensor output,int rows,int tokens,int hidden,int top_k,float scale) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for (int token=begin[1]; token<end[1]; ++token) for (int block=begin[0]; block<end[0]; ++block) {
        float128 sum={0,0};
        for (int k=0; k<top_k; ++k) {
            float128 value=load_f32(coords(token*top_k+k,block*128),input);
            sum.v1+=value.v1; sum.v2+=value.v2;
        }
        sum.v1*=scale; sum.v2*=scale;
        store_f32(coords(token,block*128),output,sum);
    }
}
#endif
