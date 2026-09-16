#include "tensor_io.h"

static inline int5 coords(int row, int channel) { int5 c = {channel,row,0,0,0}; return c; }
static inline float64 sigmoid(float64 value) { return v_reciprocal_f32(1.0f+v_exp_f32(-value)); }
static inline int boundary(int token, int length, int sequences, tensor cu, int end) {
#if ARENO_KIND == 1
    int lo=0, hi=sequences;
    while (lo+1 < hi) {
        int mid=lo+(hi-lo)/2;
        int5 c={mid,0,0,0,0};
        if (s_i32_ld_g(gen_addr(c,cu)) <= token) lo=mid; else hi=mid;
    }
    int5 c={lo+end,0,0,0,0};
    return s_i32_ld_g(gen_addr(c,cu));
#else
    return (token/length+end)*length;
#endif
}
#if ARENO_DIRECTION == 1
static inline float128 grad_preact(int row, int channel, tensor grad, tensor preact) {
    float128 value=load_fp32_pair(coords(row,channel),preact), g=load_f32(coords(row,channel),grad);
    float64 s1=sigmoid(value.v1), s2=sigmoid(value.v2);
    g.v1 *= s1*(1.0f+value.v1*(1.0f-s1));
    g.v2 *= s2*(1.0f+value.v2*(1.0f-s2));
    return g;
}
#endif

void main(tensor input, tensor weight,
#if ARENO_DIRECTION == 1
          tensor grad, tensor preact,
#endif
#if ARENO_KIND == 1
          tensor cu,
#elif ARENO_KIND == 2
          tensor history,
#endif
          tensor output,
#if ARENO_DIRECTION == 1
          tensor grad_weight,
#else
          tensor preact,
#endif
          int tokens, int channels, int kernel, int length, int sequences) {
#if ARENO_KIND == 1
#define BOUNDARIES cu
#else
#define BOUNDARIES input
#endif
    const int5 start=get_index_space_offset(), end=start+get_index_space_size();
    for (int row=start[1]; row<end[1]; ++row) for (int block=start[0]; block<end[0]; ++block) {
        int channel=block*128;
        float128 acc;
        acc.v1=0; acc.v2=0;
#if ARENO_DIRECTION == 0
#if ARENO_KIND != 2
        int first=boundary(row,length,sequences,BOUNDARIES,0);
#endif
        for (int tap=0; tap<kernel; ++tap) {
            float128 value;
#if ARENO_KIND == 2
            if (tap == kernel-1) value=load_f32(coords(row,channel),input);
            else value=load_f32(coords(row*(kernel-1)+tap,channel),history);
#else
            int source=row-(kernel-1-tap);
            if (source < first) continue;
            value=load_f32(coords(source,channel),input);
#endif
            float128 w=load_fp32_pair(coords(tap,channel),weight);
            acc.v1 += value.v1*w.v1; acc.v2 += value.v2*w.v2;
        }
        store_fp32_pair(coords(row,channel),preact,acc);
        acc.v1 *= sigmoid(acc.v1); acc.v2 *= sigmoid(acc.v2);
        store_f32(coords(row,channel),output,acc);
#else
        if (row < tokens) {
            int last=boundary(row,length,sequences,BOUNDARIES,1);
            for (int shift=0; shift<kernel && shift<last-row; ++shift) {
                float128 g=grad_preact(row+shift,channel,grad,preact);
                float128 w=load_fp32_pair(coords(kernel-1-shift,channel),weight);
                acc.v1 += g.v1*w.v1; acc.v2 += g.v2*w.v2;
            }
            store_f32(coords(row,channel),output,acc);
        }
        if (row < kernel) {
            acc.v1=0; acc.v2=0;
            for (int token=0; token<tokens; ++token) {
                int source=token-(kernel-1-row);
                if (source < boundary(token,length,sequences,BOUNDARIES,0)) continue;
                float128 g=grad_preact(token,channel,grad,preact), x=load_f32(coords(source,channel),input);
                acc.v1 += g.v1*x.v1; acc.v2 += g.v2*x.v2;
            }
            store_fp32_pair(coords(row,channel),grad_weight,acc);
        }
#endif
    }
#undef BOUNDARIES
}
