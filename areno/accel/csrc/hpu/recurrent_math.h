#pragma once
#include "tensor_io.h"
static inline int5 rc(int row,int column) { int5 c={column,row,0,0,0}; return c; }
static inline float rexp(float x) { float64 v=x; v=v_exp_f32(v); return v[0]; }
static inline float rlog(float x) { float64 v=x; v=v_log_f32(v); return v[0]; }
static inline float rinv(float x) { float64 v=x; v=v_reciprocal_f32(v); return v[0]; }
static inline float rrsqrt(float x) { float64 v=x; v=v_rsqrt_f32(v); return v[0]; }
static inline float rsigmoid(float x) { return x >= 0 ? rinv(1+rexp(-x)) : rexp(x)*rinv(1+rexp(x)); }
static inline float rsoftplus(float x,float beta,float threshold) {
    float z=x*beta;
    if (z > threshold) return x;
    // Avoid overflow in the inactive branch and cancellation for large x.
    return ((z > 0 ? z : 0)+rlog(1+rexp(z > 0 ? -z : z)))/beta;
}
static inline float rround(float x,int dtype) {
    if (dtype == 1) return s_convert_bf16_to_f32(s_convert_f32_to_bf16(x,SW_RHNE));
    if (dtype == 2) return s_convert_f16_to_f32(s_convert_f32_to_f16(x,SW_RHNE));
    return x;
}
static inline void rstore(tensor target,int row,int column,float value) { s_f32_st_g(gen_addr(rc(row,column),target),value); }
static inline float rload(tensor source,int row,int column) { return load_scalar_f32(rc(row,column),source); }
static inline void radd(tensor target,int row,int column,float value) {
    float64 values=0; values[column%64]=value;
    v_f32_st_tnsr_rmw(rc(row,column-column%64),target,values,RMW_SET|RMW_OP_ADD|RMW_DT_FP32);
}
