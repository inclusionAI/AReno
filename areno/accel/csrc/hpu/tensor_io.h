// TPC tensor loads/stores in blocks of 128 values, computing in FP32.
#pragma once

static inline float128 load_fp32_pair(int5 coords, tensor input) {
    float128 value;
    value.v1 = v_f32_ld_tnsr_b(coords, input);
    coords[0] += 64;
    value.v2 = v_f32_ld_tnsr_b(coords, input);
    return value;
}

static inline void store_fp32_pair(int5 coords, tensor output, float128 value) {
    v_f32_st_tnsr(coords, output, value.v1);
    coords[0] += 64;
    v_f32_st_tnsr(coords, output, value.v2);
}

static inline float128 load_f32(int5 coords, tensor input) {
#if ARENO_DTYPE == 1
    return v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, input));
#elif ARENO_DTYPE == 2
    return v_convert_f16_to_f32_all_b(v_f16_ld_tnsr_b(coords, input));
#else
    return load_fp32_pair(coords, input);
#endif
}

static inline void store_f32(int5 coords, tensor output, float128 value) {
#if ARENO_DTYPE == 1
    v_bf16_st_tnsr(coords, output, v_convert_f32_to_bf16_all_b(value, SW_RHNE));
#elif ARENO_DTYPE == 2
    v_f16_st_tnsr(coords, output, v_convert_f32_to_f16_all_b(value, SW_RHNE));
#else
    store_fp32_pair(coords, output, value);
#endif
}

static inline float load_scalar_f32(int5 coords, tensor input) {
    return s_f32_ld_g(gen_addr(coords, input));
}

static inline float load_scalar(int5 coords, tensor input) {
#if ARENO_DTYPE == 1
    return s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(coords,input)));
#elif ARENO_DTYPE == 2
    return s_convert_f16_to_f32(s_f16_ld_g(gen_addr(coords,input)));
#else
    return load_scalar_f32(coords,input);
#endif
}
static inline void store_scalar(int5 coords, tensor output, float value) {
#if ARENO_DTYPE == 1
    s_bf16_st_g(gen_addr(coords,output),s_convert_f32_to_bf16(value,SW_RHNE));
#elif ARENO_DTYPE == 2
    s_f16_st_g(gen_addr(coords,output),s_convert_f32_to_f16(value,SW_RHNE));
#else
    s_f32_st_g(gen_addr(coords,output),value);
#endif
}

static inline void copy_storage(int5 source,tensor input,int5 target,tensor output) {
#if ARENO_DTYPE == 1
    v_bf16_st_tnsr(target,output,v_bf16_ld_tnsr_b(source,input));
#elif ARENO_DTYPE == 2
    v_f16_st_tnsr(target,output,v_f16_ld_tnsr_b(source,input));
#else
    store_fp32_pair(target,output,load_fp32_pair(source,input));
#endif
}
