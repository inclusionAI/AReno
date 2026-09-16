// Host interpretation of the FP32 intrinsics used by AReno's TPC source.
// This checks algorithms and tensor indexing, not TPC compilation or ISA accuracy.
#pragma once
#include <cmath>
#include <stdexcept>
#include <cstdint>
#define __global

typedef float float64 __attribute__((ext_vector_type(64)));
typedef int int5 __attribute__((ext_vector_type(5)));
struct float128 { float64 v1, v2; };
typedef unsigned short bf16;
typedef bf16 bfloat128 __attribute__((ext_vector_type(128)));
typedef _Float16 half128 __attribute__((ext_vector_type(128)));
struct tensor { void* data; int width; int rows; int element_size = 4; };
constexpr int SW_RHNE = 0;
constexpr int RMW_SET = 1, RMW_OP_ADD = 2, RMW_DT_FP32 = 4, RMW_DT_BF16 = 8, RMW_DT_FP16 = 16;
static int5 index_space_extent;
static int5 index_space_origin;

inline int5 get_index_space_offset() { return index_space_origin; }
inline int5 get_index_space_size() { return index_space_extent; }
inline float64 v_f32_ld_tnsr_b(int5 coords, tensor t) {
    float64 value = 0.0f;
    for (int lane = 0; lane < 64; ++lane) {
        const int col = coords[0] + lane, row = coords[1];
        if (col >= 0 && col < t.width && row >= 0 && row < t.rows) value[lane] = static_cast<float*>(t.data)[row * t.width + col];
    }
    return value;
}
inline void v_f32_st_tnsr(int5 coords, tensor t, float64 value) {
    for (int lane = 0; lane < 64; ++lane) {
        const int col = coords[0] + lane, row = coords[1];
        if (col >= 0 && col < t.width && row >= 0 && row < t.rows) static_cast<float*>(t.data)[row * t.width + col] = value[lane];
    }
}
inline void* gen_addr(int5 coords, tensor t) {
    if (coords[0] < 0 || coords[0] >= t.width || coords[1] < 0 || coords[1] >= t.rows) {
        throw std::out_of_range("TPC reference scalar tensor access");
    }
    return static_cast<unsigned char*>(t.data) + (coords[1] * t.width + coords[0]) * t.element_size;
}
inline float s_f32_ld_g(void* address) { return *static_cast<float*>(address); }
inline unsigned short s_u16_ld_g(void* address) { return *static_cast<unsigned short*>(address); }
inline unsigned char s_u8_ld_g(void* address) { return *static_cast<unsigned char*>(address); }
inline unsigned int s_u32_ld_g(void* address) { return *static_cast<unsigned int*>(address); }
inline int s_i32_ld_g(void* address) { return *static_cast<int*>(address); }
inline bf16 s_bf16_ld_g(void* address) { return s_u16_ld_g(address); }
inline void s_f32_st_g(void* address, float value) { *static_cast<float*>(address) = value; }
inline void s_u16_st_g(void* address, unsigned short value) { *static_cast<unsigned short*>(address) = value; }
inline void s_u8_st_g(void* address, unsigned char value) { *static_cast<unsigned char*>(address) = value; }
inline void s_u32_st_g(void* address, unsigned int value) { *static_cast<unsigned int*>(address) = value; }
inline void s_i32_st_g(void* address, int value) { *static_cast<int*>(address) = value; }
inline void s_bf16_st_g(void* address, bf16 value) { s_u16_st_g(address, value); }
inline _Float16 s_f16_ld_g(void* address) { return *static_cast<_Float16*>(address); }
inline void s_f16_st_g(void* address, _Float16 value) { *static_cast<_Float16*>(address) = value; }
inline float s_convert_f16_to_f32(_Float16 value) { return value; }
inline _Float16 s_convert_f32_to_f16(float value, int = 0) { return value; }
inline int s_convert_f32_to_i32(float value, int = 0) { return static_cast<int>(std::nearbyint(value)); }
inline float s_convert_bf16_to_f32(bf16 value) {
    union { unsigned int bits; float value; } word;
    word.bits = static_cast<unsigned int>(value) << 16;
    return word.value;
}
inline bf16 s_convert_f32_to_bf16(float value, int = 0) {
    union { unsigned int bits; float value; } word;
    word.value = value;
    if (std::isnan(value)) return 0x7fc0;
    return static_cast<bf16>((word.bits + 0x7fff + ((word.bits >> 16) & 1)) >> 16);
}
inline bfloat128 v_bf16_ld_tnsr_b(int5 coords, tensor t) {
    bfloat128 value = 0;
    for (int lane = 0; lane < 128; ++lane) {
        int col = coords[0] + lane, row = coords[1];
        if (col >= 0 && col < t.width && row >= 0 && row < t.rows) value[lane] = static_cast<bf16*>(t.data)[row*t.width+col];
    }
    return value;
}
inline void v_bf16_st_tnsr(int5 coords, tensor t, bfloat128 value) {
    for (int lane = 0; lane < 128; ++lane) {
        int col = coords[0] + lane, row = coords[1];
        if (col >= 0 && col < t.width && row >= 0 && row < t.rows) static_cast<bf16*>(t.data)[row*t.width+col] = value[lane];
    }
}
inline float128 v_convert_bf16_to_f32_all_b(bfloat128 input) {
    float128 output;
    for (int i=0; i<64; ++i) {
        output.v1[i] = s_convert_bf16_to_f32(input[i]);
        output.v2[i] = s_convert_bf16_to_f32(input[64+i]);
    }
    return output;
}
inline bfloat128 v_convert_f32_to_bf16_all_b(float128 input, int) {
    bfloat128 output;
    for (int i=0; i<64; ++i) {
        output[i] = s_convert_f32_to_bf16(input.v1[i]);
        output[64+i] = s_convert_f32_to_bf16(input.v2[i]);
    }
    return output;
}
inline float64 v_f32_reduce_add(float64 x) {
    float sum = 0;
    for (int lane = 0; lane < 64; ++lane) sum += x[lane];
    return float64{} + sum;
}
inline float64 v_exp_f32(float64 x) {
    for (int lane = 0; lane < 64; ++lane) x[lane] = std::exp(x[lane]);
    return x;
}
inline float64 v_log_f32(float64 x) {
    for (int lane=0; lane<64; ++lane) x[lane]=std::log(x[lane]);
    return x;
}
inline float64 v_reciprocal_f32(float64 x) { return 1.0f / x; }
inline float64 v_rsqrt_f32(float64 x) {
    for (int lane = 0; lane < 64; ++lane) x[lane] = 1.0f / std::sqrt(x[lane]);
    return x;
}
inline float64 v_sqrt_f32(float64 x) {
    for (int lane = 0; lane < 64; ++lane) x[lane] = std::sqrt(x[lane]);
    return x;
}
inline half128 v_f16_ld_tnsr_b(int5 coords, tensor t) {
    half128 value = 0;
    for (int lane=0; lane<128; ++lane) {
        int col = coords[0]+lane, row = coords[1];
        if (col >= 0 && col < t.width && row >= 0 && row < t.rows) value[lane] = static_cast<_Float16*>(t.data)[row*t.width+col];
    }
    return value;
}
inline void v_f16_st_tnsr(int5 coords, tensor t, half128 value) {
    for (int lane=0; lane<128; ++lane) {
        int col = coords[0]+lane, row = coords[1];
        if (col >= 0 && col < t.width && row >= 0 && row < t.rows) static_cast<_Float16*>(t.data)[row*t.width+col] = value[lane];
    }
}
inline float128 v_convert_f16_to_f32_all_b(half128 input) {
    float128 output;
    for (int i=0; i<64; ++i) { output.v1[i]=input[i]; output.v2[i]=input[64+i]; }
    return output;
}
inline half128 v_convert_f32_to_f16_all_b(float128 input, int) {
    half128 output;
    for (int i=0; i<64; ++i) { output[i]=input.v1[i]; output[64+i]=input.v2[i]; }
    return output;
}
inline void v_f32_st_tnsr_rmw(int5 coords, tensor t, float64 value, int) {
    v_f32_st_tnsr(coords, t, v_f32_ld_tnsr_b(coords, t)+value);
}
inline void v_bf16_st_tnsr_rmw(int5 coords, tensor t, bfloat128 value, int) {
    auto old = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, t));
    auto addition = v_convert_bf16_to_f32_all_b(value);
    old.v1 += addition.v1; old.v2 += addition.v2;
    v_bf16_st_tnsr(coords, t, v_convert_f32_to_bf16_all_b(old, SW_RHNE));
}
inline void v_f16_st_tnsr_rmw(int5 coords, tensor t, half128 value, int) {
    v_f16_st_tnsr(coords, t, v_f16_ld_tnsr_b(coords, t)+value);
}
