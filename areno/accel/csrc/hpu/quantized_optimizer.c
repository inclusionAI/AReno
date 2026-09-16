// Block ownership matches the CUDA quantizers: each block either commits all
// weights/states or preserves all old values when a nonfinite value occurs.
#include "tensor_io.h"

static inline int5 coordinate(int index) { int5 c = {index, 0, 0, 0, 0}; return c; }
static inline float load_model_scalar(tensor model, int index) {
#if ARENO_DTYPE == 1
    return s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(coordinate(index), model)));
#else
    return load_scalar_f32(coordinate(index), model);
#endif
}
static inline void store_model_scalar(tensor model, int index, float value) {
#if ARENO_DTYPE == 1
    s_bf16_st_g(gen_addr(coordinate(index), model), s_convert_f32_to_bf16(value, SW_RHNE));
#else
    s_f32_st_g(gen_addr(coordinate(index), model), value);
#endif
}
static inline void copy_model_scalar(tensor output, tensor input, int index) {
    // Preserve NaN payloads and every storage bit when CUDA would skip a block.
#if ARENO_DTYPE == 1
    s_u16_st_g(gen_addr(coordinate(index), output), s_u16_ld_g(gen_addr(coordinate(index), input)));
#else
    s_u32_st_g(gen_addr(coordinate(index), output), s_u32_ld_g(gen_addr(coordinate(index), input)));
#endif
}
static inline float load_grad_scalar(tensor grad, int index) {
#if ARENO_GRAD_DTYPE == 1
    return s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(coordinate(index), grad)));
#else
    return load_scalar_f32(coordinate(index), grad);
#endif
}
static inline int finite(float value) {
    union { float value; unsigned int bits; } word;
    word.value = value;
    return (word.bits & 0x7f800000u) != 0x7f800000u;
}
static inline float absolute(float value) { return value < 0 ? -value : value; }
static inline float max_value(float a, float b) { return a > b ? a : b; }
#if ARENO_KIND != 8
static inline float signed4_value(int code) {
    switch (code) {
        case 0: return -0.8875f; case 1: return -0.6625f; case 2: return -0.4375f; case 3: return -0.2125f;
        case 4: return -0.0775f; case 5: return -0.0325f; case 6: return -0.0055f; case 7: return 0.0f;
        case 8: return 0.0055f; case 9: return 0.0325f; case 10: return 0.0775f; case 11: return 0.2125f;
        case 12: return 0.4375f; case 13: return 0.6625f; case 14: return 0.8875f; default: return 1.0f;
    }
}
static inline int nearest_signed4(float value) {
    int best = 0;
    float distance = absolute(value - signed4_value(0));
    for (int code = 1; code < 16; ++code) {
        float candidate = absolute(value - signed4_value(code));
        if (candidate < distance) { best = code; distance = candidate; }
    }
    return best;
}
#else
static inline int nearest_code(float value, tensor codebook) {
    int lower = 0, upper = 255;
    while (lower < upper) {
        int middle = (lower + upper) >> 1;
        if (load_scalar_f32(coordinate(middle), codebook) < value) lower = middle + 1;
        else upper = middle;
    }
    if (lower == 0) return 0;
    float left = absolute(value - load_scalar_f32(coordinate(lower - 1), codebook));
    float right = absolute(load_scalar_f32(coordinate(lower), codebook) - value);
    return left <= right ? lower - 1 : lower;
}
#endif
static inline int load_code(tensor q, int index) {
#if ARENO_KIND != 8
    int byte = s_u8_ld_g(gen_addr(coordinate(index / 2), q));
    return (byte >> ((index & 1) * 4)) & 15;
#else
    return s_u8_ld_g(gen_addr(coordinate(index), q));
#endif
}
void main(tensor model, tensor grad, tensor moment_q, tensor moment_scale,
#if ARENO_KIND == 3
          tensor factors, tensor row_mean, tensor invalid_flag,
#else
          tensor variance_q, tensor variance_scale,
#endif
#if ARENO_KIND == 8
          tensor signed_codebook, tensor unsigned_codebook,
#endif
          tensor updated_model, tensor updated_mq, tensor updated_ms,
#if ARENO_KIND != 3
          tensor updated_vq, tensor updated_vs,
#endif
          int count, int block_size, float beta1, float beta2, float effective_lr, float weight_decay,
          float eps, float step_size, float bias_correction2_sqrt,
          int parameter_shard_start, int rows, int columns) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int block = start[0]; block < end[0]; ++block) {
        int first = block * block_size;
        int last = count - first > block_size ? first + block_size : count;
        float old_ms = load_scalar_f32(coordinate(block), moment_scale);
#if ARENO_KIND != 3
        float old_vs = load_scalar_f32(coordinate(block), variance_scale);
        float new_vs = 0;
#endif
        float new_ms = 0;
        int invalid = 0;
#if ARENO_KIND == 3
        invalid = s_i32_ld_g(gen_addr(coordinate(0), invalid_flag)) != 0;
        float mean = max_value(load_scalar_f32(coordinate(0), row_mean), 1e-30f);
#endif
        // Recompute after reduction, as in CUDA, without full FP32 state arrays.
        for (int pass = 0; pass < 2; ++pass) {
#if ARENO_KIND != 8
            int pending_m = 7;
#endif
#if ARENO_KIND == 4
            int pending_v = 0;
#endif
            for (int index = first; index < last; ++index) {
                int mc = load_code(moment_q, index);
#if ARENO_KIND != 3
                int vc = load_code(variance_q, index);
#endif
#if ARENO_KIND == 3
                int parameter_index = parameter_shard_start + index;
                float m = signed4_value(mc) * old_ms;
                float v = load_scalar_f32(coordinate(parameter_index / columns), factors)
                    * load_scalar_f32(coordinate(rows + parameter_index % columns), factors) / mean;
#elif ARENO_KIND == 4
                float m = signed4_value(mc) * old_ms;
                float v = ((float)vc + 1.0f) * old_vs / 16.0f;
#else
                float m = load_scalar_f32(coordinate(mc), signed_codebook) * old_ms;
                float v = load_scalar_f32(coordinate(vc), unsigned_codebook) * old_vs;
#endif
                float g = load_grad_scalar(grad, index);
                float w = load_model_scalar(model, index);
                m = beta1 * m + (1.0f - beta1) * g;
#if ARENO_KIND != 3
                v = beta2 * v + (1.0f - beta2) * g * g;
#endif
                if (weight_decay != 0.0f) w *= 1.0f - effective_lr * weight_decay;
                float64 variance_vector = v;
                float64 reciprocal = v_reciprocal_f32(v_sqrt_f32(variance_vector) / bias_correction2_sqrt + eps);
                w -= step_size * m * reciprocal[0];
                if (pass == 0) {
                    invalid |= !finite(g) || !finite(m) || !finite(v) || !finite(w);
                    new_ms = max_value(new_ms, absolute(m));
#if ARENO_KIND != 3
                    new_vs = max_value(new_vs, v);
#endif
                    continue;
                }
                if (!invalid) {
                    float nm = m / max_value(new_ms, 1e-30f);
#if ARENO_KIND != 3
                    float nv = v / max_value(new_vs, 1e-30f);
#endif
#if ARENO_KIND != 8
                    mc = nearest_signed4(nm);
#if ARENO_KIND == 4
                    vc = s_convert_f32_to_i32(nv * 16.0f - 1.0f, SW_RHNE);
                    vc = vc < 0 ? 0 : vc > 15 ? 15 : vc;
#endif
#else
                    mc = nearest_code(nm, signed_codebook);
                    vc = nearest_code(nv, unsigned_codebook);
#endif
                }
                if (invalid) copy_model_scalar(updated_model, model, index);
                else store_model_scalar(updated_model, index, w);
#if ARENO_KIND != 8
                if ((index & 1) == 0) {
                    pending_m = mc;
#if ARENO_KIND == 4
                    pending_v = vc;
#endif
                }
                if ((index & 1) != 0 || index == last - 1) {
                    int high_m = (index & 1) != 0 ? mc : 7;
                    int mbyte = pending_m | (high_m << 4);
#if ARENO_KIND == 4
                    int high_v = (index & 1) != 0 ? vc : 0;
                    int vbyte = pending_v | (high_v << 4);
#endif
                    if (invalid) {
                        mbyte = s_u8_ld_g(gen_addr(coordinate(index / 2), moment_q));
#if ARENO_KIND != 3
                        vbyte = s_u8_ld_g(gen_addr(coordinate(index / 2), variance_q));
#endif
                    }
                    s_u8_st_g(gen_addr(coordinate(index / 2), updated_mq), (unsigned char)mbyte);
#if ARENO_KIND != 3
                    s_u8_st_g(gen_addr(coordinate(index / 2), updated_vq), (unsigned char)vbyte);
#endif
                }
#else
                s_u8_st_g(gen_addr(coordinate(index), updated_mq), (unsigned char)mc);
                s_u8_st_g(gen_addr(coordinate(index), updated_vq), (unsigned char)vc);
#endif
            }
        }
        s_f32_st_g(gen_addr(coordinate(block), updated_ms), invalid ? old_ms : new_ms);
#if ARENO_KIND != 3
        s_f32_st_g(gen_addr(coordinate(block), updated_vs), invalid ? old_vs : new_vs);
#endif
    }
}
