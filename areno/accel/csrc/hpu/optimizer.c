// FP32-state AdamW and lossless BF16 + low-bits/carry FP32 master updates.
#include "tensor_io.h"

static inline float128 load_gradient(int5 coords, tensor grad) {
#if ARENO_GRAD_DTYPE == 1
    return v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coords, grad));
#else
    return load_fp32_pair(coords, grad);
#endif
}

#if ARENO_KIND == 0
void main(tensor model, tensor grad, tensor exp_avg, tensor exp_avg_sq,
          tensor updated_model, tensor updated_avg, tensor updated_avg_sq,
          int count, int carry_offset, float beta1, float beta2,
          float effective_lr, float weight_decay, float eps, float step_size, float bias_correction2_sqrt) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int block = start[0]; block < end[0]; ++block) {
        int5 coords = {block * 128, 0, 0, 0, 0};
        float128 w = load_f32(coords, model);
        float128 g = load_gradient(coords, grad);
        float128 m = load_fp32_pair(coords, exp_avg);
        float128 v = load_fp32_pair(coords, exp_avg_sq);
        m.v1 = beta1 * m.v1 + (1.0f - beta1) * g.v1;
        m.v2 = beta1 * m.v2 + (1.0f - beta1) * g.v2;
        v.v1 = beta2 * v.v1 + (1.0f - beta2) * g.v1 * g.v1;
        v.v2 = beta2 * v.v2 + (1.0f - beta2) * g.v2 * g.v2;
        if (weight_decay != 0.0f) {
            w.v1 *= 1.0f - effective_lr * weight_decay;
            w.v2 *= 1.0f - effective_lr * weight_decay;
        }
        w.v1 -= step_size * m.v1 * v_reciprocal_f32(v_sqrt_f32(v.v1) / bias_correction2_sqrt + eps);
        w.v2 -= step_size * m.v2 * v_reciprocal_f32(v_sqrt_f32(v.v2) / bias_correction2_sqrt + eps);
        store_f32(coords, updated_model, w);
        store_fp32_pair(coords, updated_avg, m);
        store_fp32_pair(coords, updated_avg_sq, v);
    }
}
#else
// One program owns a complete carry byte, including partial first/last bytes.
// This mirrors CUDA's byte ownership and prevents overlapping bit updates.
void main(tensor model, tensor grad, tensor low_bits, tensor carry_bits, tensor exp_avg, tensor exp_avg_sq,
          tensor updated_model, tensor updated_low, tensor updated_carry, tensor updated_avg, tensor updated_avg_sq,
          int count, int carry_offset, float beta1, float beta2,
          float effective_lr, float weight_decay, float eps, float step_size, float bias_correction2_sqrt) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int byte = start[0]; byte < end[0]; ++byte) {
        int5 bc = {byte, 0, 0, 0, 0};
        unsigned char carries = s_u8_ld_g(gen_addr(bc, carry_bits));
        for (int bit = 0; bit < 8; ++bit) {
            const int index = byte * 8 + bit - carry_offset;
            if (index < 0 || index >= count) continue;
            int5 coords = {index, 0, 0, 0, 0};
            unsigned int high = s_u16_ld_g(gen_addr(coords, model));
            unsigned int carry = (carries >> bit) & 1u;
            union { unsigned int bits; float value; } master;
            master.bits = (((high - carry) & 0xffffu) << 16) | s_u16_ld_g(gen_addr(coords, low_bits));
#if ARENO_GRAD_DTYPE == 1
            float g = s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(coords, grad)));
#else
            float g = load_scalar_f32(coords, grad);
#endif
            float m = beta1 * load_scalar_f32(coords, exp_avg) + (1.0f - beta1) * g;
            float v = beta2 * load_scalar_f32(coords, exp_avg_sq) + (1.0f - beta2) * g * g;
            if (weight_decay != 0.0f) master.value *= 1.0f - effective_lr * weight_decay;
            float64 variance = v;
            float64 denom = v_sqrt_f32(variance) / bias_correction2_sqrt + eps;
            float64 reciprocal = v_reciprocal_f32(denom);
            master.value -= step_size * m * reciprocal[0];
            union { bf16 value; unsigned short bits; } rounded;
            rounded.value = s_convert_f32_to_bf16(master.value, SW_RHNE);
            s_u16_st_g(gen_addr(coords, updated_model), rounded.bits);
            s_u16_st_g(gen_addr(coords, updated_low), (unsigned short)(master.bits & 0xffffu));
            s_f32_st_g(gen_addr(coords, updated_avg), m);
            s_f32_st_g(gen_addr(coords, updated_avg_sq), v);
            unsigned char mask = (unsigned char)(1u << bit);
            if (rounded.bits != (unsigned short)(master.bits >> 16)) carries |= mask;
            else carries &= (unsigned char)~mask;
        }
        s_u8_st_g(gen_addr(bc, updated_carry), carries);
    }
}
#endif
