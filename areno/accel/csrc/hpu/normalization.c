// One TPC program per row (forward/dx), or channel block (dw).
// Storage is [rows, hidden]; tensor coordinates reverse that order.
#include "tensor_io.h"

static inline float64 sigmoid(float64 x) {
    return v_reciprocal_f32(1.0f + v_exp_f32(-x));
}

#if ARENO_DIRECTION == 0
void main(tensor input,
#if ARENO_KIND >= 1
          tensor weight,
#endif
#if ARENO_KIND >= 2
          tensor gate,
#endif
          tensor output, tensor inv_rms, int hidden, int rows, float epsilon
#if ARENO_KIND == 3
          , int groups
#endif
          ) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int row = start[0]; row < end[0]; ++row) {
        float64 sum = 0.0f;
        for (int col = 0; col < hidden; col += 128) {
            int5 coords = {col, row, 0, 0, 0};
            float128 x = load_f32(coords, input);
            sum += x.v1 * x.v1 + x.v2 * x.v2;
        }
        float64 inv = v_rsqrt_f32(v_f32_reduce_add(sum) / (float)hidden + epsilon);
        int5 inv_coords = {0, row, 0, 0, 0};
        v_f32_st_tnsr(inv_coords, inv_rms, inv);
        for (int col = 0; col < hidden; col += 128) {
            int5 coords = {col, row, 0, 0, 0};
            float128 x = load_f32(coords, input);
            float128 y = {x.v1 * inv, x.v2 * inv};
#if ARENO_KIND >= 1
            int5 wc = {col, 0, 0, 0, 0};
#if ARENO_KIND == 3
            wc[1] = row % groups;
#endif
            float128 w = load_fp32_pair(wc, weight);
            y.v1 *= w.v1;
            y.v2 *= w.v2;
#endif
#if ARENO_KIND == 2
            float128 g = load_f32(coords, gate);
            y.v1 *= g.v1 * sigmoid(g.v1);
            y.v2 *= g.v2 * sigmoid(g.v2);
#elif ARENO_KIND == 3
            float128 g = load_f32(coords, gate);
            y.v1 *= sigmoid(g.v1);
            y.v2 *= sigmoid(g.v2);
#endif
            store_f32(coords, output, y);
        }
    }
}
#elif ARENO_DIRECTION == 1
void main(tensor input, tensor grad_output, tensor inv_rms,
#if ARENO_KIND >= 1
          tensor weight,
#endif
#if ARENO_KIND == 2
          tensor gate,
#endif
          tensor grad_input,
#if ARENO_KIND == 2
          tensor grad_gate,
#endif
          int hidden, int rows, float epsilon) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int row = start[0]; row < end[0]; ++row) {
        int5 inv_coords = {0, row, 0, 0, 0};
        float64 inv = load_scalar_f32(inv_coords, inv_rms);
        float64 dot = 0.0f;
        for (int col = 0; col < hidden; col += 128) {
            int5 coords = {col, row, 0, 0, 0};
            float128 x = load_f32(coords, input);
            float128 dy = load_f32(coords, grad_output);
#if ARENO_KIND >= 1
            int5 wc = {col, 0, 0, 0, 0};
            float128 w = load_fp32_pair(wc, weight);
            dy.v1 *= w.v1;
            dy.v2 *= w.v2;
#endif
#if ARENO_KIND == 2
            float128 g = load_f32(coords, gate);
            dy.v1 *= g.v1 * sigmoid(g.v1);
            dy.v2 *= g.v2 * sigmoid(g.v2);
#endif
            dot += dy.v1 * x.v1 + dy.v2 * x.v2;
        }
        float64 correction = v_f32_reduce_add(dot) * inv * inv / (float)hidden;
        for (int col = 0; col < hidden; col += 128) {
            int5 coords = {col, row, 0, 0, 0};
            float128 x = load_f32(coords, input);
            float128 dy = load_f32(coords, grad_output);
#if ARENO_KIND >= 1
            int5 wc = {col, 0, 0, 0, 0};
            float128 w = load_fp32_pair(wc, weight);
            dy.v1 *= w.v1;
            dy.v2 *= w.v2;
#endif
#if ARENO_KIND == 2
            float128 g = load_f32(coords, gate);
            float128 s = {sigmoid(g.v1), sigmoid(g.v2)};
            float128 dg = {
                dy.v1 * x.v1 * inv * s.v1 * (1.0f + g.v1 * (1.0f - s.v1)),
                dy.v2 * x.v2 * inv * s.v2 * (1.0f + g.v2 * (1.0f - s.v2))
            };
            store_f32(coords, grad_gate, dg);
            dy.v1 *= g.v1 * s.v1;
            dy.v2 *= g.v2 * s.v2;
#endif
            float128 dx = {inv * (dy.v1 - x.v1 * correction), inv * (dy.v2 - x.v2 * correction)};
            store_f32(coords, grad_input, dx);
        }
    }
}
#else
void main(tensor input, tensor grad_output, tensor inv_rms,
#if ARENO_KIND == 2
          tensor gate,
#endif
          tensor grad_weight, int hidden, int rows, float epsilon) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int block = start[0]; block < end[0]; ++block) {
        float128 dw = {0.0f, 0.0f};
        for (int row = 0; row < rows; ++row) {
            int5 coords = {block * 128, row, 0, 0, 0};
            int5 inv_coords = {0, row, 0, 0, 0};
            float64 inv = load_scalar_f32(inv_coords, inv_rms);
            float128 x = load_f32(coords, input);
            float128 dy = load_f32(coords, grad_output);
            dy.v1 *= x.v1 * inv;
            dy.v2 *= x.v2 * inv;
#if ARENO_KIND == 2
            float128 g = load_f32(coords, gate);
            dy.v1 *= g.v1 * sigmoid(g.v1);
            dy.v2 *= g.v2 * sigmoid(g.v2);
#endif
            dw.v1 += dy.v1;
            dw.v2 += dy.v2;
        }
        int5 coords = {block * 128, 0, 0, 0, 0};
        store_fp32_pair(coords, grad_weight, dw);
    }
}
#endif
