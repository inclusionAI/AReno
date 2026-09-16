// TPC-C, compiled separately for each activation, direction, and storage dtype.
// Tensor layout is [rows, channels, width] in PyTorch, reversed in TPC.
// channels=2 for gated inputs. Keeping it as a tensor dimension prevents
// vector-tail stores from crossing the gate/up boundary for odd widths.

#define ARENO_FLOAT float64
#define ARENO_EXP(x) v_exp_f32(x)
#define ARENO_LOG(x) v_log_f32(x)
#define ARENO_RECIP(x) v_reciprocal_f32(x)
#define ARENO_TANH(x) v_tanh_f32(x)
#define ARENO_SELECT_GT(a, b, x, y) v_f32_sel_grt_f32_b(a, b, x, y)
#define ARENO_SELECT_EQ(a, b, x, y) v_f32_sel_eq_f32_b(a, b, x, y)
#include "activation_math.h"
#include "tensor_io.h"

#if ARENO_BACKWARD
void main(tensor input, tensor grad_output, tensor output) {
#else
void main(tensor input, tensor output) {
#endif
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int row = start[1]; row < end[1]; ++row) {
        for (int block = start[0]; block < end[0]; ++block) {
            int5 coords = {block * 128, 0, row, 0, 0};
            float128 x = load_f32(coords, input);
            float128 up = {1.0f, 1.0f};
#if ARENO_GATED
            int5 up_coords = coords;
            up_coords[1] = 1;
            up = load_f32(up_coords, input);
#endif
            float128 value;
#if ARENO_BACKWARD
            float128 grad = load_f32(coords, grad_output);
            value.v1 = areno_activation_grad_x(ARENO_KIND, x.v1, up.v1, grad.v1);
            value.v2 = areno_activation_grad_x(ARENO_KIND, x.v2, up.v2, grad.v2);
            store_f32(coords, output, value);
#if ARENO_GATED
            value.v1 = areno_activation_grad_up(ARENO_KIND, x.v1, grad.v1);
            value.v2 = areno_activation_grad_up(ARENO_KIND, x.v2, grad.v2);
            store_f32(up_coords, output, value);
#endif
#else
            value.v1 = areno_activation_value(ARENO_KIND, x.v1, up.v1);
            value.v2 = areno_activation_value(ARENO_KIND, x.v2, up.v2);
            store_f32(coords, output, value);
#endif
        }
    }
}
