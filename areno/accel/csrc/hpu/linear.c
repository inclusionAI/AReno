// Native TPC bias add and bias-gradient reduction around the MME GEMM node.
#include "tensor_io.h"

#if ARENO_DIRECTION == 0
void main(tensor input, tensor bias, tensor output, int hidden, int rows, float unused) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int row = start[1]; row < end[1]; ++row) {
        for (int block = start[0]; block < end[0]; ++block) {
            int5 coords = {block * 128, row, 0, 0, 0};
            int5 bc = {block * 128, 0, 0, 0, 0};
            float128 x = load_f32(coords, input);
            float128 b = load_f32(bc, bias);
            float128 result = {x.v1 + b.v1, x.v2 + b.v2};
            store_f32(coords, output, result);
        }
    }
}
#else
void main(tensor grad_output, tensor grad_bias, int hidden, int rows, float unused) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int block = start[0]; block < end[0]; ++block) {
        float128 sum = {0.0f, 0.0f};
        for (int row = 0; row < rows; ++row) {
            int5 coords = {block * 128, row, 0, 0, 0};
            float128 value = load_f32(coords, grad_output);
            sum.v1 += value.v1;
            sum.v2 += value.v2;
        }
        int5 coords = {block * 128, 0, 0, 0, 0};
        store_f32(coords, grad_bias, sum);
    }
}
#endif
