// Each program owns one row/column sum. No atomics or input mutation is needed.
#include "tensor_io.h"

static inline int5 coordinate(int index) { int5 c = {index, 0, 0, 0, 0}; return c; }

#if ARENO_KIND == 0
void main(tensor grad, tensor sums, tensor updated_sums, tensor invalid_mask,
          int count, int parameter_shard_start, int rows, int columns) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int factor = start[0]; factor < end[0]; ++factor) {
        int first, last, stride;
        if (factor < rows) {
            first = factor * columns - parameter_shard_start;
            last = first + columns;
            first = first < 0 ? 0 : first;
            last = last > count ? count : last;
            stride = 1;
        } else {
            first = factor - rows - parameter_shard_start % columns;
            if (first < 0) first += columns;
            last = count;
            stride = columns;
        }
        float sum = load_scalar_f32(coordinate(factor), sums);
        int invalid = 0;
        for (int index = first; index < last;) {
#if ARENO_DTYPE == 1
            float value = s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(coordinate(index), grad)));
#else
            float value = load_scalar_f32(coordinate(index), grad);
#endif
            union { float value; unsigned int bits; } squared;
            squared.value = value * value;
            if ((squared.bits & 0x7f800000u) == 0x7f800000u) invalid = 1;
            else sum += squared.value;
            if (last - index <= stride) break;
            index += stride;
        }
        s_f32_st_g(gen_addr(coordinate(factor), updated_sums), sum);
        s_i32_st_g(gen_addr(coordinate(factor), invalid_mask), invalid);
    }
}
#else
void main(tensor invalid_mask, tensor invalid, tensor updated_invalid,
          int count, int parameter_shard_start, int rows, int columns) {
    int flag = s_i32_ld_g(gen_addr(coordinate(0), invalid));
    for (int index = 0; index < rows + columns; ++index) {
        if (s_i32_ld_g(gen_addr(coordinate(index), invalid_mask)) != 0) {
            flag = 1;
            break;
        }
    }
    s_i32_st_g(gen_addr(coordinate(0), updated_invalid), flag);
}
#endif
