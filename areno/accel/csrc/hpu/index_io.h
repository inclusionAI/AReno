// TPC scalar registers are 32-bit. Read both words of int64 tensor storage so
// a large/negative ID never aliases an unrelated valid low-32-bit index.
#pragma once
static inline int load_signed_index64(int5 coords, tensor input) {
    __global unsigned int* address = (__global unsigned int*)gen_addr(coords, input);
    unsigned int low = s_u32_ld_g(address);
    unsigned int high = s_u32_ld_g(address + 1);
    if (high == 0 && low <= 0x7fffffffu) return (int)low;
    if (high == 0xffffffffu && low >= 0x80000000u) return (int)low;
    return (-2147483647-1);
}
static inline int load_index64(int5 coords, tensor input) {
    int value=load_signed_index64(coords,input);
    return value >= 0 ? value : -1;
}

static inline void store_index64(int5 coords, tensor output, int value) {
    __global unsigned int* address = (__global unsigned int*)gen_addr(coords, output);
    s_u32_st_g(address, (unsigned int)value);
    s_u32_st_g(address + 1, value < 0 ? 0xffffffffu : 0u);
}
