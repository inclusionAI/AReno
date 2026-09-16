// Causal attention with FP32 online softmax, matching the native CUDA path.
// Width programs recompute scores, keeping scratch independent of sequence length.
#include "tensor_io.h"

static inline int5 coords(int row, int column) { int5 c = {column, row, 0, 0, 0}; return c; }
static inline float exp_scalar(float x) { float64 v = x; v = v_exp_f32(v); return v[0]; }
static inline float reciprocal_scalar(float x) { float64 v = x; v = v_reciprocal_f32(v); return v[0]; }
static inline float maximum(float a, float b) { return a > b ? a : b; }
#if ARENO_KIND == 2
static inline int cache_row(tensor table, int batch, int key, int block_size, int kv_heads, int head) {
    int block = s_i32_ld_g(gen_addr(coords(batch, key/block_size), table));
    return (block*block_size+key%block_size)*kv_heads+head;
}
#endif

static inline float dot(tensor a, int ar, tensor b, int br, int hidden) {
    float64 sum = 0;
    for (int d=0; d<hidden; d+=128) {
        float128 av = load_f32(coords(ar, d), a), bv = load_f32(coords(br, d), b);
        sum += av.v1 * bv.v1 + av.v2 * bv.v2;
    }
    sum = v_f32_reduce_add(sum);
    return sum[0];
}
#if ARENO_DIRECTION == 1
static inline float gradient_dot(tensor grad, int qr, tensor v, int kr, tensor output, int hidden) {
    float64 sum = 0;
    for (int d=0; d<hidden; d+=128) {
        float128 g = load_f32(coords(qr, d), grad), value = load_f32(coords(kr, d), v), o = load_f32(coords(qr, d), output);
        sum += g.v1 * (value.v1-o.v1) + g.v2 * (value.v2-o.v2);
    }
    sum = v_f32_reduce_add(sum);
    return sum[0];
}
static inline void atomic_add_pair(int5 c, tensor output, float128 value) {
    v_f32_st_tnsr_rmw(c, output, value.v1, RMW_SET | RMW_OP_ADD | RMW_DT_FP32);
    c[0] += 64;
    v_f32_st_tnsr_rmw(c, output, value.v2, RMW_SET | RMW_OP_ADD | RMW_DT_FP32);
}
#endif

void main(tensor q, tensor k, tensor v,
#if ARENO_DIRECTION == 1
          tensor grad, tensor output,
#endif
#if ARENO_KIND == 1
          tensor cu_seqlens,
#elif ARENO_KIND == 2
          tensor block_table, tensor cache_seqlens,
#endif
#if ARENO_DIRECTION == 1
          tensor dq, tensor dk, tensor dv,
#else
          tensor output,
#endif
          int q_rows, int k_rows, int hidden, int q_heads, int kv_heads, int q_length,
          int k_length, int query_start, int window_left, int sequences, float scale, int block_size, int max_blocks) {
    const int5 start = get_index_space_offset(), end = start + get_index_space_size();
    for (int qr=start[1]; qr<end[1]; ++qr) {
        int first, last;
#if ARENO_KIND != 2
        int base, stride;
#endif
#if ARENO_KIND == 1
        int token = qr / q_heads, head = qr % q_heads;
        int lo = 0, hi = sequences;
        while (lo + 1 < hi) {
            int mid = lo + (hi-lo)/2;
            int5 sc = {mid,0,0,0,0};
            if (s_i32_ld_g(gen_addr(sc, cu_seqlens)) <= token) lo = mid;
            else hi = mid;
        }
        int5 sc = {lo,0,0,0,0};
        first = s_i32_ld_g(gen_addr(sc, cu_seqlens));
        last = token;
        base = head / (q_heads/kv_heads);
        stride = kv_heads;
#elif ARENO_KIND == 2
        int batch = qr/q_heads, head = (qr%q_heads)/(q_heads/kv_heads);
        first = 0;
        int5 lc = {batch,0,0,0,0};
        last = s_i32_ld_g(gen_addr(lc, cache_seqlens));
#else
        first = 0;
        last = query_start + qr % q_length;
        base = (qr/q_length)*k_length;
        stride = 1;
#endif
        if (window_left >= 0 && last-first > window_left) first = last-window_left;
        for (int block=start[0]; block<end[0]; ++block) {
            int d = block*128;
            float max_score = 0, denom = 0;
            float128 accumulator;
            accumulator.v1 = 0; accumulator.v2 = 0;
            // Online log-sum-exp keeps exponent arguments nonpositive.
            for (int key=first; key<=last; ++key) {
#if ARENO_KIND == 2
                int kr = cache_row(block_table, batch, key, block_size, kv_heads, head);
#else
                int kr = base + key*stride;
#endif
                float score = dot(q, qr, k, kr, hidden)*scale;
                float next_max = denom == 0 ? score : maximum(max_score, score);
                float alpha = denom == 0 ? 0 : exp_scalar(max_score-next_max);
                float beta = exp_scalar(score-next_max);
                denom = denom*alpha + beta;
                max_score = next_max;
#if ARENO_DIRECTION == 0
                float128 value = load_f32(coords(kr, d), v);
                accumulator.v1 = accumulator.v1*alpha + value.v1*beta;
                accumulator.v2 = accumulator.v2*alpha + value.v2*beta;
#endif
            }
            float inverse = reciprocal_scalar(denom);
#if ARENO_DIRECTION == 0
            accumulator.v1 *= inverse;
            accumulator.v2 *= inverse;
            store_f32(coords(qr, d), output, accumulator);
#else
            float128 query = load_f32(coords(qr, d), q), gradient = load_f32(coords(qr, d), grad);
            for (int key=first; key<=last; ++key) {
                int kr = base+key*stride;
                float probability = exp_scalar(dot(q, qr, k, kr, hidden)*scale-max_score)*inverse;
                float ds = probability*gradient_dot(grad, qr, v, kr, output, hidden)*scale;
                float128 key_value = load_f32(coords(kr, d), k), kg, vg;
                accumulator.v1 += ds*key_value.v1;
                accumulator.v2 += ds*key_value.v2;
                kg.v1 = ds*query.v1; kg.v2 = ds*query.v2;
                vg.v1 = probability*gradient.v1; vg.v2 = probability*gradient.v2;
                atomic_add_pair(coords(kr, d), dk, kg);
                atomic_add_pair(coords(kr, d), dv, vg);
            }
            store_fp32_pair(coords(qr, d), dq, accumulator);
#endif
        }
    }
}
