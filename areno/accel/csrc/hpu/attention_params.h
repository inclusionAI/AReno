#pragma once
struct AttentionParams {
    int q_rows;
    int k_rows;
    int hidden;
    int q_heads;
    int kv_heads;
    int q_length;
    int k_length;
    int query_start;
    int window_left;
    int sequences;
    float scale;
    int block_size;
    int max_blocks;
};
