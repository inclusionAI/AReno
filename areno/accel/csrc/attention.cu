#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int kAttentionThreads = 256;
constexpr int kAttentionTileN = 16;

int64_t attention_tile_n(int64_t head_dim) {
  if (head_dim <= 256) {
    return 16;
  }
  if (head_dim <= 512) {
    return 8;
  }
  return 4;
}

__device__ __forceinline__ int64_t find_sequence_id(const int32_t* __restrict__ cu_seqlens, int64_t num_seqs, int64_t token_idx) {
  int64_t lo = 0;
  int64_t hi = num_seqs;
  while (lo + 1 < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (static_cast<int64_t>(cu_seqlens[mid]) <= token_idx) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename scalar_t>
__global__ void causal_attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t heads,
    int64_t q_len,
    int64_t k_len,
    int64_t head_dim,
    int64_t tile_n,
    int64_t query_start,
    int64_t window_left,
    float softmax_scale) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* acc_s = q_s + head_dim;
  float* partial_s = acc_s + head_dim;
  float* k_tile = partial_s + blockDim.x;
  float* v_tile = k_tile + tile_n * head_dim;

  const int64_t row = blockIdx.x;
  const int64_t b = row / (heads * q_len);
  const int64_t rem = row - b * heads * q_len;
  const int64_t h = rem / q_len;
  const int64_t qi = rem - h * q_len;
  const int64_t abs_q = query_start + qi;

  const int64_t q_base = ((b * heads + h) * q_len + qi) * head_dim;
  const int64_t k_base = (b * heads + h) * k_len * head_dim;
  const int64_t v_base = (b * heads + h) * k_len * head_dim;
  const int64_t out_base = q_base;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = static_cast<float>(q[q_base + d]);
    acc_s[d] = 0.0f;
  }
  __syncthreads();

  float m = -std::numeric_limits<float>::infinity();
  float l = 0.0f;

  for (int64_t tile_start = 0; tile_start < k_len; tile_start += tile_n) {
    const int64_t remaining = k_len - tile_start;
    const int64_t tile_len = remaining < tile_n ? remaining : tile_n;
    const int64_t tile_elements = tile_len * head_dim;
    for (int64_t idx = threadIdx.x; idx < tile_elements; idx += blockDim.x) {
      const int64_t ki = tile_start + idx / head_dim;
      const int64_t d = idx % head_dim;
      k_tile[idx] = static_cast<float>(k[k_base + ki * head_dim + d]);
      v_tile[idx] = static_cast<float>(v[v_base + ki * head_dim + d]);
    }
    __syncthreads();

    for (int64_t tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
      const int64_t ki = tile_start + tile_idx;
      const bool valid = ki <= abs_q && (window_left < 0 || ki >= abs_q - window_left);

      float thread_dot = 0.0f;
      for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        thread_dot += q_s[d] * k_tile[tile_idx * head_dim + d];
      }
      partial_s[threadIdx.x] = thread_dot;
      __syncthreads();

      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        float alpha = 1.0f;
        float beta = 0.0f;
        if (valid) {
          const float score = partial_s[0] * softmax_scale;
          const float new_m = fmaxf(m, score);
          alpha = l == 0.0f ? 0.0f : expf(m - new_m);
          beta = expf(score - new_m);
          l = l * alpha + beta;
          m = new_m;
        }
        partial_s[0] = alpha;
        partial_s[1] = beta;
        partial_s[2] = l;
      }
      __syncthreads();

      const float alpha = partial_s[0];
      const float beta = partial_s[1];
      for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc_s[d] = acc_s[d] * alpha + beta * v_tile[tile_idx * head_dim + d];
      }
      __syncthreads();
    }
  }

  const float denom = partial_s[2];
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    out[out_base + d] = static_cast<scalar_t>(acc_s[d] / denom);
  }
}

template <typename scalar_t>
__global__ void varlen_causal_attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    int64_t total_tokens,
    int64_t num_seqs,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t tile_n,
    int64_t window_left,
    float softmax_scale) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* acc_s = q_s + head_dim;
  float* partial_s = acc_s + head_dim;
  float* k_tile = partial_s + blockDim.x;
  float* v_tile = k_tile + tile_n * head_dim;

  const int64_t row = blockIdx.x;
  const int64_t token_idx = row / q_heads;
  const int64_t h = row - token_idx * q_heads;
  const int64_t kv_group = q_heads / kv_heads;
  const int64_t kv_h = h / kv_group;
  const int64_t seq_id = find_sequence_id(cu_seqlens, num_seqs, token_idx);
  const int64_t seq_start = static_cast<int64_t>(cu_seqlens[seq_id]);
  const int64_t local_q = token_idx - seq_start;
  const int64_t first_key = window_left >= 0 && local_q > window_left ? token_idx - window_left : seq_start;
  const int64_t last_key = token_idx;

  const int64_t q_base = (token_idx * q_heads + h) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = static_cast<float>(q[q_base + d]);
    acc_s[d] = 0.0f;
  }
  __syncthreads();

  float m = -std::numeric_limits<float>::infinity();
  float l = 0.0f;

  for (int64_t tile_start = first_key; tile_start <= last_key; tile_start += tile_n) {
    const int64_t remaining = last_key - tile_start + 1;
    const int64_t tile_len = remaining < tile_n ? remaining : tile_n;
    const int64_t tile_elements = tile_len * head_dim;
    for (int64_t idx = threadIdx.x; idx < tile_elements; idx += blockDim.x) {
      const int64_t ki = tile_start + idx / head_dim;
      const int64_t d = idx % head_dim;
      const int64_t kv_offset = (ki * kv_heads + kv_h) * head_dim + d;
      k_tile[idx] = static_cast<float>(k[kv_offset]);
      v_tile[idx] = static_cast<float>(v[kv_offset]);
    }
    __syncthreads();

    for (int64_t tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
      float thread_dot = 0.0f;
      for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        thread_dot += q_s[d] * k_tile[tile_idx * head_dim + d];
      }
      partial_s[threadIdx.x] = thread_dot;
      __syncthreads();

      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        const float score = partial_s[0] * softmax_scale;
        const float new_m = fmaxf(m, score);
        const float alpha = l == 0.0f ? 0.0f : expf(m - new_m);
        const float beta = expf(score - new_m);
        l = l * alpha + beta;
        m = new_m;
        partial_s[0] = alpha;
        partial_s[1] = beta;
        partial_s[2] = l;
      }
      __syncthreads();

      const float alpha = partial_s[0];
      const float beta = partial_s[1];
      for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc_s[d] = acc_s[d] * alpha + beta * v_tile[tile_idx * head_dim + d];
      }
      __syncthreads();
    }
  }

  const float denom = partial_s[2];
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    out[q_base + d] = static_cast<scalar_t>(acc_s[d] / denom);
  }
}

template <typename scalar_t>
__global__ void causal_attention_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ out,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int64_t batch,
    int64_t heads,
    int64_t q_len,
    int64_t k_len,
    int64_t head_dim,
    int64_t query_start,
    int64_t window_left,
    float softmax_scale) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* go_s = q_s + head_dim;
  float* out_s = go_s + head_dim;
  float* dq_s = out_s + head_dim;
  float* partial_s = dq_s + head_dim;

  const int64_t row = blockIdx.x;
  const int64_t b = row / (heads * q_len);
  const int64_t rem = row - b * heads * q_len;
  const int64_t h = rem / q_len;
  const int64_t qi = rem - h * q_len;
  const int64_t abs_q = query_start + qi;

  const int64_t q_base = ((b * heads + h) * q_len + qi) * head_dim;
  const int64_t k_base = (b * heads + h) * k_len * head_dim;
  const int64_t v_base = (b * heads + h) * k_len * head_dim;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = static_cast<float>(q[q_base + d]);
    go_s[d] = static_cast<float>(grad_out[q_base + d]);
    out_s[d] = static_cast<float>(out[q_base + d]);
    dq_s[d] = 0.0f;
  }
  __syncthreads();

  float max_score = -std::numeric_limits<float>::infinity();
  for (int64_t ki = 0; ki < k_len; ++ki) {
    if (ki > abs_q || (window_left >= 0 && ki < abs_q - window_left)) {
      continue;
    }
    float thread_dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      thread_dot += q_s[d] * static_cast<float>(k[k_base + ki * head_dim + d]);
    }
    partial_s[threadIdx.x] = thread_dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      max_score = fmaxf(max_score, partial_s[0] * softmax_scale);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_s[0] = max_score;
  }
  __syncthreads();
  max_score = partial_s[0];
  __syncthreads();

  float denom = 0.0f;
  for (int64_t ki = 0; ki < k_len; ++ki) {
    if (ki > abs_q || (window_left >= 0 && ki < abs_q - window_left)) {
      continue;
    }
    float thread_dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      thread_dot += q_s[d] * static_cast<float>(k[k_base + ki * head_dim + d]);
    }
    partial_s[threadIdx.x] = thread_dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      denom += expf(partial_s[0] * softmax_scale - max_score);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_s[0] = denom;
  }
  __syncthreads();
  denom = partial_s[0];
  __syncthreads();

  for (int64_t ki = 0; ki < k_len; ++ki) {
    if (ki > abs_q || (window_left >= 0 && ki < abs_q - window_left)) {
      continue;
    }
    float thread_qk = 0.0f;
    float thread_go_v_minus_out = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const float k_val = static_cast<float>(k[k_base + ki * head_dim + d]);
      const float v_val = static_cast<float>(v[v_base + ki * head_dim + d]);
      thread_qk += q_s[d] * k_val;
      thread_go_v_minus_out += go_s[d] * (v_val - out_s[d]);
    }
    partial_s[threadIdx.x] = thread_qk;
    partial_s[blockDim.x + threadIdx.x] = thread_go_v_minus_out;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
        partial_s[blockDim.x + threadIdx.x] += partial_s[blockDim.x + threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float p = expf(partial_s[0] * softmax_scale - max_score) / denom;
    const float ds = p * partial_s[blockDim.x];
    const float scaled_ds = ds * softmax_scale;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const int64_t offset = k_base + ki * head_dim + d;
      const float q_val = q_s[d];
      const float k_val = static_cast<float>(k[offset]);
      const float go_val = go_s[d];
      dq_s[d] += scaled_ds * k_val;
      atomicAdd(grad_k + offset, scaled_ds * q_val);
      atomicAdd(grad_v + offset, p * go_val);
    }
    __syncthreads();
  }

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    grad_q[q_base + d] = dq_s[d];
  }
}

template <typename scalar_t>
__global__ void varlen_causal_attention_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int64_t total_tokens,
    int64_t num_seqs,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t window_left,
    float softmax_scale) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* go_s = q_s + head_dim;
  float* out_s = go_s + head_dim;
  float* dq_s = out_s + head_dim;
  float* partial_s = dq_s + head_dim;

  const int64_t row = blockIdx.x;
  const int64_t token_idx = row / q_heads;
  const int64_t h = row - token_idx * q_heads;
  const int64_t kv_group = q_heads / kv_heads;
  const int64_t kv_h = h / kv_group;
  const int64_t seq_id = find_sequence_id(cu_seqlens, num_seqs, token_idx);
  const int64_t seq_start = static_cast<int64_t>(cu_seqlens[seq_id]);
  const int64_t local_q = token_idx - seq_start;
  const int64_t first_key = window_left >= 0 && local_q > window_left ? token_idx - window_left : seq_start;
  const int64_t last_key = token_idx;
  const int64_t q_base = (token_idx * q_heads + h) * head_dim;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = static_cast<float>(q[q_base + d]);
    go_s[d] = static_cast<float>(grad_out[q_base + d]);
    out_s[d] = static_cast<float>(out[q_base + d]);
    dq_s[d] = 0.0f;
  }
  __syncthreads();

  float max_score = -std::numeric_limits<float>::infinity();
  for (int64_t ki = first_key; ki <= last_key; ++ki) {
    float thread_dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      thread_dot += q_s[d] * static_cast<float>(k[(ki * kv_heads + kv_h) * head_dim + d]);
    }
    partial_s[threadIdx.x] = thread_dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      max_score = fmaxf(max_score, partial_s[0] * softmax_scale);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_s[0] = max_score;
  }
  __syncthreads();
  max_score = partial_s[0];
  __syncthreads();

  float denom = 0.0f;
  for (int64_t ki = first_key; ki <= last_key; ++ki) {
    float thread_dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      thread_dot += q_s[d] * static_cast<float>(k[(ki * kv_heads + kv_h) * head_dim + d]);
    }
    partial_s[threadIdx.x] = thread_dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      denom += expf(partial_s[0] * softmax_scale - max_score);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_s[0] = denom;
  }
  __syncthreads();
  denom = partial_s[0];
  __syncthreads();

  for (int64_t ki = first_key; ki <= last_key; ++ki) {
    float thread_qk = 0.0f;
    float thread_go_v_minus_out = 0.0f;
    const int64_t kv_base = (ki * kv_heads + kv_h) * head_dim;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const float k_val = static_cast<float>(k[kv_base + d]);
      const float v_val = static_cast<float>(v[kv_base + d]);
      thread_qk += q_s[d] * k_val;
      thread_go_v_minus_out += go_s[d] * (v_val - out_s[d]);
    }
    partial_s[threadIdx.x] = thread_qk;
    partial_s[blockDim.x + threadIdx.x] = thread_go_v_minus_out;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
        partial_s[blockDim.x + threadIdx.x] += partial_s[blockDim.x + threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float p = expf(partial_s[0] * softmax_scale - max_score) / denom;
    const float ds = p * partial_s[blockDim.x];
    const float scaled_ds = ds * softmax_scale;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      const float q_val = q_s[d];
      const float k_val = static_cast<float>(k[kv_base + d]);
      const float go_val = go_s[d];
      dq_s[d] += scaled_ds * k_val;
      atomicAdd(grad_k + kv_base + d, scaled_ds * q_val);
      atomicAdd(grad_v + kv_base + d, p * go_val);
    }
    __syncthreads();
  }

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    grad_q[q_base + d] = dq_s[d];
  }
}

template <typename scalar_t>
__global__ void paged_decode_update_kernel(
    const scalar_t* __restrict__ k_update,
    const scalar_t* __restrict__ v_update,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cache_seqlens,
    int64_t batch,
    int64_t kv_heads,
    int64_t block_size,
    int64_t max_blocks,
    int64_t head_dim) {
  const int64_t row = blockIdx.x;
  const int64_t b = row / kv_heads;
  const int64_t kv_h = row - b * kv_heads;
  const int64_t length = static_cast<int64_t>(cache_seqlens[b]) + 1;
  const int64_t abs_q = length - 1;
  const int64_t update_block_col = abs_q / block_size;
  const int64_t update_block_offset = abs_q - update_block_col * block_size;
  const int64_t update_block_id = static_cast<int64_t>(block_table[b * max_blocks + update_block_col]);
  const int64_t update_cache_base = ((update_block_id * block_size + update_block_offset) * kv_heads + kv_h) * head_dim;
  const int64_t update_base = (b * kv_heads + kv_h) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    k_cache[update_cache_base + d] = k_update[update_base + d];
    v_cache[update_cache_base + d] = v_update[update_base + d];
  }
}

template <typename scalar_t>
__global__ void paged_causal_attention_decode_split_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cache_seqlens,
    float* __restrict__ split_m,
    float* __restrict__ split_l,
    float* __restrict__ split_acc,
    int64_t batch,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t block_size,
    int64_t max_blocks,
    int64_t head_dim,
    int64_t window_left,
    int64_t num_splits,
    float softmax_scale) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* acc_s = q_s + head_dim;
  float* partial_s = acc_s + head_dim;

  const int64_t row = blockIdx.x / num_splits;
  const int64_t split = blockIdx.x - row * num_splits;
  const int64_t b = row / q_heads;
  const int64_t h = row - b * q_heads;
  const int64_t kv_group = q_heads / kv_heads;
  const int64_t kv_h = h / kv_group;
  const int64_t q_base = (b * q_heads + h) * head_dim;
  const int64_t length = static_cast<int64_t>(cache_seqlens[b]) + 1;
  const int64_t abs_q = length - 1;
  const int64_t window_start = abs_q - window_left;
  const int64_t first_allowed = window_left >= 0 ? (window_start > 0 ? window_start : static_cast<int64_t>(0)) : static_cast<int64_t>(0);
  const int64_t split_size = (length + num_splits - 1) / num_splits;
  const int64_t split_start = split * split_size;
  const int64_t split_end_raw = split_start + split_size;
  const int64_t split_end = split_end_raw < length ? split_end_raw : length;
  const int64_t start = split_start > first_allowed ? split_start : first_allowed;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = static_cast<float>(q[q_base + d]);
    acc_s[d] = 0.0f;
  }
  __syncthreads();

  float m = -std::numeric_limits<float>::infinity();
  float l = 0.0f;

  for (int64_t ki = start; ki < split_end; ++ki) {
    const int64_t block_col = ki / block_size;
    const int64_t block_offset = ki - block_col * block_size;
    const int64_t block_id = static_cast<int64_t>(block_table[b * max_blocks + block_col]);
    const int64_t cache_base = ((block_id * block_size + block_offset) * kv_heads + kv_h) * head_dim;

    float thread_dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      thread_dot += q_s[d] * static_cast<float>(k_cache[cache_base + d]);
    }
    partial_s[threadIdx.x] = thread_dot;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial_s[threadIdx.x] += partial_s[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const float score = partial_s[0] * softmax_scale;
      const float new_m = fmaxf(m, score);
      const float alpha = l == 0.0f ? 0.0f : expf(m - new_m);
      const float beta = expf(score - new_m);
      l = l * alpha + beta;
      m = new_m;
      partial_s[0] = alpha;
      partial_s[1] = beta;
      partial_s[2] = l;
    }
    __syncthreads();

    const float alpha = partial_s[0];
    const float beta = partial_s[1];
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      acc_s[d] = acc_s[d] * alpha + beta * static_cast<float>(v_cache[cache_base + d]);
    }
    __syncthreads();
  }

  const int64_t split_base = (row * num_splits + split);
  if (threadIdx.x == 0) {
    split_m[split_base] = m;
    split_l[split_base] = l;
  }
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    split_acc[split_base * head_dim + d] = acc_s[d];
  }
}

template <typename scalar_t>
__global__ void paged_causal_attention_decode_reduce_kernel(
    const float* __restrict__ split_m,
    const float* __restrict__ split_l,
    const float* __restrict__ split_acc,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t head_dim,
    int64_t num_splits) {
  extern __shared__ float shared[];
  float* acc_s = shared;
  float* state_s = acc_s + head_dim;

  const int64_t row = blockIdx.x;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    acc_s[d] = 0.0f;
  }
  __syncthreads();

  float m = -std::numeric_limits<float>::infinity();
  float l = 0.0f;
  for (int64_t split = 0; split < num_splits; ++split) {
    const int64_t split_base = row * num_splits + split;
    const float part_l = split_l[split_base];
    if (part_l == 0.0f) {
      continue;
    }
    const float part_m = split_m[split_base];
    const float new_m = fmaxf(m, part_m);
    const float alpha = l == 0.0f ? 0.0f : expf(m - new_m);
    const float beta = expf(part_m - new_m);
    const float new_l = l * alpha + part_l * beta;
    state_s[0] = alpha;
    state_s[1] = beta;
    __syncthreads();
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      acc_s[d] = acc_s[d] * state_s[0] + split_acc[split_base * head_dim + d] * state_s[1];
    }
    __syncthreads();
    m = new_m;
    l = new_l;
  }

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    out[row * head_dim + d] = static_cast<scalar_t>(acc_s[d] / l);
  }
}

}  // namespace

torch::Tensor areno_causal_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t query_start,
    int64_t window_left,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "areno_causal_attention q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "areno_causal_attention k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "areno_causal_attention v must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "areno_causal_attention q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "areno_causal_attention k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "areno_causal_attention v must be contiguous");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "areno_causal_attention expects 4D tensors");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "areno_causal_attention dtype mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "areno_causal_attention batch mismatch");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "areno_causal_attention head mismatch");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "areno_causal_attention head dim mismatch");
  TORCH_CHECK(k.size(2) == v.size(2), "areno_causal_attention key/value length mismatch");
  TORCH_CHECK(query_start >= 0, "areno_causal_attention query_start must be non-negative");
  TORCH_CHECK(query_start + q.size(2) <= k.size(2), "areno_causal_attention query positions exceed key length");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto out = torch::empty_like(q);
  const int64_t rows = q.size(0) * q.size(1) * q.size(2);
  const int64_t tile_n = attention_tile_n(q.size(3));
  const int64_t shared_floats = 2 * q.size(3) + kAttentionThreads + 2 * tile_n * q.size(3);
  const size_t shared_bytes = static_cast<size_t>(shared_floats) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "areno_causal_attention_forward", [&] {
    causal_attention_forward_kernel<scalar_t><<<static_cast<int>(rows), kAttentionThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        q.size(0),
        q.size(1),
        q.size(2),
        k.size(2),
        q.size(3),
        tile_n,
        query_start,
        window_left,
        static_cast<float>(softmax_scale));
  });
  return out;
}

std::vector<torch::Tensor> areno_causal_attention_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    int64_t query_start,
    int64_t window_left,
    double softmax_scale) {
  TORCH_CHECK(grad_out.is_cuda(), "areno_causal_attention_backward grad_out must be CUDA");
  TORCH_CHECK(q.is_cuda(), "areno_causal_attention_backward q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "areno_causal_attention_backward k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "areno_causal_attention_backward v must be CUDA");
  TORCH_CHECK(out.is_cuda(), "areno_causal_attention_backward out must be CUDA");
  TORCH_CHECK(grad_out.is_contiguous(), "areno_causal_attention_backward grad_out must be contiguous");
  TORCH_CHECK(q.is_contiguous(), "areno_causal_attention_backward q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "areno_causal_attention_backward k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "areno_causal_attention_backward v must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "areno_causal_attention_backward out must be contiguous");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "areno_causal_attention_backward expects 4D q/k/v");
  TORCH_CHECK(grad_out.sizes() == q.sizes() && out.sizes() == q.sizes(), "areno_causal_attention_backward grad/out shape mismatch");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "areno_causal_attention_backward dtype mismatch");
  TORCH_CHECK(grad_out.scalar_type() == q.scalar_type() && out.scalar_type() == q.scalar_type(), "areno_causal_attention_backward grad/out dtype mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "areno_causal_attention_backward batch mismatch");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "areno_causal_attention_backward head mismatch");
  TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "areno_causal_attention_backward head dim mismatch");
  TORCH_CHECK(k.size(2) == v.size(2), "areno_causal_attention_backward key/value length mismatch");
  TORCH_CHECK(query_start >= 0, "areno_causal_attention_backward query_start must be non-negative");
  TORCH_CHECK(query_start + q.size(2) <= k.size(2), "areno_causal_attention_backward query positions exceed key length");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto grad_options = q.options().dtype(torch::kFloat32);
  auto grad_q = torch::empty(q.sizes(), grad_options);
  auto grad_k = torch::zeros(k.sizes(), grad_options);
  auto grad_v = torch::zeros(v.sizes(), grad_options);
  const int64_t rows = q.size(0) * q.size(1) * q.size(2);
  const int64_t shared_floats = 4 * q.size(3) + 2 * kAttentionThreads;
  const size_t shared_bytes = static_cast<size_t>(shared_floats) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "areno_causal_attention_backward", [&] {
    causal_attention_backward_kernel<scalar_t><<<static_cast<int>(rows), kAttentionThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        grad_q.data_ptr<float>(),
        grad_k.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        q.size(0),
        q.size(1),
        q.size(2),
        k.size(2),
        q.size(3),
        query_start,
        window_left,
        static_cast<float>(softmax_scale));
  });
  return {grad_q, grad_k, grad_v};
}

torch::Tensor areno_varlen_causal_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens,
    int64_t window_left,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "areno_varlen_causal_attention q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "areno_varlen_causal_attention k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "areno_varlen_causal_attention v must be CUDA");
  TORCH_CHECK(cu_seqlens.is_cuda(), "areno_varlen_causal_attention cu_seqlens must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "areno_varlen_causal_attention q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "areno_varlen_causal_attention k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "areno_varlen_causal_attention v must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "areno_varlen_causal_attention cu_seqlens must be contiguous");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "areno_varlen_causal_attention expects 3D q/k/v");
  TORCH_CHECK(cu_seqlens.dim() == 1, "areno_varlen_causal_attention cu_seqlens must be 1D");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "areno_varlen_causal_attention cu_seqlens must be int32");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "areno_varlen_causal_attention dtype mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "areno_varlen_causal_attention token count mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "areno_varlen_causal_attention key/value head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "areno_varlen_causal_attention q heads must be divisible by kv heads");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "areno_varlen_causal_attention head dim mismatch");
  TORCH_CHECK(cu_seqlens.size(0) >= 2, "areno_varlen_causal_attention cu_seqlens must contain at least one sequence");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto out = torch::empty_like(q);
  const int64_t rows = q.size(0) * q.size(1);
  const int64_t tile_n = attention_tile_n(q.size(2));
  const int64_t shared_floats = 2 * q.size(2) + kAttentionThreads + 2 * tile_n * q.size(2);
  const size_t shared_bytes = static_cast<size_t>(shared_floats) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "areno_varlen_causal_attention_forward", [&] {
    varlen_causal_attention_forward_kernel<scalar_t><<<static_cast<int>(rows), kAttentionThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        q.size(0),
        cu_seqlens.size(0) - 1,
        q.size(1),
        k.size(1),
        q.size(2),
        tile_n,
        window_left,
        static_cast<float>(softmax_scale));
  });
  return out;
}

std::vector<torch::Tensor> areno_varlen_causal_attention_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor cu_seqlens,
    int64_t window_left,
    double softmax_scale) {
  TORCH_CHECK(grad_out.is_cuda(), "areno_varlen_causal_attention_backward grad_out must be CUDA");
  TORCH_CHECK(q.is_cuda(), "areno_varlen_causal_attention_backward q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "areno_varlen_causal_attention_backward k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "areno_varlen_causal_attention_backward v must be CUDA");
  TORCH_CHECK(out.is_cuda(), "areno_varlen_causal_attention_backward out must be CUDA");
  TORCH_CHECK(cu_seqlens.is_cuda(), "areno_varlen_causal_attention_backward cu_seqlens must be CUDA");
  TORCH_CHECK(grad_out.is_contiguous(), "areno_varlen_causal_attention_backward grad_out must be contiguous");
  TORCH_CHECK(q.is_contiguous(), "areno_varlen_causal_attention_backward q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "areno_varlen_causal_attention_backward k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "areno_varlen_causal_attention_backward v must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "areno_varlen_causal_attention_backward out must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "areno_varlen_causal_attention_backward cu_seqlens must be contiguous");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "areno_varlen_causal_attention_backward expects 3D q/k/v");
  TORCH_CHECK(cu_seqlens.dim() == 1, "areno_varlen_causal_attention_backward cu_seqlens must be 1D");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "areno_varlen_causal_attention_backward cu_seqlens must be int32");
  TORCH_CHECK(grad_out.sizes() == q.sizes() && out.sizes() == q.sizes(), "areno_varlen_causal_attention_backward grad/out shape mismatch");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "areno_varlen_causal_attention_backward dtype mismatch");
  TORCH_CHECK(grad_out.scalar_type() == q.scalar_type() && out.scalar_type() == q.scalar_type(), "areno_varlen_causal_attention_backward grad/out dtype mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "areno_varlen_causal_attention_backward token count mismatch");
  TORCH_CHECK(k.size(1) == v.size(1), "areno_varlen_causal_attention_backward key/value head mismatch");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "areno_varlen_causal_attention_backward q heads must be divisible by kv heads");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "areno_varlen_causal_attention_backward head dim mismatch");
  TORCH_CHECK(cu_seqlens.size(0) >= 2, "areno_varlen_causal_attention_backward cu_seqlens must contain at least one sequence");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto grad_options = q.options().dtype(torch::kFloat32);
  auto grad_q = torch::empty(q.sizes(), grad_options);
  auto grad_k = torch::zeros(k.sizes(), grad_options);
  auto grad_v = torch::zeros(v.sizes(), grad_options);
  const int64_t rows = q.size(0) * q.size(1);
  const int64_t shared_floats = 4 * q.size(2) + 2 * kAttentionThreads;
  const size_t shared_bytes = static_cast<size_t>(shared_floats) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "areno_varlen_causal_attention_backward", [&] {
    varlen_causal_attention_backward_kernel<scalar_t><<<static_cast<int>(rows), kAttentionThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        grad_q.data_ptr<float>(),
        grad_k.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        q.size(0),
        cu_seqlens.size(0) - 1,
        q.size(1),
        k.size(1),
        q.size(2),
        window_left,
        static_cast<float>(softmax_scale));
  });
  return {grad_q, grad_k, grad_v};
}

torch::Tensor areno_paged_causal_attention_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_update,
    torch::Tensor v_update,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor cache_seqlens,
    int64_t window_left,
    int64_t num_splits,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "areno_paged_causal_attention_decode q must be CUDA");
  TORCH_CHECK(k_update.is_cuda(), "areno_paged_causal_attention_decode k_update must be CUDA");
  TORCH_CHECK(v_update.is_cuda(), "areno_paged_causal_attention_decode v_update must be CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "areno_paged_causal_attention_decode k_cache must be CUDA");
  TORCH_CHECK(v_cache.is_cuda(), "areno_paged_causal_attention_decode v_cache must be CUDA");
  TORCH_CHECK(block_table.is_cuda(), "areno_paged_causal_attention_decode block_table must be CUDA");
  TORCH_CHECK(cache_seqlens.is_cuda(), "areno_paged_causal_attention_decode cache_seqlens must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "areno_paged_causal_attention_decode q must be contiguous");
  TORCH_CHECK(k_update.is_contiguous(), "areno_paged_causal_attention_decode k_update must be contiguous");
  TORCH_CHECK(v_update.is_contiguous(), "areno_paged_causal_attention_decode v_update must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous(), "areno_paged_causal_attention_decode k_cache must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "areno_paged_causal_attention_decode v_cache must be contiguous");
  TORCH_CHECK(block_table.is_contiguous(), "areno_paged_causal_attention_decode block_table must be contiguous");
  TORCH_CHECK(cache_seqlens.is_contiguous(), "areno_paged_causal_attention_decode cache_seqlens must be contiguous");
  TORCH_CHECK(q.dim() == 3, "areno_paged_causal_attention_decode q must be shaped (batch, heads, head_dim)");
  TORCH_CHECK(k_update.dim() == 3 && v_update.dim() == 3, "areno_paged_causal_attention_decode updates must be 3D");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "areno_paged_causal_attention_decode cache tensors must be 4D");
  TORCH_CHECK(block_table.dim() == 2, "areno_paged_causal_attention_decode block_table must be 2D");
  TORCH_CHECK(cache_seqlens.dim() == 1, "areno_paged_causal_attention_decode cache_seqlens must be 1D");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(), "areno_paged_causal_attention_decode dtype mismatch");
  TORCH_CHECK(k_update.scalar_type() == q.scalar_type() && v_update.scalar_type() == q.scalar_type(), "areno_paged_causal_attention_decode update dtype mismatch");
  TORCH_CHECK(block_table.scalar_type() == at::kInt, "areno_paged_causal_attention_decode block_table must be int32");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "areno_paged_causal_attention_decode cache_seqlens must be int32");
  TORCH_CHECK(q.size(0) == block_table.size(0) && q.size(0) == cache_seqlens.size(0), "areno_paged_causal_attention_decode batch mismatch");
  TORCH_CHECK(k_update.size(0) == q.size(0) && v_update.size(0) == q.size(0), "areno_paged_causal_attention_decode update batch mismatch");
  TORCH_CHECK(k_cache.size(0) == v_cache.size(0) && k_cache.size(1) == v_cache.size(1), "areno_paged_causal_attention_decode cache block shape mismatch");
  TORCH_CHECK(k_cache.size(2) == v_cache.size(2), "areno_paged_causal_attention_decode cache head mismatch");
  TORCH_CHECK(k_update.size(1) == k_cache.size(2) && v_update.size(1) == v_cache.size(2), "areno_paged_causal_attention_decode update head mismatch");
  TORCH_CHECK(q.size(2) == k_cache.size(3) && q.size(2) == v_cache.size(3), "areno_paged_causal_attention_decode head dim mismatch");
  TORCH_CHECK(k_update.size(2) == k_cache.size(3) && v_update.size(2) == v_cache.size(3), "areno_paged_causal_attention_decode update head dim mismatch");
  TORCH_CHECK(q.size(1) % k_cache.size(2) == 0, "areno_paged_causal_attention_decode q heads must be divisible by kv heads");
  TORCH_CHECK(num_splits >= 1, "areno_paged_causal_attention_decode num_splits must be >= 1");

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto out = torch::empty_like(q);
  const int64_t rows = q.size(0) * q.size(1);
  const int64_t shared_floats = 2 * q.size(2) + kAttentionThreads;
  const size_t shared_bytes = static_cast<size_t>(shared_floats) * sizeof(float);
  auto split_options = q.options().dtype(torch::kFloat32);
  auto split_m = torch::empty({rows, num_splits}, split_options);
  auto split_l = torch::empty({rows, num_splits}, split_options);
  auto split_acc = torch::empty({rows, num_splits, q.size(2)}, split_options);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "areno_paged_causal_attention_decode_forward", [&] {
    paged_decode_update_kernel<scalar_t><<<static_cast<int>(q.size(0) * k_cache.size(2)), kAttentionThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        k_update.data_ptr<scalar_t>(),
        v_update.data_ptr<scalar_t>(),
        k_cache.data_ptr<scalar_t>(),
        v_cache.data_ptr<scalar_t>(),
        block_table.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        q.size(0),
        k_cache.size(2),
        k_cache.size(1),
        block_table.size(1),
        q.size(2));
    paged_causal_attention_decode_split_kernel<scalar_t><<<static_cast<int>(rows * num_splits), kAttentionThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<scalar_t>(),
        k_cache.data_ptr<scalar_t>(),
        v_cache.data_ptr<scalar_t>(),
        block_table.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        split_m.data_ptr<float>(),
        split_l.data_ptr<float>(),
        split_acc.data_ptr<float>(),
        q.size(0),
        q.size(1),
        k_cache.size(2),
        k_cache.size(1),
        block_table.size(1),
        q.size(2),
        window_left,
        num_splits,
        static_cast<float>(softmax_scale));
    paged_causal_attention_decode_reduce_kernel<scalar_t><<<static_cast<int>(rows), kAttentionThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        split_m.data_ptr<float>(),
        split_l.data_ptr<float>(),
        split_acc.data_ptr<float>(),
        out.data_ptr<scalar_t>(),
        rows,
        q.size(2),
        num_splits);
  });
  return out;
}
