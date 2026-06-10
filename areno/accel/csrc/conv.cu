#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace areno_accel {

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float d_silu(float x) {
  float sigmoid = 1.0f / (1.0f + expf(-x));
  return sigmoid * (1.0f + x * (1.0f - sigmoid));
}

template <typename scalar_t>
__global__ void depthwise_causal_conv1d_silu_forward_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    scalar_t* __restrict__ output,
    float* __restrict__ preact,
    int batch,
    int seqlen,
    int channels,
    int kernel_size) {
  int64_t total = static_cast<int64_t>(batch) * seqlen * channels;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int channel = linear % channels;
    int token = (linear / channels) % seqlen;
    int batch_idx = static_cast<int>(linear / (static_cast<int64_t>(seqlen) * channels));
    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
      int src_token = token + k - (kernel_size - 1);
      if (src_token >= 0) {
        int64_t input_idx = (static_cast<int64_t>(batch_idx) * seqlen + src_token) * channels + channel;
        acc += to_float(input[input_idx]) * weight[channel * kernel_size + k];
      }
    }
    preact[linear] = acc;
    output[linear] = from_float<scalar_t>(silu(acc));
  }
}

template <typename scalar_t>
__global__ void depthwise_causal_conv1d_silu_decode_kernel(
    const scalar_t* __restrict__ current,
    const scalar_t* __restrict__ history,
    const float* __restrict__ weight,
    scalar_t* __restrict__ output,
    float* __restrict__ preact,
    int rows,
    int channels,
    int kernel_size) {
  int64_t total = static_cast<int64_t>(rows) * channels;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int channel = linear % channels;
    int row = static_cast<int>(linear / channels);
    float acc = 0.0f;
    for (int k = 0; k < kernel_size - 1; ++k) {
      acc += to_float(history[(static_cast<int64_t>(row) * channels + channel) * (kernel_size - 1) + k]) * weight[channel * kernel_size + k];
    }
    acc += to_float(current[linear]) * weight[channel * kernel_size + kernel_size - 1];
    preact[linear] = acc;
    output[linear] = from_float<scalar_t>(silu(acc));
  }
}

template <typename scalar_t>
__global__ void depthwise_causal_conv1d_silu_grad_input_kernel(
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ preact,
    const float* __restrict__ weight,
    scalar_t* __restrict__ grad_input,
    int batch,
    int seqlen,
    int channels,
    int kernel_size) {
  int64_t total = static_cast<int64_t>(batch) * seqlen * channels;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int channel = linear % channels;
    int token = (linear / channels) % seqlen;
    int batch_idx = static_cast<int>(linear / (static_cast<int64_t>(seqlen) * channels));
    float grad = 0.0f;
    int max_out = min(seqlen - 1, token + kernel_size - 1);
    for (int out_token = token; out_token <= max_out; ++out_token) {
      int k = token - out_token + kernel_size - 1;
      int64_t out_idx = (static_cast<int64_t>(batch_idx) * seqlen + out_token) * channels + channel;
      grad += to_float(grad_output[out_idx]) * d_silu(preact[out_idx]) * weight[channel * kernel_size + k];
    }
    grad_input[linear] = from_float<scalar_t>(grad);
  }
}

template <typename scalar_t>
__global__ void depthwise_causal_conv1d_silu_grad_weight_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ preact,
    float* __restrict__ grad_weight,
    int batch,
    int seqlen,
    int channels,
    int kernel_size) {
  int64_t total = static_cast<int64_t>(channels) * kernel_size;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int k = linear % kernel_size;
    int channel = static_cast<int>(linear / kernel_size);
    float grad = 0.0f;
    for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
      for (int out_token = 0; out_token < seqlen; ++out_token) {
        int src_token = out_token + k - (kernel_size - 1);
        if (src_token >= 0) {
          int64_t out_idx = (static_cast<int64_t>(batch_idx) * seqlen + out_token) * channels + channel;
          int64_t in_idx = (static_cast<int64_t>(batch_idx) * seqlen + src_token) * channels + channel;
          grad += to_float(grad_output[out_idx]) * d_silu(preact[out_idx]) * to_float(input[in_idx]);
        }
      }
    }
    grad_weight[linear] = grad;
  }
}

__device__ __forceinline__ int find_segment(const int32_t* __restrict__ cu_seqlens, int segments, int token) {
  int lo = 0;
  int hi = segments;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (cu_seqlens[mid] <= token) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename scalar_t>
__global__ void packed_depthwise_causal_conv1d_silu_forward_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ output,
    float* __restrict__ preact,
    int tokens,
    int channels,
    int kernel_size,
    int segments) {
  int64_t total = static_cast<int64_t>(tokens) * channels;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int channel = linear % channels;
    int token = static_cast<int>(linear / channels);
    int seq = find_segment(cu_seqlens, segments, token);
    int seq_start = cu_seqlens[seq];
    int local_token = token - seq_start;
    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
      int src_local = local_token + k - (kernel_size - 1);
      if (src_local >= 0) {
        int64_t input_idx = (static_cast<int64_t>(seq_start + src_local) * channels) + channel;
        acc += to_float(input[input_idx]) * weight[channel * kernel_size + k];
      }
    }
    preact[linear] = acc;
    output[linear] = from_float<scalar_t>(silu(acc));
  }
}

template <typename scalar_t>
__global__ void packed_depthwise_causal_conv1d_silu_grad_input_kernel(
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ preact,
    const float* __restrict__ weight,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ grad_input,
    int tokens,
    int channels,
    int kernel_size,
    int segments) {
  int64_t total = static_cast<int64_t>(tokens) * channels;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int channel = linear % channels;
    int token = static_cast<int>(linear / channels);
    int seq = find_segment(cu_seqlens, segments, token);
    int seq_start = cu_seqlens[seq];
    int seq_end = cu_seqlens[seq + 1];
    int local_token = token - seq_start;
    int max_out_local = min(seq_end - seq_start - 1, local_token + kernel_size - 1);
    float grad = 0.0f;
    for (int out_local = local_token; out_local <= max_out_local; ++out_local) {
      int k = local_token - out_local + kernel_size - 1;
      int64_t out_idx = (static_cast<int64_t>(seq_start + out_local) * channels) + channel;
      grad += to_float(grad_output[out_idx]) * d_silu(preact[out_idx]) * weight[channel * kernel_size + k];
    }
    grad_input[linear] = from_float<scalar_t>(grad);
  }
}

template <typename scalar_t>
__global__ void packed_depthwise_causal_conv1d_silu_grad_weight_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ preact,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ grad_weight,
    int channels,
    int kernel_size,
    int segments) {
  int64_t total = static_cast<int64_t>(channels) * kernel_size;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int k = linear % kernel_size;
    int channel = static_cast<int>(linear / kernel_size);
    float grad = 0.0f;
    for (int seq = 0; seq < segments; ++seq) {
      int seq_start = cu_seqlens[seq];
      int seq_end = cu_seqlens[seq + 1];
      for (int out_token = seq_start; out_token < seq_end; ++out_token) {
        int src_token = out_token + k - (kernel_size - 1);
        if (src_token >= seq_start) {
          int64_t out_idx = (static_cast<int64_t>(out_token) * channels) + channel;
          int64_t in_idx = (static_cast<int64_t>(src_token) * channels) + channel;
          grad += to_float(grad_output[out_idx]) * d_silu(preact[out_idx]) * to_float(input[in_idx]);
        }
      }
    }
    grad_weight[linear] = grad;
  }
}

}  // namespace areno_accel

std::vector<torch::Tensor> areno_depthwise_causal_conv1d_silu_forward_cuda(torch::Tensor input, torch::Tensor weight) {
  TORCH_CHECK(input.is_cuda(), "areno_depthwise_causal_conv1d_silu input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_depthwise_causal_conv1d_silu weight must be CUDA");
  TORCH_CHECK(input.dim() == 3, "areno_depthwise_causal_conv1d_silu input must be (batch, seqlen, channels)");
  TORCH_CHECK(weight.dim() == 3 && weight.size(1) == 1, "areno_depthwise_causal_conv1d_silu weight must be (channels, 1, kernel)");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_depthwise_causal_conv1d_silu weight must be float32");
  TORCH_CHECK(input.size(2) == weight.size(0), "areno_depthwise_causal_conv1d_silu channel mismatch");
  auto output = torch::empty_like(input);
  auto preact = torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));
  int batch = static_cast<int>(input.size(0));
  int seqlen = static_cast<int>(input.size(1));
  int channels = static_cast<int>(input.size(2));
  int kernel_size = static_cast<int>(weight.size(2));
  int threads = 256;
  int blocks = static_cast<int>(std::min<int64_t>((input.numel() + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_depthwise_causal_conv1d_silu_forward", [&] {
    areno_accel::depthwise_causal_conv1d_silu_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        preact.data_ptr<float>(),
        batch,
        seqlen,
        channels,
        kernel_size);
  });
  return {output, preact};
}

std::vector<torch::Tensor> areno_depthwise_causal_conv1d_silu_decode_cuda(
    torch::Tensor current,
    torch::Tensor history,
    torch::Tensor weight) {
  TORCH_CHECK(current.is_cuda(), "areno_depthwise_causal_conv1d_silu_decode current must be CUDA");
  TORCH_CHECK(history.is_cuda(), "areno_depthwise_causal_conv1d_silu_decode history must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_depthwise_causal_conv1d_silu_decode weight must be CUDA");
  TORCH_CHECK(current.dim() == 2, "areno_depthwise_causal_conv1d_silu_decode current must be (rows, channels)");
  TORCH_CHECK(history.dim() == 3, "areno_depthwise_causal_conv1d_silu_decode history must be (rows, channels, kernel - 1)");
  TORCH_CHECK(weight.dim() == 3 && weight.size(1) == 1, "areno_depthwise_causal_conv1d_silu_decode weight must be (channels, 1, kernel)");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_depthwise_causal_conv1d_silu_decode weight must be float32");
  TORCH_CHECK(current.size(0) == history.size(0), "areno_depthwise_causal_conv1d_silu_decode row mismatch");
  TORCH_CHECK(current.size(1) == history.size(1) && current.size(1) == weight.size(0), "areno_depthwise_causal_conv1d_silu_decode channel mismatch");
  TORCH_CHECK(history.size(2) + 1 == weight.size(2), "areno_depthwise_causal_conv1d_silu_decode kernel mismatch");
  auto output = torch::empty_like(current);
  auto preact = torch::empty(current.sizes(), current.options().dtype(torch::kFloat32));
  int rows = static_cast<int>(current.size(0));
  int channels = static_cast<int>(current.size(1));
  int kernel_size = static_cast<int>(weight.size(2));
  int threads = 256;
  int blocks = static_cast<int>(std::min<int64_t>((current.numel() + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(current));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, current.scalar_type(), "areno_depthwise_causal_conv1d_silu_decode", [&] {
    areno_accel::depthwise_causal_conv1d_silu_decode_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        current.data_ptr<scalar_t>(),
        history.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        preact.data_ptr<float>(),
        rows,
        channels,
        kernel_size);
  });
  return {output, preact};
}

std::vector<torch::Tensor> areno_depthwise_causal_conv1d_silu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor preact) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_depthwise_causal_conv1d_silu grad_output must be CUDA");
  TORCH_CHECK(input.is_cuda(), "areno_depthwise_causal_conv1d_silu input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_depthwise_causal_conv1d_silu weight must be CUDA");
  TORCH_CHECK(preact.is_cuda(), "areno_depthwise_causal_conv1d_silu preact must be CUDA");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_depthwise_causal_conv1d_silu weight must be float32");
  auto grad_input = torch::empty_like(input);
  auto grad_weight = torch::empty_like(weight);
  int batch = static_cast<int>(input.size(0));
  int seqlen = static_cast<int>(input.size(1));
  int channels = static_cast<int>(input.size(2));
  int kernel_size = static_cast<int>(weight.size(2));
  int threads = 256;
  int blocks_input = static_cast<int>(std::min<int64_t>((input.numel() + threads - 1) / threads, 4096));
  int blocks_weight = static_cast<int>(std::min<int64_t>((weight.numel() + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_depthwise_causal_conv1d_silu_backward", [&] {
    areno_accel::depthwise_causal_conv1d_silu_grad_input_kernel<scalar_t><<<blocks_input, threads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        preact.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_input.data_ptr<scalar_t>(),
        batch,
        seqlen,
        channels,
        kernel_size);
    areno_accel::depthwise_causal_conv1d_silu_grad_weight_kernel<scalar_t><<<blocks_weight, threads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        preact.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        batch,
        seqlen,
        channels,
        kernel_size);
  });
  return {grad_input, grad_weight};
}

std::vector<torch::Tensor> areno_packed_depthwise_causal_conv1d_silu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor cu_seqlens) {
  TORCH_CHECK(input.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu weight must be CUDA");
  TORCH_CHECK(cu_seqlens.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu cu_seqlens must be CUDA");
  TORCH_CHECK(input.dim() == 3 && input.size(0) == 1, "areno_packed_depthwise_causal_conv1d_silu input must be (1, tokens, channels)");
  TORCH_CHECK(weight.dim() == 3 && weight.size(1) == 1, "areno_packed_depthwise_causal_conv1d_silu weight must be (channels, 1, kernel)");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == at::kInt, "areno_packed_depthwise_causal_conv1d_silu cu_seqlens must be int32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_packed_depthwise_causal_conv1d_silu weight must be float32");
  TORCH_CHECK(input.size(2) == weight.size(0), "areno_packed_depthwise_causal_conv1d_silu channel mismatch");
  TORCH_CHECK(cu_seqlens.size(0) >= 2, "areno_packed_depthwise_causal_conv1d_silu cu_seqlens must contain at least one segment");
  auto output = torch::empty_like(input);
  auto preact = torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));
  int tokens = static_cast<int>(input.size(1));
  int channels = static_cast<int>(input.size(2));
  int kernel_size = static_cast<int>(weight.size(2));
  int segments = static_cast<int>(cu_seqlens.size(0) - 1);
  int threads = 256;
  int blocks = static_cast<int>(std::min<int64_t>((input.numel() + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_packed_depthwise_causal_conv1d_silu_forward", [&] {
    areno_accel::packed_depthwise_causal_conv1d_silu_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        cu_seqlens.data_ptr<int32_t>(),
        output.data_ptr<scalar_t>(),
        preact.data_ptr<float>(),
        tokens,
        channels,
        kernel_size,
        segments);
  });
  return {output, preact};
}

std::vector<torch::Tensor> areno_packed_depthwise_causal_conv1d_silu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor cu_seqlens,
    torch::Tensor preact) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu grad_output must be CUDA");
  TORCH_CHECK(input.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu weight must be CUDA");
  TORCH_CHECK(cu_seqlens.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu cu_seqlens must be CUDA");
  TORCH_CHECK(preact.is_cuda(), "areno_packed_depthwise_causal_conv1d_silu preact must be CUDA");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "areno_packed_depthwise_causal_conv1d_silu cu_seqlens must be int32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_packed_depthwise_causal_conv1d_silu weight must be float32");
  auto grad_input = torch::empty_like(input);
  auto grad_weight = torch::empty_like(weight);
  int tokens = static_cast<int>(input.size(1));
  int channels = static_cast<int>(input.size(2));
  int kernel_size = static_cast<int>(weight.size(2));
  int segments = static_cast<int>(cu_seqlens.size(0) - 1);
  int threads = 256;
  int blocks_input = static_cast<int>(std::min<int64_t>((input.numel() + threads - 1) / threads, 4096));
  int blocks_weight = static_cast<int>(std::min<int64_t>((weight.numel() + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_packed_depthwise_causal_conv1d_silu_backward", [&] {
    areno_accel::packed_depthwise_causal_conv1d_silu_grad_input_kernel<scalar_t><<<blocks_input, threads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        preact.data_ptr<float>(),
        weight.data_ptr<float>(),
        cu_seqlens.data_ptr<int32_t>(),
        grad_input.data_ptr<scalar_t>(),
        tokens,
        channels,
        kernel_size,
        segments);
    areno_accel::packed_depthwise_causal_conv1d_silu_grad_weight_kernel<scalar_t><<<blocks_weight, threads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        preact.data_ptr<float>(),
        cu_seqlens.data_ptr<int32_t>(),
        grad_weight.data_ptr<float>(),
        channels,
        kernel_size,
        segments);
  });
  return {grad_input, grad_weight};
}
