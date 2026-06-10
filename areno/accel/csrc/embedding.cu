#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "atomic_utils.cuh"

namespace areno_accel {

template <typename scalar_t>
__global__ void vocab_embedding_forward_kernel(
    const int64_t* __restrict__ input_ids,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t elements,
    int hidden,
    int64_t vocab_start,
    int64_t vocab_end) {
  int64_t total = elements * hidden;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int hidden_idx = linear % hidden;
    int64_t token = linear / hidden;
    int64_t id = input_ids[token];
    if (id >= vocab_start && id < vocab_end) {
      output[linear] = weight[(id - vocab_start) * hidden + hidden_idx];
    } else {
      output[linear] = static_cast<scalar_t>(0);
    }
  }
}

template <typename scalar_t>
__global__ void vocab_embedding_backward_kernel(
    const int64_t* __restrict__ input_ids,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    int64_t elements,
    int hidden,
    int64_t vocab_start,
    int64_t vocab_end) {
  int64_t total = elements * hidden;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int hidden_idx = linear % hidden;
    int64_t token = linear / hidden;
    int64_t id = input_ids[token];
    if (id >= vocab_start && id < vocab_end) {
      atomic_add(grad_weight + (id - vocab_start) * hidden + hidden_idx, grad_output[linear]);
    }
  }
}

}  // namespace areno_accel

torch::Tensor areno_vocab_embedding_forward_cuda(torch::Tensor input_ids, torch::Tensor weight, int64_t vocab_start, int64_t vocab_end) {
  TORCH_CHECK(input_ids.is_cuda(), "areno_vocab_embedding input_ids must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_vocab_embedding weight must be CUDA");
  TORCH_CHECK(input_ids.scalar_type() == at::kLong, "areno_vocab_embedding input_ids must be int64");
  TORCH_CHECK(weight.dim() == 2, "areno_vocab_embedding weight must be 2D");
  auto output_shape = input_ids.sizes().vec();
  output_shape.push_back(weight.size(1));
  auto output = torch::empty(output_shape, weight.options());
  int64_t elements = input_ids.numel();
  int hidden = static_cast<int>(weight.size(1));
  int threads = 256;
  int blocks = static_cast<int>(std::min<int64_t>((elements * hidden + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(weight));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "areno_vocab_embedding_forward", [&] {
    areno_accel::vocab_embedding_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input_ids.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        elements,
        hidden,
        vocab_start,
        vocab_end);
  });
  return output;
}

torch::Tensor areno_vocab_embedding_backward_cuda(torch::Tensor grad_output, torch::Tensor input_ids, torch::Tensor weight, int64_t vocab_start, int64_t vocab_end) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_vocab_embedding grad_output must be CUDA");
  TORCH_CHECK(input_ids.is_cuda(), "areno_vocab_embedding input_ids must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_vocab_embedding weight must be CUDA");
  grad_output = grad_output.contiguous();
  auto grad_weight = torch::zeros_like(weight);
  int64_t elements = input_ids.numel();
  int hidden = static_cast<int>(weight.size(1));
  int threads = 256;
  int blocks = static_cast<int>(std::min<int64_t>((elements * hidden + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(weight));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "areno_vocab_embedding_backward", [&] {
    areno_accel::vocab_embedding_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input_ids.data_ptr<int64_t>(),
        grad_output.data_ptr<scalar_t>(),
        grad_weight.data_ptr<scalar_t>(),
        elements,
        hidden,
        vocab_start,
        vocab_end);
  });
  return grad_weight;
}
