#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>

namespace areno_accel {

constexpr int kBiasThreads = 256;

cudaDataType_t cuda_type(at::ScalarType dtype) {
  if (dtype == at::kHalf) {
    return CUDA_R_16F;
  }
  if (dtype == at::kBFloat16) {
    return CUDA_R_16BF;
  }
  if (dtype == at::kFloat) {
    return CUDA_R_32F;
  }
  TORCH_CHECK(false, "areno_linear only supports fp16, bf16, and fp32 tensors");
}

template <typename scalar_t>
__global__ void add_bias_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ bias, int64_t elements, int64_t width) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[idx] = static_cast<scalar_t>(static_cast<float>(output[idx]) + static_cast<float>(bias[idx % width]));
  }
}

template <typename scalar_t>
__global__ void bias_grad_kernel(const scalar_t* __restrict__ grad_output, scalar_t* __restrict__ grad_bias, int64_t rows, int64_t cols) {
  int col = blockIdx.x;
  float thread_sum = 0.0f;
  for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    thread_sum += static_cast<float>(grad_output[row * cols + col]);
  }
  __shared__ float scratch[256];
  scratch[threadIdx.x] = thread_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    grad_bias[col] = static_cast<scalar_t>(scratch[0]);
  }
}

template <>
__global__ void add_bias_kernel<c10::Half>(c10::Half* __restrict__ output, const c10::Half* __restrict__ bias, int64_t elements, int64_t width) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[idx] = static_cast<c10::Half>(static_cast<float>(output[idx]) + static_cast<float>(bias[idx % width]));
  }
}

template <>
__global__ void add_bias_kernel<c10::BFloat16>(c10::BFloat16* __restrict__ output, const c10::BFloat16* __restrict__ bias, int64_t elements, int64_t width) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[idx] = static_cast<c10::BFloat16>(static_cast<float>(output[idx]) + static_cast<float>(bias[idx % width]));
  }
}

void gemm_row_major(
    cublasHandle_t handle,
    cublasOperation_t op_a,
    cublasOperation_t op_b,
    int64_t m,
    int64_t n,
    int64_t k,
    const void* a,
    cudaDataType_t a_type,
    int64_t lda,
    const void* b,
    cudaDataType_t b_type,
    int64_t ldb,
    void* c,
    cudaDataType_t c_type,
    int64_t ldc) {
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      op_a,
      op_b,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      &alpha,
      a,
      a_type,
      static_cast<int>(lda),
      b,
      b_type,
      static_cast<int>(ldb),
      &beta,
      c,
      c_type,
      static_cast<int>(ldc),
      compute_type,
      algo));
}

void maybe_add_bias(torch::Tensor output, torch::Tensor bias) {
  if (bias.numel() == 0) {
    return;
  }
  int64_t elements = output.numel();
  int64_t width = output.size(-1);
  int blocks = static_cast<int>(std::min<int64_t>((elements + kBiasThreads - 1) / kBiasThreads, 4096));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, output.scalar_type(), "areno_linear_bias", [&] {
    add_bias_kernel<scalar_t><<<blocks, kBiasThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        elements,
        width);
  });
}

torch::Tensor reduce_bias_grad(torch::Tensor grad_output) {
  auto grad_bias = torch::empty({grad_output.size(-1)}, grad_output.options());
  int64_t rows = grad_output.numel() / grad_output.size(-1);
  int64_t cols = grad_output.size(-1);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad_output.scalar_type(), "areno_linear_bias_grad", [&] {
    bias_grad_kernel<scalar_t><<<static_cast<int>(cols), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output.data_ptr<scalar_t>(),
        grad_bias.data_ptr<scalar_t>(),
        rows,
        cols);
  });
  return grad_bias;
}

}  // namespace areno_accel

torch::Tensor areno_linear_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, bool use_bias) {
  TORCH_CHECK(input.is_cuda(), "areno_linear input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_linear weight must be CUDA");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "areno_linear input and weight dtype must match");
  TORCH_CHECK(input.dim() >= 2, "areno_linear input must have at least 2 dimensions");
  TORCH_CHECK(weight.dim() == 2, "areno_linear weight must be 2D");
  TORCH_CHECK(input.size(-1) == weight.size(1), "areno_linear input and weight shape mismatch");
  if (use_bias) {
    TORCH_CHECK(bias.is_cuda(), "areno_linear bias must be CUDA");
    TORCH_CHECK(bias.scalar_type() == input.scalar_type(), "areno_linear bias dtype must match input");
    TORCH_CHECK(bias.numel() == weight.size(0), "areno_linear bias size mismatch");
  }

  auto out_shape = input.sizes().vec();
  out_shape.back() = weight.size(0);
  auto output = torch::empty(out_shape, input.options());
  int64_t k = input.size(-1);
  int64_t m = input.numel() / k;
  int64_t n = weight.size(0);

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto dtype = areno_accel::cuda_type(input.scalar_type());
  areno_accel::gemm_row_major(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      weight.data_ptr(),
      dtype,
      k,
      input.data_ptr(),
      dtype,
      k,
      output.data_ptr(),
      dtype,
      n);
  if (use_bias) {
    areno_accel::maybe_add_bias(output, bias);
  }
  return output;
}

std::vector<torch::Tensor> areno_linear_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    bool use_bias) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_linear grad_output must be CUDA");
  TORCH_CHECK(input.is_cuda(), "areno_linear input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_linear weight must be CUDA");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "areno_linear input and weight dtype must match");
  TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(), "areno_linear grad dtype must match input");

  auto grad_input = torch::empty_like(input);
  auto grad_weight = torch::empty_like(weight);
  auto grad_bias = use_bias ? areno_accel::reduce_bias_grad(grad_output) : torch::empty({0}, grad_output.options());
  int64_t k = input.size(-1);
  int64_t m = input.numel() / k;
  int64_t n = weight.size(0);

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto dtype = areno_accel::cuda_type(input.scalar_type());

  areno_accel::gemm_row_major(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      k,
      m,
      n,
      weight.data_ptr(),
      dtype,
      k,
      grad_output.data_ptr(),
      dtype,
      n,
      grad_input.data_ptr(),
      dtype,
      k);

  areno_accel::gemm_row_major(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      k,
      n,
      m,
      input.data_ptr(),
      dtype,
      k,
      grad_output.data_ptr(),
      dtype,
      n,
      grad_weight.data_ptr(),
      dtype,
      k);

  return {grad_input, grad_weight, grad_bias};
}

torch::Tensor areno_grouped_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<int64_t> tokens_per_expert) {
  TORCH_CHECK(input.is_cuda(), "areno_grouped_linear input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_grouped_linear weight must be CUDA");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "areno_grouped_linear input and weight dtype must match");
  TORCH_CHECK(input.dim() == 2, "areno_grouped_linear input must be 2D");
  TORCH_CHECK(weight.dim() == 3, "areno_grouped_linear weight must be 3D");
  TORCH_CHECK(static_cast<int64_t>(tokens_per_expert.size()) == weight.size(0), "tokens_per_expert must match expert count");
  TORCH_CHECK(input.size(1) == weight.size(2), "areno_grouped_linear input and weight shape mismatch");

  int64_t experts = weight.size(0);
  int64_t k = input.size(1);
  int64_t n = weight.size(1);
  int64_t total_tokens = 0;
  for (int64_t count : tokens_per_expert) {
    TORCH_CHECK(count >= 0, "tokens_per_expert values must be non-negative");
    total_tokens += count;
  }
  TORCH_CHECK(total_tokens == input.size(0), "tokens_per_expert sum must match input rows");

  auto output = torch::empty({input.size(0), n}, input.options());
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto dtype = areno_accel::cuda_type(input.scalar_type());

  int64_t offset = 0;
  char* input_base = static_cast<char*>(input.data_ptr());
  char* output_base = static_cast<char*>(output.data_ptr());
  char* weight_base = static_cast<char*>(weight.data_ptr());
  int64_t elem_size = input.element_size();
  for (int64_t expert = 0; expert < experts; ++expert) {
    int64_t m = tokens_per_expert[expert];
    if (m > 0) {
      const void* expert_weight = weight_base + expert * n * k * elem_size;
      const void* expert_input = input_base + offset * k * elem_size;
      void* expert_output = output_base + offset * n * elem_size;
      areno_accel::gemm_row_major(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          n,
          m,
          k,
          expert_weight,
          dtype,
          k,
          expert_input,
          dtype,
          k,
          expert_output,
          dtype,
          n);
    }
    offset += m;
  }
  return output;
}

torch::Tensor areno_grouped_linear_forward_counts_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor tokens_per_expert) {
  TORCH_CHECK(tokens_per_expert.is_cuda(), "areno_grouped_linear tokens_per_expert must be CUDA");
  TORCH_CHECK(tokens_per_expert.dim() == 1, "areno_grouped_linear tokens_per_expert must be 1D");
  TORCH_CHECK(tokens_per_expert.scalar_type() == at::kLong || tokens_per_expert.scalar_type() == at::kInt, "areno_grouped_linear tokens_per_expert must be int32 or int64");
  TORCH_CHECK(tokens_per_expert.numel() == weight.size(0), "tokens_per_expert must match expert count");
  auto counts_cpu = tokens_per_expert.to(torch::kCPU);
  std::vector<int64_t> counts(static_cast<size_t>(counts_cpu.numel()));
  if (counts_cpu.scalar_type() == at::kLong) {
    const int64_t* ptr = counts_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) {
      counts[static_cast<size_t>(i)] = ptr[i];
    }
  } else {
    const int32_t* ptr = counts_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) {
      counts[static_cast<size_t>(i)] = static_cast<int64_t>(ptr[i]);
    }
  }
  return areno_grouped_linear_forward_cuda(input, weight, counts);
}

std::vector<torch::Tensor> areno_grouped_linear_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<int64_t> tokens_per_expert) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_grouped_linear grad_output must be CUDA");
  TORCH_CHECK(input.is_cuda(), "areno_grouped_linear input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_grouped_linear weight must be CUDA");
  TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(), "areno_grouped_linear grad dtype must match input");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "areno_grouped_linear input and weight dtype must match");
  TORCH_CHECK(input.dim() == 2, "areno_grouped_linear input must be 2D");
  TORCH_CHECK(grad_output.dim() == 2, "areno_grouped_linear grad_output must be 2D");
  TORCH_CHECK(weight.dim() == 3, "areno_grouped_linear weight must be 3D");
  TORCH_CHECK(static_cast<int64_t>(tokens_per_expert.size()) == weight.size(0), "tokens_per_expert must match expert count");
  TORCH_CHECK(input.size(1) == weight.size(2), "areno_grouped_linear input and weight shape mismatch");
  TORCH_CHECK(grad_output.size(0) == input.size(0), "areno_grouped_linear grad_output rows must match input");
  TORCH_CHECK(grad_output.size(1) == weight.size(1), "areno_grouped_linear grad_output cols must match output size");

  int64_t experts = weight.size(0);
  int64_t k = input.size(1);
  int64_t n = weight.size(1);
  int64_t total_tokens = 0;
  for (int64_t count : tokens_per_expert) {
    TORCH_CHECK(count >= 0, "tokens_per_expert values must be non-negative");
    total_tokens += count;
  }
  TORCH_CHECK(total_tokens == input.size(0), "tokens_per_expert sum must match input rows");

  auto grad_input = torch::empty_like(input);
  auto grad_weight = torch::zeros_like(weight);
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto dtype = areno_accel::cuda_type(input.scalar_type());

  int64_t offset = 0;
  char* input_base = static_cast<char*>(input.data_ptr());
  char* grad_output_base = static_cast<char*>(grad_output.data_ptr());
  char* weight_base = static_cast<char*>(weight.data_ptr());
  char* grad_input_base = static_cast<char*>(grad_input.data_ptr());
  char* grad_weight_base = static_cast<char*>(grad_weight.data_ptr());
  int64_t elem_size = input.element_size();
  for (int64_t expert = 0; expert < experts; ++expert) {
    int64_t m = tokens_per_expert[expert];
    if (m > 0) {
      const void* expert_weight = weight_base + expert * n * k * elem_size;
      const void* expert_input = input_base + offset * k * elem_size;
      const void* expert_grad_output = grad_output_base + offset * n * elem_size;
      void* expert_grad_input = grad_input_base + offset * k * elem_size;
      void* expert_grad_weight = grad_weight_base + expert * n * k * elem_size;

      areno_accel::gemm_row_major(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          k,
          m,
          n,
          expert_weight,
          dtype,
          k,
          expert_grad_output,
          dtype,
          n,
          expert_grad_input,
          dtype,
          k);

      areno_accel::gemm_row_major(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          k,
          n,
          m,
          expert_input,
          dtype,
          k,
          expert_grad_output,
          dtype,
          n,
          expert_grad_weight,
          dtype,
          k);
    }
    offset += m;
  }
  return {grad_input, grad_weight};
}

std::vector<torch::Tensor> areno_grouped_linear_backward_counts_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor tokens_per_expert) {
  TORCH_CHECK(tokens_per_expert.is_cuda(), "areno_grouped_linear tokens_per_expert must be CUDA");
  TORCH_CHECK(tokens_per_expert.dim() == 1, "areno_grouped_linear tokens_per_expert must be 1D");
  TORCH_CHECK(tokens_per_expert.scalar_type() == at::kLong || tokens_per_expert.scalar_type() == at::kInt, "areno_grouped_linear tokens_per_expert must be int32 or int64");
  TORCH_CHECK(tokens_per_expert.numel() == weight.size(0), "tokens_per_expert must match expert count");
  auto counts_cpu = tokens_per_expert.to(torch::kCPU);
  std::vector<int64_t> counts(static_cast<size_t>(counts_cpu.numel()));
  if (counts_cpu.scalar_type() == at::kLong) {
    const int64_t* ptr = counts_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) {
      counts[static_cast<size_t>(i)] = ptr[i];
    }
  } else {
    const int32_t* ptr = counts_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) {
      counts[static_cast<size_t>(i)] = static_cast<int64_t>(ptr[i]);
    }
  }
  return areno_grouped_linear_backward_cuda(grad_output, input, weight, counts);
}
