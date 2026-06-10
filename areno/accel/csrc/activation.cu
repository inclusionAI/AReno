#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace areno_accel {

template <typename scalar_t>
__device__ __forceinline__ float read_as_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t cast_from_float(float value) {
  return static_cast<scalar_t>(value);
}

struct SiluMul {
  __device__ __forceinline__ float operator()(float gate, float up) const {
    return gate / (1.0f + expf(-gate)) * up;
  }
};

struct GeluTanhMul {
  __device__ __forceinline__ float operator()(float gate, float up) const {
    constexpr float cubic = 0.044715f;
    constexpr float scale = 0.7978845608028654f;
    float cdf = 0.5f * (1.0f + tanhf(scale * (gate + cubic * gate * gate * gate)));
    return gate * cdf * up;
  }
};

struct SiluMulGrad {
  __device__ __forceinline__ void operator()(float gate, float up, float grad, float* dgate, float* dup) const {
    float sigmoid = 1.0f / (1.0f + expf(-gate));
    float silu = gate * sigmoid;
    *dgate = grad * up * sigmoid * (1.0f + gate * (1.0f - sigmoid));
    *dup = grad * silu;
  }
};

struct GeluTanhMulGrad {
  __device__ __forceinline__ void operator()(float gate, float up, float grad, float* dgate, float* dup) const {
    constexpr float cubic = 0.044715f;
    constexpr float scale = 0.7978845608028654f;
    float gate2 = gate * gate;
    float inner = scale * (gate + cubic * gate * gate2);
    float tanh_inner = tanhf(inner);
    float cdf = 0.5f * (1.0f + tanh_inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float d_inner = scale * (1.0f + 3.0f * cubic * gate2);
    float d_gelu = cdf + 0.5f * gate * sech2 * d_inner;
    *dgate = grad * up * d_gelu;
    *dup = grad * gate * cdf;
  }
};

template <typename scalar_t, typename Activation>
__global__ void gated_product_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ input, int half_width) {
  const int64_t row = blockIdx.x;
  const int64_t in_base = row * half_width * 2;
  const int64_t out_base = row * half_width;
  Activation activation;
  for (int col = threadIdx.x; col < half_width; col += blockDim.x) {
    float gate = read_as_float(input[in_base + col]);
    float up = read_as_float(input[in_base + half_width + col]);
    output[out_base + col] = cast_from_float<scalar_t>(activation(gate, up));
  }
}

template <typename scalar_t, typename ActivationGrad>
__global__ void gated_product_backward_kernel(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    int half_width) {
  const int64_t row = blockIdx.x;
  const int64_t in_base = row * half_width * 2;
  const int64_t out_base = row * half_width;
  ActivationGrad activation_grad;
  for (int col = threadIdx.x; col < half_width; col += blockDim.x) {
    float gate = read_as_float(input[in_base + col]);
    float up = read_as_float(input[in_base + half_width + col]);
    float grad = read_as_float(grad_output[out_base + col]);
    float dgate;
    float dup;
    activation_grad(gate, up, grad, &dgate, &dup);
    grad_input[in_base + col] = cast_from_float<scalar_t>(dgate);
    grad_input[in_base + half_width + col] = cast_from_float<scalar_t>(dup);
  }
}

template <typename scalar_t>
__global__ void silu_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ input, int64_t elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float x = read_as_float(input[idx]);
    output[idx] = cast_from_float<scalar_t>(x / (1.0f + expf(-x)));
  }
}

template <typename scalar_t>
__global__ void sigmoid_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ input, int64_t elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float x = read_as_float(input[idx]);
    output[idx] = cast_from_float<scalar_t>(1.0f / (1.0f + expf(-x)));
  }
}

template <typename scalar_t>
__global__ void sigmoid_backward_kernel(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ output,
    int64_t elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float y = read_as_float(output[idx]);
    grad_input[idx] = cast_from_float<scalar_t>(read_as_float(grad_output[idx]) * y * (1.0f - y));
  }
}

template <typename scalar_t>
__global__ void softplus_kernel(scalar_t* __restrict__ output, const scalar_t* __restrict__ input, int64_t elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float x = read_as_float(input[idx]);
    output[idx] = cast_from_float<scalar_t>(x > 20.0f ? x : log1pf(expf(x)));
  }
}

template <typename scalar_t>
__global__ void softplus_backward_kernel(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    int64_t elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float x = read_as_float(input[idx]);
    float sigmoid = x > 20.0f ? 1.0f : 1.0f / (1.0f + expf(-x));
    grad_input[idx] = cast_from_float<scalar_t>(read_as_float(grad_output[idx]) * sigmoid);
  }
}

template <typename scalar_t>
__global__ void silu_backward_kernel(
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    int64_t elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float x = read_as_float(input[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-x));
    float grad = read_as_float(grad_output[idx]) * sigmoid * (1.0f + x * (1.0f - sigmoid));
    grad_input[idx] = cast_from_float<scalar_t>(grad);
  }
}

template <typename Kernel>
void run_unary(torch::Tensor output, torch::Tensor input, const char* op_name, Kernel kernel) {
  TORCH_CHECK(input.is_cuda(), op_name, " input must be CUDA");
  TORCH_CHECK(output.is_cuda(), op_name, " output must be CUDA");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), op_name, " dtype mismatch");
  TORCH_CHECK(input.numel() == output.numel(), op_name, " shape mismatch");

  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((input.numel() + threads - 1) / threads, 4096));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_unary", [&] {
    kernel.template operator()<scalar_t>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), input.numel(), stream, blocks, threads);
  });
}

struct SiluLauncher {
  template <typename scalar_t>
  void operator()(scalar_t* output, const scalar_t* input, int64_t elements, cudaStream_t stream, int blocks, int threads) const {
    silu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(output, input, elements);
  }
};

struct SigmoidLauncher {
  template <typename scalar_t>
  void operator()(scalar_t* output, const scalar_t* input, int64_t elements, cudaStream_t stream, int blocks, int threads) const {
    sigmoid_kernel<scalar_t><<<blocks, threads, 0, stream>>>(output, input, elements);
  }
};

struct SoftplusLauncher {
  template <typename scalar_t>
  void operator()(scalar_t* output, const scalar_t* input, int64_t elements, cudaStream_t stream, int blocks, int threads) const {
    softplus_kernel<scalar_t><<<blocks, threads, 0, stream>>>(output, input, elements);
  }
};

void run_silu(torch::Tensor output, torch::Tensor input, const char* op_name) {
  run_unary(output, input, op_name, SiluLauncher{});
}

void run_sigmoid(torch::Tensor output, torch::Tensor input, const char* op_name) {
  run_unary(output, input, op_name, SigmoidLauncher{});
}

void run_softplus(torch::Tensor output, torch::Tensor input, const char* op_name) {
  run_unary(output, input, op_name, SoftplusLauncher{});
}

void run_silu_backward(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input, const char* op_name) {
  TORCH_CHECK(input.is_cuda(), op_name, " input must be CUDA");
  TORCH_CHECK(grad_output.is_cuda(), op_name, " grad_output must be CUDA");
  TORCH_CHECK(grad_input.is_cuda(), op_name, " grad_input must be CUDA");
  TORCH_CHECK(input.scalar_type() == grad_output.scalar_type(), op_name, " dtype mismatch");
  TORCH_CHECK(input.scalar_type() == grad_input.scalar_type(), op_name, " dtype mismatch");
  TORCH_CHECK(input.numel() == grad_output.numel(), op_name, " shape mismatch");
  TORCH_CHECK(input.numel() == grad_input.numel(), op_name, " shape mismatch");

  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((input.numel() + threads - 1) / threads, 4096));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_silu_backward", [&] {
    silu_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), input.numel());
  });
}

void run_sigmoid_backward(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor output, const char* op_name) {
  TORCH_CHECK(output.is_cuda(), op_name, " output must be CUDA");
  TORCH_CHECK(grad_output.is_cuda(), op_name, " grad_output must be CUDA");
  TORCH_CHECK(grad_input.is_cuda(), op_name, " grad_input must be CUDA");
  TORCH_CHECK(output.scalar_type() == grad_output.scalar_type(), op_name, " dtype mismatch");
  TORCH_CHECK(output.scalar_type() == grad_input.scalar_type(), op_name, " dtype mismatch");
  TORCH_CHECK(output.numel() == grad_output.numel(), op_name, " shape mismatch");
  TORCH_CHECK(output.numel() == grad_input.numel(), op_name, " shape mismatch");

  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((output.numel() + threads - 1) / threads, 4096));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard guard(device_of(output));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, output.scalar_type(), "areno_sigmoid_backward", [&] {
    sigmoid_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), output.numel());
  });
}

void run_softplus_backward(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input, const char* op_name) {
  TORCH_CHECK(input.is_cuda(), op_name, " input must be CUDA");
  TORCH_CHECK(grad_output.is_cuda(), op_name, " grad_output must be CUDA");
  TORCH_CHECK(grad_input.is_cuda(), op_name, " grad_input must be CUDA");
  TORCH_CHECK(input.scalar_type() == grad_output.scalar_type(), op_name, " dtype mismatch");
  TORCH_CHECK(input.scalar_type() == grad_input.scalar_type(), op_name, " dtype mismatch");
  TORCH_CHECK(input.numel() == grad_output.numel(), op_name, " shape mismatch");
  TORCH_CHECK(input.numel() == grad_input.numel(), op_name, " shape mismatch");

  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((input.numel() + threads - 1) / threads, 4096));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_softplus_backward", [&] {
    softplus_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), input.numel());
  });
}

template <typename Activation>
void run_gated_product(torch::Tensor output, torch::Tensor input, const char* op_name) {
  TORCH_CHECK(input.is_cuda(), op_name, " input must be CUDA");
  TORCH_CHECK(output.is_cuda(), op_name, " output must be CUDA");
  TORCH_CHECK(input.dim() > 0, op_name, " input must have at least one dimension");
  TORCH_CHECK(input.size(-1) % 2 == 0, op_name, " input last dimension must be even");
  TORCH_CHECK(output.numel() * 2 == input.numel(), op_name, " output shape does not match input");

  const int half_width = input.size(-1) / 2;
  const int64_t rows = input.numel() / input.size(-1);
  const int threads = std::min(1024, std::max(32, ((half_width + 31) / 32) * 32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_gated_product", [&] {
    gated_product_kernel<scalar_t, Activation><<<rows, threads, 0, stream>>>(
        output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), half_width);
  });
}

template <typename ActivationGrad>
void run_gated_product_backward(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input, const char* op_name) {
  TORCH_CHECK(input.is_cuda(), op_name, " input must be CUDA");
  TORCH_CHECK(grad_output.is_cuda(), op_name, " grad_output must be CUDA");
  TORCH_CHECK(grad_input.is_cuda(), op_name, " grad_input must be CUDA");
  TORCH_CHECK(input.size(-1) % 2 == 0, op_name, " input last dimension must be even");
  TORCH_CHECK(grad_input.numel() == input.numel(), op_name, " grad_input shape does not match input");
  TORCH_CHECK(grad_output.numel() * 2 == input.numel(), op_name, " grad_output shape does not match input");

  const int half_width = input.size(-1) / 2;
  const int64_t rows = input.numel() / input.size(-1);
  const int threads = std::min(1024, std::max(32, ((half_width + 31) / 32) * 32));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_gated_product_backward", [&] {
    gated_product_backward_kernel<scalar_t, ActivationGrad><<<rows, threads, 0, stream>>>(
        grad_input.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), half_width);
  });
}

}  // namespace areno_accel

void areno_silu_and_mul_cuda(torch::Tensor output, torch::Tensor input) {
  areno_accel::run_gated_product<areno_accel::SiluMul>(output, input, "areno_silu_and_mul");
}

void areno_gelu_tanh_and_mul_cuda(torch::Tensor output, torch::Tensor input) {
  areno_accel::run_gated_product<areno_accel::GeluTanhMul>(output, input, "areno_gelu_tanh_and_mul");
}

torch::Tensor areno_silu_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input);
  areno_accel::run_silu(output, input, "areno_silu");
  return output;
}

torch::Tensor areno_sigmoid_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input);
  areno_accel::run_sigmoid(output, input, "areno_sigmoid");
  return output;
}

torch::Tensor areno_softplus_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input);
  areno_accel::run_softplus(output, input, "areno_softplus");
  return output;
}

void areno_d_silu_and_mul_cuda(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input) {
  areno_accel::run_gated_product_backward<areno_accel::SiluMulGrad>(grad_input, grad_output, input, "areno_d_silu_and_mul");
}

void areno_d_gelu_tanh_and_mul_cuda(torch::Tensor grad_input, torch::Tensor grad_output, torch::Tensor input) {
  areno_accel::run_gated_product_backward<areno_accel::GeluTanhMulGrad>(
      grad_input, grad_output, input, "areno_d_gelu_tanh_and_mul");
}

torch::Tensor areno_d_silu_cuda(torch::Tensor grad_output, torch::Tensor input) {
  auto grad_input = torch::empty_like(input);
  areno_accel::run_silu_backward(grad_input, grad_output, input, "areno_d_silu");
  return grad_input;
}

torch::Tensor areno_d_sigmoid_cuda(torch::Tensor grad_output, torch::Tensor output) {
  auto grad_input = torch::empty_like(output);
  areno_accel::run_sigmoid_backward(grad_input, grad_output, output, "areno_d_sigmoid");
  return grad_input;
}

torch::Tensor areno_d_softplus_cuda(torch::Tensor grad_output, torch::Tensor input) {
  auto grad_input = torch::empty_like(input);
  areno_accel::run_softplus_backward(grad_input, grad_output, input, "areno_d_softplus");
  return grad_input;
}
