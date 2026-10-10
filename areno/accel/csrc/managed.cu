#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

torch::Tensor areno_managed_empty_like(torch::Tensor prototype) {
  TORCH_CHECK(prototype.is_cuda(), "paged optimizer requires CUDA");
  TORCH_CHECK(prototype.is_contiguous(), "managed allocation requires a contiguous tensor");
  const int device = prototype.get_device();
  c10::cuda::CUDAGuard guard(device);
  int supported = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrManagedMemory, device));
  TORCH_CHECK(supported, "this CUDA device does not support managed memory");
  int concurrent = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device));
  TORCH_CHECK(concurrent, "paged optimizer requires concurrent managed access");
  if (prototype.numel() == 0) return torch::empty_like(prototype);
  void* pointer = nullptr;
  C10_CUDA_CHECK(cudaMallocManaged(&pointer, prototype.nbytes(), cudaMemAttachGlobal));
  return torch::from_blob(pointer, prototype.sizes(), [device](void* ptr) {
    c10::cuda::CUDAGuard free_guard(device);
    // cudaFree retains the allocation until pending device users finish.
    cudaFree(ptr);
  }, prototype.options().requires_grad(false));
}

bool areno_is_managed(torch::Tensor tensor) {
  if (!tensor.is_cuda() || tensor.numel() == 0) return false;
  c10::cuda::CUDAGuard guard(tensor.device());
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, tensor.data_ptr()));
  return attributes.type == cudaMemoryTypeManaged;
}
