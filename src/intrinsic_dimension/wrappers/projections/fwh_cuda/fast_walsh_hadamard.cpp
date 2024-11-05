// The codes are from Armen Aghajanyan from facebook, from paper
// Intrinsic Dimensionality Explains the Effectiveness of Language Model
// Fine-Tuning https://arxiv.org/abs/2012.13255

#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void fast_walsh_hadamard_transform_cuda_kernel(const int n_params, const int half_size, torch::Tensor in, torch::Tensor out, bool normalize);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor fast_walsh_hadamard_transform(torch::Tensor input, bool normalize) {
    CHECK_INPUT(input);
    const int n_params = input.numel();
    torch::Tensor output_flat = input.clone();
    int size = 1;
    while (size < n_params) {
        size *= 2;
    }
    const int half_size = size / 2;
    fast_walsh_hadamard_transform_cuda_kernel(n_params, half_size, input, output_flat, normalize);
    return output_flat;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_walsh_hadamard_transform", &fast_walsh_hadamard_transform, "Fast Walsh Hadamard Transform (CUDA)");
}