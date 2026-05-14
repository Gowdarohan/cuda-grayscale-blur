// gaussian_processing.cu

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "gaussian_kernels.cuh"

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// gaussian blur wrapper
torch::Tensor gaussian_blur(torch::Tensor img, int blur_size) {
    TORCH_CHECK(img.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(img.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(img.dim() == 3, "Input must be CHW format");

    const int channels = img.size(0);
    const int height = img.size(1);
    const int width = img.size(2);

    const int block_z = std::min(4, channels);

    dim3 dimBlock(16, 16, block_z);
    dim3 dimGrid(
        cdiv(width, dimBlock.x),
        cdiv(height, dimBlock.y),
        cdiv(channels, block_z)
    );

    auto result = torch::empty_like(img);

    blur_kernel<<<dimGrid, dimBlock>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width,
        height,
        blur_size,
        channels
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "CUDA error: ", cudaGetErrorString(err));

    return result;
}

// grayscale wrapper
torch::Tensor rgb_to_grayscale(torch::Tensor img) {
    TORCH_CHECK(img.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(img.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(img.dim() == 3, "Input must be 3D (C, H, W)");
    TORCH_CHECK(img.size(0) == 3,
        "Input must have 3 channels (RGB)");

    const int height = img.size(1);
    const int width = img.size(2);

    dim3 dimBlock(32, 32);
}