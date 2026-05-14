// gaussian_processing.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// ========== GAUSSIAN BLUR KERNEL (3D Parallel Channels) ==========
__global__ void blur_kernel(
    unsigned char* Pin,
    unsigned char* Pout,
    int width,
    int height,
    int blur_size,
    int channels
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;

    if(col >= width || row >= height || ch >= channels) return;

    int baseoffset = ch * width * height;
    int pixval = 0;
    int pixels = 0;

    for(int blur_row = -blur_size; blur_row <= blur_size; blur_row++) {
        for(int blur_col = -blur_size; blur_col <= blur_size; blur_col++) {
            int r = row + blur_row;
            int c = col + blur_col;
            if(r >= 0 && r < height && c >= 0 && c < width) {
                pixval += Pin[baseoffset + r * width + c];
                pixels++;
            }
        }
    }

    if(pixels > 0) {
        Pout[baseoffset + row * width + col] = (unsigned char)(pixval / pixels);
    }
}

// ========== RGB TO GRAYSCALE KERNEL (Planar layout) ==========
__global__ void rgbToGrayscaleKernel(
    unsigned char* Pin,
    unsigned char* Pout,
    int width,
    int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;

        // Access each channel from planar layout (CHW)
        unsigned char r = Pin[0 * height * width + idx];
        unsigned char g = Pin[1 * height * width + idx];
        unsigned char b = Pin[2 * height * width + idx];

        Pout[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// ========== BLUR WRAPPER ==========
torch::Tensor gaussian_blur(torch::Tensor img, int blur_size) {
    TORCH_CHECK(img.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(img.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(img.dim() == 3, "Input must be CHW format");

    const int channels = img.size(0);
    const int height = img.size(1);
    const int width = img.size(2);

    // Limit block_z to max 4 or channels (whichever is smaller)
    const int block_z = std::min(4, channels);

    dim3 dimBlock(16, 16, block_z);
    dim3 dimGrid(
        cdiv(width, dimBlock.x),
        cdiv(height, dimBlock.y),
        cdiv(channels, block_z)
    );

    auto result = torch::empty_like(img);

    blur_kernel<<<dimGrid, dimBlock, 0>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width, height, blur_size, channels
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return result;
}

// ========== GRAYSCALE WRAPPER ==========
torch::Tensor rgb_to_grayscale(torch::Tensor img) {
    TORCH_CHECK(img.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(img.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(img.dim() == 3, "Input must be 3D (C, H, W)");
    TORCH_CHECK(img.size(0) == 3, "Input must have 3 channels (RGB)");

    const int height = img.size(1);
    const int width = img.size(2);

    dim3 dimBlock(32, 32);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width},
        torch::TensorOptions().dtype(torch::kUInt8).device(img.device()));

    rgbToGrayscaleKernel<<<dimGrid, dimBlock, 0>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width, height
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_blur", &gaussian_blur, "Gaussian blur using CUDA");
    m.def("rgb_to_grayscale", &rgb_to_grayscale, "Convert RGB to Grayscale using CUDA");
}