#include <cuda_runtime.h>
// the kernels have been splited

// gaussian blur kernel
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

    if (col >= width || row >= height || ch >= channels)
        return;

    int baseoffset = ch * width * height;
    int pixval = 0;
    int pixels = 0;

    for (int blur_row = -blur_size; blur_row <= blur_size; blur_row++) {
        for (int blur_col = -blur_size; blur_col <= blur_size; blur_col++) {
            int r = row + blur_row;
            int c = col + blur_col;

            if (r >= 0 && r < height && c >= 0 && c < width) {
                pixval += Pin[baseoffset + r * width + c];
                pixels++;
            }
        }
    }

    if (pixels > 0) {
        Pout[baseoffset + row * width + col] =
            (unsigned char)(pixval / pixels);
    }
}

// grayscale scale kernel
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

        unsigned char r = Pin[0 * height * width + idx];
        unsigned char g = Pin[1 * height * width + idx];
        unsigned char b = Pin[2 * height * width + idx];

        Pout[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}