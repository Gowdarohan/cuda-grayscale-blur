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

// kernel for edge sharpening the image
__global__ void edge_sharpining(unsigned char* Pin, unsigned char* Pout, int width, int height){

    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(col > 0 && col < width - 1 && row > 0 && row < height - 1){
        int top = Pin[(row-1)*width + col];
        int bottom = Pin[(row + 1)*width + col];
        int left = Pin[row*width + (col-1)];
        int right = Pin[row*width + (col + 1)];
        int center = Pin[row*width + col];

        int value = 5*center - top - bottom - left - right;
        value = max(0,min(255,value));
        Pout[row*width + col] = (unsigned char)value;
    }
    else{
        Pout[row*width + col] = Pin[row*width + col];
    }
}