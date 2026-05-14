# CUDA Image Processing using PyTorch & CUDA Kernels

A high-performance GPU accelerated image processing project built using **CUDA, PyTorch, and C++ CUDA extensions**.

This project performs real-time image operations such as:

- Gaussian Blur
- RGB to Grayscale Conversion
- Grayscale → Blur Processing
- Blur → Grayscale Processing
- Performance Benchmarking on GPU

The project demonstrates parallel image processing using custom CUDA kernels for faster execution compared to CPU-based implementations.

---

# Features

- CUDA accelerated image filtering
- Custom Gaussian Blur CUDA kernel
- Parallel grayscale conversion
- PyTorch CUDA extension integration
- Performance comparison visualization
- Supports both URL images and local images
- Automatic RGB conversion for RGBA images

---

# Technologies Used

- Python
- CUDA C++
- PyTorch
- torchvision
- matplotlib
- NumPy
- PIL (Python Imaging Library)

---

# Project Structure

```bash
cuda-grayscale-blur/
│
├── main.py
├── gaussian_processing.cu
├── gaussian_kernels.cuh
├── requirements.txt
├── README.md
└── .gitattributes
```

---

# Output Examples

## Original Image

![Original Image](original_image.png)

---

## Grayscale Image

![Grayscale Image](grayscale_image.png)

---

## Gaussian Blurred Image

![Blurred Image](gaussian_blured_image.png)

---

## Performance Comparison

![Performance Comparison](performance_comparision.png)

---

# Performance Summary

The project compares execution times for different image operations on GPU.

| Operation      | Time   |
|----------------|--------|
| Color Blur     | 0.9 ms |
| Grayscale      | 0.3 ms |
| Gray → Blur    | 3.3 ms |
| Blur → Gray    | 2.3 ms |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/cuda-image-processing.git

cd cuda-image-processing
```

---

## Install Dependencies

```bash
pip install torch torchvision matplotlib pillow numpy requests ninja
```

---

# Requirements

- NVIDIA GPU
- CUDA Toolkit Installed
- PyTorch with CUDA Support
- Python 3.8+

Check CUDA availability:

```python
import torch
print(torch.cuda.is_available())
```

---

# Run the Project

```bash
python main.py
```

Then:

1. Enter image URL or local image path
2. Enter Gaussian blur kernel size
3. CUDA kernels compile automatically
4. Results and performance charts are displayed

---

# CUDA Operations Implemented

## 1. Gaussian Blur

Applies Gaussian filtering using parallel CUDA kernels for image smoothing.

## 2. RGB to Grayscale

Converts RGB image into grayscale using weighted channel computation.

## 3. Combined Operations

- Grayscale → Blur
- Blur → Grayscale

Used to compare execution performance and processing order effects.

---

# Key Learning Outcomes

- CUDA kernel programming
- GPU memory handling
- Parallel image processing
- PyTorch CUDA extensions
- Performance benchmarking
- Computer vision preprocessing

---

# Future Improvements

- Edge detection filters
- Sharpening filters
- Video frame processing
- Real-time webcam support
- CUDA shared memory optimization
- Stream parallelism

---

# Author

**ROHAN J**

---

# License

This project is open-source and available under the MIT License.