# main.py - Complete CUDA Image Processing for VS Code
import torch
from torch.utils.cpp_extension import load
from torchvision.io import read_image, write_png
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import time
import numpy as np
import os

print("=" * 60)
print("CUDA Image Processing - 3D Kernel (Parallel Channel Memory Layout)")
print("=" * 60)

if not torch.cuda.is_available():
    print("CUDA is not available! Please check your CUDA installation.")
    print("Make sure you have:")
    print("  - NVIDIA GPU")
    print("  - CUDA Toolkit installed")
    print("  - PyTorch with CUDA support")
    exit()

print(f"CUDA available: {torch.cuda.get_device_name(0)}")

# Getting image URL or local image path
print("\nEnter image URL (or path to local image):")
image_input = input().strip()

if not image_input:
    image_input = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    print(f"Using default: {image_input}")

# Load image from URL or local file
print(f"\nLoading image...")
try:
    if image_input.startswith(('http://', 'https://')):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_input, headers=headers, timeout=30)
        response.raise_for_status()
        pil_img = Image.open(io.BytesIO(response.content))
    else:
        pil_img = Image.open(image_input)

    if pil_img.mode == 'RGBA':
        print("RGBA detected - converting to RGB...")
        background = Image.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    pil_img.save("temp_image.jpg", "JPEG")
    print("Image loaded and converted to RGB")
    
except Exception as e:
    print(f"Failed to load image: {e}")
    exit()

# Load image to GPU
print(f"\nLoading to GPU...")
x_rgb = read_image("temp_image.jpg").contiguous().cuda()
print(f"  Loaded RGB: {x_rgb.shape} (C x H x W)")
print(f"  Channels: {x_rgb.shape[0]}, Size: {x_rgb.shape[2]}x{x_rgb.shape[1]}")

# input blur size
print("\n" + "=" * 60)
blur_size = int(input("Enter blur size (odd number, 1-31): "))
if blur_size % 2 == 0:
    blur_size += 1
    print(f"Using odd number: {blur_size}")

print(f"\nCompiling CUDA kernels from gaussian_processing.cu...")

if not os.path.exists("gaussian_processing.cu"):
    print("Error: gaussian_processing.cu not found!")
    print("Please ensure the CUDA kernel file is in the same directory.")
    exit()

try:
    import shutil
    if os.path.exists("gaussian_processing"):
        shutil.rmtree("gaussian_processing")

    ext = load(
        name="gaussian_processing",
        sources=["gaussian_processing.cu"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"]
    )
    print("  CUDA kernels compiled successfully!")
except Exception as e:
    print(f"Compilation failed: {e}")
    print("\nTroubleshooting tips:")
    print("  1. Make sure CUDA toolkit is installed: nvcc --version")
    print("  2. Make sure PyTorch has CUDA support: torch.cuda.is_available()")
    print("  3. Install ninja: pip install ninja")
    exit()

print("\n" + "=" * 60)
print("Processing Original Image with Different Effects")
print("=" * 60)

performance_metrics = {}

print(f"\n1. Applying Gaussian Blur to Original Color Image (size {blur_size})...")
num_runs = 10
blur_times = []
for i in range(num_runs):
    torch.cuda.synchronize()
    start = time.time()
    y_blur_rgb = ext.gaussian_blur(x_rgb, blur_size)
    torch.cuda.synchronize()
    blur_times.append((time.time() - start) * 1000)
blur_time = np.mean(blur_times)
blur_time_std = np.std(blur_times)
print(f"     Color blur complete - Avg: {blur_time:.2f}ms ± {blur_time_std:.2f}ms (over {num_runs} runs)")

print(f"\n2. Converting Original Image to Grayscale...")
start = time.time()
y_grayscale = ext.rgb_to_grayscale(x_rgb)
torch.cuda.synchronize()
gray_time = (time.time() - start) * 1000
print(f"Grayscale conversion complete in {gray_time:.1f}ms")


print(f"\n3. Converting to Grayscale then Applying Blur (size {blur_size})...")
start = time.time()
y_gray_first = ext.rgb_to_grayscale(x_rgb)
torch.cuda.synchronize()
y_gray_first_blurred = ext.gaussian_blur(y_gray_first.unsqueeze(0), blur_size)
torch.cuda.synchronize()
gray_blur_time = (time.time() - start) * 1000
print(f"Grayscale + Blur complete in {gray_blur_time:.1f}ms")

# 6.4: Blur then Grayscale
print(f"\n4. Applying Blur then Grayscale (size {blur_size})...")
start_blur = time.time()
y_blur_first = ext.gaussian_blur(x_rgb, blur_size)
torch.cuda.synchronize()
blur_step_time = (time.time() - start_blur) * 1000

start_gray = time.time()
y_blur_first_gray = ext.rgb_to_grayscale(y_blur_first)
torch.cuda.synchronize()
gray_step_time = (time.time() - start_gray) * 1000
total_time = blur_step_time + gray_step_time

print(f"     Blur + Grayscale complete:")
print(f"      - Blur step: {blur_step_time:.2f}ms")
print(f"      - Grayscale step: {gray_step_time:.2f}ms")
print(f"      - Total: {total_time:.2f}ms")
print(f"      - Overhead: {(total_time - (blur_time + gray_time)):.2f}ms")

# Step 7: Save all results
print("\n Saving all results...")

output_files = []

file1 = f"original_blurred_size{blur_size}.png"
write_png(y_blur_rgb.cpu(), file1)
output_files.append(file1)
print(f"     Saved: {file1}")

file2 = f"grayscale.png"
write_png(y_grayscale.unsqueeze(0).cpu(), file2)
output_files.append(file2)
print(f"    Saved: {file2}")

file3 = f"grayscale_then_blur_size{blur_size}.png"
write_png(y_gray_first_blurred.cpu(), file3)
output_files.append(file3)
print(f"     Saved: {file3}")

file4 = f"blur_then_grayscale_size{blur_size}.png"
write_png(y_blur_first_gray.unsqueeze(0).cpu(), file4)
output_files.append(file4)
print(f"     Saved: {file4}")

# Display results
print("\nDisplaying Results...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(x_rgb.cpu().permute(1,2,0).numpy())
axes[0, 0].set_title(f'Original Image\n{x_rgb.shape[2]}x{x_rgb.shape[1]}')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_blur_rgb.cpu().permute(1,2,0).numpy())
axes[0, 1].set_title(f'Color Blur (size={blur_size})\nAvg: {blur_time:.2f}ms')
axes[0, 1].axis('off')

axes[0, 2].imshow(y_grayscale.cpu().numpy(), cmap='gray')
axes[0, 2].set_title(f'Grayscale\n{gray_time:.1f}ms')
axes[0, 2].axis('off')

axes[1, 0].imshow(y_gray_first_blurred.squeeze(0).cpu().numpy(), cmap='gray')
axes[1, 0].set_title(f'Grayscale → Blur\n{gray_blur_time:.1f}ms')
axes[1, 0].axis('off')

axes[1, 1].imshow(y_blur_first_gray.cpu().numpy(), cmap='gray')
axes[1, 1].set_title(f'Blur → Grayscale\nTotal: {total_time:.2f}ms', fontsize=9)
axes[1, 1].axis('off')

ax = axes[1, 2]
times = [blur_time, gray_time, gray_blur_time, total_time]
labels = ['Color Blur', 'Grayscale', 'Gray→Blur', 'Blur→Gray']
bars = ax.bar(labels, times, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
ax.set_ylabel('Time (ms)')
ax.set_title('Performance Comparison')
ax.grid(True, alpha=0.3, axis='y')

for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
            f'{t:.1f}ms', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# performance
print("\n" + "=" * 60)
print("DETAILED PERFORMANCE SUMMARY")
print("=" * 60)
print(f"Image Size: {x_rgb.shape[2]} x {x_rgb.shape[1]} pixels")
print(f"Blur Kernel Size: {blur_size} x {blur_size}")
print(f"\n{'Operation':<20} {'Time (ms)':<15}")
print("-" * 35)
print(f"{'Color Blur':<20} {blur_time:<15.2f}")
print(f"{'Grayscale':<20} {gray_time:<15.2f}")
print(f"{'Gray→Blur':<20} {gray_blur_time:<15.2f}")
print(f"{'Blur→Gray':<20} {total_time:<15.2f}")

os.remove("temp_image.jpg")

print("\nComplete!")