# main.py - Complete CUDA Image Processing with Split Files
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

# Step 1: Install ninja
print("\nInstalling ninja...")
!pip install ninja -q
print("Ninja installed")

# Step 2: Get image URL
print("\nEnter image URL:")
image_url = input().strip()

if not image_url:
    image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    print(f"Using default: {image_url}")

# Step 3: Download and convert RGBA to RGB
print(f"\nDownloading...")
try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(image_url, headers=headers, timeout=30)
    response.raise_for_status()

    pil_img = Image.open(io.BytesIO(response.content))

    if pil_img.mode == 'RGBA':
        print("RGBA detected - converting to RGB...")
        background = Image.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    pil_img.save("input_image.jpg", "JPEG")
    print("Converted to RGB")

except Exception as e:
    print(f"Failed: {e}")
    !wget -q https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg -O input_image.jpg
    print("✓ Using default image")

# Step 4: Load image to GPU
print(f"\n📸 Loading to GPU...")
x_rgb = read_image("input_image.jpg").contiguous().cuda()
print(f"Loaded RGB: {x_rgb.shape} (C x H x W)")
print(f"Channels: {x_rgb.shape[0]}, Size: {x_rgb.shape[2]}x{x_rgb.shape[1]}")

# Step 5: Get blur size
print("\n" + "=" * 60)
blur_size = int(input("Enter blur size (odd number, 1-31): "))
if blur_size % 2 == 0:
    blur_size += 1
    print(f"  Using odd number: {blur_size}")

# Step 6: Check if CUDA files exist
print(f"\nLoading CUDA extension...")
if not os.path.exists("gaussian_processing.cu"):
    print("Error: gaussian_processing.cu not found!")
    exit()
if not os.path.exists("gaussian_kernels.cuh"):
    print("Error: gaussian_kernels.cuh not found!")
    exit()

# Step 7: Compile CUDA extension from .cu file
try:
    ext = load(
        name="gaussian_processing",
        sources=["gaussian_processing.cu"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"]
    )
    print("CUDA kernels compiled successfully!")
except Exception as e:
    print(f"Compilation failed: {e}")
    exit()

# Step 8: Process all versions
print("\n" + "=" * 60)
print("Processing Original Image with Different Effects")
print("=" * 60)

performance_metrics = {}

# 8.1: Original RGB Blur
print(f"\n1. Applying Gaussian Blur to Original Color Image (size {blur_size})")
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
print(f" Color blur complete - Avg: {blur_time:.2f}ms +/- {blur_time_std:.2f}ms (over {num_runs} runs)")

# 8.2: Grayscale
print(f"\n2. Converting Original Image to Grayscale")
start = time.time()
y_grayscale = ext.rgb_to_grayscale(x_rgb)
torch.cuda.synchronize()
gray_time = (time.time() - start) * 1000
print(f"  Grayscale conversion complete in {gray_time:.1f}ms")

# 8.3: Grayscale then Blur
print(f"\n3. Converting to Grayscale then Applying Blur (size {blur_size})")
start = time.time()
y_gray_first = ext.rgb_to_grayscale(x_rgb)
torch.cuda.synchronize()
y_gray_first_blurred = ext.gaussian_blur(y_gray_first.unsqueeze(0), blur_size)
torch.cuda.synchronize()
gray_blur_time = (time.time() - start) * 1000
print(f"  Grayscale + Blur complete in {gray_blur_time:.1f}ms")

# 8.4: Blur then Grayscale
print(f"\n4. Applying Blur then Grayscale (size {blur_size})")
start_blur = time.time()
y_blur_first = ext.gaussian_blur(x_rgb, blur_size)
torch.cuda.synchronize()
blur_step_time = (time.time() - start_blur) * 1000

start_gray = time.time()
y_blur_first_gray = ext.rgb_to_grayscale(y_blur_first)
torch.cuda.synchronize()
gray_step_time = (time.time() - start_gray) * 1000

# 8.5: Edge Sharpening
print(f"\n5. Applying Edge Sharpening")
start = time.time()
y_edge_sharp = ext.edge_sharpening(x_rgb)
torch.cuda.synchronize()
edge_time = (time.time() - start) * 1000
print(f"  Edge sharpening complete in {edge_time:.1f}ms")

total_time = blur_step_time + gray_step_time
print(f"\nPerformance Summary:")
print(f" - Blur step: {blur_step_time:.2f}ms")
print(f" - Grayscale step: {gray_step_time:.2f}ms")
print(f" - Edge sharpening: {edge_time:.2f}ms")
print(f" - Blur->Gray total: {total_time:.2f}ms")
print(f" - Overhead: {(total_time - (blur_time + gray_time)):.2f}ms")

# Step 9: Save all results
print("\nSaving all results...")

output_files = []

file1 = f"original_blurred_size{blur_size}.png"
write_png(y_blur_rgb.cpu(), file1)
output_files.append(file1)
print(f"  Saved: {file1}")

file2 = f"grayscale.png"
write_png(y_grayscale.unsqueeze(0).cpu(), file2)
output_files.append(file2)
print(f"  Saved: {file2}")

file3 = f"grayscale_then_blur_size{blur_size}.png"
write_png(y_gray_first_blurred.cpu(), file3)
output_files.append(file3)
print(f"  Saved: {file3}")

file4 = f"blur_then_grayscale_size{blur_size}.png"
write_png(y_blur_first_gray.unsqueeze(0).cpu(), file4)
output_files.append(file4)
print(f"  Saved: {file4}")

file5 = f"edge_sharpened.png"
write_png(y_edge_sharp.cpu(), file5)
output_files.append(file5)
print(f"  Saved: {file5}")

# download option
print("\n" + "=" * 60)
from google.colab import files

download_choice = input("Download all results? (y/n): ").lower().strip()
if download_choice == 'y':
    for f in output_files:
        print(f"  Downloading: {f}")
        files.download(f)
    print("All files downloaded!")

# diaplay images
print("\nDisplaying Results...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(x_rgb.cpu().permute(1,2,0).numpy())
axes[0, 0].set_title(f'Original Image\n{x_rgb.shape[2]}x{x_rgb.shape[1]}')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_blur_rgb.cpu().permute(1,2,0).numpy())
axes[0, 1].set_title(f'Color Blur (size={blur_size})\n{blur_time:.2f}ms')
axes[0, 1].axis('off')

axes[0, 2].imshow(y_grayscale.cpu().numpy(), cmap='gray')
axes[0, 2].set_title(f'Grayscale\n{gray_time:.1f}ms')
axes[0, 2].axis('off')

axes[1, 0].imshow(y_gray_first_blurred.squeeze(0).cpu().numpy(), cmap='gray')
axes[1, 0].set_title(f'Grayscale → Blur\n{gray_blur_time:.1f}ms')
axes[1, 0].axis('off')

axes[1, 1].imshow(y_edge_sharp.cpu().permute(1,2,0).numpy())
axes[1, 1].set_title(f'Edge Sharpening\n{edge_time:.1f}ms')
axes[1, 1].axis('off')

ax = axes[1, 2]
times = [blur_time, gray_time, gray_blur_time, edge_time]
labels = ['Color Blur', 'Grayscale', 'Gray→Blur', 'Edge Sharpening']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = ax.bar(labels, times, color=colors, alpha=0.8)
ax.set_ylabel('Time (ms)')
ax.set_title('Performance Comparison')
ax.grid(True, alpha=0.3, axis='y')

for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
            f'{t:.1f}ms', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()



print("\nComplete! Using split CUDA files structure!")
print(f" - gaussian_kernels.cuh (kernel definitions)")
print(f" - gaussian_processing.cu (CUDA wrappers)")
print(f" - main.py (Python main file)")