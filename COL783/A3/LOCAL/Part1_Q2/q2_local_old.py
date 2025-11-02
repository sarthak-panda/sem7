#!/usr/bin/env python3
"""
haar_q2.py
Modular pipeline for Parts A, B, C of the Haar transform exercise.

Saves outputs under ./Q2_Output and creates Q2_Output.zip in /mnt/data.

Requirements:
  pip install numpy pillow matplotlib
Run:
  python3 haar_q2.py
"""

import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import zipfile

# --- Paths ---
OUT_ROOT = "./Q2_Output"
PART_A_DIR = os.path.join(OUT_ROOT, "PartA")
PART_B_DIR = os.path.join(OUT_ROOT, "PartB")
PART_C_DIR = os.path.join(OUT_ROOT, "PartC")
for d in (OUT_ROOT, PART_A_DIR, PART_B_DIR, PART_C_DIR):
    os.makedirs(d, exist_ok=True)

# --- Helpers ---
def save_image(img_arr, path, cmap='gray', vmin=None, vmax=None):
    plt.figure(figsize=(6,6))
    if img_arr.ndim == 2:
        plt.imshow(img_arr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
    elif img_arr.ndim == 1:
        plt.plot(img_arr)
        plt.grid(True)
        plt.xlabel("Index")
        plt.ylabel("Value")
    else:
        plt.imshow(img_arr.astype(np.uint8))
        plt.axis('off')
    plt.tight_layout(pad=0.1)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64))**2)

def psnr(a, b, data_range=255.0):
    m = mse(a, b)
    if m == 0:
        return float('inf')
    return 10 * math.log10((data_range**2) / m)

# --- Haar 1D (orthonormal) ---
def haar1d_forward_clear(arr):
    a = arr.astype(np.float64).copy()
    n = a.size
    if n & (n-1):
        raise ValueError("Length must be power of two")
    coeffs = np.zeros_like(a)
    length = n
    temp = a.copy()
    details_stack = []
    while length > 1:
        half = length // 2
        averages = np.zeros(half)
        details = np.zeros(half)
        for i in range(half):
            x = temp[2*i]
            y = temp[2*i+1]
            averages[i] = (x + y) / math.sqrt(2.0)
            details[i] = (x - y) / math.sqrt(2.0)
        details_stack.append(details)
        temp[:half] = averages
        length = half
    coeffs[0] = temp[0]
    pos = 1
    for details in reversed(details_stack):
        L = details.size
        coeffs[pos:pos+L] = details
        pos += L
    return coeffs

def haar1d_inverse(coeffs):
    coeffs = coeffs.astype(np.float64).copy()
    n = coeffs.size
    if n & (n-1):
        raise ValueError("Length must be power of two")
    levels = int(math.log2(n))
    a = coeffs[0:1].copy()
    pos = 1
    details = []
    for lev in range(levels, 0, -1):
        size = 2**(lev-1)
        details.append(coeffs[pos:pos+size].copy())
        pos += size
    details.reverse()
    temp = a.copy()
    for d in details:
        new = np.zeros(temp.size * 2)
        for i in range(temp.size):
            avg = temp[i]
            diff = d[i]
            x = (avg + diff) / math.sqrt(2.0)
            y = (avg - diff) / math.sqrt(2.0)
            new[2*i] = x
            new[2*i+1] = y
        temp = new
    return temp

# --- 2D Haar via separability ---
def haar2d_forward(img):
    img = img.astype(np.float64)
    N, M = img.shape
    if N != M:
        raise ValueError("Image must be square for this simple implementation")
    if N & (N-1):
        raise ValueError("Size must be power of two")
    temp = np.zeros_like(img)
    for i in range(N):
        temp[i, :] = haar1d_forward_clear(img[i, :])
    out = np.zeros_like(temp)
    for j in range(N):
        out[:, j] = haar1d_forward_clear(temp[:, j])
    return out

def haar2d_inverse(coeffs2d):
    coeffs2d = coeffs2d.astype(np.float64)
    N, M = coeffs2d.shape
    temp = np.zeros_like(coeffs2d)
    for j in range(N):
        temp[:, j] = haar1d_inverse(coeffs2d[:, j])
    out = np.zeros_like(temp)
    for i in range(N):
        out[i, :] = haar1d_inverse(temp[i, :])
    return out

def keep_top_k_coeffs(arr, k):
    out = np.zeros_like(arr)
    flat = arr.flatten()
    idx = np.argsort(np.abs(flat))[::-1]
    k = min(k, flat.size)
    top_idx = idx[:k]
    flat2 = np.zeros_like(flat)
    flat2[top_idx] = flat[top_idx]
    return flat2.reshape(arr.shape)

def downsample_then_upsample(img, factor):
    N = img.shape[0]
    small_N = max(1, N // factor)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    small = pil.resize((small_N, small_N), resample=Image.BICUBIC)
    up = small.resize((N, N), resample=Image.BICUBIC)
    return np.array(up, dtype=np.float64)

# --- Input image selection ---
print("Looking for input image 'q2_input.png' in working directory...")
input_candidates = [
    "q2_input.png",
    "./q2_input.png",
    "/mnt/data/q2_input.png"
]
input_path = None
for p in input_candidates:
    if os.path.exists(p):
        input_path = p
        break

if input_path is None:
    print("No input image found. Generating synthetic image (checkerboard + gradient) of size 256x256.")
    N = 256
    checker = np.indices((N,N)).sum(axis=0) % 2
    gradient = np.linspace(0, 255, N).reshape(1, N).repeat(N, axis=0)
    img = (checker * 120 + gradient * 0.5).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img.save(os.path.join(OUT_ROOT, "generated_q2_input.png"))
else:
    print("Found input image:", input_path)
    pil_img = Image.open(input_path).convert("L")
    w,h = pil_img.size
    N = min(w,h)
    pow2 = 1 << (N.bit_length()-1)
    if pow2 != N:
        print(f"Input size {N} is not power of two; cropping to {pow2}x{pow2}.")
        pil_img = pil_img.crop((0,0,pow2,pow2))
        N = pow2
    pil_img = pil_img.resize((N,N), resample=Image.BICUBIC)

img_arr = np.array(pil_img, dtype=np.float64)
Image.fromarray(np.clip(img_arr,0,255).astype(np.uint8)).save(os.path.join(OUT_ROOT, "q2_input_used.png"))

# --- Part A ---
print("======Part-A processing...======")
N = img_arr.shape[0]
middle_row = img_arr[N//2, :].astype(np.float64)
L = middle_row.size
if L & (L-1):
    raise RuntimeError("Middle row length not power of two after preprocessing")

coeffs_1d = haar1d_forward_clear(middle_row)
recon_1d = haar1d_inverse(coeffs_1d)
norm_orig = np.linalg.norm(middle_row)
norm_trans = np.linalg.norm(coeffs_1d)
norm_recon = np.linalg.norm(recon_1d)
with open(os.path.join(PART_A_DIR, "partA_report.txt"), "w") as f:
    f.write("Part A: 1D Haar transform numerical verification\n")
    f.write(f"Original signal norm: {norm_orig:.12f}\n")
    f.write(f"Transform coefficients norm: {norm_trans:.12f}\n")
    f.write(f"Reconstructed signal norm: {norm_recon:.12f}\n")
    f.write("Reconstruction max abs error: {:.6e}\n".format(np.max(np.abs(middle_row - recon_1d))))
save_image(middle_row, os.path.join(PART_A_DIR, "middle_row_original.png"))
save_image(coeffs_1d, os.path.join(PART_A_DIR, "middle_row_transform.png"))
save_image(recon_1d, os.path.join(PART_A_DIR, "middle_row_reconstructed.png"))
with open(os.path.join(PART_A_DIR, "proof_summary.txt"), "w") as f:
    f.write("Proof (summary): The Haar transform pairs consecutive samples and maps (x,y) -> ((x+y)/sqrt(2), (x-y)/sqrt(2)).\n")
    f.write("This is an orthonormal rotation in R^2; repeated dyadic application preserves the Euclidean norm.\n")

# --- Part B ---
print("======Part-B processing...======")
haar2d = haar2d_forward(img_arr)
denom = max(1.0, np.abs(haar2d).max())
vis = 127 + (haar2d / denom) * 127
vis = np.clip(vis, 0, 255)
recon_img = haar2d_inverse(haar2d)
recon_img_clipped = np.clip(recon_img, 0, 255)
save_image(img_arr, os.path.join(PART_B_DIR, "input_image.png"))
save_image(vis, os.path.join(PART_B_DIR, "haar2d_visual_shifted.png"))
save_image(recon_img_clipped, os.path.join(PART_B_DIR, "reconstructed_from_haar2d.png"))
with open(os.path.join(PART_B_DIR, "partB_report.txt"), "w") as f:
    f.write("Part B: 2D Haar transform and reconstruction\n")
    f.write(f"Input image shape: {img_arr.shape}\n")
    f.write(f"Max abs(diff) between input and reconstruction: {np.max(np.abs(img_arr - recon_img_clipped)):.6e}\n")
    f.write(f"PSNR between input and reconstruction: {psnr(img_arr, recon_img_clipped):.6f} dB\n")

# --- Part C ---
print("======Part-C processing...======")
N = img_arr.shape[0]
total_coeffs = N * N
k1 = total_coeffs // 16
top_k1 = keep_top_k_coeffs(haar2d, k1)
rec1 = haar2d_inverse(top_k1)
rec1_clipped = np.clip(rec1, 0, 255)
rec_ds1 = downsample_then_upsample(img_arr, factor=4)
save_image(rec1_clipped, os.path.join(PART_C_DIR, f"recon_top_{k1}_coeffs.png"))
save_image(rec_ds1, os.path.join(PART_C_DIR, f"recon_downsample_{N//4}x{N//4}_upsampled.png"))
mse1 = mse(img_arr, rec1_clipped)
mse_ds1 = mse(img_arr, rec_ds1)
psnr1 = psnr(img_arr, rec1_clipped)
psnr_ds1 = psnr(img_arr, rec_ds1)
with open(os.path.join(PART_C_DIR, "partC_experiment1.txt"), "w") as f:
    f.write("Experiment 1: keep top N^2/16 coefficients vs downsample to N/4 x N/4\n")
    f.write(f"Image size: {N}x{N}\n")
    f.write(f"k (coefficients kept): {k1}\n")
    f.write(f"MSE (haar top-k): {mse1:.6f}\n")
    f.write(f"PSNR (haar top-k): {psnr1:.6f} dB\n")
    f.write(f"MSE (downsample then upsample): {mse_ds1:.6f}\n")
    f.write(f"PSNR (downsample then upsample): {psnr_ds1:.6f} dB\n")

k2 = max(1, total_coeffs // 256)
top_k2 = keep_top_k_coeffs(haar2d, k2)
rec2 = haar2d_inverse(top_k2)
rec2_clipped = np.clip(rec2, 0, 255)
rec_ds2 = downsample_then_upsample(img_arr, factor=16)
save_image(rec2_clipped, os.path.join(PART_C_DIR, f"recon_top_{k2}_coeffs.png"))
save_image(rec_ds2, os.path.join(PART_C_DIR, f"recon_downsample_{N//16}x{N//16}_upsampled.png"))
mse2 = mse(img_arr, rec2_clipped)
mse_ds2 = mse(img_arr, rec_ds2)
psnr2 = psnr(img_arr, rec2_clipped)
psnr_ds2 = psnr(img_arr, rec_ds2)
with open(os.path.join(PART_C_DIR, "partC_experiment2.txt"), "w") as f:
    f.write("Experiment 2: keep top N^2/256 coefficients vs downsample to N/16 x N/16\n")
    f.write(f"k (coefficients kept): {k2}\n")
    f.write(f"MSE (haar top-k): {mse2:.6f}\n")
    f.write(f"PSNR (haar top-k): {psnr2:.6f} dB\n")
    f.write(f"MSE (downsample then upsample): {mse_ds2:.6f}\n")
    f.write(f"PSNR (downsample then upsample): {psnr_ds2:.6f} dB\n")

np.save(os.path.join(OUT_ROOT, "haar2d_raw.npy"), haar2d)

zipf = os.path.join(".", "Q2_Output.zip")
if os.path.exists(zipf):
    os.remove(zipf)
with zipfile.ZipFile(zipf, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT_ROOT):
        for file in files:
            full = os.path.join(root, file)
            arcname = os.path.relpath(full, OUT_ROOT)
            zf.write(full, arcname)

print("All processing complete. Outputs zipped to", zipf)
