#!/usr/bin/env python3
"""
Fixed Gaussian + Laplacian pyramid blending pipeline.

Saves outputs under ./Q1_Output/PartA..PartD
"""

from PIL import Image
import numpy as np
import os, sys

# Try to use OpenCV (preferred)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

OUT_DIR = "./Q1_Output"
PART_A_DIR = os.path.join(OUT_DIR, "PartA")
PART_B_DIR = os.path.join(OUT_DIR, "PartB")
PART_C_DIR = os.path.join(OUT_DIR, "PartC")
PART_D_DIR = os.path.join(OUT_DIR, "PartD")
for d in [OUT_DIR, PART_A_DIR, PART_B_DIR, PART_C_DIR, PART_D_DIR]:
    os.makedirs(d, exist_ok=True)

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required image not found: {path}. Please put it in the working directory.")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

# change to your input paths if needed
f = load_image("../Testcases/q1_apple.jpg")
g = load_image("../Testcases/q1_orange.jpg")

# center-crop to matching size if needed
if f.shape != g.shape:
    min_h = min(f.shape[0], g.shape[0])
    min_w = min(f.shape[1], g.shape[1])
    def center_crop(a, h, w):
        y0 = max((a.shape[0] - h)//2, 0)
        x0 = max((a.shape[1] - w)//2, 0)
        return a[y0:y0+h, x0:x0+w]
    f = center_crop(f, min_h, min_w)
    g = center_crop(g, min_h, min_w)

H, W, C = f.shape

# 1D Gaussian kernel (for fallback separable conv)
kernel_1d = np.array([1,4,6,4,1], dtype=np.float32) / 16.0

# ----- convolution helpers (fallback when cv2 not present) -----
def convolve_separable(img, kernel):
    pad = len(kernel)//2
    H,W,C = img.shape
    tmp = np.zeros_like(img, dtype=np.float32)
    # rows
    for ch in range(C):
        for i in range(H):
            row = img[i,:,ch]
            row_p = np.pad(row, pad, mode='reflect')
            tmp[i,:,ch] = np.convolve(row_p, kernel, mode='valid')
    out = np.zeros_like(tmp, dtype=np.float32)
    # cols
    for ch in range(C):
        for j in range(W):
            col = tmp[:,j,ch]
            col_p = np.pad(col, pad, mode='reflect')
            out[:,j,ch] = np.convolve(col_p, kernel, mode='valid')
    return out

def gaussian_blur(img, kernel):
    if _HAS_CV2:
        k2d = np.outer(kernel, kernel).astype(np.float32)
        # cv2.filter2D expects single-channel (H,W) or multi-channel (H,W,C) - works with float32
        out = cv2.filter2D(img, -1, k2d, borderType=cv2.BORDER_REFLECT)
        return out
    else:
        return convolve_separable(img, kernel)

# ----- consistent downsample / upsample -----
if _HAS_CV2:
    # use pyramid functions (recommended)
    def downsample(img):
        # cv2.pyrDown accepts float32 arrays; result is half sized
        return cv2.pyrDown(img)

    def upsample(img, out_shape):
        # cv2.pyrUp doubles each dim; if sizes mismatch, crop/pad to out_shape
        up = cv2.pyrUp(img)
        H_out, W_out = out_shape[0], out_shape[1]
        # If pyrUp overshoots by 1 due to odd sizes, crop
        up = up[:H_out, :W_out, :]
        # If undersized (shouldn't happen), pad
        if up.shape[0] < H_out or up.shape[1] < W_out:
            padded = np.zeros((H_out, W_out, img.shape[2]), dtype=np.float32)
            padded[:up.shape[0], :up.shape[1], :] = up
            up = padded
        return up
else:
    # fallback: use separable conv + zero insertion (keeps previous behavior)
    def downsample(img):
        blurred = gaussian_blur(img, kernel_1d)
        return blurred[::2, ::2, :]

    def upsample(img, out_shape):
        H_out, W_out = out_shape[0], out_shape[1]
        up = np.zeros((H_out, W_out, img.shape[2]), dtype=np.float32)
        up[::2, ::2, :] = img
        k = kernel_1d * 4.0
        up_blur = gaussian_blur(up, k)
        # crop/pad just in case
        up_blur = up_blur[:H_out, :W_out, :]
        if up_blur.shape[0] < H_out or up_blur.shape[1] < W_out:
            padded = np.zeros((H_out, W_out, img.shape[2]), dtype=np.float32)
            padded[:up_blur.shape[0], :up_blur.shape[1], :] = up_blur
            up_blur = padded
        return up_blur

# ----- pyramid builders -----
def build_gaussian_pyramid(img, max_levels=None):
    gp = [img]
    cur = img
    level = 0
    while True:
        if max_levels is not None and level+1 >= max_levels:
            break
        if cur.shape[0] < 16 or cur.shape[1] < 16:
            break
        nxt = downsample(cur)
        if nxt.shape[0] < 1 or nxt.shape[1] < 1:
            break
        gp.append(nxt)
        cur = nxt
        level += 1
    return gp

def build_laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp)-1):
        expanded = upsample(gp[i+1], gp[i].shape[:2])
        if expanded.shape != gp[i].shape:
            expanded = expanded[:gp[i].shape[0], :gp[i].shape[1], :]
        lap = gp[i] - expanded
        lp.append(lap)
    lp.append(gp[-1].copy())
    return lp

def reconstruct_from_laplacian(lp):
    cur = lp[-1]
    for i in range(len(lp)-2, -1, -1):
        up = upsample(cur, lp[i].shape[:2])
        if up.shape != lp[i].shape:
            up = up[:lp[i].shape[0], :lp[i].shape[1], :]
        cur = up + lp[i]
    return cur

# ----- image save helpers -----
def float_to_uint8(img):
    arr = np.clip(img*255.0 + 0.5, 0, 255).astype(np.uint8)
    return arr

def save_image(arr, path):
    Image.fromarray(float_to_uint8(arr)).save(path)

def save_pyramid_images(gp, folder, prefix="gauss"):
    for i, im in enumerate(gp):
        save_image(im, os.path.join(folder, f"{prefix}_level_{i}.png"))

def save_laplacian_images(lp, folder, prefix="lap"):
    for i, im in enumerate(lp):
        # display-friendly normalization for each level (not used in blending)
        vmin = float(im.min())
        vmax = float(im.max())
        if vmax - vmin > 1e-8:
            disp = (im - vmin) / (vmax - vmin)
        else:
            disp = np.clip(im, 0, 1)
        save_image(disp, os.path.join(folder, f"{prefix}_level_{i}.png"))

# ----------------- PART A -----------------
print("======Part-A processing...======")
gp_f = build_gaussian_pyramid(f)
save_pyramid_images(gp_f, PART_A_DIR, prefix="f_gauss")
print(f"Saved Gaussian pyramid for f (levels={len(gp_f)}) under {PART_A_DIR}")

# ----------------- PART B -----------------
print("======Part-B processing...======")
lp_f = build_laplacian_pyramid(gp_f)
save_laplacian_images(lp_f, PART_B_DIR, prefix="f_lap")
recon_f = reconstruct_from_laplacian(lp_f)
max_diff = np.max(np.abs(recon_f - f))
save_image(recon_f, os.path.join(PART_B_DIR, "reconstructed_f.png"))
print(f"Reconstructed f saved under {PART_B_DIR}. Max absolute difference: {max_diff:.6e}")

# ----------------- PART C -----------------
print("======Part-C processing...======")
m = np.zeros((H, W, 1), dtype=np.float32)
m[:, W//2:, 0] = 1.0
save_image(np.repeat(m, 3, axis=2), os.path.join(PART_C_DIR, "mask_binary.png"))
hard = (1.0 - m) * f + m * g
save_image(hard, os.path.join(PART_C_DIR, "hard_composite.png"))
print(f"Saved binary mask and hard composite under {PART_C_DIR}")

# ----------------- PART D -----------------
print("======Part-D processing...======")

def build_gaussian_pyramid_mask(mask, max_levels=None):
    # ensure mask is (H,W,1)
    cur = mask if mask.ndim == 3 else mask[..., np.newaxis]
    gp = [cur]
    level = 0
    while True:
        if max_levels is not None and level+1 >= max_levels:
            break
        if cur.shape[0] < 16 or cur.shape[1] < 16:
            break
        nxt = downsample(cur)
        # ensure channel last
        if nxt.ndim == 2:
            nxt = nxt[..., np.newaxis]
        gp.append(nxt)
        cur = nxt
        level += 1
    return gp

gp_m = build_gaussian_pyramid_mask(m)
# build gaussian for g with same maximum levels as f for alignment
gp_g = build_gaussian_pyramid(g, max_levels=len(gp_f))
lp_g = build_laplacian_pyramid(gp_g)

# align lengths (take min)
min_levels = min(len(lp_f), len(lp_g), len(gp_m))
lp_f = lp_f[:min_levels]
lp_g = lp_g[:min_levels]
gp_m = gp_m[:min_levels]

# Blend
blended_lp = []
for i in range(min_levels):
    mask_level = gp_m[i]
    # ensure 3-channel mask for blending (repeat single channel)
    mask3 = mask_level if mask_level.shape[2] == 3 else np.repeat(mask_level[..., :1], 3, axis=2)
    # Clip mask to [0,1] just in case and ensure float32
    mask3 = np.clip(mask3.astype(np.float32), 0.0, 1.0)
    blended = (1.0 - mask3) * lp_f[i] + mask3 * lp_g[i]
    blended_lp.append(blended)

save_laplacian_images(blended_lp, PART_D_DIR, prefix="blended_lap")

final = reconstruct_from_laplacian(blended_lp)
# debug: print range before clipping
print("Final reconstructed range before clipping: min=", float(final.min()), " max=", float(final.max()))
# clip to [0,1] to avoid weird saturations
final = np.clip(final, 0.0, 1.0)
save_image(final, os.path.join(PART_D_DIR, "final_blended.png"))
print(f"Saved blended pyramid and final blended image under {PART_D_DIR}")

# side-by-side
cmp = np.hstack([f, hard, final])
save_image(cmp, os.path.join(OUT_DIR, "comparison_f_hard_blended.png"))

print("Pipeline finished. Outputs written to ./Q1_Output (PartA, PartB, PartC, PartD).")