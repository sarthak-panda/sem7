#!/usr/bin/env python3
"""
Gaussian + Laplacian pyramid blending pipeline.

Expect apple.png and orange.png in the working directory.
Outputs written to:
  ./Q1_Output/
    PartA/   -> Gaussian pyramid images for f (apple)
    PartB/   -> Laplacian pyramid images and reconstruction for f
    PartC/   -> Binary mask and hard composite
    PartD/   -> Blended laplacian images and final blended image

Prints:
  "======Part-A processing...======" and similar markers for parts B, C, D.
"""

from PIL import Image
import numpy as np
import os
import sys

# Try to use OpenCV for faster convolution if available (optional)
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

# load input images (expects 'apple.png' and 'orange.png')
f = load_image("../Testcases/q1_apple.jpg")
g = load_image("../Testcases/q1_orange.jpg")

# If shapes differ, center-crop larger image to the smaller image size
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

# Gaussian kernel 1D (as specified): w = (1/16) * [1,4,6,4,1]
kernel_1d = np.array([1,4,6,4,1], dtype=np.float32) / 16.0

# fallback separable convolution (1D on rows then cols) if cv2 not available
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
        out = cv2.filter2D(img, -1, k2d, borderType=cv2.BORDER_REFLECT)
        return out
    else:
        return convolve_separable(img, kernel)

def downsample(img):
    blurred = gaussian_blur(img, kernel_1d)
    return blurred[::2, ::2, :]

def upsample(img, out_shape):
    H_out, W_out = out_shape[0], out_shape[1]
    up = np.zeros((H_out, W_out, img.shape[2]), dtype=np.float32)
    up[::2, ::2, :] = img
    k = kernel_1d * 4.0    # typical factor to compensate energy after insertion of zeros
    up_blur = gaussian_blur(up, k)
    return up_blur

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
    lp.append(gp[-1])  # top residual
    return lp

def reconstruct_from_laplacian(lp):
    cur = lp[-1]
    for i in range(len(lp)-2, -1, -1):
        up = upsample(cur, lp[i].shape[:2])
        if up.shape != lp[i].shape:
            up = up[:lp[i].shape[0], :lp[i].shape[1], :]
        cur = up + lp[i]
    return cur

def float_to_uint8(img):
    arr = np.clip(img*255.0, 0, 255).astype(np.uint8)
    return arr

def save_image(arr, path):
    Image.fromarray(float_to_uint8(arr)).save(path)

def save_pyramid_images(gp, folder, prefix="gauss"):
    for i, im in enumerate(gp):
        save_image(im, os.path.join(folder, f"{prefix}_level_{i}.png"))

def save_laplacian_images(lp, folder, prefix="lap"):
    for i, im in enumerate(lp):
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
m = np.zeros((H,W,1), dtype=np.float32)
m[:, W//2:, 0] = 1.0   # mask: 0 for left (f), 1 for right (g)
save_image(np.repeat(m, 3, axis=2), os.path.join(PART_C_DIR, "mask_binary.png"))
hard = (1.0 - m) * f + m * g
save_image(hard, os.path.join(PART_C_DIR, "hard_composite.png"))
print(f"Saved binary mask and hard composite under {PART_C_DIR}")

# ----------------- PART D -----------------
print("======Part-D processing...======")

def build_gaussian_pyramid_mask(mask, max_levels=None):
    """
    Build gaussian pyramid for a single-channel mask.
    Ensures blurred arrays always have shape (H, W, 1).
    """
    gp = [mask if mask.ndim == 3 else mask[..., np.newaxis]]
    cur = gp[0]
    level = 0
    while True:
        if max_levels is not None and level+1 >= max_levels:
            break
        if cur.shape[0] < 16 or cur.shape[1] < 16:
            break

        # blur mask (keep it single-channel with trailing dim)
        if _HAS_CV2:
            # cv2.filter2D will return 2D for single-channel input,
            # so force shape back to (H, W, 1) after filtering.
            k2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)
            # cv2 expects single-channel to be HxW (not HxWx1) so pass squeezed
            blurred2d = cv2.filter2D(cur[..., 0], -1, k2d, borderType=cv2.BORDER_REFLECT)
            blurred = blurred2d[..., np.newaxis]
        else:
            # separable conv keeping the channel dimension
            pad = len(kernel_1d)//2
            Hc, Wc = cur.shape[:2]
            tmp = np.zeros_like(cur, dtype=np.float32)
            # rows
            for i in range(Hc):
                row = cur[i, :, 0]
                row_p = np.pad(row, pad, mode='reflect')
                tmp[i, :, 0] = np.convolve(row_p, kernel_1d, mode='valid')
            # cols
            blurred = np.zeros_like(tmp, dtype=np.float32)
            for j in range(Wc):
                col = tmp[:, j, 0]
                col_p = np.pad(col, pad, mode='reflect')
                blurred[:, j, 0] = np.convolve(col_p, kernel_1d, mode='valid')

        # downsample and ensure the result keeps the channel axis
        nxt = blurred[::2, ::2]
        if nxt.ndim == 2:
            nxt = nxt[..., np.newaxis]
        gp.append(nxt)
        cur = nxt
        level += 1

    return gp

gp_m = build_gaussian_pyramid_mask(m)
gp_g = build_gaussian_pyramid(g, max_levels=len(gp_f))
lp_g = build_laplacian_pyramid(gp_g)

# Align levels
min_levels = min(len(lp_f), len(lp_g), len(gp_m))
lp_f = lp_f[:min_levels]
lp_g = lp_g[:min_levels]
gp_m = gp_m[:min_levels]

# Blend Laplacian levels using gaussian mask levels
blended_lp = []
for i in range(min_levels):
    mask_level = gp_m[i]
    if mask_level.shape[2] == 1:
        mask3 = np.repeat(mask_level, 3, axis=2)
    else:
        mask3 = mask_level[..., :3]
    blended = (1.0 - mask3) * lp_f[i] + mask3 * lp_g[i]
    blended_lp.append(blended)

save_laplacian_images(blended_lp, PART_D_DIR, prefix="blended_lap")
final = reconstruct_from_laplacian(blended_lp)
save_image(final, os.path.join(PART_D_DIR, "final_blended.png"))
print(f"Saved blended pyramid and final blended image under {PART_D_DIR}")

# Side-by-side comparison image
cmp = np.hstack([f, hard, final])
save_image(cmp, os.path.join(OUT_DIR, "comparison_f_hard_blended.png"))

print("Pipeline finished. Outputs written to ./Q1_Output (PartA, PartB, PartC, PartD).")
