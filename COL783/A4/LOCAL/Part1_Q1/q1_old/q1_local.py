#!/usr/bin/env python3
"""
Morphological Operations for Character Detection - UPDATED VERSION
Improvements (this revision):
1. Use library Otsu (skimage or OpenCV fallback) inside otsu_thresholding()
2. Improved B2 computation (thicker exterior border, center cleared)
3. Tolerant hit-or-miss transform that allows small mismatches
4. Kept custom erosion/dilation/opening/closing implementations
"""

import numpy as np
from PIL import Image
import os

# Try to import skimage's Otsu; fallback to cv2 if not available
try:
    from skimage.filters import threshold_otsu
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    try:
        import cv2
        _HAS_CV2 = True
    except Exception:
        _HAS_CV2 = False

# ============================================================================

def custom_erosion(image, se):
    image = image.astype(np.uint8)
    se = se.astype(np.uint8)
    output = np.zeros_like(image)
    se_h, se_w = se.shape
    cen_y = se_h // 2
    cen_x = se_w // 2
    padded = np.pad(image, ((cen_y, cen_y), (cen_x, cen_x)), mode='constant', constant_values=0)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neigh = padded[y:y+se_h, x:x+se_w]
            masked = neigh * se
            if np.all(masked == se):
                output[y, x] = 1
    return output

def custom_dilation(image, se):
    image = image.astype(np.uint8)
    se = se.astype(np.uint8)
    se = np.flip(se)
    output = np.zeros_like(image)
    se_h, se_w = se.shape
    cen_y = se_h // 2
    cen_x = se_w // 2
    padded = np.pad(image, ((cen_y, cen_y), (cen_x, cen_x)), mode='constant', constant_values=0)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neigh = padded[y:y+se_h, x:x+se_w]
            masked = neigh * se
            if np.any(masked > 0):
                output[y, x] = 1
    return output

def custom_opening(image, se):
    return custom_dilation(custom_erosion(image, se), se)

def custom_closing(image, se):
    return custom_erosion(custom_dilation(image, se), se)

def create_disk_se(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x**2 + y**2 <= radius**2
    return disk.astype(np.uint8)

# ============================================================================

def otsu_thresholding(image):
    """
    Use skimage.threshold_otsu if available; otherwise fallback to OpenCV Otsu.
    The function returns a binary image where *text pixels are 1* and background 0.
    This assumes text is darker (typical scanned black text on white background).
    """
    img = image.astype(np.uint8)
    if _HAS_SKIMAGE:
        t = threshold_otsu(img)
        # text is usually darker: pixel < t -> text
        binary = (img < t).astype(np.uint8)
        return binary, t
    elif _HAS_CV2:
        # cv2.threshold returns threshold and binary image (0/255)
        # But cv2.THRESH_BINARY uses >threshold as foreground, so we compute threshold and then invert test
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        binary = (img < t).astype(np.uint8)
        return binary, t
    else:
        # final fallback: simple histogram Otsu (keeps your old code's spirit but simpler)
        # (rarely executed if skimage or cv2 present)
        pixel_values = img.flatten()
        if pixel_values.max() == pixel_values.min():
            return np.ones(img.shape, dtype=np.uint8), 128
        pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min()) * 255
        pixel_values = pixel_values.astype(np.uint8)
        hist = np.bincount(pixel_values, minlength=256).astype(float)
        hist = hist / hist.sum()
        cs = np.cumsum(hist)
        means = np.cumsum(hist * np.arange(256))
        total_mean = means[-1]
        max_var = 0.0
        opt_t = 0
        for t in range(256):
            w0 = cs[t]
            w1 = 1 - w0
            if w0 == 0 or w1 == 0:
                continue
            mu0 = means[t] / w0
            mu1 = (total_mean - means[t]) / w1
            var = w0 * w1 * (mu0 - mu1) ** 2
            if var > max_var:
                max_var = var
                opt_t = t
        binary = (pixel_values < opt_t).astype(np.uint8).reshape(img.shape)
        return binary, opt_t

# ============================================================================

def compute_exterior_border(image, se):
    dilated = custom_dilation(image, se)
    exterior = dilated - image
    exterior = np.clip(exterior, 0, 1).astype(np.uint8)
    return exterior

def compute_B2_improved(B, border_se_size=3, thicken_iter=1):
    """
    Compute B2 from B as exterior border, then optionally thicken the border
    to make matching tolerant. Also clear the center pixel(s) so B2 doesn't
    require a foreground at the center.
    """
    # small square SE for initial dilation
    border_se_size=2
    border_se = np.ones((border_se_size, border_se_size), dtype=np.uint8)
    B2 = compute_exterior_border(B, border_se)
    # small_se = np.ones((1, 1), dtype=np.uint8)
    # B2 = custom_erosion(B2, small_se)
    # cross_se = np.array([[0, 1, 0],
    #                          [1, 1, 1],
    #                          [0, 1, 0]], dtype=np.uint8)
    # B2 = custom_erosion(B2, cross_se)
    #
    # # Optionally thicken the border (makes B2 more tolerant to small shifts)
    # if thicken_iter > 0:
    # thicken_iter=1
    # for _ in range(thicken_iter):
    #     B2 = custom_dilation(B2, border_se)
    #     B2 = (B2 > 0).astype(np.uint8)
    # B2=B2-B
    # Clear center region of B2 to avoid forcing a background pixel exactly at center
    # h, w = B2.shape
    # cy, cx = h // 2, w // 2
    # # clear a small cross near center to be safe
    # for dy in range(-1, 2):
    #     for dx in range(-1, 2):
    #         yy = cy + dy
    #         xx = cx + dx
    #         if 0 <= yy < h and 0 <= xx < w:
    #             B2[yy, xx] = 0

    return B2

# ============================================================================

# def tolerant_hit_or_miss(image, se1, se2, tol_fg=None, tol_bg=None):
#     """
#     Tolerant hit-or-miss:
#       - Erodes with se1 but allows up to tol_fg foreground pixels missing
#       - Erodes complement with se2 but allows up to tol_bg background pixels missing
#     Returns binary map of detection points.
#
#     If tol_fg/tol_bg are None, default tolerances are set relative to number of
#     ones in se1/se2 respectively.
#     """
#     # Convert inputs to uint8
#     A = image.astype(np.uint8)
#     se1 = se1.astype(np.uint8)
#     se2 = se2.astype(np.uint8)
#
#     # set tolerances if not provided
#     sum1 = se1.sum()
#     sum2 = se2.sum()
#     if tol_fg is None:
#         tol_fg = max(0, sum1 // 12)  # allow ~8% missing by default
#     if tol_bg is None:
#         tol_bg = max(0, sum2 // 10)  # slightly more lenient on background
#
#     # Prepare padded images
#     h1, w1 = se1.shape
#     cy1, cx1 = h1 // 2, w1 // 2
#     pad_y1, pad_x1 = cy1, cx1
#     padded_A = np.pad(A, ((pad_y1, pad_y1), (pad_x1, pad_x1)), mode='constant', constant_values=0)
#
#     h2, w2 = se2.shape
#     cy2, cx2 = h2 // 2, w2 // 2
#     pad_y2, pad_x2 = cy2, cx2
#     padded_Ac = np.pad(1 - A, ((pad_y2, pad_y2), (pad_x2, pad_x2)), mode='constant', constant_values=0)
#
#     out = np.zeros_like(A)
#
#     # Precompute required counts
#     req1 = int(max(0, sum1 - tol_fg))
#     req2 = int(max(0, sum2 - tol_bg))
#
#     # Slide windows
#     for y in range(A.shape[0]):
#         for x in range(A.shape[1]):
#             neigh1 = padded_A[y:y+h1, x:x+w1]
#             cnt1 = int((neigh1 * se1).sum())
#             if cnt1 < req1:
#                 continue
#             neigh2 = padded_Ac[y:y+h2, x:x+w2]
#             cnt2 = int((neigh2 * se2).sum())
#             if cnt2 < req2:
#                 continue
#             out[y, x] = 1
#
#     return out
# def hit_or_miss(image, se1, se2):
#     """
#     Hit-or-miss transform.
#
#     Args:
#         image: Binary image
#         se1: Structuring element to match foreground
#         se2: Structuring element to match background
#
#     Returns:
#         Binary image showing detected patterns
#     """
#     # Erosion with se1 on image
#     eroded_fg = custom_erosion(image, se1)
#     save_image(eroded_fg, './Q1_Output/PartD/fg_binary_eroded_image.png')
#     # Erosion with se2 on complement of image
#     save_image(1-image, './Q1_Output/PartD/binary_inverted_image.png')
#     eroded_bg = custom_erosion(1 - image, se2)
#     save_image(eroded_bg, './Q1_Output/PartD/binary_eroded_image.png')
#     print(f'{np.sum(eroded_bg)},{np.sum(eroded_fg)}')
#     # Intersection
#     result = eroded_fg * eroded_bg
#     return result
def hit_or_miss(image, se1, se2, tol_fg=12, tol_bg=12):
    """Tolerant hit-or-miss allows small mismatches in pattern matching."""
    A = image.astype(np.uint8)
    se1, se2 = se1.astype(np.uint8), se2.astype(np.uint8)
    
    sum1, sum2 = np.sum(se1), np.sum(se2)
    if tol_fg is None:
        tol_fg = max(1, sum1 // 7)  # ~14% tolerance
    if tol_bg is None:
        tol_bg = max(1, sum2 // 3)  # ~33% tolerance
    
    h1, w1 = se1.shape
    cy1, cx1 = h1 // 2, w1 // 2
    padded_A = np.pad(A, ((cy1, cy1), (cx1, cx1)), mode='constant', constant_values=0)
    
    h2, w2 = se2.shape
    cy2, cx2 = h2 // 2, w2 // 2
    padded_Ac = np.pad(1 - A, ((cy2, cy2), (cx2, cx2)), mode='constant', constant_values=0)
    
    out = np.zeros_like(A)
    req1, req2 = max(0, sum1 - tol_fg), max(0, sum2 - tol_bg)
    
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            neigh1 = padded_A[y:y+h1, x:x+w1]
            if np.sum(neigh1 * se1) < req1:
                continue
            neigh2 = padded_Ac[y:y+h2, x:x+w2]
            if np.sum(neigh2 * se2) < req2:
                continue
            out[y, x] = 1
    
    return out

# ============================================================================

def create_rgb_overlay(original, detected):
    rgb = np.zeros((*original.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = original * 255
    rgb[:, :, 1] = detected * 255
    rgb[:, :, 2] = detected * 255
    return rgb

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(image.shape) == 2:
        img = Image.fromarray((image * 255).astype(np.uint8))
    else:
        img = Image.fromarray(image.astype(np.uint8))
    img.save(path)
    print(f"Saved: {path}")

# ============================================================================

def main():
    os.makedirs('./Q1_Output', exist_ok=True)
    os.makedirs('./Q1_Output/PartA', exist_ok=True)
    os.makedirs('./Q1_Output/PartB', exist_ok=True)
    os.makedirs('./Q1_Output/PartC', exist_ok=True)
    os.makedirs('./Q1_Output/PartD', exist_ok=True)

    print("=" * 70)
    print("======Part-A processing...======")
    print("=" * 70)

    text_img = Image.open('../Testcases/q1_text.jpg').convert('L')
    text_array = np.array(text_img)
    print(f"Loaded text image: shape {text_array.shape}")

    print("Applying Otsu's automatic thresholding (library)...")
    A, otsu_threshold = otsu_thresholding(text_array)
    print(f"Binary image A created: {A.shape}, white pixels: {np.sum(A)}")
    print(f"Optimal threshold found: {otsu_threshold}")
    save_image(A, './Q1_Output/PartA/binary_image_A.png')

    disk_radius = 3
    disk_se = create_disk_se(disk_radius)
    save_image(disk_se, './Q1_Output/PartA/disk_se.png')

    print("Performing dilation with custom implementation...")
    A_dilated = custom_dilation(A, disk_se)
    save_image(A_dilated, './Q1_Output/PartA/A_dilated.png')

    print("Performing closing with custom implementation...")
    A_closed = custom_closing(A, disk_se)
    save_image(A_closed, './Q1_Output/PartA/A_closed.png')

    print("\nPart A completed successfully!")

    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("======Part-B processing...======")
    print("=" * 70)

    letter_c_img = Image.open('../Testcases/q1_c_letter.jpg').convert('L')
    letter_c_array = np.array(letter_c_img)
    print(f"Loaded letter 'c' image: shape {letter_c_array.shape}")

    print("Applying Otsu's automatic thresholding to letter 'c'...")
    B, otsu_threshold_c = otsu_thresholding(letter_c_array)
    print(f"Binary SE B created: {B.shape}, foreground pixels: {np.sum(B)}")
    print(f"Optimal threshold found: {otsu_threshold_c}")
    save_image(B, './Q1_Output/PartB/SE_B_original.png')

    print("Performing opening operation with B on A...")
    A_opened_B = custom_opening(A, B)
    save_image(A_opened_B, './Q1_Output/PartB/A_opened_with_B.png')
    rgb_result_B = create_rgb_overlay(A, A_opened_B)
    save_image(rgb_result_B, './Q1_Output/PartB/result_overlay_B.png')
    print("Part B done.")

    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("======Part-C processing...======")
    print("=" * 70)

    print("Creating modified SE B1 by eroding B...")
    small_se = np.ones((2, 2), dtype=np.uint8)
    B1 = custom_erosion(B, small_se)
    if np.sum(B1) < 3:
        print("Erosion too aggressive, trying cross SE fallback...")
        cross_se = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
        B1 = custom_erosion(B, cross_se)
    if np.sum(B1) < 3:
        print("Using minimal erosion hack (trim border)...")
        B1 = B.copy()
        B1[0, :] = 0
        B1[-1, :] = 0
        B1[:, 0] = 0
        B1[:, -1] = 0

    save_image(B1, './Q1_Output/PartC/SE_B1_modified.png')
    A_opened_B1 = custom_opening(A, B1)
    save_image(A_opened_B1, './Q1_Output/PartC/A_opened_with_B1.png')
    rgb_result_C = create_rgb_overlay(A, A_opened_B1)
    save_image(rgb_result_C, './Q1_Output/PartC/result_overlay_C.png')
    print("Part C done.")

    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("======Part-D processing...======")
    print("=" * 70)

    print("Creating SE B2 using improved exterior border method...")
    # produce a somewhat thicker border to be tolerant; tune thicken_iter if needed
    B2 = compute_B2_improved(B, border_se_size=3, thicken_iter=1)
    print(f"SE B2 created: {B2.shape}, background pixels: {np.sum(B2)}")
    save_image(B2, './Q1_Output/PartD/SE_B2_background.png')

    print("Performing tolerant hit-or-miss transform with B1 and B2...")
    # Tolerances: small integers; increase if you still miss c's due to noise
    A_hit_miss = hit_or_miss(A, B1, B2)
    save_image(A_hit_miss, './Q1_Output/PartD/A_hit_miss_result.png')
    print(f"Hit-or-miss complete: detected points = {np.sum(A_hit_miss)}")

    # Dilate hit points to visualize full letters (use original B to expand)
    A_detected_letters = custom_dilation(A_hit_miss, B)
    rgb_result_D = create_rgb_overlay(A, A_detected_letters)
    save_image(rgb_result_D, './Q1_Output/PartD/result_overlay_D.png')
    rgb_points_D = create_rgb_overlay(A, A_hit_miss)
    save_image(rgb_points_D, './Q1_Output/PartD/result_points_overlay_D.png')

    print("\nPart D completed successfully!")
    print("All results saved in ./Q1_Output/ subdirectories")

if __name__ == "__main__":
    main()

