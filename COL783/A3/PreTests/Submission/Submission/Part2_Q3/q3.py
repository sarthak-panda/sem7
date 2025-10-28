import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
output_dirs = {
    'base': './Q3_Output',
    'partA': './Q3_Output/PartA',
    'partB': './Q3_Output/PartB',
    'partC': './Q3_Output/PartC'
}
for dir_path in output_dirs.values():
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def calculate_psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def add_gaussian_noise_for_psnr(image, target_psnr):
    max_pixel = 255.0
    target_mse = (max_pixel / (10 ** (target_psnr / 20.0))) ** 2
    sigma = np.sqrt(target_mse)
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    actual_psnr = calculate_psnr(image, noisy_image)
    print(f"Target PSNR: {target_psnr} dB, Actual PSNR: {actual_psnr:.2f} dB")
    return noisy_image, noise

def haar_wavelet_1d(signal):
    n = len(signal)
    if n % 2 != 0:
        raise ValueError("Signal length must be even")
    approx = (signal[::2] + signal[1::2]) / np.sqrt(2)
    detail = (signal[::2] - signal[1::2]) / np.sqrt(2)
    return np.concatenate([approx, detail])

def haar_wavelet_1d_inverse(coeffs):
    n = len(coeffs)
    half = n // 2
    approx = coeffs[:half]
    detail = coeffs[half:]
    signal = np.zeros(n)
    signal[::2] = (approx + detail) / np.sqrt(2)
    signal[1::2] = (approx - detail) / np.sqrt(2)
    return signal

def haar_2d_single_level(image):
    rows, cols = image.shape
    temp = np.zeros_like(image)
    for i in range(rows):
        temp[i, :] = haar_wavelet_1d(image[i, :])
    result = np.zeros_like(image)
    for j in range(cols):
        result[:, j] = haar_wavelet_1d(temp[:, j])
    return result

def haar_2d_single_level_inverse(coeffs):
    rows, cols = coeffs.shape
    temp = np.zeros_like(coeffs)
    for j in range(cols):
        temp[:, j] = haar_wavelet_1d_inverse(coeffs[:, j])
    result = np.zeros_like(coeffs)
    for i in range(rows):
        result[i, :] = haar_wavelet_1d_inverse(temp[i, :])
    return result

def haar_multilevel_transform(image, levels):
    result = image.copy()
    rows, cols = image.shape
    current_rows, current_cols = rows, cols
    for level in range(levels):
        if current_rows < 2 or current_cols < 2:
            print(f"Warning: Cannot perform level {level+1}, dimensions too small")
            break
        upper_left = result[:current_rows, :current_cols]
        transformed = haar_2d_single_level(upper_left)
        result[:current_rows, :current_cols] = transformed
        current_rows = current_rows // 2
        current_cols = current_cols // 2
    return result

def haar_multilevel_inverse(coeffs, levels):
    result = coeffs.copy()
    rows, cols = coeffs.shape
    dimensions = [(rows, cols)]
    for _ in range(levels - 1):
        dimensions.append((dimensions[-1][0] // 2, dimensions[-1][1] // 2))
    dimensions.reverse()
    for level, (current_rows, current_cols) in enumerate(dimensions):
        if current_rows < 2 or current_cols < 2:
            continue
        upper_left = result[:current_rows, :current_cols]
        inverse_transformed = haar_2d_single_level_inverse(upper_left)
        result[:current_rows, :current_cols] = inverse_transformed
    return result

def get_db2_coefficients():
    h = np.array([
        (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
        (1 - np.sqrt(3)) / (4 * np.sqrt(2))
    ])
    g = ((-1.)**np.arange(len(h))) * h[::-1]
    h = np.asarray(h).ravel()
    g = np.asarray(g).ravel()
    return h, g

def db2_wavelet_1d_forward(signal, h, g, mode='periodic'):
    n = len(signal)
    if mode == 'periodic':
        extended = np.concatenate([signal[-3:], signal, signal[:3]])
    # elif mode == 'symmetric':
    #     extended = np.concatenate([signal[2::-1], signal, signal[:-4:-1]])
    else:
        # extended = np.pad(signal, (3, 3), mode='constant')
        raise NotImplementedError(f"Mode '{mode}' not implemented yet, will be added later.")
    approx = []
    detail = []
    for i in range(n):
        a = sum(h[j] * extended[i + 3 - j] for j in range(4))
        approx.append(a)
        d = sum(g[j] * extended[i + 3 - j] for j in range(4))
        detail.append(d)
    approx = np.array(approx[::2])
    detail = np.array(detail[::2])
    return np.concatenate([approx, detail])

# def db2_wavelet_1d_inverse(coeffs, h, g, mode='periodic'):
#     n = len(coeffs)
#     half = n // 2
#     approx = coeffs[:half]
#     detail = coeffs[half:]
#     h_r = h[::-1]
#     g_r = g[::-1]
#     approx_up = np.zeros(n)
#     detail_up = np.zeros(n)
#     approx_up[::2] = approx
#     detail_up[::2] = detail
#     if mode == 'periodic':
#         approx_ext = np.concatenate([approx_up[-3:], approx_up, approx_up[:3]])
#         detail_ext = np.concatenate([detail_up[-3:], detail_up, detail_up[:3]])
#     elif mode == 'symmetric':
#         approx_ext = np.concatenate([approx_up[2::-1], approx_up, approx_up[:-4:-1]])
#         detail_ext = np.concatenate([detail_up[2::-1], detail_up, detail_up[:-4:-1]])
#     else:
#         approx_ext = np.pad(approx_up, (3, 3), mode='constant')
#         detail_ext = np.pad(detail_up, (3, 3), mode='constant')
#     signal = np.zeros(n)
#     for i in range(n):
#         signal[i] = sum(h_r[j] * approx_ext[i + j + 3] for j in range(4))
#         signal[i] += sum(g_r[j] * detail_ext[i + j + 3] for j in range(4))
#     return signal

def db2_wavelet_1d_inverse(coeffs, h, g, mode='periodic'):
    n = len(coeffs)
    half = n // 2
    approx = coeffs[:half]
    detail = coeffs[half:]
    h_syn = h[::-1]
    g_syn = g[::-1]
    approx_up = np.zeros(n)
    detail_up = np.zeros(n)
    approx_up[::2] = approx
    detail_up[::2] = detail
    signal = np.zeros(n)
    filter_len = len(h)
    for i in range(n):
        for k in range(filter_len):
            idx = (i - k) % n
            signal[i] += h_syn[k] * approx_up[idx]
            signal[i] += g_syn[k] * detail_up[idx]
    if mode == 'periodic':
        signal = np.roll(signal, -(filter_len - 1))
    return signal

def db2_2d_single_level(image, h, g, mode='periodic'):
    rows, cols = image.shape
    temp = np.zeros_like(image)
    for i in range(rows):
        temp[i, :] = db2_wavelet_1d_forward(image[i, :], h, g, mode)
    result = np.zeros_like(image)
    for j in range(cols):
        result[:, j] = db2_wavelet_1d_forward(temp[:, j], h, g, mode)
    return result

def db2_2d_single_level_inverse(coeffs, h, g, mode='periodic'):
    rows, cols = coeffs.shape
    temp = np.zeros_like(coeffs)
    for j in range(cols):
        temp[:, j] = db2_wavelet_1d_inverse(coeffs[:, j], h, g, mode)
    result = np.zeros_like(coeffs)
    for i in range(rows):
        result[i, :] = db2_wavelet_1d_inverse(temp[i, :], h, g, mode)
    return result

def db2_multilevel_transform(image, levels, mode='periodic'):
    h, g = get_db2_coefficients()
    result = image.copy()
    rows, cols = image.shape
    current_rows, current_cols = rows, cols
    for level in range(levels):
        if current_rows < 4 or current_cols < 4:
            print(f"Cannot perform level {level+1}, dimensions too small")
            break
        upper_left = result[:current_rows, :current_cols]
        transformed = db2_2d_single_level(upper_left, h, g, mode)
        result[:current_rows, :current_cols] = transformed
        current_rows = current_rows // 2
        current_cols = current_cols // 2
    return result

def db2_multilevel_inverse(coeffs, levels, mode='periodic'):
    h, g = get_db2_coefficients()
    result = coeffs.copy()
    rows, cols = coeffs.shape
    dimensions = [(rows, cols)]
    for _ in range(levels - 1):
        dimensions.append((dimensions[-1][0] // 2, dimensions[-1][1] // 2))
    dimensions.reverse()
    for level, (current_rows, current_cols) in enumerate(dimensions):
        if current_rows < 4 or current_cols < 4:
            continue
        upper_left = result[:current_rows, :current_cols]
        inverse_transformed = db2_2d_single_level_inverse(upper_left, h, g, mode)
        result[:current_rows, :current_cols] = inverse_transformed
    return result

def hard_threshold(coeffs, threshold):
    result = coeffs.copy()
    result[np.abs(result) < threshold] = 0
    return result

def soft_threshold(coeffs, threshold):
    result = np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    return result

def extract_detail_coefficients(coeffs, levels, rows, cols):
    details = []
    current_rows, current_cols = rows, cols
    for level in range(levels):
        current_rows = current_rows // 2
        current_cols = current_cols // 2
        if current_rows < 1 or current_cols < 1:
            break
        lh = coeffs[:current_rows, current_cols:2*current_cols].flatten()
        hl = coeffs[current_rows:2*current_rows, :current_cols].flatten()
        hh = coeffs[current_rows:2*current_rows, current_cols:2*current_cols].flatten()
        details.extend(lh)
        details.extend(hl)
        details.extend(hh)
    return np.array(details)

def visualize_wavelet_coeffs(coeffs, levels, rows, cols):
    result = coeffs.copy()
    current_rows, current_cols = rows, cols
    for level in range(levels):
        current_rows = current_rows // 2
        current_cols = current_cols // 2
        if current_rows < 1 or current_cols < 1:
            break
        result[:current_rows, current_cols:2*current_cols] += 128
        result[current_rows:2*current_rows, :current_cols] += 128
        result[current_rows:2*current_rows, current_cols:2*current_cols] += 128
    return np.clip(result, 0, 255)

print("=" * 60)
print("======Part-A processing...======")
print("=" * 60)
img = Image.open('../Testcases/q3_input.png').convert('L')
f = np.array(img, dtype=np.float64)
rows, cols = f.shape
print(f"Loaded image: {rows} x {cols}")
np.random.seed(42)
g, noise = add_gaussian_noise_for_psnr(f, target_psnr=20.0)
Image.fromarray(g.astype(np.uint8)).save('./Q3_Output/PartA/noisy_image.png')
print("Saved noisy image")
levels = 5
print(f"Performing {levels}-level Haar wavelet transform...")
f_wavelet = haar_multilevel_transform(f, levels)
g_wavelet = haar_multilevel_transform(g, levels)
noise_wavelet = haar_multilevel_transform(noise, levels)
print("Haar wavelet transforms completed")
f_wavelet_vis = visualize_wavelet_coeffs(f_wavelet, levels, rows, cols)
g_wavelet_vis = visualize_wavelet_coeffs(g_wavelet, levels, rows, cols)
noise_wavelet_vis = visualize_wavelet_coeffs(noise_wavelet, levels, rows, cols)
Image.fromarray(f_wavelet_vis.astype(np.uint8)).save('./Q3_Output/PartA/f_wavelet_coeffs.png')
Image.fromarray(g_wavelet_vis.astype(np.uint8)).save('./Q3_Output/PartA/g_wavelet_coeffs.png')
Image.fromarray(noise_wavelet_vis.astype(np.uint8)).save('./Q3_Output/PartA/noise_wavelet_coeffs.png')
print("Saved wavelet coefficient visualizations")
f_details = extract_detail_coefficients(f_wavelet, levels, rows, cols)
g_details = extract_detail_coefficients(g_wavelet, levels, rows, cols)
noise_details = extract_detail_coefficients(noise_wavelet, levels, rows, cols)
print(f"Number of detail coefficients: {len(f_details)}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax0, ax1, ax2 = axes
ax0.hist(f_details, bins=100, color='blue', alpha=0.7, edgecolor='black')
ax0.set_title('Histogram of Detail Coefficients (f)')
ax0.set_xlabel('Coefficient Value')
ax0.set_ylabel('Frequency')
ax0.grid(True, alpha=0.3)
ax1.hist(g_details, bins=100, color='red', alpha=0.7, edgecolor='black')
ax1.set_title('Histogram of Detail Coefficients (g)')
ax1.set_xlabel('Coefficient Value')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)
ax2.hist(noise_details, bins=100, color='green', alpha=0.7, edgecolor='black')
ax2.set_title('Histogram of Detail Coefficients (η = g - f)')
ax2.set_xlabel('Coefficient Value')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Q3_Output/PartA/detail_coeffs_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved detail coefficient histograms")
print("Part A completed successfully!\n")

print("=" * 60)
print("======Part-B processing...======")
print("=" * 60)
threshold_values = np.linspace(0, 50, 100)
psnr_hard = []
psnr_soft = []
print(f"Testing {len(threshold_values)} threshold values...")
for i, t in enumerate(threshold_values):
    g_wavelet_hard = hard_threshold(g_wavelet.copy(), t)
    denoised_hard = haar_multilevel_inverse(g_wavelet_hard, levels)
    denoised_hard = np.clip(denoised_hard, 0, 255)
    psnr_hard.append(calculate_psnr(f, denoised_hard))
    g_wavelet_soft = soft_threshold(g_wavelet.copy(), t)
    denoised_soft = haar_multilevel_inverse(g_wavelet_soft, levels)
    denoised_soft = np.clip(denoised_soft, 0, 255)
    psnr_soft.append(calculate_psnr(f, denoised_soft))
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(threshold_values)} thresholds")
print("Threshold testing completed")
best_idx_hard = np.argmax(psnr_hard)
best_idx_soft = np.argmax(psnr_soft)
best_t_hard = threshold_values[best_idx_hard]
best_t_soft = threshold_values[best_idx_soft]
best_psnr_hard = psnr_hard[best_idx_hard]
best_psnr_soft = psnr_soft[best_idx_soft]
print(f"\nHard Thresholding - Best threshold: {best_t_hard:.2f}, Best PSNR: {best_psnr_hard:.2f} dB")
print(f"Soft Thresholding - Best threshold: {best_t_soft:.2f}, Best PSNR: {best_psnr_soft:.2f} dB")
g_wavelet_hard_best = hard_threshold(g_wavelet.copy(), best_t_hard)
denoised_hard_best = haar_multilevel_inverse(g_wavelet_hard_best, levels)
denoised_hard_best = np.clip(denoised_hard_best, 0, 255)
g_wavelet_soft_best = soft_threshold(g_wavelet.copy(), best_t_soft)
denoised_soft_best = haar_multilevel_inverse(g_wavelet_soft_best, levels)
denoised_soft_best = np.clip(denoised_soft_best, 0, 255)
Image.fromarray(denoised_hard_best.astype(np.uint8)).save('./Q3_Output/PartB/denoised_hard_best.png')
Image.fromarray(denoised_soft_best.astype(np.uint8)).save('./Q3_Output/PartB/denoised_soft_best.png')
print("Saved best denoised images")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(threshold_values, psnr_hard, label='Hard Thresholding', linewidth=2, color='blue')
ax.plot(threshold_values, psnr_soft, label='Soft Thresholding', linewidth=2, color='red')
ax.scatter([best_t_hard], [best_psnr_hard], color='blue', s=100, marker='o', 
           label=f'Best Hard (t={best_t_hard:.2f}, PSNR={best_psnr_hard:.2f} dB)', zorder=5)
ax.scatter([best_t_soft], [best_psnr_soft], color='red', s=100, marker='s', 
           label=f'Best Soft (t={best_t_soft:.2f}, PSNR={best_psnr_soft:.2f} dB)', zorder=5)
ax.set_xlabel('Threshold (t)', fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)
ax.set_title('PSNR vs Threshold for Hard and Soft Thresholding (Haar Wavelets)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Q3_Output/PartB/psnr_vs_threshold_haar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved PSNR vs threshold plot")
print("Part B completed successfully!\n")

print("=" * 60)
print("======Part-C processing...======")
print("=" * 60)
print(f"Performing {levels}-level db2 wavelet transform...")
f_wavelet_db2 = db2_multilevel_transform(f, levels, mode='periodic')
g_wavelet_db2 = db2_multilevel_transform(g, levels, mode='periodic')
noise_wavelet_db2 = db2_multilevel_transform(noise, levels, mode='periodic')
print("db2 wavelet transforms completed")
f_wavelet_db2_vis = visualize_wavelet_coeffs(f_wavelet_db2, levels, rows, cols)
g_wavelet_db2_vis = visualize_wavelet_coeffs(g_wavelet_db2, levels, rows, cols)
noise_wavelet_db2_vis = visualize_wavelet_coeffs(noise_wavelet_db2, levels, rows, cols)
Image.fromarray(f_wavelet_db2_vis.astype(np.uint8)).save('./Q3_Output/PartC/f_wavelet_coeffs_db2.png')
Image.fromarray(g_wavelet_db2_vis.astype(np.uint8)).save('./Q3_Output/PartC/g_wavelet_coeffs_db2.png')
Image.fromarray(noise_wavelet_db2_vis.astype(np.uint8)).save('./Q3_Output/PartC/noise_wavelet_coeffs_db2.png')
print("Saved db2 wavelet coefficient visualizations")
f_details_db2 = extract_detail_coefficients(f_wavelet_db2, levels, rows, cols)
g_details_db2 = extract_detail_coefficients(g_wavelet_db2, levels, rows, cols)
noise_details_db2 = extract_detail_coefficients(noise_wavelet_db2, levels, rows, cols)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax0, ax1, ax2 = axes
ax0.hist(f_details_db2, bins=100, color='blue', alpha=0.7, edgecolor='black')
ax0.set_title('Histogram of Detail Coefficients (f) - db2')
ax0.set_xlabel('Coefficient Value')
ax0.set_ylabel('Frequency')
ax0.grid(True, alpha=0.3)
ax1.hist(g_details_db2, bins=100, color='red', alpha=0.7, edgecolor='black')
ax1.set_title('Histogram of Detail Coefficients (g) - db2')
ax1.set_xlabel('Coefficient Value')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)
ax2.hist(noise_details_db2, bins=100, color='green', alpha=0.7, edgecolor='black')
ax2.set_title('Histogram of Detail Coefficients (η = g - f) - db2')
ax2.set_xlabel('Coefficient Value')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Q3_Output/PartC/detail_coeffs_histograms_db2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved db2 detail coefficient histograms")
threshold_values_db2 = np.linspace(0, 50, 100)
psnr_hard_db2 = []
psnr_soft_db2 = []
print(f"Testing {len(threshold_values_db2)} threshold values with db2...")
for i, t in enumerate(threshold_values_db2):
    g_wavelet_db2_hard = hard_threshold(g_wavelet_db2.copy(), t)
    denoised_hard_db2 = db2_multilevel_inverse(g_wavelet_db2_hard, levels, mode='periodic')
    denoised_hard_db2 = np.clip(denoised_hard_db2, 0, 255)
    psnr_hard_db2.append(calculate_psnr(f, denoised_hard_db2))
    g_wavelet_db2_soft = soft_threshold(g_wavelet_db2.copy(), t)
    denoised_soft_db2 = db2_multilevel_inverse(g_wavelet_db2_soft, levels, mode='periodic')
    denoised_soft_db2 = np.clip(denoised_soft_db2, 0, 255)
    psnr_soft_db2.append(calculate_psnr(f, denoised_soft_db2))
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(threshold_values_db2)} thresholds")
print("db2 threshold testing completed")
best_idx_hard_db2 = np.argmax(psnr_hard_db2)
best_idx_soft_db2 = np.argmax(psnr_soft_db2)
best_t_hard_db2 = threshold_values_db2[best_idx_hard_db2]
best_t_soft_db2 = threshold_values_db2[best_idx_soft_db2]
best_psnr_hard_db2 = psnr_hard_db2[best_idx_hard_db2]
best_psnr_soft_db2 = psnr_soft_db2[best_idx_soft_db2]
print(f"\ndb2 Hard Thresholding - Best threshold: {best_t_hard_db2:.2f}, Best PSNR: {best_psnr_hard_db2:.2f} dB")
print(f"db2 Soft Thresholding - Best threshold: {best_t_soft_db2:.2f}, Best PSNR: {best_psnr_soft_db2:.2f} dB")
g_wavelet_db2_hard_best = hard_threshold(g_wavelet_db2.copy(), best_t_hard_db2)
denoised_hard_db2_best = db2_multilevel_inverse(g_wavelet_db2_hard_best, levels, mode='periodic')
denoised_hard_db2_best = np.clip(denoised_hard_db2_best, 0, 255)
g_wavelet_db2_soft_best = soft_threshold(g_wavelet_db2.copy(), best_t_soft_db2)
denoised_soft_db2_best = db2_multilevel_inverse(g_wavelet_db2_soft_best, levels, mode='periodic')
denoised_soft_db2_best = np.clip(denoised_soft_db2_best, 0, 255)
Image.fromarray(denoised_hard_db2_best.astype(np.uint8)).save('./Q3_Output/PartC/denoised_hard_best_db2.png')
Image.fromarray(denoised_soft_db2_best.astype(np.uint8)).save('./Q3_Output/PartC/denoised_soft_best_db2.png')
print("Saved best db2 denoised images")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(threshold_values_db2, psnr_hard_db2, label='Hard Thresholding (db2)', linewidth=2, color='blue')
ax.plot(threshold_values_db2, psnr_soft_db2, label='Soft Thresholding (db2)', linewidth=2, color='red')
ax.scatter([best_t_hard_db2], [best_psnr_hard_db2], color='blue', s=100, marker='o', 
           label=f'Best Hard (t={best_t_hard_db2:.2f}, PSNR={best_psnr_hard_db2:.2f} dB)', zorder=5)
ax.scatter([best_t_soft_db2], [best_psnr_soft_db2], color='red', s=100, marker='s', 
           label=f'Best Soft (t={best_t_soft_db2:.2f}, PSNR={best_psnr_soft_db2:.2f} dB)', zorder=5)
ax.set_xlabel('Threshold (t)', fontsize=12)
ax.set_ylabel('PSNR (dB)', fontsize=12)
ax.set_title('PSNR vs Threshold for Hard and Soft Thresholding (db2 Wavelets)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Q3_Output/PartC/psnr_vs_threshold_db2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved PSNR vs threshold plot for db2")
print("Part C completed successfully!\n")
print("=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
print("\n--- HAAR WAVELETS ---")
print(f"Hard Thresholding - Best threshold: {best_t_hard:.2f}, Best PSNR: {best_psnr_hard:.2f} dB")
print(f"Soft Thresholding - Best threshold: {best_t_soft:.2f}, Best PSNR: {best_psnr_soft:.2f} dB")
print("\n--- DB2 WAVELETS ---")
print(f"Hard Thresholding - Best threshold: {best_t_hard_db2:.2f}, Best PSNR: {best_psnr_hard_db2:.2f} dB")
print(f"Soft Thresholding - Best threshold: {best_t_soft_db2:.2f}, Best PSNR: {best_psnr_soft_db2:.2f} dB")
print("\n--- COMPARISON ---")
print(f"Noisy image PSNR: 20.00 dB")
print(f"Improvement with Haar (best): {max(best_psnr_hard, best_psnr_soft) - 20:.2f} dB")
print(f"Improvement with db2 (best): {max(best_psnr_hard_db2, best_psnr_soft_db2) - 20:.2f} dB")
print("\n" + "=" * 60)
print("ALL PROCESSING COMPLETE")
print("=" * 60)
