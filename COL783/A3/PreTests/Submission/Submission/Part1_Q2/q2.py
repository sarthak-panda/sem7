import numpy as np
from PIL import Image
import os
output_dirs = ['./Q2_Output', './Q2_Output/PartA', './Q2_Output/PartB', './Q2_Output/PartC']
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

def haar_1d(f):
    n = len(f)
    if n == 1:
        return f.copy()
    t = np.zeros(n, dtype=float)
    half = n // 2
    for i in range(half):
        t[i] = (f[2*i] + f[2*i+1]) / np.sqrt(2)
        t[half + i] = (f[2*i] - f[2*i+1]) / np.sqrt(2)
    t[:half] = haar_1d(t[:half])
    return t

def inverse_haar_1d(t):
    n = len(t)
    if n == 1:
        return t.copy()
    half = n // 2
    t_copy = t.copy()
    t_copy[:half] = inverse_haar_1d(t_copy[:half])
    f = np.zeros(n, dtype=float)
    for i in range(half):
        f[2*i] = (t_copy[i] + t_copy[half + i]) / np.sqrt(2)
        f[2*i+1] = (t_copy[i] - t_copy[half + i]) / np.sqrt(2)
    return f

def verify_orthogonality(f):
    t = haar_1d(f)
    norm_f = np.linalg.norm(f)
    norm_t = np.linalg.norm(t)
    return norm_f, norm_t, abs(norm_f - norm_t)

def haar_2d(f):
    t = np.zeros_like(f, dtype=float)
    for i in range(f.shape[0]):
        t[i, :] = haar_1d(f[i, :])
    for j in range(t.shape[1]):
        t[:, j] = haar_1d(t[:, j])
    return t

def inverse_haar_2d(t):
    f = np.zeros_like(t, dtype=float)
    for j in range(t.shape[1]):
        f[:, j] = inverse_haar_1d(t[:, j])
    for i in range(f.shape[0]):
        f[i, :] = inverse_haar_1d(f[i, :])
    return f

def keep_top_coefficients(haar_coeffs, num_coeffs):
    coeffs_copy = haar_coeffs.copy()
    flat_coeffs = coeffs_copy.flatten()
    abs_coeffs = np.abs(flat_coeffs)
    if num_coeffs < len(flat_coeffs):
        threshold_idx = len(flat_coeffs) - num_coeffs
        sorted_abs = np.sort(abs_coeffs)
        threshold = sorted_abs[threshold_idx]
        mask = abs_coeffs < threshold
        flat_coeffs[mask] = 0
    return flat_coeffs.reshape(coeffs_copy.shape)

def downsample_image(img, factor):
    h, w = img.shape
    new_h, new_w = h // factor, w // factor
    img_pil = Image.fromarray(img.astype(np.uint8))
    downsampled = img_pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(downsampled, dtype=float)

print("======Part-A processing...======")
test_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
t_test = haar_1d(test_signal)
reconstructed = inverse_haar_1d(t_test)
norm_f, norm_t, diff = verify_orthogonality(test_signal)
print(f"Test signal: {test_signal}")
print(f"Haar transform: {t_test}")
print(f"Reconstruction error: {np.linalg.norm(test_signal - reconstructed):.2e}")
print(f"||f|| = {norm_f:.6f}, ||t|| = {norm_t:.6f}, Difference: {diff:.2e}")
img = Image.open('../Testcases/q2_input_new.png').convert('L')
img_array = np.array(img.resize((256,256), Image.LANCZOS), dtype=float)
middle_row = img_array[img_array.shape[0]//2, :]
t_row = haar_1d(middle_row)
reconstructed_row = inverse_haar_1d(t_row)
np.savetxt('./Q2_Output/PartA/middle_row_original.txt', middle_row, fmt='%.6f')
np.savetxt('./Q2_Output/PartA/middle_row_haar.txt', t_row, fmt='%.6f')
print("Part A completed. Results saved to ./Q2_Output/PartA/")

print("="*60)
print("======Part-B processing...======")
img_to_save = Image.fromarray(img_array.astype(np.uint8))
img_to_save.save('./Q2_Output/PartB/input_image.png')
haar_coeffs = haar_2d(img_array)
img = haar_coeffs.copy().astype(float)
den = max(abs(img.min()), abs(img.max()), 1e-12)
img /= den   
h, w = img.shape
out = np.zeros((h, w, 3), dtype=float)
a = 0.005
b = 1.0 - a
c = 0.5
neg_mask = img < 0
pos_mask = img > 0
neg_denom = img.min() - 0.001
pos_denom = img.max() + 0.001
if abs(neg_denom) < 1e-12: neg_denom = -1e-12
if abs(pos_denom) < 1e-12: pos_denom = 1e-12
out[:, :, 0] = 0.0
out[:, :, 2] = 0.0
if neg_mask.any():
    base_neg = img[neg_mask] / neg_denom
    base_neg = np.clip(base_neg, 0.0, None)   
    out[..., 0][neg_mask] = a + b * np.power(base_neg, c)
if pos_mask.any():
    base_pos = img[pos_mask] / pos_denom
    base_pos = np.clip(base_pos, 0.0, None)
    out[..., 2][pos_mask] = a + b * np.power(base_pos, c)
gamma = 2.2
out = np.power(out, 1.0 / gamma)          
haar_img = Image.fromarray((np.clip(out * 255.0, 0, 255)).astype(np.uint8))
haar_img.save('./Q2_Output/PartB/haar_transform.png')
reconstructed = inverse_haar_2d(haar_coeffs)
reconstructed_img = Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8))
reconstructed_img.save('./Q2_Output/PartB/reconstructed_image.png')
print("Part B completed. Images saved to ./Q2_Output/PartB/")

print("="*60)
print("======Part-C processing...======")
N = img_array.shape[0]
num_coeffs_16 = (N * N) // 16
haar_sparse_16 = keep_top_coefficients(haar_coeffs, num_coeffs_16)
reconstructed_sparse_16 = np.clip(inverse_haar_2d(haar_sparse_16), 0, 255)
Image.fromarray(reconstructed_sparse_16.astype(np.uint8)).save('./Q2_Output/PartC/reconstructed_N2_div_16.png')
downsampled_4_up = np.array(Image.fromarray(
    downsample_image(img_array, 4).astype(np.uint8)).resize((N, N), Image.LANCZOS), dtype=float)
Image.fromarray(downsampled_4_up.astype(np.uint8)).save('./Q2_Output/PartC/downsampled_N_div_4.png')
num_coeffs_256 = (N * N) // 256
haar_sparse_256 = keep_top_coefficients(haar_coeffs, num_coeffs_256)
reconstructed_sparse_256 = np.clip(inverse_haar_2d(haar_sparse_256), 0, 255)
Image.fromarray(reconstructed_sparse_256.astype(np.uint8)).save('./Q2_Output/PartC/reconstructed_N2_div_256.png')
downsampled_16_up = np.array(Image.fromarray(
    downsample_image(img_array, 16).astype(np.uint8)).resize((N, N), Image.LANCZOS), dtype=float)
Image.fromarray(downsampled_16_up.astype(np.uint8)).save('./Q2_Output/PartC/downsampled_N_div_16.png')
print("Part C completed. Results saved to ./Q2_Output/PartC/")
print("="*60)