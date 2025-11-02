import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import os
from pathlib import Path

def dct2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def calculate_psnr(original, reconstructed, max_intensity=255):
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_intensity / np.sqrt(mse))
    return psnr

def calculate_rmse(original, reconstructed):
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    return np.sqrt(mse)

def load_and_crop_image(filename, block_size=16):
    img = Image.open(filename).convert('L')  
    img_array = np.array(img)
    h, w = img_array.shape
    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size
    cropped = img_array[:new_h, :new_w]
    print(f"  Loaded {filename}: Original size {h}x{w}, Cropped (New size) {new_h}x{new_w}")
    return cropped

def save_image(img_array, filename):
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"  Saved: {filename}")

def create_error_image(original, reconstructed, output_path, scale_factor=10):
    error = np.abs(original.astype(float) - reconstructed.astype(float))
    error_scaled = np.clip(error * scale_factor, 0, 255)
    save_image(error_scaled, output_path)
    return error

def apply_block_dct(image, block_size=16):
    h, w = image.shape
    dct_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = dct2d(block.astype(float))
            dct_blocks.append(dct_block)
    return dct_blocks, (h // block_size, w // block_size)

def apply_inverse_block_dct(dct_blocks, grid_shape, block_size=16):
    rows, cols = grid_shape
    h = rows * block_size
    w = cols * block_size
    reconstructed = np.zeros((h, w))
    block_idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            idct_block = idct2d(dct_blocks[block_idx])
            reconstructed[i:i+block_size, j:j+block_size] = idct_block
            block_idx += 1
    return reconstructed

def apply_full_dct(image):
    return dct2d(image.astype(float))

def apply_inverse_full_dct(dct_coeffs):
    return idct2d(dct_coeffs)

def threshold_block_coefficients(dct_block, target_rmse, block_size=16, max_intensity=255):
    thresholded = dct_block.copy()
    abs_coeffs = np.abs(thresholded.flatten())
    sorted_indices = np.argsort(-abs_coeffs)
    sorted_coeffs = abs_coeffs[sorted_indices]
    target_error_energy = (target_rmse ** 2) * (block_size ** 2)
    cumulative_energy = np.cumsum(sorted_coeffs[::-1] ** 2)
    num_to_zero = np.searchsorted(cumulative_energy, target_error_energy)
    num_to_keep = block_size * block_size - num_to_zero
    flat_thresh = thresholded.flatten()
    indices_to_zero = sorted_indices[num_to_keep:]
    flat_thresh[indices_to_zero] = 0
    thresholded = flat_thresh.reshape(block_size, block_size)
    return thresholded, num_to_keep

def threshold_full_coefficients(dct_coeffs, target_rmse, max_intensity=255):
    h, w = dct_coeffs.shape
    thresholded = dct_coeffs.copy()
    abs_coeffs = np.abs(thresholded.flatten())
    sorted_indices = np.argsort(-abs_coeffs)
    sorted_coeffs = abs_coeffs[sorted_indices]
    target_error_energy = (target_rmse ** 2) * (h * w)
    cumulative_energy = np.cumsum(sorted_coeffs[::-1] ** 2)
    num_to_zero = np.searchsorted(cumulative_energy, target_error_energy)
    num_to_keep = h * w - num_to_zero
    flat_thresh = thresholded.flatten()
    indices_to_zero = sorted_indices[num_to_keep:]
    flat_thresh[indices_to_zero] = 0
    thresholded = flat_thresh.reshape(h, w)
    return thresholded, num_to_keep

def part_a():
    print("\n" + "="*60)
    print("======Part-A processing...======")
    print("="*60 + "\n")
    print("Loading images...")
    f1 = load_and_crop_image('../Testcases/q4_q5_f1.jpg')
    f2 = load_and_crop_image('../Testcases/q4_q5_input_f2.png')
    print("\nProcessing f1 (image 1)...")
    dct_blocks_f1, grid_shape_f1 = apply_block_dct(f1, block_size=16)
    print(f"  Number of 16x16 blocks: {len(dct_blocks_f1)}")
    print(f"  Grid shape: {grid_shape_f1[0]} rows x {grid_shape_f1[1]} cols")
    reconstructed_f1 = apply_inverse_block_dct(dct_blocks_f1, grid_shape_f1, block_size=16)
    max_diff_f1 = np.max(np.abs(f1 - reconstructed_f1))
    psnr_f1 = calculate_psnr(f1, reconstructed_f1)
    print(f"  Verification - Max pixel difference: {max_diff_f1:.2e}")
    print(f"  Verification - PSNR: {psnr_f1:.2f} dB")
    save_image(reconstructed_f1, './Q4_Output/PartA/f1_reconstructed.png')
    save_image(f1, './Q4_Output/PartA/f1_original.png')
    print("\nProcessing f2 (image 2)...")
    dct_blocks_f2, grid_shape_f2 = apply_block_dct(f2, block_size=16)
    print(f"  Number of 16x16 blocks: {len(dct_blocks_f2)}")
    print(f"  Grid shape: {grid_shape_f2[0]} rows x {grid_shape_f2[1]} cols")
    reconstructed_f2 = apply_inverse_block_dct(dct_blocks_f2, grid_shape_f2, block_size=16)
    max_diff_f2 = np.max(np.abs(f2 - reconstructed_f2))
    psnr_f2 = calculate_psnr(f2, reconstructed_f2)
    print(f"  Verification - Max pixel difference: {max_diff_f2:.2e}")
    print(f"  Verification - PSNR: {psnr_f2:.2f} dB")
    save_image(reconstructed_f2, './Q4_Output/PartA/f2_reconstructed.png')
    save_image(f2, './Q4_Output/PartA/f2_original.png')
    print("\n  Part A completed")
    print("  Images are identical after DCT-IDCT transformation")
    print("  (within floating point precision)")
    return f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2

def part_b(f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2):
    print("\n" + "="*60)
    print("======Part-B processing...======")
    print("="*60 + "\n")
    target_psnr = 20  
    max_intensity = 255
    target_rmse_per_block = 0.1 * max_intensity  
    print(f"Target: PSNR >= {target_psnr} dB")
    print(f"Target: RMSE <= {target_rmse_per_block:.1f} per block\n")
    print("Processing f1 (image 1) - Block-wise compression...")
    thresholded_blocks_f1 = []
    total_nonzero_f1 = 0
    for block in dct_blocks_f1:
        thresh_block, num_kept = threshold_block_coefficients(
            block, target_rmse_per_block, block_size=16, max_intensity=max_intensity
        )
        thresholded_blocks_f1.append(thresh_block)
        total_nonzero_f1 += num_kept
    f1_hat = apply_inverse_block_dct(thresholded_blocks_f1, grid_shape_f1, block_size=16)
    psnr_f1 = calculate_psnr(f1, f1_hat, max_intensity)
    rmse_f1 = calculate_rmse(f1, f1_hat)
    total_coeffs_f1 = len(dct_blocks_f1) * 16 * 16
    compression_ratio_f1 = total_nonzero_f1 / total_coeffs_f1
    print(f"  PSNR: {psnr_f1:.2f} dB")
    print(f"  RMSE: {rmse_f1:.2f}")
    print(f"  Total coefficients: {total_coeffs_f1}")
    print(f"  Nonzero coefficients: {total_nonzero_f1}")
    print(f"  Percentage kept: {compression_ratio_f1*100:.2f}%")
    save_image(f1_hat, './Q4_Output/PartB/f1_compressed.png')
    error_f1 = create_error_image(f1, f1_hat, './Q4_Output/PartB/f1_error.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f1):.2f}")
    print("\nProcessing f2 (image 2) - Block-wise compression...")
    thresholded_blocks_f2 = []
    total_nonzero_f2 = 0
    for block in dct_blocks_f2:
        thresh_block, num_kept = threshold_block_coefficients(
            block, target_rmse_per_block, block_size=16, max_intensity=max_intensity
        )
        thresholded_blocks_f2.append(thresh_block)
        total_nonzero_f2 += num_kept
    f2_hat = apply_inverse_block_dct(thresholded_blocks_f2, grid_shape_f2, block_size=16)
    psnr_f2 = calculate_psnr(f2, f2_hat, max_intensity)
    rmse_f2 = calculate_rmse(f2, f2_hat)
    total_coeffs_f2 = len(dct_blocks_f2) * 16 * 16
    compression_ratio_f2 = total_nonzero_f2 / total_coeffs_f2
    print(f"  PSNR: {psnr_f2:.2f} dB")
    print(f"  RMSE: {rmse_f2:.2f}")
    print(f"  Total coefficients: {total_coeffs_f2}")
    print(f"  Nonzero coefficients: {total_nonzero_f2}")
    print(f"  Percentage kept: {compression_ratio_f2*100:.2f}%")
    save_image(f2_hat, './Q4_Output/PartB/f2_compressed.png')
    error_f2 = create_error_image(f2, f2_hat, './Q4_Output/PartB/f2_error.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f2):.2f}")
    print("\n  Part B completed")
    with open('./Q4_Output/PartB/summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("PART B: Block-wise DCT Compression Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Target: PSNR >= {target_psnr} dB (RMSE <= {target_rmse_per_block:.1f})\n\n")
        f.write("Image f1 (q4_input_1.png):\n")
        f.write(f"  PSNR: {psnr_f1:.2f} dB\n")
        f.write(f"  RMSE: {rmse_f1:.2f}\n")
        f.write(f"  Total coefficients: {total_coeffs_f1}\n")
        f.write(f"  Nonzero coefficients: {total_nonzero_f1}\n")
        f.write(f"  Percentage kept: {compression_ratio_f1*100:.2f}%\n")
        f.write(f"  Max error: {np.max(error_f1):.2f}\n\n")
        f.write("Image f2 (q4_input_2.png):\n")
        f.write(f"  PSNR: {psnr_f2:.2f} dB\n")
        f.write(f"  RMSE: {rmse_f2:.2f}\n")
        f.write(f"  Total coefficients: {total_coeffs_f2}\n")
        f.write(f"  Nonzero coefficients: {total_nonzero_f2}\n")
        f.write(f"  Percentage kept: {compression_ratio_f2*100:.2f}%\n")
        f.write(f"  Max error: {np.max(error_f2):.2f}\n")
    print("  Summary saved to: ./Q4_Output/PartB/summary.txt")

def part_c(f1, f2):
    print("\n" + "="*60)
    print("======Part-C processing...======")
    print("="*60 + "\n")
    target_psnr = 20  
    max_intensity = 255
    target_rmse = 0.1 * max_intensity  
    print(f"Target: PSNR >= {target_psnr} dB")
    print(f"Target: RMSE <= {target_rmse:.1f} for entire image\n")
    print("Processing f1 (image 1) - Full image DCT compression...")
    dct_full_f1 = apply_full_dct(f1)
    total_coeffs_f1 = f1.shape[0] * f1.shape[1]
    print(f"  Total DCT coefficients: {total_coeffs_f1}")
    thresholded_dct_f1, nonzero_f1 = threshold_full_coefficients(
        dct_full_f1, target_rmse, max_intensity
    )
    f1_hat_full = apply_inverse_full_dct(thresholded_dct_f1)
    psnr_f1 = calculate_psnr(f1, f1_hat_full, max_intensity)
    rmse_f1 = calculate_rmse(f1, f1_hat_full)
    compression_ratio_f1 = nonzero_f1 / total_coeffs_f1
    print(f"  PSNR: {psnr_f1:.2f} dB")
    print(f"  RMSE: {rmse_f1:.2f}")
    print(f"  Nonzero coefficients: {nonzero_f1}")
    print(f"  Percentage kept: {compression_ratio_f1*100:.2f}%")
    save_image(f1_hat_full, './Q4_Output/PartC/f1_compressed_full.png')
    error_f1 = create_error_image(f1, f1_hat_full, './Q4_Output/PartC/f1_error_full.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f1):.2f}")
    print("\nProcessing f2 (image 2) - Full image DCT compression...")
    dct_full_f2 = apply_full_dct(f2)
    total_coeffs_f2 = f2.shape[0] * f2.shape[1]
    print(f"  Total DCT coefficients: {total_coeffs_f2}")
    thresholded_dct_f2, nonzero_f2 = threshold_full_coefficients(
        dct_full_f2, target_rmse, max_intensity
    )
    f2_hat_full = apply_inverse_full_dct(thresholded_dct_f2)
    psnr_f2 = calculate_psnr(f2, f2_hat_full, max_intensity)
    rmse_f2 = calculate_rmse(f2, f2_hat_full)
    compression_ratio_f2 = nonzero_f2 / total_coeffs_f2
    print(f"  PSNR: {psnr_f2:.2f} dB")
    print(f"  RMSE: {rmse_f2:.2f}")
    print(f"  Nonzero coefficients: {nonzero_f2}")
    print(f"  Percentage kept: {compression_ratio_f2*100:.2f}%")
    save_image(f2_hat_full, './Q4_Output/PartC/f2_compressed_full.png')
    error_f2 = create_error_image(f2, f2_hat_full, './Q4_Output/PartC/f2_error_full.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f2):.2f}")
    print("\n  Part C completed")
    with open('./Q4_Output/PartC/summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("PART C: Full Image DCT Compression Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Target: PSNR >= {target_psnr} dB (RMSE <= {target_rmse:.1f})\n\n")
        f.write("Image f1 (q4_input_1.png):\n")
        f.write(f"  PSNR: {psnr_f1:.2f} dB\n")
        f.write(f"  RMSE: {rmse_f1:.2f}\n")
        f.write(f"  Total coefficients: {total_coeffs_f1}\n")
        f.write(f"  Nonzero coefficients: {nonzero_f1}\n")
        f.write(f"  Percentage kept: {compression_ratio_f1*100:.2f}%\n")
        f.write(f"  Max error: {np.max(error_f1):.2f}\n\n")
        f.write("Image f2 (q4_input_2.png):\n")
        f.write(f"  PSNR: {psnr_f2:.2f} dB\n")
        f.write(f"  RMSE: {rmse_f2:.2f}\n")
        f.write(f"  Total coefficients: {total_coeffs_f2}\n")
        f.write(f"  Nonzero coefficients: {nonzero_f2}\n")
        f.write(f"  Percentage kept: {compression_ratio_f2*100:.2f}%\n")
        f.write(f"  Max error: {np.max(error_f2):.2f}\n\n")
    print("  Summary saved to: ./Q4_Output/PartC/summary.txt")

def main():
	output_dirs = [
		'./Q4_Output',
		'./Q4_Output/PartA',
		'./Q4_Output/PartB',
		'./Q4_Output/PartC'
	]
	for dir_path in output_dirs:
		Path(dir_path).mkdir(parents=True, exist_ok=True)
	f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2 = part_a()
	part_b(f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2)
	part_c(f1, f2)

if __name__ == "__main__":
    main()
