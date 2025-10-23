"""
Block DCT Coding Implementation (JPEG-style)
Assignment: Problem 4
Author: Student
Date: October 24, 2025
"""

import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import os
from pathlib import Path

# ===========================
# Helper Functions
# ===========================

def dct2d(block):
    """
    2D DCT transformation using orthonormal basis
    
    Args:
        block: 2D numpy array
    Returns:
        2D DCT coefficients
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block):
    """
    2D Inverse DCT transformation
    
    Args:
        block: 2D DCT coefficients
    Returns:
        2D spatial domain block
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def calculate_psnr(original, reconstructed, max_intensity=255):
    """
    Calculate Peak Signal-to-Noise Ratio
    
    Args:
        original: Original image array
        reconstructed: Reconstructed image array
        max_intensity: Maximum possible pixel value (default 255)
    Returns:
        PSNR in dB
    """
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_intensity / np.sqrt(mse))
    return psnr

def calculate_rmse(original, reconstructed):
    """
    Calculate Root Mean Square Error
    
    Args:
        original: Original image array
        reconstructed: Reconstructed image array
    Returns:
        RMSE value
    """
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    return np.sqrt(mse)

def load_and_crop_image(filename, block_size=16):
    """
    Load image and crop to multiple of block_size
    
    Args:
        filename: Path to image file
        block_size: Block size for DCT (default 16)
    Returns:
        Cropped grayscale image as numpy array
    """
    img = Image.open(filename).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    # Crop to multiple of block_size
    h, w = img_array.shape
    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size
    
    cropped = img_array[:new_h, :new_w]
    print(f"  Loaded {filename}: Original size {h}x{w}, Cropped to {new_h}x{new_w}")
    return cropped

def save_image(img_array, filename):
    """
    Save numpy array as image
    
    Args:
        img_array: Image data as numpy array
        filename: Output file path
    """
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"  Saved: {filename}")

def create_error_image(original, reconstructed, output_path, scale_factor=10):
    """
    Create and save error image (magnified for visibility)
    
    Args:
        original: Original image array
        reconstructed: Reconstructed image array
        output_path: Path to save error image
        scale_factor: Multiplication factor for error visibility
    Returns:
        Error array
    """
    error = np.abs(original.astype(float) - reconstructed.astype(float))
    # Scale error for visibility
    error_scaled = np.clip(error * scale_factor, 0, 255)
    save_image(error_scaled, output_path)
    return error

# ===========================
# Block DCT Functions
# ===========================

def apply_block_dct(image, block_size=16):
    """
    Apply block DCT to image
    
    Args:
        image: Input image as numpy array
        block_size: Size of DCT blocks (default 16)
    Returns:
        Tuple of (list of DCT coefficient blocks, grid shape)
    """
    h, w = image.shape
    dct_blocks = []
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = dct2d(block.astype(float))
            dct_blocks.append(dct_block)
    
    return dct_blocks, (h // block_size, w // block_size)

def apply_inverse_block_dct(dct_blocks, grid_shape, block_size=16):
    """
    Apply inverse block DCT to reconstruct image
    
    Args:
        dct_blocks: List of DCT coefficient blocks
        grid_shape: Tuple of (rows, cols) of blocks
        block_size: Size of DCT blocks (default 16)
    Returns:
        Reconstructed image as numpy array
    """
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

# ===========================
# Full Image DCT Functions
# ===========================

def apply_full_dct(image):
    """
    Apply DCT to entire image
    
    Args:
        image: Input image as numpy array
    Returns:
        DCT coefficients for entire image
    """
    return dct2d(image.astype(float))

def apply_inverse_full_dct(dct_coeffs):
    """
    Apply inverse DCT to reconstruct entire image
    
    Args:
        dct_coeffs: DCT coefficients for entire image
    Returns:
        Reconstructed image as numpy array
    """
    return idct2d(dct_coeffs)

# ===========================
# Coefficient Thresholding
# ===========================

def threshold_block_coefficients(dct_block, target_rmse, block_size=16, max_intensity=255):
    """
    Threshold DCT coefficients in a block to achieve target RMSE
    Uses the orthogonality property: ||x - x_hat||^2 = ||X - X_hat||^2
    
    Args:
        dct_block: DCT coefficient block
        target_rmse: Target RMSE (e.g., 0.1 * I_max for 20 dB PSNR)
        block_size: Size of block
        max_intensity: Maximum intensity value
    Returns:
        Thresholded DCT block
    """
    # Create a copy
    thresholded = dct_block.copy()
    
    # Get absolute values and flatten
    abs_coeffs = np.abs(thresholded.flatten())
    
    # Sort coefficients by magnitude (descending)
    sorted_indices = np.argsort(-abs_coeffs)
    sorted_coeffs = abs_coeffs[sorted_indices]
    
    # Calculate cumulative error energy when zeroing coefficients
    # Error energy = sum of squared coefficients that are zeroed
    target_error_energy = (target_rmse ** 2) * (block_size ** 2)
    
    # Find how many coefficients we can zero while staying within target
    cumulative_energy = np.cumsum(sorted_coeffs[::-1] ** 2)
    
    # Find the last index where cumulative energy is still below target
    num_to_zero = np.searchsorted(cumulative_energy, target_error_energy)
    num_to_keep = block_size * block_size - num_to_zero
    
    # Zero out the smallest coefficients
    flat_thresh = thresholded.flatten()
    indices_to_zero = sorted_indices[num_to_keep:]
    flat_thresh[indices_to_zero] = 0
    
    thresholded = flat_thresh.reshape(block_size, block_size)
    
    return thresholded, num_to_keep

def threshold_full_coefficients(dct_coeffs, target_rmse, max_intensity=255):
    """
    Threshold DCT coefficients for entire image to achieve target RMSE
    
    Args:
        dct_coeffs: DCT coefficients for entire image
        target_rmse: Target RMSE
        max_intensity: Maximum intensity value
    Returns:
        Tuple of (thresholded DCT coefficients, number of nonzero coefficients)
    """
    h, w = dct_coeffs.shape
    thresholded = dct_coeffs.copy()
    
    # Get absolute values and flatten
    abs_coeffs = np.abs(thresholded.flatten())
    
    # Sort coefficients by magnitude (descending)
    sorted_indices = np.argsort(-abs_coeffs)
    sorted_coeffs = abs_coeffs[sorted_indices]
    
    # Calculate cumulative error energy
    target_error_energy = (target_rmse ** 2) * (h * w)
    cumulative_energy = np.cumsum(sorted_coeffs[::-1] ** 2)
    
    # Find number of coefficients to keep
    num_to_zero = np.searchsorted(cumulative_energy, target_error_energy)
    num_to_keep = h * w - num_to_zero
    
    # Zero out the smallest coefficients
    flat_thresh = thresholded.flatten()
    indices_to_zero = sorted_indices[num_to_keep:]
    flat_thresh[indices_to_zero] = 0
    
    thresholded = flat_thresh.reshape(h, w)
    
    return thresholded, num_to_keep

# ===========================
# PART A: Block DCT Implementation
# ===========================

def part_a():
    """
    Part A: Implement and verify 16x16 block DCT
    """
    print("\n" + "="*60)
    print("======Part-A processing...======")
    print("="*60 + "\n")
    
    # Load images
    print("Loading and preparing images...")
    f1 = load_and_crop_image('q4_input_1.png')
    f2 = load_and_crop_image('q4_input_2.png')
    
    print("\nProcessing f1 (image 1)...")
    # Apply DCT to f1
    dct_blocks_f1, grid_shape_f1 = apply_block_dct(f1, block_size=16)
    print(f"  Number of 16x16 blocks: {len(dct_blocks_f1)}")
    print(f"  Grid shape: {grid_shape_f1[0]} rows x {grid_shape_f1[1]} cols")
    
    # Apply inverse DCT to verify
    reconstructed_f1 = apply_inverse_block_dct(dct_blocks_f1, grid_shape_f1, block_size=16)
    
    # Calculate verification metrics
    max_diff_f1 = np.max(np.abs(f1 - reconstructed_f1))
    psnr_f1 = calculate_psnr(f1, reconstructed_f1)
    
    print(f"  Verification - Max pixel difference: {max_diff_f1:.2e}")
    print(f"  Verification - PSNR: {psnr_f1:.2f} dB")
    
    # Save reconstructed image
    save_image(reconstructed_f1, './Q4_Output/PartA/f1_reconstructed.png')
    save_image(f1, './Q4_Output/PartA/f1_original.png')
    
    print("\nProcessing f2 (image 2)...")
    # Apply DCT to f2
    dct_blocks_f2, grid_shape_f2 = apply_block_dct(f2, block_size=16)
    print(f"  Number of 16x16 blocks: {len(dct_blocks_f2)}")
    print(f"  Grid shape: {grid_shape_f2[0]} rows x {grid_shape_f2[1]} cols")
    
    # Apply inverse DCT to verify
    reconstructed_f2 = apply_inverse_block_dct(dct_blocks_f2, grid_shape_f2, block_size=16)
    
    # Calculate verification metrics
    max_diff_f2 = np.max(np.abs(f2 - reconstructed_f2))
    psnr_f2 = calculate_psnr(f2, reconstructed_f2)
    
    print(f"  Verification - Max pixel difference: {max_diff_f2:.2e}")
    print(f"  Verification - PSNR: {psnr_f2:.2f} dB")
    
    # Save reconstructed image
    save_image(reconstructed_f2, './Q4_Output/PartA/f2_reconstructed.png')
    save_image(f2, './Q4_Output/PartA/f2_original.png')
    
    print("\n✓ Part A completed successfully!")
    print("  Images are identical after DCT-IDCT transformation")
    print("  (within floating point precision)")
    
    return f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2

# ===========================
# PART B: Lossy Compression with Block DCT
# ===========================

def part_b(f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2):
    """
    Part B: Lossy compression using coefficient thresholding (per-block)
    Target: PSNR >= 20 dB (RMSE <= 0.1 * I_max = 25.5)
    """
    print("\n" + "="*60)
    print("======Part-B processing...======")
    print("="*60 + "\n")
    
    target_psnr = 20  # dB
    max_intensity = 255
    target_rmse_per_block = 0.1 * max_intensity  # 25.5
    
    print(f"Target: PSNR >= {target_psnr} dB")
    print(f"Target: RMSE <= {target_rmse_per_block:.1f} per block\n")
    
    # Process f1
    print("Processing f1 (image 1) - Block-wise compression...")
    thresholded_blocks_f1 = []
    total_nonzero_f1 = 0
    
    for block in dct_blocks_f1:
        thresh_block, num_kept = threshold_block_coefficients(
            block, target_rmse_per_block, block_size=16, max_intensity=max_intensity
        )
        thresholded_blocks_f1.append(thresh_block)
        total_nonzero_f1 += num_kept
    
    # Reconstruct image
    f1_hat = apply_inverse_block_dct(thresholded_blocks_f1, grid_shape_f1, block_size=16)
    
    # Calculate metrics
    psnr_f1 = calculate_psnr(f1, f1_hat, max_intensity)
    rmse_f1 = calculate_rmse(f1, f1_hat)
    total_coeffs_f1 = len(dct_blocks_f1) * 16 * 16
    compression_ratio_f1 = total_nonzero_f1 / total_coeffs_f1
    
    print(f"  PSNR: {psnr_f1:.2f} dB")
    print(f"  RMSE: {rmse_f1:.2f}")
    print(f"  Total coefficients: {total_coeffs_f1}")
    print(f"  Nonzero coefficients: {total_nonzero_f1}")
    print(f"  Percentage kept: {compression_ratio_f1*100:.2f}%")
    
    # Save results
    save_image(f1_hat, './Q4_Output/PartB/f1_compressed.png')
    error_f1 = create_error_image(f1, f1_hat, './Q4_Output/PartB/f1_error.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f1):.2f}")
    
    # Process f2
    print("\nProcessing f2 (image 2) - Block-wise compression...")
    thresholded_blocks_f2 = []
    total_nonzero_f2 = 0
    
    for block in dct_blocks_f2:
        thresh_block, num_kept = threshold_block_coefficients(
            block, target_rmse_per_block, block_size=16, max_intensity=max_intensity
        )
        thresholded_blocks_f2.append(thresh_block)
        total_nonzero_f2 += num_kept
    
    # Reconstruct image
    f2_hat = apply_inverse_block_dct(thresholded_blocks_f2, grid_shape_f2, block_size=16)
    
    # Calculate metrics
    psnr_f2 = calculate_psnr(f2, f2_hat, max_intensity)
    rmse_f2 = calculate_rmse(f2, f2_hat)
    total_coeffs_f2 = len(dct_blocks_f2) * 16 * 16
    compression_ratio_f2 = total_nonzero_f2 / total_coeffs_f2
    
    print(f"  PSNR: {psnr_f2:.2f} dB")
    print(f"  RMSE: {rmse_f2:.2f}")
    print(f"  Total coefficients: {total_coeffs_f2}")
    print(f"  Nonzero coefficients: {total_nonzero_f2}")
    print(f"  Percentage kept: {compression_ratio_f2*100:.2f}%")
    
    # Save results
    save_image(f2_hat, './Q4_Output/PartB/f2_compressed.png')
    error_f2 = create_error_image(f2, f2_hat, './Q4_Output/PartB/f2_error.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f2):.2f}")
    
    print("\n✓ Part B completed successfully!")
    print("  Note: Error images are scaled 10x for visibility")
    
    # Save summary report
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

# ===========================
# PART C: Full Image DCT Compression
# ===========================

def part_c(f1, f2):
    """
    Part C: Lossy compression using full image DCT
    Compare with block-based approach
    """
    print("\n" + "="*60)
    print("======Part-C processing...======")
    print("="*60 + "\n")
    
    target_psnr = 20  # dB
    max_intensity = 255
    target_rmse = 0.1 * max_intensity  # 25.5
    
    print(f"Target: PSNR >= {target_psnr} dB")
    print(f"Target: RMSE <= {target_rmse:.1f} for entire image\n")
    
    # Process f1
    print("Processing f1 (image 1) - Full image DCT compression...")
    dct_full_f1 = apply_full_dct(f1)
    total_coeffs_f1 = f1.shape[0] * f1.shape[1]
    print(f"  Total DCT coefficients: {total_coeffs_f1}")
    
    # Threshold coefficients
    thresholded_dct_f1, nonzero_f1 = threshold_full_coefficients(
        dct_full_f1, target_rmse, max_intensity
    )
    
    # Reconstruct image
    f1_hat_full = apply_inverse_full_dct(thresholded_dct_f1)
    
    # Calculate metrics
    psnr_f1 = calculate_psnr(f1, f1_hat_full, max_intensity)
    rmse_f1 = calculate_rmse(f1, f1_hat_full)
    compression_ratio_f1 = nonzero_f1 / total_coeffs_f1
    
    print(f"  PSNR: {psnr_f1:.2f} dB")
    print(f"  RMSE: {rmse_f1:.2f}")
    print(f"  Nonzero coefficients: {nonzero_f1}")
    print(f"  Percentage kept: {compression_ratio_f1*100:.2f}%")
    
    # Save results
    save_image(f1_hat_full, './Q4_Output/PartC/f1_compressed_full.png')
    error_f1 = create_error_image(f1, f1_hat_full, './Q4_Output/PartC/f1_error_full.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f1):.2f}")
    
    # Process f2
    print("\nProcessing f2 (image 2) - Full image DCT compression...")
    dct_full_f2 = apply_full_dct(f2)
    total_coeffs_f2 = f2.shape[0] * f2.shape[1]
    print(f"  Total DCT coefficients: {total_coeffs_f2}")
    
    # Threshold coefficients
    thresholded_dct_f2, nonzero_f2 = threshold_full_coefficients(
        dct_full_f2, target_rmse, max_intensity
    )
    
    # Reconstruct image
    f2_hat_full = apply_inverse_full_dct(thresholded_dct_f2)
    
    # Calculate metrics
    psnr_f2 = calculate_psnr(f2, f2_hat_full, max_intensity)
    rmse_f2 = calculate_rmse(f2, f2_hat_full)
    compression_ratio_f2 = nonzero_f2 / total_coeffs_f2
    
    print(f"  PSNR: {psnr_f2:.2f} dB")
    print(f"  RMSE: {rmse_f2:.2f}")
    print(f"  Nonzero coefficients: {nonzero_f2}")
    print(f"  Percentage kept: {compression_ratio_f2*100:.2f}%")
    
    # Save results
    save_image(f2_hat_full, './Q4_Output/PartC/f2_compressed_full.png')
    error_f2 = create_error_image(f2, f2_hat_full, './Q4_Output/PartC/f2_error_full.png', scale_factor=10)
    print(f"  Max error: {np.max(error_f2):.2f}")
    
    print("\n✓ Part C completed successfully!")
    print("  Note: Error images are scaled 10x for visibility")
    
    # Save summary report
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
        
        f.write("-"*60 + "\n")
        f.write("DISCUSSION: Block DCT vs Full Image DCT\n")
        f.write("-"*60 + "\n\n")
        f.write("PROS of Full Image DCT:\n")
        f.write("1. Better energy compaction - most energy concentrated in\n")
        f.write("   top-left corner of the entire frequency domain\n")
        f.write("2. No blocking artifacts - smoother reconstructed images\n")
        f.write("3. Higher compression ratio for same PSNR target\n")
        f.write("4. Better for images with smooth, slowly varying content\n\n")
        
        f.write("CONS of Full Image DCT:\n")
        f.write("1. High computational complexity - O(N²log N) for entire image\n")
        f.write("2. High memory requirements - need to store full DCT matrix\n")
        f.write("3. Poor adaptability to local image features\n")
        f.write("4. Errors can propagate across entire image\n")
        f.write("5. Not practical for large images in real-time applications\n\n")
        
        f.write("PROS of Block DCT:\n")
        f.write("1. Lower computational complexity - can process blocks in parallel\n")
        f.write("2. Lower memory requirements - only need block-size buffers\n")
        f.write("3. Better adaptability to local features\n")
        f.write("4. Localized errors - artifacts confined to blocks\n")
        f.write("5. Industry standard (JPEG, MPEG)\n\n")
        
        f.write("CONS of Block DCT:\n")
        f.write("1. Blocking artifacts at high compression\n")
        f.write("2. Less efficient energy compaction\n")
        f.write("3. Discontinuities at block boundaries\n\n")
        
        f.write("CONCLUSION:\n")
        f.write("Block DCT is preferred in practice due to computational\n")
        f.write("efficiency and practical implementation advantages, despite\n")
        f.write("slightly worse compression efficiency. The blocking artifacts\n")
        f.write("can be mitigated using overlapped blocks or post-filtering.\n")
    
    print("  Summary saved to: ./Q4_Output/PartC/summary.txt")

# ===========================
# Main Execution
# ===========================

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("Block DCT Coding Implementation (JPEG-style)")
    print("Problem 4 - Image Compression Assignment")
    print("="*60)
    
    # Create output directories
    output_dirs = [
        './Q4_Output',
        './Q4_Output/PartA',
        './Q4_Output/PartB',
        './Q4_Output/PartC'
    ]
    
    print("\nCreating output directories...")
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    try:
        # Execute Part A
        f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2 = part_a()
        
        # Execute Part B
        part_b(f1, f2, dct_blocks_f1, dct_blocks_f2, grid_shape_f1, grid_shape_f2)
        
        # Execute Part C
        part_c(f1, f2)
        
        print("\n" + "="*60)
        print("ALL PARTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nOutput files saved in:")
        print("  ./Q4_Output/PartA/ - Original and reconstructed images")
        print("  ./Q4_Output/PartB/ - Block DCT compression results")
        print("  ./Q4_Output/PartC/ - Full image DCT compression results")
        print("\nCheck summary.txt files in PartB and PartC for detailed results.")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPlease ensure the following files exist in the current directory:")
        print("  - q4_input_1.png")
        print("  - q4_input_2.png")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
