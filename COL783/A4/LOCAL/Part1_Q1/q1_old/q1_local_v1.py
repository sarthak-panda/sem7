#!/usr/bin/env python3
"""
Morphological Operations for Character Detection - UPDATED VERSION
Assignment: Q1 - Binary Morphology and Hit-or-Miss Transform

Improvements:
1. Custom erosion and dilation implementations (from scratch, no library functions)
2. Otsu's automatic thresholding instead of manual threshold
3. Improved B2 computation using exterior border method: B2 = (B ⊕ se) \ B
"""

import numpy as np
from PIL import Image
import os

# ============================================================================
# CUSTOM BINARY MORPHOLOGICAL OPERATIONS (FROM SCRATCH)
# ============================================================================

def custom_erosion(image, se):
    """
    Binary erosion with user-specified structuring element.
    Implemented from scratch WITHOUT using scipy/library functions.
    
    Erosion: output pixel is 1 only if ALL pixels under SE are 1 in input
    Formula: A ⊖ B = {z | B_z ⊆ A}
    
    Args:
        image: Binary image (numpy array with 0s and 1s)
        se: Structuring element (numpy array with 0s and 1s)
    
    Returns:
        Eroded binary image
    """
    image = image.astype(np.uint8)
    se = se.astype(np.uint8)
    output = np.zeros_like(image)
    
    # Get SE dimensions and center
    se_height, se_width = se.shape
    center_y = int(se_height // 2)
    center_x = int(se_width // 2)
    
    # Pad image to handle boundaries
    pad_y, pad_x = int(center_y), int(center_x)
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), 
                    mode='constant', constant_values=0)
    
    # Apply erosion: slide SE over image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Extract neighborhood
            neighborhood = padded[y:y+se_height, x:x+se_width]
            # Erosion: check if SE pattern is fully contained in neighborhood
            masked = neighborhood * se
            if np.all(masked == se):
                output[y, x] = 1
    
    return output


def custom_dilation(image, se):
    """
    Binary dilation with user-specified structuring element.
    Implemented from scratch WITHOUT using scipy/library functions.
    
    Dilation: output pixel is 1 if ANY pixel under SE is 1 in input
    Formula: A ⊕ B = {z | (B_z) ∩ A ≠ ∅}
    
    Args:
        image: Binary image (numpy array with 0s and 1s)
        se: Structuring element (numpy array with 0s and 1s)
    
    Returns:
        Dilated binary image
    """
    image = image.astype(np.uint8)
    se = se.astype(np.uint8)
    output = np.zeros_like(image)
    
    # Get SE dimensions and center
    se_height, se_width = se.shape
    center_y = int(se_height // 2)
    center_x = int(se_width // 2)
    
    # Pad image to handle boundaries
    pad_y, pad_x = int(center_y), int(center_x)
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), 
                    mode='constant', constant_values=0)
    
    # Apply dilation: slide SE over image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Extract neighborhood
            neighborhood = padded[y:y+se_height, x:x+se_width]
            # Dilation: check if SE has any overlap with neighborhood
            masked = neighborhood * se
            if np.any(masked > 0):
                output[y, x] = 1
    
    return output


def custom_opening(image, se):
    """
    Opening: erosion followed by dilation.
    Formula: A ∘ B = (A ⊖ B) ⊕ B
    
    Opens small holes while preserving overall structure.
    
    Args:
        image: Binary image
        se: Structuring element
    
    Returns:
        Opened binary image
    """
    eroded = custom_erosion(image, se)
    opened = custom_dilation(eroded, se)
    return opened


def custom_closing(image, se):
    """
    Closing: dilation followed by erosion.
    Formula: A • B = (A ⊕ B) ⊖ B
    
    Closes small holes while preserving overall structure.
    
    Args:
        image: Binary image
        se: Structuring element
    
    Returns:
        Closed binary image
    """
    dilated = custom_dilation(image, se)
    closed = custom_erosion(dilated, se)
    return closed


def create_disk_se(radius):
    """
    Create a disk-shaped structuring element.
    All points within Euclidean distance <= radius are included.
    
    Args:
        radius: Radius of the disk
    
    Returns:
        Binary disk-shaped structuring element
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x**2 + y**2 <= radius**2
    return disk.astype(np.uint8)


def hit_or_miss(image, se1, se2):
    """
    Hit-or-Miss Transform.
    
    Finds locations where se1 matches foreground and se2 matches background.
    Formula: A ⊗ (B1, B2) = (A ⊖ B1) ∩ (A^c ⊖ B2)
    
    Args:
        image: Binary image
        se1: Structuring element to match foreground
        se2: Structuring element to match background
    
    Returns:
        Binary image showing detected patterns
    """
    # Erosion with se1 on image
    eroded_fg = custom_erosion(image, se1)
    # Erosion with se2 on complement of image
    eroded_bg = custom_erosion(1 - image, se2)
    # Intersection
    result = eroded_fg * eroded_bg
    return result


def compute_exterior_border(image, se):
    """
    Compute exterior border of an object using dilation.
    Formula: border = (image ⊕ se) - image
    
    This extracts only the boundary pixels added by dilation.
    
    Args:
        image: Binary image
        se: Structuring element
    
    Returns:
        Binary image showing only the exterior border
    """
    # Dilate the image
    dilated = custom_dilation(image, se)
    
    # Exterior border = dilated - original
    exterior = dilated - image
    exterior = np.clip(exterior, 0, 1).astype(np.uint8)
    
    return exterior


def compute_B2_improved(B):
    """
    Compute B2 as the exterior border of B.
    
    B2 = (B ⊕ se) \ B
    
    This captures the background pattern immediately outside B,
    which helps distinguish 'c' from 'e' and 'o' in hit-or-miss transform.
    
    Args:
        B: Structuring element representing letter 'c'
    
    Returns:
        B2: Background structuring element (exterior border)
    """
    # Create a small SE for dilation to get the border
    border_se = np.ones((3, 3), dtype=np.uint8)
    
    # Compute exterior border
    B2 = compute_exterior_border(B, border_se)
    
    return B2


# ============================================================================
# OTSU'S AUTOMATIC THRESHOLDING
# ============================================================================

def otsu_thresholding(image):
    """
    Otsu's automatic thresholding method.
    
    Finds the optimal threshold that maximizes between-class variance,
    which minimizes intra-class variance.
    
    Args:
        image: Grayscale image (numpy array)
    
    Returns:
        binary_image: Binary image using optimal threshold (preserves shape)
        threshold_value: The optimal threshold computed
    """
    original_shape = image.shape
    
    # Flatten image and normalize to 0-255
    pixel_values = image.flatten()
    if pixel_values.max() == pixel_values.min():
        return np.ones(original_shape, dtype=np.uint8), 128
    
    pixel_values = (pixel_values - pixel_values.min()) / \
                   (pixel_values.max() - pixel_values.min()) * 255
    pixel_values = pixel_values.astype(np.uint8)
    
    # Calculate histogram
    histogram = np.bincount(pixel_values, minlength=256)
    histogram = histogram / histogram.sum()  # Normalize
    
    # Calculate cumulative sum and mean
    cs = np.cumsum(histogram)
    mean = np.cumsum(histogram * np.arange(256))
    
    # Total mean
    total_mean = mean[-1]
    
    # Calculate between-class variance for each threshold
    max_variance = 0
    optimal_threshold = 0
    
    for t in range(256):
        w0 = cs[t]  # Weight of background
        w1 = 1 - w0  # Weight of foreground
        
        # Skip if either class is empty
        if w0 == 0 or w1 == 0:
            continue
        
        mu0 = mean[t] / w0  # Mean of background
        mu1 = (total_mean - mean[t]) / w1  # Mean of foreground
        
        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t
    
    # Apply threshold and reshape back to original shape
    binary_image = (pixel_values < optimal_threshold).astype(np.uint8)
    binary_image = binary_image.reshape(original_shape)
    
    return binary_image, optimal_threshold


# ============================================================================
# VISUALIZATION AND UTILITY FUNCTIONS
# ============================================================================

def create_rgb_overlay(original, detected):
    """
    Create RGB image for visualization.
    
    Convention:
    - Red component: original image A (all text)
    - Green and Blue components: detected letters
    
    Result: detected letters appear WHITE, undetected letters appear RED
    
    Args:
        original: Original binary image
        detected: Detected regions binary image
    
    Returns:
        RGB image with overlay
    """
    rgb = np.zeros((*original.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = original * 255  # Red channel
    rgb[:, :, 1] = detected * 255  # Green channel
    rgb[:, :, 2] = detected * 255  # Blue channel
    return rgb


def save_image(image, path):
    """
    Save image using PIL.
    
    Args:
        image: Numpy array (2D or 3D)
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if len(image.shape) == 2:
        # Binary or grayscale
        img = Image.fromarray((image * 255).astype(np.uint8))
    else:
        # RGB
        img = Image.fromarray(image.astype(np.uint8))
    img.save(path)
    print(f"Saved: {path}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main function to execute all parts of the assignment."""
    
    # Create output directories
    os.makedirs('./Q1_Output', exist_ok=True)
    os.makedirs('./Q1_Output/PartA', exist_ok=True)
    os.makedirs('./Q1_Output/PartB', exist_ok=True)
    os.makedirs('./Q1_Output/PartC', exist_ok=True)
    os.makedirs('./Q1_Output/PartD', exist_ok=True)
    
    # ========================================================================
    # PART A: Dilation and Closing with Disk Structuring Element
    # ========================================================================
    
    print("=" * 70)
    print("======Part-A processing...======")
    print("=" * 70)
    
    # Load the input text image
    text_img = Image.open('../Testcases/q1_text.jpg').convert('L')
    text_array = np.array(text_img)
    print(f"Loaded text image: shape {text_array.shape}")
    
    # Use Otsu thresholding instead of manual threshold
    print("Applying Otsu's automatic thresholding...")
    A, otsu_threshold = otsu_thresholding(text_array)
    print(f"Binary image A created: {A.shape}, white pixels: {np.sum(A)}")
    print(f"Optimal threshold found: {otsu_threshold}")
    save_image(A, './Q1_Output/PartA/binary_image_A.png')
    
    # Create disk-shaped structuring element (moderately sized)
    disk_radius = 3
    disk_se = create_disk_se(disk_radius)
    print(f"Created disk SE with radius {disk_radius}: shape {disk_se.shape}")
    save_image(disk_se, './Q1_Output/PartA/disk_se.png')
    
    # Perform dilation
    print("Performing dilation with custom implementation...")
    A_dilated = custom_dilation(A, disk_se)
    save_image(A_dilated, './Q1_Output/PartA/A_dilated.png')
    print(f"Dilation complete: white pixels = {np.sum(A_dilated)}")
    
    # Perform closing
    print("Performing closing with custom implementation...")
    A_closed = custom_closing(A, disk_se)
    save_image(A_closed, './Q1_Output/PartA/A_closed.png')
    print(f"Closing complete: white pixels = {np.sum(A_closed)}")
    
    print("\nPart A completed successfully!")
    
    # ========================================================================
    # PART B: Opening with Letter 'c' as Structuring Element
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("======Part-B processing...======")
    print("=" * 70)
    
    # Load the letter 'c' image
    letter_c_img = Image.open('../Testcases/q1_c_letter.jpg').convert('L')
    letter_c_array = np.array(letter_c_img)
    print(f"Loaded letter 'c' image: shape {letter_c_array.shape}")
    
    # Use Otsu thresholding for letter C
    print("Applying Otsu's automatic thresholding to letter 'c'...")
    B, otsu_threshold_c = otsu_thresholding(letter_c_array)
    print(f"Binary SE B created: {B.shape}, foreground pixels: {np.sum(B)}")
    print(f"Optimal threshold found: {otsu_threshold_c}")
    save_image(B, './Q1_Output/PartB/SE_B_original.png')
    
    # Perform opening with B
    print("Performing opening operation with B on A...")
    A_opened_B = custom_opening(A, B)
    save_image(A_opened_B, './Q1_Output/PartB/A_opened_with_B.png')
    print(f"Opening complete: detected pixels = {np.sum(A_opened_B)}")
    
    # Create RGB overlay for visualization
    rgb_result_B = create_rgb_overlay(A, A_opened_B)
    save_image(rgb_result_B, './Q1_Output/PartB/result_overlay_B.png')
    print("RGB overlay created (detected = white, undetected = red)")
    
    print("\nPart B completed successfully!")
    print("Note: Not all c's are found because:")
    print("  - The SE B must match exactly (pixel-by-pixel)")
    print("  - Small variations in font rendering, size, position cause mismatches")
    print("  - Opening requires perfect alignment and identical shape/size")
    
    # ========================================================================
    # PART C: Modified SE to Find All c's
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("======Part-C processing...======")
    print("=" * 70)
    
    print("Creating modified SE B1 by eroding B...")
    
    # Erode B to make it smaller and more flexible
    small_se = np.ones((2, 2), dtype=np.uint8)
    B1 = custom_erosion(B, small_se)
    
    # Fallback strategies if erosion is too aggressive
    if np.sum(B1) < 3:
        print("Erosion too aggressive, using alternative approach...")
        cross_se = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        B1 = custom_erosion(B, cross_se)
    
    if np.sum(B1) < 3:
        print("Using minimal erosion approach...")
        B1 = B.copy()
        B1[0, :] = 0
        B1[-1, :] = 0
        B1[:, 0] = 0
        B1[:, -1] = 0
    
    print(f"Modified SE B1 created: {B1.shape}, foreground pixels: {np.sum(B1)}")
    save_image(B1, './Q1_Output/PartC/SE_B1_modified.png')
    
    # Perform opening with B1
    print("Performing opening operation with B1 on A...")
    A_opened_B1 = custom_opening(A, B1)
    save_image(A_opened_B1, './Q1_Output/PartC/A_opened_with_B1.png')
    print(f"Opening complete: detected pixels = {np.sum(A_opened_B1)}")
    
    # Create RGB overlay
    rgb_result_C = create_rgb_overlay(A, A_opened_B1)
    save_image(rgb_result_C, './Q1_Output/PartC/result_overlay_C.png')
    print("RGB overlay created")
    
    print("\nPart C completed successfully!")
    print("Explanation:")
    print("  - B1 was obtained by eroding B with a small structuring element")
    print("  - This makes B1 smaller and more flexible")
    print("  - Allows matching variations in c's throughout the image")
    print("  - Also detects supersets of c (like 'e', 'o')")
    
    # ========================================================================
    # PART D: Hit-or-Miss Transform to Find Only c's
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("======Part-D processing...======")
    print("=" * 70)
    
    print("Creating SE B2 using exterior border method...")
    print("B2 = (B dilation SE) - B")
    
    # Use improved B2 computation: exterior border method
    B2 = compute_B2_improved(B)
    print(f"SE B2 created: {B2.shape}, background pixels: {np.sum(B2)}")
    save_image(B2, './Q1_Output/PartD/SE_B2_background.png')
    
    # Perform hit-or-miss transform
    print("Performing hit-or-miss transform with B1 and B2...")
    print("Pattern: (A erosion B1) intersection (complement(A) erosion B2)")
    A_hit_miss = hit_or_miss(A, B1, B2)
    save_image(A_hit_miss, './Q1_Output/PartD/A_hit_miss_result.png')
    print(f"Hit-or-miss complete: detected points = {np.sum(A_hit_miss)}")
    
    # Dilate the result to show full letters for visualization
    A_detected_letters = custom_dilation(A_hit_miss, B)
    
    # Create RGB overlays
    rgb_result_D = create_rgb_overlay(A, A_detected_letters)
    save_image(rgb_result_D, './Q1_Output/PartD/result_overlay_D.png')
    
    rgb_points_D = create_rgb_overlay(A, A_hit_miss)
    save_image(rgb_points_D, './Q1_Output/PartD/result_points_overlay_D.png')
    print("RGB overlays created")
    
    print("\nPart D completed successfully!")
    print("Explanation:")
    print("  - B2 was obtained as exterior border of B using (B dilation) - B")
    print("  - B1 matches the interior structure of 'c'")
    print("  - B2 matches the background in the opening region")
    print("  - Hit-or-miss finds locations where:")
    print("    * B1 matches the foreground (letter interior)")
    print("    * B2 matches the background (opening region)")
    print("  - This distinguishes 'c' from 'e' (middle bar) and 'o' (closed)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("ALL PARTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nAll results saved in ./Q1_Output/ subdirectories")
    print("Check PartA, PartB, PartC, and PartD folders for outputs")
    print("\nKey Improvements:")
    print("  1. Custom erosion/dilation from scratch (no library functions)")
    print("  2. Otsu's automatic thresholding instead of manual threshold")
    print("  3. B2 using exterior border method: B2 = (B ⊕ se) \\ B")
    print("=" * 70)


if __name__ == "__main__":
    main()
