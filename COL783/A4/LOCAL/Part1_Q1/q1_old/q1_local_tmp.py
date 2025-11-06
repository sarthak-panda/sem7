#!/usr/bin/env python3
"""
Morphological Operations for Character Detection
Assignment: Q1 - Binary Morphology and Hit-or-Miss Transform
"""

import numpy as np
from PIL import Image
import os
# from scipy.ndimage import binary_erosion, binary_dilation
#
# # ============================================================================
# # MORPHOLOGICAL OPERATIONS MODULE
# # ============================================================================
#
# def custom_erosion(image, se):
#     """
#     Binary erosion with user-specified structuring element.
#
#     Args:
#         image: Binary image (numpy array with 0s and 1s)
#         se: Structuring element (numpy array with 0s and 1s)
#
#     Returns:
#         Eroded binary image
#     """
#     return binary_erosion(image, structure=se).astype(np.uint8)
#
#
# def custom_dilation(image, se):
#     """
#     Binary dilation with user-specified structuring element.
#
#     Args:
#         image: Binary image (numpy array with 0s and 1s)
#         se: Structuring element (numpy array with 0s and 1s)
#
#     Returns:
#         Dilated binary image
#     """
#     return binary_dilation(image, structure=se).astype(np.uint8)
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
    """
    Opening: erosion followed by dilation.
    
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
    Hit-or-miss transform.
    
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


def create_rgb_overlay(original, detected):
    """
    Create RGB image where:
    - Red component: original image A
    - Green and Blue components: detected letters
    Result: detected letters are white, undetected are red
    
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
    
    print("=" * 50)
    print("======Part-A processing...======")
    print("=" * 50)
    
    # Load the input text image
    text_img = Image.open('../Testcases/q1_text.jpg').convert('L')
    text_array = np.array(text_img)
    print(f"Loaded text image: shape {text_array.shape}")
    
    # Threshold and negate to get binary image A
    threshold = 128
    A = (text_array < threshold).astype(np.uint8)
    
    print(f"Binary image A created: {A.shape}, white pixels: {np.sum(A)}")
    save_image(A, './Q1_Output/PartA/binary_image_A.png')
    
    # Create disk-shaped structuring element (moderately sized)
    disk_radius = 3
    disk_se = create_disk_se(disk_radius)
    print(f"Created disk SE with radius {disk_radius}: shape {disk_se.shape}")
    save_image(disk_se, './Q1_Output/PartA/disk_se.png')
    
    # Perform dilation
    print("Performing dilation...")
    A_dilated = custom_dilation(A, disk_se)
    save_image(A_dilated, './Q1_Output/PartA/A_dilated.png')
    print(f"Dilation complete: white pixels = {np.sum(A_dilated)}")
    
    # Perform closing
    print("Performing closing...")
    A_closed = custom_closing(A, disk_se)
    save_image(A_closed, './Q1_Output/PartA/A_closed.png')
    print(f"Closing complete: white pixels = {np.sum(A_closed)}")
    
    print("\nPart A completed successfully!")
    
    # ========================================================================
    # PART B: Opening with Letter 'c' as Structuring Element
    # ========================================================================
    
    print("\n" + "=" * 50)
    print("======Part-B processing...======")
    print("=" * 50)
    
    # Load the letter 'c' image
    letter_c_img = Image.open('../Testcases/q1_c_letter.jpg').convert('L')
    letter_c_array = np.array(letter_c_img)
    print(f"Loaded letter 'c' image: shape {letter_c_array.shape}")
    
    # Threshold to get binary structuring element B
    B = (letter_c_array < 128).astype(np.uint8)
    print(f"Binary SE B created: {B.shape}, foreground pixels: {np.sum(B)}")
    save_image(B, './Q1_Output/PartB/SE_B_original.png')
    
    # Perform opening with B
    print("Performing opening operation with B on A...")
    A_opened_B = custom_opening(A, B)
    save_image(A_opened_B, './Q1_Output/PartB/A_opened_with_B.png')
    print(f"Opening complete: detected pixels = {np.sum(A_opened_B)}")
    
    # Create RGB overlay for visualization
    rgb_result_B = create_rgb_overlay(A, A_opened_B)
    save_image(rgb_result_B, './Q1_Output/PartB/result_overlay_B.png')
    print("RGB overlay created")
    
    print("\nPart B completed successfully!")
    print("Note: Not all c's are found because the SE must match exactly")
    
    # ========================================================================
    # PART C: Modified SE to Find All c's
    # ========================================================================
    
    print("\n" + "=" * 50)
    print("======Part-C processing...======")
    print("=" * 50)
    
    print("Creating modified SE B1 by eroding B...")
    
    # Erode B to make it smaller and more flexible
    small_se = np.ones((2, 2), dtype=np.uint8)
    B1 = custom_erosion(B, small_se)
    
    # Fallback strategies if erosion is too aggressive
    if np.sum(B1) < 3:
        print("Using alternative erosion approach...")
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
    print("B1 was obtained by eroding B to make it more flexible")
    
    # ========================================================================
    # PART D: Hit-or-Miss Transform to Find Only c's
    # ========================================================================
    
    print("\n" + "=" * 50)
    print("======Part-D processing...======")
    print("=" * 50)
    
    print("Creating SE B2 for background matching...")
    
    # Create B2 to capture the opening region of 'c'
    dilate_se = np.ones((3, 3), dtype=np.uint8)
    B_dilated = custom_dilation(B, dilate_se)
    
    # B2 should be in the "opening" region of 'c'
    B2 = B_dilated.copy()
    B2[B > 0] = 0
    
    # Focus on right side opening
    width_B = B2.shape[1]
    height_B = B2.shape[0]
    B2[:, :width_B//2] = 0
    B2[:height_B//3, :] = 0
    B2[2*height_B//3:, :] = 0
    
    print(f"SE B2 created: {B2.shape}, background pixels: {np.sum(B2)}")
    save_image(B2, './Q1_Output/PartD/SE_B2_background.png')
    
    # Perform hit-or-miss transform
    print("Performing hit-or-miss transform with B1 and B2...")
    A_hit_miss = hit_or_miss(A, B1, B2)
    save_image(A_hit_miss, './Q1_Output/PartD/A_hit_miss_result.png')
    print(f"Hit-or-miss complete: detected points = {np.sum(A_hit_miss)}")
    
    # Dilate result to show full letters
    A_detected_letters = custom_dilation(A_hit_miss, B)
    
    # Create RGB overlays
    rgb_result_D = create_rgb_overlay(A, A_detected_letters)
    save_image(rgb_result_D, './Q1_Output/PartD/result_overlay_D.png')
    
    rgb_points_D = create_rgb_overlay(A, A_hit_miss)
    save_image(rgb_points_D, './Q1_Output/PartD/result_points_overlay_D.png')
    print("RGB overlays created")
    
    print("\nPart D completed successfully!")
    print("B2 matches the background in the opening region of 'c'")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("ALL PARTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nAll results saved in ./Q1_Output/ subdirectories")
    print("Check PartA, PartB, PartC, and PartD folders for outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()
