import cv2
import numpy as np
import os

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def save_pyramid_images(pyramid, output_dir, prefix):
    ensure_directory(output_dir)
    for i, level in enumerate(pyramid):
        filename = os.path.join(output_dir, f"{prefix}_level_{i}.png")
        if level.dtype == np.float32 or level.dtype == np.float64:
            normalized = cv2.normalize(level, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(filename, normalized.astype(np.uint8))
        else:
            cv2.imwrite(filename, level)
    print(f"  Saved {len(pyramid)} levels to {output_dir}")

def construct_gaussian_pyramid(image, levels, kernel):
    pyramid = [image]
    current = image.astype(np.float32).copy()
    current = image.copy()
    for i in range(levels - 1):
        blurred = cv2.sepFilter2D(current, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_REFLECT)
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    return pyramid

def upsample_and_convolve(image, target_shape, kernel):
    h, w = target_shape[:2]
    image = image.astype(np.float32)
    if len(image.shape) == 3:
        upsampled = np.zeros((h, w, image.shape[2]), dtype=np.float32)
    else:
        upsampled = np.zeros((h, w), dtype=np.float32)
    upsampled[::2, ::2] = image
    upsampled *= 4.0
    result = cv2.sepFilter2D(upsampled, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_REFLECT)
    return result

def construct_laplacian_pyramid(gaussian_pyramid, kernel):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        current = gaussian_pyramid[i].astype(np.float32)
        next_level = gaussian_pyramid[i + 1]
        upsampled = upsample_and_convolve(next_level, current.shape, kernel)
        laplacian = current - upsampled
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1].astype(np.float32))
    return laplacian_pyramid

def reconstruct_from_laplacian(laplacian_pyramid, kernel):
    reconstructed = laplacian_pyramid[-1].copy()
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        upsampled = upsample_and_convolve(reconstructed, laplacian_pyramid[i].shape, kernel)
        reconstructed = upsampled + laplacian_pyramid[i]
    return reconstructed

def blend_pyramids(laplacian_f, laplacian_g, mask_pyramid):
    blended = []
    for i in range(len(laplacian_f)):
        mask = mask_pyramid[i].astype(np.float32) / 255.0
        if len(laplacian_f[i].shape) == 3 and len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        blended_level = (1 - mask) * laplacian_f[i] + mask * laplacian_g[i]
        blended.append(blended_level)
    return blended

def main():
    f_path = "../Testcases/q1_apple.jpg"
    g_path = "../Testcases/q1_orange.jpg"
    num_levels = 6
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    output_base = "./Q1_Output"
    output_partA = os.path.join(output_base, "PartA")
    output_partB = os.path.join(output_base, "PartB")
    output_partC = os.path.join(output_base, "PartC")
    output_partD = os.path.join(output_base, "PartD")
    print("Loading images...")
    f = cv2.imread(f_path)
    g = cv2.imread(g_path)
    if f is None or g is None:
        print(f"Error: Could not load images from {f_path} and {g_path}")
        return
    f = f.astype(np.float32)
    g = g.astype(np.float32)
    print(f"Loaded f: {f.shape}, g: {g.shape}")
    if f.shape != g.shape:
        print("Warning: Images have different sizes. Resizing to match...")
        h = min(f.shape[0], g.shape[0])
        w = min(f.shape[1], g.shape[1])
        f = cv2.resize(f, (w, h))
        g = cv2.resize(g, (w, h))
        print(f"Resized to: {f.shape}")
    
    print("\n" + "="*50)
    print("======Part-A processing...======")
    print("="*50)
    print("Constructing Gaussian pyramid for f...")
    gaussian_f = construct_gaussian_pyramid(f, num_levels, kernel)
    save_pyramid_images(gaussian_f, output_partA, "gaussian_f")
    print("Constructing Gaussian pyramid for g...")
    gaussian_g = construct_gaussian_pyramid(g, num_levels, kernel)
    save_pyramid_images(gaussian_g, output_partA, "gaussian_g")
    print(f"Part A complete. Gaussian pyramids with {num_levels} levels created.")

    print("\n" + "="*50)
    print("======Part-B processing...======")
    print("="*50)
    print("Constructing Laplacian pyramid for f...")
    laplacian_f = construct_laplacian_pyramid(gaussian_f, kernel)
    save_pyramid_images(laplacian_f, output_partB, "laplacian_f")
    print("Constructing Laplacian pyramid for g...")
    laplacian_g = construct_laplacian_pyramid(gaussian_g, kernel)
    save_pyramid_images(laplacian_g, output_partB, "laplacian_g")
    print("Reconstructing f from its Laplacian pyramid...")
    reconstructed_f = reconstruct_from_laplacian(laplacian_f, kernel)
    reconstructed_f = np.clip(reconstructed_f, 0, 255).astype(np.uint8)
    diff = cv2.absdiff(f.astype(np.uint8), reconstructed_f)
    max_error = np.max(diff)
    mean_error = np.mean(diff)
    print(f"Reconstruction verification:")
    print(f"  Max pixel error: {max_error}")
    print(f"  Mean pixel error: {mean_error:.6f}")
    ensure_directory(output_partB)
    cv2.imwrite(os.path.join(output_partB, "reconstructed_f.png"), reconstructed_f)
    cv2.imwrite(os.path.join(output_partB, "reconstruction_error.png"), diff * 10)  
    print("Part B complete. Laplacian pyramids created and reconstruction verified.")

    print("\n" + "="*50)
    print("======Part-C processing...======")
    print("="*50)
    print("Creating binary mask (vertical split)...")
    h, w = f.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, w//2:] = 255  
    ensure_directory(output_partC)
    cv2.imwrite(os.path.join(output_partC, "binary_mask.png"), mask)
    print("Creating hard transition blend...")
    mask_float = mask.astype(np.float32) / 255.0
    mask_float_3ch = mask_float[:, :, np.newaxis]
    hard_blend = ((1 - mask_float_3ch) * f.astype(np.float32) + 
                  mask_float_3ch * g.astype(np.float32))
    hard_blend = np.clip(hard_blend, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_partC, "hard_transition.png"), hard_blend)
    print("Part C complete. Hard transition blend created.")

    print("\n" + "="*50)
    print("======Part-D processing...======")
    print("="*50)
    print("Creating Gaussian pyramid of the mask...")
    mask = mask.astype(np.float32) 
    mask_pyramid = construct_gaussian_pyramid(mask, num_levels, kernel)
    save_pyramid_images(mask_pyramid, output_partD, "gaussian_mask")
    print("Blending Laplacian pyramids...")
    blended_laplacian = blend_pyramids(laplacian_f, laplacian_g, mask_pyramid)
    save_pyramid_images(blended_laplacian, output_partD, "blended_laplacian")
    print("Reconstructing final image with smooth transition...")
    final_blend = reconstruct_from_laplacian(blended_laplacian, kernel)
    final_blend = np.clip(final_blend, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_partD, "smooth_transition.png"), final_blend)
    print("Part D complete. Smooth transition blend created.")
    print("\n" + "="*50)
    print("ALL PARTS COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Output directory: {output_base}")
    print(f"  Part A (Gaussian pyramids): {output_partA}")
    print(f"  Part B (Laplacian pyramids): {output_partB}")
    print(f"  Part C (Hard transition): {output_partC}")
    print(f"  Part D (Smooth transition): {output_partD}")

if __name__ == "__main__":
    main()