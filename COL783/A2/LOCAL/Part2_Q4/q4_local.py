from pathlib import Path
import argparse
import math
import time
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional
from scipy.signal import correlate2d

USE_SCIPY = True
try:
    import scipy
    from scipy.ndimage import median_filter as scipy_median_filter
    from scipy.ndimage import uniform_filter
except Exception:
    USE_SCIPY = False

OUT_DIR = Path("./q4_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_image_or_generate(path: Path, fallback_size=(512, 384)) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
        print(f"Loaded image: {path} size={img.size}")
        return img
    print(f"Warning: {path} not found. Generating fallback natural-like image.")
    W, H = fallback_size
    base = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(base)
    for y in range(H):
        v = int(30 + 225 * (y / (H - 1)))
        draw.line([(0, y), (W, y)], fill=(v, int(v*0.9), int(v*0.7)))
    rng = np.random.RandomState(1)
    for _ in range(35):
        cx = int(rng.randint(0, W))
        cy = int(rng.randint(0, H))
        r = int(rng.randint(8, 60))
        col = tuple(int(rng.randint(0, 256)) for _ in range(3))
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw.ellipse(bbox, fill=col)
    fallback_path = OUT_DIR / "fallback_natural_Q4_generated.png"
    base.save(fallback_path)
    print(f"Saved fallback image to {fallback_path}")
    return base

def save_img_from_array(arr: np.ndarray, fname: str) -> Path:
    """Save a float or uint8 image array to OUT_DIR and return Path."""
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr_clipped)
    outp = OUT_DIR / fname
    outp.parent.mkdir(parents=True, exist_ok=True)
    im.save(outp)
    return outp

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

def psnr(a: np.ndarray, b: np.ndarray, maxval: float = 255.0) -> float:
    mmse = mse(a, b)
    if mmse == 0:
        return float("inf")
    return 10.0 * math.log10((maxval ** 2) / mmse)

def desired_mse_for_psnr(psnr_db: float, maxval: float = 255.0) -> float:
    return (maxval ** 2) / (10.0 ** (psnr_db / 10.0))

def add_gaussian_noise_target_psnr(img: np.ndarray, target_psnr_db: float):
    # Add zero-mean Gaussian noise with variance = desired MSE
    desired_mse = desired_mse_for_psnr(target_psnr_db)
    sigma = math.sqrt(desired_mse)
    noise = np.random.normal(0.0, sigma, size=img.shape)
    return img + noise, sigma

def add_uniform_noise_target_psnr(img: np.ndarray, target_psnr_db: float):
    # Uniform noise in [-a, a] with var = a^2 / 3 = desired_mse
    desired_mse = desired_mse_for_psnr(target_psnr_db)
    a = math.sqrt(3 * desired_mse)
    noise = np.random.uniform(-a, a, size=img.shape)
    return img + noise, a

def add_salt_pepper_target_psnr(img: np.ndarray, target_psnr_db: float):
    """
    Salt-and-pepper noise tuned to reach target PSNR approximately.
    We compute the expected squared-error per element if a pixel is set to 0 or 255,
    average over both possibilities and over channels, then solve for p.
    """
    desired_mse = desired_mse_for_psnr(target_psnr_db)
    imgf = img.astype(np.float32)
    err_per_element = np.mean(((0.0 - imgf) ** 2 + (255.0 - imgf) ** 2) / 2.0)
    p = desired_mse / err_per_element
    if p > 1.0:
        print("desired PSNR too low/target unreachable with salt-pepper, clipping p to 1.0")
        p = 1.0
    H, W = img.shape[:2]
    rng = np.random.RandomState()
    mask = rng.rand(H, W) < p
    salt = rng.rand(H, W) < 0.5
    out = imgf.copy()
    if out.ndim == 3:#applying to all channels
        for c in range(out.shape[2]):
            ch = out[:, :, c]
            ch[mask & salt] = 255.0
            ch[mask & (~salt)] = 0.0
            out[:, :, c] = ch
    else:
        out[mask & salt] = 255.0
        out[mask & (~salt)] = 0.0
    return out, float(p)

def box_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize == 1:
        return img.copy()
    if USE_SCIPY:
        # scipy's uniform_filter works per-channel; size applies to each axis
        if img.ndim == 3:
            out = np.empty_like(img)
            for c in range(img.shape[2]):
                out[:, :, c] = uniform_filter(img[:, :, c], size=ksize, mode='reflect')
            return out
        else:
            return uniform_filter(img, size=ksize, mode='reflect')
    #Recursively do for each channel
    if img.ndim == 3:
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = box_filter(img[:, :, c], ksize)
        return out
    H, W = img.shape
    pad = ksize // 2
    p = np.pad(img, pad, mode='reflect').astype(np.float64)
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = p.cumsum(axis=0).cumsum(axis=1)#2d prefix sum approach
    out = np.empty((H, W), dtype=np.float64)
    for y in range(H):
        y1 = y
        y2 = y + ksize - 1
        for x in range(W):
            x1 = x
            x2 = x + ksize - 1
            s = ii[y2+1, x2+1] - ii[y1, x2+1] - ii[y2+1, x1] + ii[y1, x1]
            out[y, x] = s / (ksize * ksize)
    return out.astype(img.dtype)

def median_filter_wrapper(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize == 1:
        return img.copy()
    if USE_SCIPY:
        if img.ndim == 3:
            out = np.empty_like(img)
            for c in range(img.shape[2]):
                out[:, :, c] = scipy_median_filter(img[:, :, c], size=ksize, mode='reflect')
            return out
        else:
            return scipy_median_filter(img, size=ksize, mode='reflect')
    # recursively do for each layer
    if img.ndim == 3:
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = median_filter_wrapper(img[:, :, c], ksize)
        return out
    # fallback using sliding window techniques
    pad = ksize // 2
    p = np.pad(img, pad, mode='reflect')
    windows = sliding_window_view(p, (ksize, ksize)) #size is H*W*K*K (4D,each cpixel containing it's neighbourhood)
    H, W = windows.shape[:2]
    reshaped = windows.reshape(H, W, ksize*ksize)#MAKING Last axis linear so it is 3d of H*W*(K*K)
    med = np.median(reshaped, axis=2)
    return med.astype(img.dtype)

def non_local_means(gray: np.ndarray,
                            patch_radius: int = 1,
                            search_radius: int = 5, 
                            h: float = None,
                            sigma: float = None,
                            fast_mode: bool = False,
                            noise_type: str = 'gaussian') -> np.ndarray:
    # Auto-estimate sigma if not provided, Estimate noise using Laplacian variance method
    if sigma is None:
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        convolved = correlate2d(gray, laplacian, mode='same', boundary='symm')
        sigma = np.sqrt(0.5 * np.pi) * np.mean(np.abs(convolved)) / 6.0
        print(f"Auto-estimated sigma: {sigma:.3f}")
    # Auto-compute h if not provided based on scikit-image recommendations
    if h is None:
        if fast_mode:
            h = 0.8 * sigma  # Conservative for fast mode
        else:
            h = 1.0 * sigma
        print(f"Auto-computed h: {h:.3f}")
    
    # Handle salt & pepper noise with hybrid approach
    if noise_type == 'saltpepper':
        print("Using hybrid approach for salt & pepper noise")
        return _nlm_salt_pepper_hybrid(gray, patch_radius, search_radius, h, sigma)
    
    # Standard NLM for Gaussian noise
    H, W = gray.shape
    pad = patch_radius + search_radius
    padded = np.pad(gray, pad, mode='reflect')
    result = np.zeros_like(gray, dtype=np.float32)
    h2 = max(h * h, 1e-10)
    patch_size = (2 * patch_radius + 1) ** 2
    
    if noise_type == 'gaussian':
        noise_correction = 2 * (sigma ** 2) * patch_size
        print(f"Noise correction applied: {noise_correction:.2f}")
    else:
        noise_correction = 0.0
        print("Noise correction skipped (not applicable for salt & pepper)")
    
    print(f"Processing {H}x{W} image")
    print(f"Patch size: {patch_size} ({2*patch_radius+1}x{2*patch_radius+1})")
    print(f"Search window: {(2*search_radius+1)**2} positions")
    print(f"Operations per pixel: ~{patch_size * (2*search_radius+1)**2}")
    
    if not fast_mode:
        # Create Gaussian kernel for patch weighting
        y_coords, x_coords = np.ogrid[-patch_radius:patch_radius+1, -patch_radius:patch_radius+1]
        gaussian_patch = np.exp(-(x_coords*x_coords + y_coords*y_coords) / (2.0 * (patch_radius/2.0)**2))
        gaussian_patch = gaussian_patch / np.sum(gaussian_patch)
        print("Using Gaussian patch weighting (slow mode)")
    else:
        gaussian_patch = None
        print("Using uniform patch weighting (fast mode)")
    
    for y in range(H):
        for x in range(W):
            y0, x0 = y + pad, x + pad
            ref_patch = padded[y0-patch_radius:y0+patch_radius+1, x0-patch_radius:x0+patch_radius+1]  
            weights_sum = 0.0
            weighted_sum = 0.0
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    yy, xx = y0 + dy, x0 + dx
                    cand_patch = padded[yy-patch_radius:yy+patch_radius+1, xx-patch_radius:xx+patch_radius+1]
                    diff = ref_patch - cand_patch
                    if fast_mode:
                        patch_distance = np.sum(diff * diff)
                    else:
                        # Slow Gaussian weighting for better quality
                        patch_distance = np.sum(gaussian_patch * diff * diff)
                    corrected_distance = max(patch_distance - noise_correction, 0.0)
                    if corrected_distance / h2 > 50:  # Compute weight with numerical stability, Prevent exp overflow.
                        weight = 0.0
                    else:
                        weight = math.exp(-corrected_distance / h2)
                    weights_sum += weight
                    weighted_sum += weight * padded[yy, xx]
            if weights_sum > 1e-10:
                result[y, x] = weighted_sum / weights_sum
            else:
                result[y, x] = gray[y, x]
        if (y % 30) == 0:
            progress = (y + 1) / H * 100
            print(f"Progress: {progress:.1f}% ({y+1}/{H} rows)")
    return result

def _adaptive_median_filter(image: np.ndarray, max_window_size: int = 7) -> np.ndarray:
    # Adaptive median filter for salt & pepper noise preprocessing.
    H, W = image.shape
    result = image.copy()
    
    for y in range(H):
        for x in range(W):
            for window_size in range(3, max_window_size + 1, 2):
                half_w = window_size // 2

                y_min = max(0, y - half_w)
                y_max = min(H, y + half_w + 1)
                x_min = max(0, x - half_w)
                x_max = min(W, x + half_w + 1)
                
                window = image[y_min:y_max, x_min:x_max]
                
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = image[y, x]
                
                # Stage A: Check if median is impulse
                A1 = z_med - z_min
                A2 = z_med - z_max
                if A1 > 0 and A2 < 0:
                    # Median is not impulse
                    # Stage B: Check if current pixel is impulse
                    B1 = z_xy - z_min
                    B2 = z_xy - z_max
                    if B1 > 0 and B2 < 0:
                        result[y, x] = z_xy  # Keep original (not impulse)
                    else:
                        result[y, x] = z_med  # Replace with median (is impulse)
                    break
                else:
                    # Median is impulse, try larger window, if not possiblle Fallback to median
                    if window_size == max_window_size:
                        result[y, x] = z_med 
    return result


def _nlm_salt_pepper_hybrid(gray: np.ndarray, patch_radius: int, search_radius: int, h: float, sigma: Optional[float]) -> np.ndarray:
    print("Step 1: Adaptive median filtering to remove salt & pepper...")
    pre_filtered = _adaptive_median_filter(gray)
    print("Step 2: Applying NLM for final smoothing...")
    # Apply NLM on pre-filtered image (now mostly Gaussian-like noise)
    result = non_local_means(
        pre_filtered, 
        patch_radius=patch_radius, 
        search_radius=search_radius,
        h=h, 
        sigma=sigma, 
        fast_mode=False,
        noise_type='gaussian'
    )
    return result

def experiment_mean_median(noisy: np.ndarray, clean: np.ndarray, w_list: list):
    """Run mean and median filters across w_list and compute PSNRs and store images."""
    mean_psnrs, median_psnrs = [], []
    mean_imgs, median_imgs = [], []
    for w in w_list:
        den_mean = box_filter(noisy, w)
        den_med = median_filter_wrapper(noisy, w)
        mean_imgs.append(den_mean)
        median_imgs.append(den_med)
        mean_psnrs.append(psnr(den_mean, clean))
        median_psnrs.append(psnr(den_med, clean))
    return mean_psnrs, median_psnrs, mean_imgs, median_imgs

def plot_psnr(w_list: list, mean_psnrs: list, median_psnrs: list, title: str, outname: str):
    plt.figure(figsize=(7,4))
    plt.plot(w_list, mean_psnrs, marker='o', label='Mean (box)')
    plt.plot(w_list, median_psnrs, marker='s', label='Median')
    plt.xlabel('Window width w')
    plt.ylabel('PSNR (dB)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    outp = OUT_DIR / outname
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {outp}")
    return outp


def main_cli(args):
    natural_img = load_image_or_generate(Path(args.input))
    natural_arr = np.asarray(natural_img).astype(np.float32)
    save_img_from_array(natural_arr, "input_natural_Q4_loaded_or_generated.png")

    H, W = natural_img.size[1], natural_img.size[0]
    const_arr = np.full((H, W, 3), 255.0/2.0, dtype=np.float32)
    save_img_from_array(const_arr, "constant_image_c.png")

    target_psnr = args.target_psnr

    noises = ["uniform", "gaussian", "saltpepper"]
    c_noisy = {}
    f_noisy = {}
    noise_params = {}

    for typ in noises:
        if typ == 'gaussian':
            nc, s = add_gaussian_noise_target_psnr(const_arr, target_psnr)
            nf, s2 = add_gaussian_noise_target_psnr(natural_arr, target_psnr)
            c_noisy[typ] = np.clip(nc, 0, 255).astype(np.float32)
            f_noisy[typ] = np.clip(nf, 0, 255).astype(np.float32)
            noise_params[typ] = {'sigma_const': s, 'sigma_natural': s2}
        elif typ == 'uniform':
            nc, a = add_uniform_noise_target_psnr(const_arr, target_psnr)
            nf, a2 = add_uniform_noise_target_psnr(natural_arr, target_psnr)
            c_noisy[typ] = np.clip(nc, 0, 255).astype(np.float32)
            f_noisy[typ] = np.clip(nf, 0, 255).astype(np.float32)
            noise_params[typ] = {'a_const': a, 'a_natural': a2}
        elif typ == 'saltpepper':
            nc, p = add_salt_pepper_target_psnr(const_arr, target_psnr)
            nf, p2 = add_salt_pepper_target_psnr(natural_arr, target_psnr)
            c_noisy[typ] = np.clip(nc, 0, 255).astype(np.float32)
            f_noisy[typ] = np.clip(nf, 0, 255).astype(np.float32)
            noise_params[typ] = {'p_const': p, 'p_natural': p2}

    for typ in noises:
        save_img_from_array(c_noisy[typ], f"PartA/c_noisy_{typ}.png")
        save_img_from_array(f_noisy[typ], f"PartA/f_noisy_{typ}.png")
        print(f"{typ}: PSNR(c_noisy vs c) = {psnr(c_noisy[typ], const_arr):.3f} dB")
        print(f"{typ}: PSNR(f_noisy vs f) = {psnr(f_noisy[typ], natural_arr):.3f} dB")

    print("Noise params:", noise_params)

    # window sizes
    w_list = [1, 3, 5, 7, 9, 11]

    # Part (b)
    for typ in noises:
        print(f"Running mean/median on constant image for noise: {typ}")
        mean_psnrs, median_psnrs, mean_imgs, median_imgs = experiment_mean_median(c_noisy[typ], const_arr, w_list)
        plot_psnr(w_list, mean_psnrs, median_psnrs, f"Const image c with {typ} noise", f"PartB/c_psnr_{typ}.png")
        # saving the best images for future rference
        best_mean_idx = int(np.argmax(mean_psnrs))
        best_med_idx = int(np.argmax(median_psnrs))
        save_img_from_array(mean_imgs[best_mean_idx], f"PartB/c_{typ}_best_mean_w{w_list[best_mean_idx]}.png")
        save_img_from_array(median_imgs[best_med_idx], f"PartB/c_{typ}_best_median_w{w_list[best_med_idx]}.png")
        print(f"Best mean w={w_list[best_mean_idx]} PSNR={mean_psnrs[best_mean_idx]:.3f}dB")
        print(f"Best median w={w_list[best_med_idx]} PSNR={median_psnrs[best_med_idx]:.3f}dB")

    # Part (c)
    for typ in noises:
        print(f"Running mean/median on natural image for noise: {typ}")
        mean_psnrs, median_psnrs, mean_imgs, median_imgs = experiment_mean_median(f_noisy[typ], natural_arr, w_list)
        plot_psnr(w_list, mean_psnrs, median_psnrs, f"Natural image f with {typ} noise", f"PartC/f_psnr_{typ}.png")
        # saving the best images for future rference
        best_mean_idx = int(np.argmax(mean_psnrs))
        best_med_idx = int(np.argmax(median_psnrs))
        save_img_from_array(mean_imgs[best_mean_idx], f"PartC/f_{typ}_best_mean_w{w_list[best_mean_idx]}.png")
        save_img_from_array(median_imgs[best_med_idx], f"PartC/f_{typ}_best_median_w{w_list[best_med_idx]}.png")
        print(f"Best mean w={w_list[best_mean_idx]} PSNR={mean_psnrs[best_mean_idx]:.3f}dB")
        print(f"Best median w={w_list[best_med_idx]} PSNR={median_psnrs[best_med_idx]:.3f}dB")

    # Part (d)
    def rgb2gray(arr):
        return (0.3333 * arr[:, :, 0] + 0.3333 * arr[:, :, 1] + 0.3333 * arr[:, :, 2])
    gray_clean = rgb2gray(natural_arr)
    nlm_summary = {}
    for typ in noises:
        gray_noisy = rgb2gray(f_noisy[typ])
        trials = [{'patch_radius': 2, 'search_radius': 6, 'noise_type': 'gaussian'}]
        if typ!='gaussian':
            trials.append({'patch_radius': 2, 'search_radius': 6, 'noise_type': typ})
        nlm_summary[typ] = []
        for t in trials:
            print(f"Running NLM for {typ} with params {t}")
            t0 = time.time()
            den = non_local_means(gray_noisy, patch_radius=t['patch_radius'], search_radius=t['search_radius'], noise_type=t['noise_type'])
            t1 = time.time()
            psnr_val = psnr(den, gray_clean)
            print("den.shape =", den.shape, "gray_clean.shape =", gray_clean.shape)
            outname = f"PartD/Noise_Type_{typ}/nlm_pr{t['patch_radius']}_sr{t['search_radius']}_de_noise_type{t['noise_type']}.png"
            save_img_from_array(den, outname)
            nlm_summary[typ].append({'params': t, 'psnr': psnr_val, 'time_s': t1 - t0, 'file': outname})
            print(f"Saved {outname} PSNR={psnr_val:.3f} time={t1-t0:.1f}s")
    print("\nAll outputs saved to:", OUT_DIR)
    print("NLM summary:")
    for k, v in nlm_summary.items():
        print(k, v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoising experiment (mean, median, NLM) tuned to PSNR=20dB by default.')
    parser.add_argument('--input', type=str, default=r'../Testcases/Q4.png', help='Path to natural image (Q4.jpg)')
    parser.add_argument('--target-psnr', type=float, default=20.0, help='Target PSNR (dB) for noisy images')
    args = parser.parse_args()
    main_cli(args)


'''
Notes:
(1) The downsampling is achieved using slice notation with a step value (also known as subsampling or nearest-neighbor downsampling).

img = gray[::downsample, ::downsample].copy()

In NumPy the slice notation [start:stop:step] is used. When start and stop are omitted, it selects elements from the beginning to the end, taking steps of size step.

gray[::downsample, ...] selects rows from the gray array.

gray[..., ::downsample] selects columns from the gray array.

If downsample is, for example, 2:

gray[::2, ::2] selects every second row and every second column of the gray array.
'''