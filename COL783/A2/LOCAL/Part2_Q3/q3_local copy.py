from pathlib import Path
import argparse
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter, zoom
from skimage import io, color, util
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import os

def load_image_gray(path: str) -> np.ndarray:
    # Load image and return grayscale float image in [0,1] format
    img = io.imread(path)
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = util.img_as_float(img)
    return gray

# def auto_crop_halftone(gray: np.ndarray, sigma: float = 3.0, percentile: float = 95.0, min_area: int = 200*200, pad: int = 20) -> tuple[np.ndarray, bool]:
#     # It finds a region with halftone dots by looking for high local variance, Returns (cropped_image, used_crop_flag).
#     local_mean = gaussian_filter(gray, sigma=sigma)
#     local_sq_mean = gaussian_filter(gray**2, sigma=sigma)
#     local_var = np.clip(local_sq_mean - local_mean**2, 0, None)
#     thr = np.percentile(local_var, percentile)
#     mask = local_var > thr
#     lab = label(mask)
#     props = regionprops(lab)
#     if not props:
#         return gray, False
#     props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
#     bbox = props_sorted[0].bbox  # (min_row, min_col, max_row, max_col)
#     r0, c0, r1, c1 = bbox
#     r0 = max(0, r0 - pad); c0 = max(0, c0 - pad)
#     r1 = min(gray.shape[0], r1 + pad); c1 = min(gray.shape[1], c1 + pad)
#     crop = gray[r0:r1, c0:c1]
#     if crop.size > min_area:
#         return crop, True
#     else:
#         return gray, False

def rescale_max_size(img: np.ndarray, max_dim: int = 1024) -> tuple[np.ndarray, float]:
    h, w = img.shape
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / max(h, w)
    img_rs = zoom(img, (scale, scale), order=1)
    return img_rs, scale

def compute_centered_spectrum(img: np.ndarray):
    """Return centered FFT (complex), magnitude and log-magnitude."""
    F = fftshift(fft2(img))
    mag = np.abs(F)
    mag_log = np.log1p(mag)
    return F, mag, mag_log

def detect_spectral_peaks(mag: np.ndarray, exclude_radius: int = 10, top_percentile: float = 99.5):
    """
    Simple spike detector: mask out small central region, threshold at percentile and return centroids.
    Returns list of (y, x, max_intensity) for each detected connected component, sorted by intensity desc.
    """
    H, W = mag.shape
    cy, cx = H//2, W//2
    yc, xc = np.ogrid[:H, :W]
    rmap = np.sqrt((yc-cy)**2 + (xc-cx)**2)
    mag_masked = mag.copy()
    mag_masked[rmap <= exclude_radius] = 0.0
    th = np.percentile(mag_masked, top_percentile)
    peaks = mag_masked > th
    lab = label(peaks)
    props = regionprops(lab, intensity_image=mag_masked)
    centers = []
    for p in props:
        ry, rx = p.centroid
        centers.append((ry, rx, float(p.max_intensity)))
    centers_sorted = sorted(centers, key=lambda x: -x[2])
    return centers_sorted

def report_peak_frequencies(centers, shape):
    """Prints frequency (cycles/pixel) and estimated spacing (pixels) for each center."""
    H, W = shape
    cy, cx = H//2, W//2
    out = []
    for (ry, rx, mi) in centers:
        dy = ry - cy
        dx = rx - cx
        fx = dx / W
        fy = dy / H
        f_mag = np.sqrt(fx*fx + fy*fy)
        spacing = (1.0 / f_mag) if f_mag > 0 else np.inf
        out.append({'centroid': (ry, rx), 'dy_dx': (dy, dx), 'fx_fy': (fx, fy), 'f_mag': f_mag, 'spacing_px': spacing, 'intensity': mi})
    return out

def build_gaussian_notch_mask(shape, centers, sigma=6.0, symmetric=True):
    H, W = shape
    Y, X = np.indices((H, W))
    mask = np.ones((H, W), dtype=float)
    cy, cx = H//2, W//2
    for (ry, rx, _) in centers:
        d2 = (Y - ry)**2 + (X - rx)**2
        g = np.exp(-d2 / (2 * sigma**2))
        mask *= (1 - g)
        if symmetric:
            ys = 2*cy - ry
            xs = 2*cx - rx
            d2s = (Y - ys)**2 + (X - xs)**2
            gs = np.exp(-d2s / (2 * sigma**2))
            mask *= (1 - gs)
    mask = np.clip(mask, 0.0, 1.0)
    return mask

def build_butterworth_notch_mask(shape, centers, D0=6.0, n=2, symmetric=True):
    """
    Butterworth notch-reject mask (multiplicative). For each notch center k:
      H_k = 1 / (1 + (D0 / D_k)^(2n))
    Overall mask is product_k H_k * H_k_sym.
    """
    Hm, Wm = shape
    Y, X = np.indices((Hm, Wm))
    mask = np.ones((Hm, Wm), dtype=float)
    cy, cx = Hm//2, Wm//2
    eps = 1e-8
    for (ry, rx, _) in centers:
        Dk = np.sqrt((Y-ry)**2 + (X-rx)**2) + eps
        Hk = 1.0 / (1.0 + (D0 / Dk)**(2*n))
        mask *= Hk
        if symmetric:
            ys = 2*cy - ry
            xs = 2*cx - rx
            Dks = np.sqrt((Y-ys)**2 + (X-xs)**2) + eps
            Hks = 1.0 / (1.0 + (D0 / Dks)**(2*n))
            mask *= Hks
    mask = np.clip(mask, 0.0, 1.0)
    return mask

def build_ideal_notch_mask(shape, centers, radius=6.0, symmetric=True):
    H, W = shape
    Y, X = np.indices((H, W))
    mask = np.ones((H, W), dtype=float)
    cy, cx = H//2, W//2
    for (ry, rx, _) in centers:
        d = np.sqrt((Y-ry)**2 + (X-rx)**2)
        mask[d <= radius] = 0.0
        if symmetric:
            ys = 2*cy - ry
            xs = 2*cx - rx
            ds = np.sqrt((Y-ys)**2 + (X-xs)**2)
            mask[ds <= radius] = 0.0
    return mask

def apply_notch_filter(F_centered, mask):
    """Apply multiplicative mask to centered spectrum F and return filtered image (real) and filtered spectrum."""
    Ff = F_centered * mask
    img_rec = np.real(ifft2(ifftshift(Ff)))
    return img_rec, Ff

def save_images(out_dir: str, **images):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, arr in images.items():
        p = Path(out_dir) / name
        # normalize float images to uint8 when saving
        arr_clip = np.clip(arr, 0, 1)
        io.imsave(str(p), util.img_as_ubyte(arr_clip))
        paths[name] = str(p)
    return paths

def show_im(title, img, cmap='gray'):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')

def main():
    parser = argparse.ArgumentParser(description="Halftone notch filtering (modular)")
    parser.add_argument('--input', required=True, help='Input image path')
    # parser.add_argument('--auto-crop', action='store_true', help='Automatically crop to halftone region')
    parser.add_argument('--max-dim', type=int, default=1024, help='Max dimension to rescale for speed')
    parser.add_argument('--method', choices=['gaussian', 'butterworth', 'ideal'], default='gaussian')
    parser.add_argument('--sigma', type=float, default=6.0, help='Gaussian sigma or ideal radius or butterworth D0')
    parser.add_argument('--butterworth-n', type=int, default=2, help='Butterworth order n (only used if method=butterworth)')
    parser.add_argument('--show', action='store_true', help='Show matplotlib figures')
    parser.add_argument('--out', default='./halftone_results', help='Output folder')
    parser.add_argument('--report-top', type=float, default=99.5, help='Percentile for peak detection threshold')
    args = parser.parse_args()

    gray = load_image_gray(args.input)

    # if args.auto_crop:
    #     gray_crop, used = auto_crop_halftone(gray)
    # else:
    # gray_crop = gray

    # gray_rs, _ = rescale_max_size(gray_crop, max_dim=args.max_dim)

    gray_rs, _ = rescale_max_size(gray, max_dim=args.max_dim)

    F, mag, mag_log = compute_centered_spectrum(gray_rs)
    centers = detect_spectral_peaks(mag, exclude_radius=10, top_percentile=args.report_top)

    # Report detected centers and estimated spacing
    report = report_peak_frequencies(centers, mag.shape)
    print("Detected spectral peaks (top to bottom):")
    for i, r in enumerate(report[:20]):
        print(f"{i+1}: centroid={r['centroid']}, dy_dx={r['dy_dx']}, fx_fy={r['fx_fy']}, f_mag={r['f_mag']:.4f}, spacing_px={r['spacing_px']:.1f}")

    # Choose filter
    if args.method == 'gaussian':
        mask = build_gaussian_notch_mask(mag.shape, centers, sigma=args.sigma)
    elif args.method == 'butterworth':
        mask = build_butterworth_notch_mask(mag.shape, centers, D0=args.sigma, n=args.butterworth_n)
    else:
        mask = build_ideal_notch_mask(mag.shape, centers, radius=args.sigma)

    # Apply
    img_filtered, F_filtered = apply_notch_filter(F, mask)
    mag_filtered_log = np.log1p(np.abs(F_filtered))

    # Prepare images for saving/display: normalized for display
    mag_log_norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-12)
    mag_filtered_norm = (mag_filtered_log - mag_filtered_log.min()) / (mag_filtered_log.max() - mag_filtered_log.min() + 1e-12)
    img_clip = np.clip(gray_rs, 0, 1)
    img_f_clip = np.clip(img_filtered, 0, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    io.imsave(out / 'input.png', util.img_as_ubyte(img_clip))
    io.imsave(out / 'spectrum.png', util.img_as_ubyte(mag_log_norm))
    io.imsave(out / f'filtered_{args.method}.png', util.img_as_ubyte(img_f_clip))
    io.imsave(out / f'spectrum_filtered_{args.method}.png', util.img_as_ubyte(mag_filtered_norm))

    print(f"Saved results to {out.resolve()}")

    if args.show:
        show_im('Input', img_clip)
        show_im('Centered log-spectrum (no padding)', mag_log_norm)
        show_im(f'Filtered spectrum ({args.method})', mag_filtered_norm)
        show_im(f'Filtered image ({args.method})', img_f_clip)
        plt.show()


if __name__ == '__main__':
    main()

# python .\q3_local.py --input ..\Testcases\Q3_Gordon_De_Lisle_in_1943.jpg --auto-crop --method gaussian --sigma 6 --show
# python .\q3_local.py --input ..\Testcases\Q3_Gordon_De_Lisle_in_1943.jpg --auto-crop --method ideal --sigma 6 --show
# python .\q3_local.py --input ..\Testcases\Q3_Gordon_De_Lisle_in_1943.jpg --auto-crop --method butterworth --sigma 6 --butterworth-n 2 --show