from pathlib import Path
import argparse
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter, zoom
from skimage import io, color, util
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle

AUTOMATED_PEAK_DETECTOR=False

def load_image_gray(path: str) -> np.ndarray:
    # Load image and return grayscale float image in [0,1] format
    img = io.imread(path)
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = util.img_as_float(img)
    return gray

def rescale_max_size(img: np.ndarray, max_dim: int = 1024) -> tuple[np.ndarray, float]:
    h, w = img.shape
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / max(h, w)
    img_rs = zoom(img, (scale, scale), order=1)
    return img_rs, scale

def compute_centered_spectrum(img: np.ndarray):
    # Return centered FFT (complex), magnitude and log-magnitude.
    F = fftshift(fft2(img))
    mag = np.abs(F)
    mag_log = np.log1p(mag)
    return F, mag, mag_log

def detect_spectral_peaks(mag: np.ndarray, exclude_radius: int = 10, top_percentile: float = 99.5, preview_radius: float = None):
    global AUTOMATED_PEAK_DETECTOR
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    def _auto_detect():
        yc, xc = np.ogrid[:H, :W]
        rmap = np.sqrt((yc - cy)**2 + (xc - cx)**2)
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

    if AUTOMATED_PEAK_DETECTOR:
        return _auto_detect()

    mag_display = np.log1p(mag)
    mag_display = (mag_display - mag_display.min())
    if mag_display.max() > 0:
        mag_display = mag_display / mag_display.max()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mag_display, cmap='gray', origin='upper')
    ax.set_title("Select spectral peaks (left-click). Press ENTER when done.\n"
                 "Press 'd' to enter drag-mode (adjust last pick), press ESC to exit drag-mode.",
                 fontsize=9)
    ax.axis('off')

    picked = []           # list of (x, y) in data coordinates (float)
    markers = []          # artist list for + marks
    circles = []          # artist list for preview circles (one per pick)
    drag_mode = False     # when True, moving mouse modifies last picked point
    preview_radius_px = preview_radius if preview_radius is not None else None

    def add_pick_visual(x, y):
        m, = ax.plot(x, y, marker='+', color='r', markersize=10, mew=1.5)
        markers.append(m)
        if preview_radius_px is not None:
            circ = Circle((x, y), preview_radius_px, fill=False, linestyle='dotted', linewidth=1.2)
            ax.add_patch(circ)
            circles.append(circ)
        fig.canvas.draw_idle()

    def update_last_visual(x, y):
        if len(markers) >= 1:
            markers[-1].set_data([x], [y])
        if len(circles) >= 1 and preview_radius_px is not None:
            circles[-1].center = (x, y)
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal drag_mode
        if event.inaxes is None:
            return
        if drag_mode:
            return
        if event.button == 1:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            picked.append((x, y))
            add_pick_visual(x, y)

    def onmove(event):
        if not drag_mode:
            return
        if event.inaxes is None:
            return
        if len(picked) == 0:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        picked[-1] = (x, y)
        update_last_visual(x, y)

    def onkey(event):
        nonlocal drag_mode
        if event.key in ('enter', 'return'):
            plt.close(fig)
        elif event.key == 'd':
            drag_mode = True
            fig.suptitle("DRAG MODE: move mouse to adjust last selection. Press ESC to exit drag mode.", color='blue')
            fig.canvas.draw_idle()
        elif event.key == 'escape':
            drag_mode = False
            try:
                fig.suptitle("")
            except Exception:
                pass
            fig.canvas.draw_idle()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', onmove)
    cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()   # blocks until enter pressed

    try:
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_move)
        fig.canvas.mpl_disconnect(cid_key)
    except Exception:
        pass

    if len(picked) == 0:
        print("No points selected — falling back to automated detection.")
        return _auto_detect()

    # Convert picked x=col, y=row coords to (row, col) and sample intensity as 3x3 neighbourhood avg
    centers = []
    for (x, y) in picked:
        cxp = int(round(x))
        ryp = int(round(y))
        cxp = int(np.clip(cxp, 0, W - 1))
        ryp = int(np.clip(ryp, 0, H - 1))
        r0 = max(0, ryp - 1); r1 = min(H, ryp + 2)
        c0 = max(0, cxp - 1); c1 = min(W, cxp + 2)
        inten = float(np.mean(mag[r0:r1, c0:c1]))
        centers.append((float(ryp), float(cxp), inten))

    centers_sorted = sorted(centers, key=lambda x: -x[2])
    return centers_sorted


def report_peak_frequencies(centers, shape):
    # Prints frequency (cycles/pixel) and estimated spacing (pixels) for each center
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
    # Butterworth notch-reject mask
    # For each notch center k:
    # H_k = 1 / (1 + (D0 / D_k)^(2n))
    # Overall mask is H_k * H_k_sym
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
    Ff = F_centered * mask
    img_rec = np.real(ifft2(ifftshift(Ff)))
    return img_rec, Ff

def save_images(out_dir: str, **images):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, arr in images.items():
        p = Path(out_dir) / name
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
    parser.add_argument('--max-dim', type=int, default=1024, help='Max dimension to rescale for speed')
    parser.add_argument('--method', choices=['gaussian', 'butterworth', 'ideal'], default='gaussian')
    parser.add_argument('--sigma', type=float, default=6.0, help='Gaussian sigma or ideal radius or butterworth D0')
    parser.add_argument('--butterworth-n', type=int, default=2, help='Butterworth order n (only used if method=butterworth)')
    parser.add_argument('--out', default='./halftone_results', help='Output folder')
    parser.add_argument('--report-top', type=float, default=99.5, help='Percentile for peak detection threshold')
    args = parser.parse_args()
    args.out += f'_{args.method}'
    args.out += f'_sigma={args.sigma}'

    gray = load_image_gray(args.input)
    gray_rs, _ = rescale_max_size(gray, max_dim=args.max_dim)
    F, mag, mag_log = compute_centered_spectrum(gray_rs)
    centers = detect_spectral_peaks(mag, exclude_radius=10, top_percentile=args.report_top,preview_radius=args.sigma)

    #detected centers and estimated spacing
    report = report_peak_frequencies(centers, mag.shape)
    print("Detected spectral peaks (top to bottom):")
    for i, r in enumerate(report[:20]):
        print(f"{i+1}: centroid={r['centroid']}, dy_dx={r['dy_dx']}, fx_fy={r['fx_fy']}, f_mag={r['f_mag']:.4f}, spacing_px={r['spacing_px']:.1f}")

    if args.method == 'gaussian':
        mask = build_gaussian_notch_mask(mag.shape, centers, sigma=args.sigma)
    elif args.method == 'butterworth':
        mask = build_butterworth_notch_mask(mag.shape, centers, D0=args.sigma, n=args.butterworth_n)
    else:
        mask = build_ideal_notch_mask(mag.shape, centers, radius=args.sigma)

    img_filtered, F_filtered = apply_notch_filter(F, mask)
    mag_filtered_log = np.log1p(np.abs(F_filtered))

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

    show_im('Input', img_clip)
    show_im('Centered log-spectrum (no padding)', mag_log_norm)
    show_im(f'Filtered spectrum ({args.method})', mag_filtered_norm)
    show_im(f'Filtered image ({args.method})', img_f_clip)
    plt.show()


if __name__ == '__main__':
    main()