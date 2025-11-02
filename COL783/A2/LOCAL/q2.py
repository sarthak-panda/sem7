import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from typing import List
import argparse 

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_gray(path: str, arr: np.ndarray):
    a = np.clip(arr, 0.0, None)
    if a.max() > 1e-12:
        a = a / (a.max())
    a8 = (255.0 * a).astype(np.uint8)
    cv2.imwrite(path, a8)

def save_rgb(path: str, arr: np.ndarray):
    a = np.clip(arr, 0.0, 1.0)
    a8 = (255.0 * a).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(a8, cv2.COLOR_RGB2BGR))

def make_star_psf(size: int = 200, num_spikes: int = 5, outer_r: int = 70, inner_r: int = 30) -> np.ndarray:
    canvas = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    angles = np.linspace(0, 2 * math.pi, num_spikes * 2, endpoint=False)
    radii = np.empty_like(angles)
    radii[::2] = outer_r
    radii[1::2] = inner_r
    pts = np.stack([
        cx + (radii * np.cos(angles)),
        cy + (radii * np.sin(angles))
    ], axis=1).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    h = canvas.astype(np.float32) / 255.0
    if h.sum() > 0:
        h = h / h.sum()
    return h.astype(np.float32)

def resize_and_normalize_psfs(h: np.ndarray, sizes: List[int]) -> List[np.ndarray]:
    out = []
    for s in sizes:
        resized = cv2.resize(h, (s, s), interpolation=cv2.INTER_AREA)
        if resized.sum() == 0:
            psf = resized.astype(np.float32)
        else:
            psf = (resized / resized.sum()).astype(np.float32)
        out.append(psf)
    return out

def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(img, gamma)

def mirror_pad_2d(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

# Spatial convolution using cv2.filter2D but fliping the kernel for convolution semantics
def spatial_blur_channel(img: np.ndarray, psf: np.ndarray, borderType=cv2.BORDER_REFLECT) -> np.ndarray:
    psf_conv = np.flipud(np.fliplr(psf)).astype(np.float32)
    out = cv2.filter2D(img.astype(np.float32), -1, psf_conv, borderType=borderType)
    out = np.clip(out, 0.0, None)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out

def spatial_blur_color(img_rgb: np.ndarray, psf: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    img_lin = gamma_correction(img_rgb, 1.0 / gamma)
    out = np.zeros_like(img_lin)
    for c in range(3):
        out[..., c] = spatial_blur_channel(img_lin[..., c], psf)
    out = np.clip(out, 0.0, None)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = gamma_correction(out, gamma)
    out = np.clip(out, 0.0, 1.0)
    return out

def frequency_blur_channel(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    h, w = img.shape
    ph, pw = psf.shape
    pad_h = ph // 2
    pad_w = pw // 2
    img_padded = mirror_pad_2d(img, pad_h, pad_w)
    padded_h, padded_w = img_padded.shape
    H_pad = np.zeros((padded_h, padded_w), dtype=np.float32)
    cy = (padded_h - ph) // 2
    cx = (padded_w - pw) // 2
    H_pad[cy:cy + ph, cx:cx + pw] = psf.astype(np.float32)
    H_for_fft = np.fft.ifftshift(H_pad)
    F = np.fft.fft2(img_padded)
    H = np.fft.fft2(H_for_fft)
    G = F * H
    g_padded = np.fft.ifft2(G).real
    g = g_padded[pad_h:pad_h + h, pad_w:pad_w + w]
    g = np.real(g).astype(np.float32)
    g = np.clip(g, 0.0, None)
    g = np.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0)
    return g

def frequency_blur_color(img_rgb: np.ndarray, psf: np.ndarray, gamma: float = 2.2, show_intermediate: bool = False) -> np.ndarray:
    img_lin = gamma_correction(img_rgb, 1.0 / gamma)
    out = np.zeros_like(img_lin)
    for c in range(3):
        out[..., c] = frequency_blur_channel(img_lin[..., c], psf)
    out = np.clip(out, 0.0, None)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = gamma_correction(out, gamma)
    out = np.clip(out, 0.0, 1.0)

    if show_intermediate:
        img_chan = img_lin[..., 0]
        ph, pw = psf.shape
        pad_h, pad_w = ph // 2, pw // 2
        img_padded = mirror_pad_2d(img_chan, pad_h, pad_w)
        padded_h, padded_w = img_padded.shape
        H_pad = np.zeros((padded_h, padded_w), dtype=np.float32)
        cy = (padded_h - ph) // 2
        cx = (padded_w - pw) // 2
        H_pad[cy:cy + ph, cx:cx + pw] = psf
        H_for_fft = np.fft.ifftshift(H_pad)
        F = np.fft.fft2(img_padded)
        H = np.fft.fft2(H_for_fft)
        G = F * H
        g_padded = np.fft.ifft2(G).real
        g = g_padded[pad_h:pad_h + img_chan.shape[0], pad_w:pad_w + img_chan.shape[1]]

        plt.figure(figsize=(15, 8))
        plt.subplot(2, 4, 1); plt.imshow(img_chan, cmap='gray'); plt.title('f: Original (chan 0)'); plt.axis('off')
        plt.subplot(2, 4, 2); plt.imshow(np.log1p(np.abs(np.fft.fftshift(F))), cmap='gray'); plt.title('|F|'); plt.axis('off')
        plt.subplot(2, 4, 3); plt.imshow(np.log1p(np.abs(np.fft.fftshift(H))), cmap='gray'); plt.title('H (fftshift)'); plt.axis('off')
        plt.subplot(2, 4, 4); plt.imshow(np.log1p(np.abs(np.fft.fftshift(G))), cmap='gray'); plt.title('G=F*H'); plt.axis('off')
        plt.subplot(2, 4, 5); plt.imshow(np.clip(g, 0.0, None), cmap='gray'); plt.title('g: After IDFT (clipped)'); plt.axis('off')
        plt.subplot(2, 4, 6); plt.imshow(img_padded, cmap='gray'); plt.title('Padded Input'); plt.axis('off')
        plt.subplot(2, 4, 7); plt.imshow(H_pad, cmap='gray'); plt.title('Positioned PSF'); plt.axis('off')
        plt.tight_layout()
        ensure_dir('./Q2_Outputs/PartC')
        plt.savefig('./Q2_Outputs/PartC/frequency_intermediates.png', bbox_inches='tight')
        plt.show()
        plt.close()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q2')
    parser.add_argument(
        '--photo_path', 
        type=str, 
        default='../Testcases/Q2_SYNTH.png', # Use the original path as the default
        help='Path to the real photograph image'
    )
    args = parser.parse_args()
    OUT_ROOT = './Q2_Outputs'
    PART_A = os.path.join(OUT_ROOT, 'PartA')
    PART_B = os.path.join(OUT_ROOT, 'PartB')
    PART_C = os.path.join(OUT_ROOT, 'PartC')
    PART_D = os.path.join(OUT_ROOT, 'PartD')
    ensure_dir(OUT_ROOT)
    ensure_dir(PART_A)
    ensure_dir(PART_B)
    ensure_dir(PART_C)
    ensure_dir(PART_D)

    # --- Part A ---
    print('====Part-A Processing...=====')
    base_star = make_star_psf(size=200, num_spikes=5, outer_r=70, inner_r=30)
    sizes = [150, 80, 40, 20, 10]
    psfs = resize_and_normalize_psfs(base_star, sizes)

    fig, axes = plt.subplots(1, len(psfs), figsize=(15, 3))
    for i, psf in enumerate(psfs):
        axes[i].imshow(psf, cmap='gray')
        axes[i].set_title(f'PSF {psf.shape[0]}x{psf.shape[1]}')
        axes[i].axis('off')
        save_gray(os.path.join(PART_A, f'psf_{psf.shape[0]}x{psf.shape[1]}.png'), psf)
    plt.tight_layout()
    plt.savefig(os.path.join(PART_A, 'psf_grid.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    # --- Part B ---
    print('====Part-B Processing...=====')
    impulse_img = np.zeros((300, 300), dtype=np.float32)
    impulses = [(75, 75), (150, 200), (220, 100)]
    for (y, x) in impulses:
        impulse_img[y, x] = 1.0
    save_gray(os.path.join(PART_B, 'impulse_original.png'), impulse_img)

    psf_small = psfs[0]
    spatial_impulse = spatial_blur_channel(impulse_img, psf_small)
    save_gray(os.path.join(PART_B, 'impulse_spatial_blur.png'), spatial_impulse)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(impulse_img, cmap='gray'); plt.title('Impulse Image'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(spatial_impulse, cmap='gray'); plt.title('Spatial Blur'); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PART_B, 'impulse_spatial_vs_original.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    # Real photograph
    PHOTO_PATH = args.photo_path 
    photo = cv2.imread(PHOTO_PATH, cv2.IMREAD_COLOR)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h0, w0 = photo.shape[:2]
    if (h0 < 900) or (w0 < 1500):
        # target 1000 x 2000 (height x width) but recall cv2.resize args are (width,height)
        photo = cv2.resize(photo, (2000, 1000), interpolation=cv2.INTER_AREA)
    save_rgb(os.path.join(PART_B, 'photo_original.png'), photo)

    spatial_photo = spatial_blur_color(photo, psf_small, gamma=2.2)
    save_rgb(os.path.join(PART_B, 'photo_spatial_blur.png'), spatial_photo)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(photo); plt.title('Original Photo'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(spatial_photo); plt.title('Spatial Blur'); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PART_B, 'photo_spatial_vs_original.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    # --- Part C ---
    print('====Part-C Processing...=====')
    freq_impulse = frequency_blur_channel(impulse_img, psf_small)
    save_gray(os.path.join(PART_C, 'impulse_frequency_blur.png'), freq_impulse)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(impulse_img, cmap='gray'); plt.title('Impulse'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(spatial_impulse, cmap='gray'); plt.title('Spatial (conv)'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(freq_impulse, cmap='gray'); plt.title('Frequency'); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PART_C, 'impulse_spatial_vs_frequency.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    freq_photo = frequency_blur_color(photo, psf_small, gamma=2.2, show_intermediate=True)
    save_rgb(os.path.join(PART_C, 'photo_frequency_blur.png'), freq_photo)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(spatial_photo); plt.title('Spatial Domain Blur'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(freq_photo); plt.title('Frequency Domain Blur'); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PART_C, 'photo_spatial_vs_frequency.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    # just doing linear-domain diff for diagnostic purposes
    img_lin = gamma_correction(photo, 1.0 / 2.2)
    spatial_no_gamma = np.zeros_like(img_lin)
    freq_no_gamma = np.zeros_like(img_lin)
    for c in range(3):
        spatial_no_gamma[..., c] = spatial_blur_channel(img_lin[..., c], psf_small)
        freq_no_gamma[..., c] = frequency_blur_channel(img_lin[..., c], psf_small)
    diff_no_gamma = np.abs(spatial_no_gamma - freq_no_gamma)
    diff_max = np.max(diff_no_gamma, axis=2)
    save_gray(os.path.join(PART_C, 'photo_linear_diff_max.png'), diff_max)
    print('Photo (linear) - max abs diff:', diff_no_gamma.max(), 'mean:', diff_no_gamma.mean())

    # --- Part D: Timing ---
    print('====Part-D Processing...=====')
    _ = spatial_blur_channel(impulse_img, psf_small)
    _ = frequency_blur_channel(impulse_img, psf_small)

    spatial_times = []
    freq_times = []
    kernel_sizes = [psf.shape[0] for psf in psfs]

    for psf in psfs:
        t0 = time.time()
        _ = spatial_blur_channel(impulse_img, psf)
        spatial_times.append(time.time() - t0)

        t0 = time.time()
        _ = frequency_blur_channel(impulse_img, psf)
        freq_times.append(time.time() - t0)

    plt.figure(figsize=(8, 6))
    plt.loglog(kernel_sizes, spatial_times, 'o-', label='Spatial Domain (conv)')
    plt.loglog(kernel_sizes, freq_times, 's-', label='Frequency Domain (FFT)')
    plt.xlabel('Kernel Size max(m,n)')
    plt.ylabel('Compute Time T(r) [s]')
    plt.title('Compute Time vs Kernel Size (Log-Log)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PART_D, 'timing_loglog.png'), bbox_inches='tight')
    plt.show()
    plt.close()

    np.save(os.path.join(PART_D, 'spatial_times.npy'), np.array(spatial_times))
    np.save(os.path.join(PART_D, 'freq_times.npy'), np.array(freq_times))

    print('All outputs saved under', OUT_ROOT)
