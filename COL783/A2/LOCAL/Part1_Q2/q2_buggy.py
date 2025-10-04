import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# synthetic PSF shape (white circle on black)

# size = 200
# psf_shape = np.zeros((size, size), dtype=np.uint8)
# cv2.circle(psf_shape, (size//2, size//2), 70, 255, -1)
# cv2.imwrite("psf_shape.png", psf_shape)

# h = cv2.imread("psf_shape.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
# h = h / h.max()
import math
size = 200
psf_shape = np.zeros((size, size), dtype=np.uint8)

# star parameters
num_spikes = 5        # number of star points (5 for classic star)
outer_r = 70          # outer radius (distance to spike tips)
inner_r = 30          # inner radius (distance to inner vertices)
cx, cy = size // 2, size // 2

# build alternating outer/inner vertices
angles = np.linspace(0, 2 * math.pi, num_spikes * 2, endpoint=False)
radii = np.empty_like(angles)
radii[::2] = outer_r   # tip
radii[1::2] = inner_r  # inner vertex

pts = np.stack([
    cx + (radii * np.cos(angles)),
    cy + (radii * np.sin(angles))
], axis=1).astype(np.int32)

# OpenCV expects shape (n,1,2) or (n,2) wrapped in a list
cv2.fillPoly(psf_shape, [pts], 255)

cv2.imwrite("psf_shape_star.png", psf_shape)
# if you want it as float like in your original:
h = cv2.imread("psf_shape_star.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
h = h / h.max()


# Resize to multiple scales

sizes = [150, 80, 40, 20, 10]
psfs = []
for s in sizes:
    resized = cv2.resize(h, (s, s), interpolation=cv2.INTER_AREA)
    psf = resized / np.sum(resized)
    psfs.append(psf)

# Visualize PSFs
fig, axes = plt.subplots(1, len(psfs), figsize=(15, 3))
for i, psf in enumerate(psfs):
    axes[i].imshow(psf, cmap="gray")
    axes[i].set_title(f"PSF {psf.shape[0]}x{psf.shape[1]}")
    axes[i].axis("off")
plt.show()

#  Gamma correction

def gamma_correction(img, gamma):
    return np.power(img, gamma)

# Mirror padding function

def mirror_pad_2d(img, pad_h, pad_w):
    """Apply mirror padding to a 2D image."""
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')


# Blurring in spatial domain (Part b)

psf_small = psfs[0]  # use the smallest PSF (10x10)

# (i) Impulse image ----
impulse_img = np.zeros((300, 300), dtype=np.float32)
impulse_img[75, 75] = 1.0
impulse_img[150, 200] = 1.0
impulse_img[220, 100] = 1.0

blurred_impulse = cv2.filter2D(impulse_img, -1, psf_small, borderType=cv2.BORDER_REFLECT)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(impulse_img, cmap="gray"); plt.title("Impulse Image"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(blurred_impulse, cmap="gray"); plt.title("Spatial Blur"); plt.axis("off")
plt.show()

#(ii) Real photograph
photo = cv2.imread("../Testcases/cameraman.png", cv2.IMREAD_COLOR)
if photo is None:
    raise FileNotFoundError("Please provide a test image as 'photo.jpg' (≈1000x2000).")
# photo = cv2.resize(photo, (1000, 2000)) use when given image is not of required size 
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

gamma = 2.2
photo_gamma = gamma_correction(photo, 1/gamma)

blurred_photo = np.zeros_like(photo_gamma)
for c in range(3):
    blurred_photo[..., c] = cv2.filter2D(photo_gamma[..., c], -1, psf_small, borderType=cv2.BORDER_REFLECT)

blurred_photo = gamma_correction(blurred_photo, gamma)
blurred_photo = np.clip(blurred_photo, 0, 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(photo); plt.title("Original Photo"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(blurred_photo); plt.title("Spatial Blur"); plt.axis("off")
plt.show()


# 4. Frequency-domain blurring (Part c) 

def frequency_blur_channel_exact(img, psf):
   
    h, w = img.shape
    ph, pw = psf.shape
    
   
    pad_h = ph // 2
    pad_w = pw // 2
    
   
    img_padded = mirror_pad_2d(img, pad_h, pad_w)
    padded_h, padded_w = img_padded.shape
    
    # Create PSF array for frequency domain, centered and zero-padded
    H_pad = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    # Place PSF at top-left corner first
    H_pad[:ph, :pw] = psf
    
    # Shift PSF to be centered for FFT 
    H_pad = np.roll(H_pad, -ph//2, axis=0)
    H_pad = np.roll(H_pad, -pw//2, axis=1)
    
   
    F = np.fft.fft2(img_padded)
    H = np.fft.fft2(H_pad)
    G = F * H
    g_padded = np.fft.ifft2(G).real
    
    # Crop back to original size (remove padding)
    g = g_padded[pad_h:pad_h+h, pad_w:pad_w+w]
    
    return g

def frequency_blur_color_exact(img_rgb, psf, gamma=2.2, show_intermediate=False):
   
    # Gamma pre-correction
    img_gamma = gamma_correction(img_rgb, 1/gamma)

    blurred = np.zeros_like(img_gamma)
    for c in range(3):
        blurred[..., c] = frequency_blur_channel_exact(img_gamma[..., c], psf)

    # Undo gamma
    blurred = gamma_correction(blurred, gamma)
    blurred = np.clip(blurred, 0, 1)

    if show_intermediate:
        # Show intermediate spectra for the first channel only
        img_chan = img_gamma[..., 0]
        h, w = img_chan.shape
        ph, pw = psf.shape
        pad_h, pad_w = ph // 2, pw // 2
        
        img_padded = mirror_pad_2d(img_chan, pad_h, pad_w)
        padded_h, padded_w = img_padded.shape
        
        F = np.fft.fft2(img_padded)
        
        H_pad = np.zeros((padded_h, padded_w), dtype=np.float32)
        H_pad[:ph, :pw] = psf
        H_pad = np.roll(H_pad, -ph//2, axis=0)
        H_pad = np.roll(H_pad, -pw//2, axis=1)
        
        H = np.fft.fft2(H_pad)
        G = F * H
        g_padded = np.fft.ifft2(G).real
        g = g_padded[pad_h:pad_h+h, pad_w:pad_w+w]

        def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 4, 1); plt.imshow(img_chan, cmap="gray"); plt.title("f: Original (channel 0)"); plt.axis("off")
        plt.subplot(2, 4, 2); plt.imshow(norm(np.log1p(np.abs(np.fft.fftshift(F)))), cmap="gray"); plt.title("DFT |F|"); plt.axis("off")
        plt.subplot(2, 4, 3); plt.imshow(norm(np.log1p(np.abs(np.fft.fftshift(H)))), cmap="gray"); plt.title("Filter H"); plt.axis("off")
        plt.subplot(2, 4, 4); plt.imshow(norm(np.log1p(np.abs(np.fft.fftshift(G)))), cmap="gray"); plt.title("Product G=FH"); plt.axis("off")
        plt.subplot(2, 4, 5); plt.imshow(norm(g), cmap="gray"); plt.title("g: After IDFT"); plt.axis("off")
        plt.subplot(2, 4, 6); plt.imshow(img_padded, cmap="gray"); plt.title("Padded Input"); plt.axis("off")
        plt.subplot(2, 4, 7); plt.imshow(H_pad, cmap="gray"); plt.title("Positioned PSF"); plt.axis("off")
        plt.tight_layout(); plt.show()

    return blurred

#Test frequency domain on impulse image
freq_blurred_impulse = frequency_blur_channel_exact(impulse_img, psf_small)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(blurred_impulse, cmap="gray"); plt.title("Spatial Blur"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(freq_blurred_impulse, cmap="gray"); plt.title("Frequency Blur"); plt.axis("off")
plt.show()

#Test frequency domain on color photo 
freq_blurred_photo_color = frequency_blur_color_exact(photo, psf_small, gamma=2.2, show_intermediate=True)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(blurred_photo); plt.title("Spatial Domain Blur"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(freq_blurred_photo_color); plt.title("Frequency Domain Blur"); plt.axis("off")
plt.show()


#Quantitative difference 

diff = np.abs(blurred_photo - freq_blurred_photo_color)
print("=== With Gamma Correction ===")
print("Max absolute difference:", diff.max())
print("Mean absolute difference:", diff.mean())

# Calculate difference without gamma correction
photo_gamma = gamma_correction(photo, 1/gamma)
spatial_no_gamma = np.zeros_like(photo_gamma)
for c in range(3):
    spatial_no_gamma[..., c] = cv2.filter2D(photo_gamma[..., c], -1, psf_small, borderType=cv2.BORDER_REFLECT)

freq_no_gamma = np.zeros_like(photo_gamma)
for c in range(3):
    freq_no_gamma[..., c] = frequency_blur_channel_exact(photo_gamma[..., c], psf_small)

diff_no_gamma = np.abs(spatial_no_gamma - freq_no_gamma)
print("\n=== Without Gamma Correction (Linear Space) ===")
print("Max absolute difference:", diff_no_gamma.max())
print("Mean absolute difference:", diff_no_gamma.mean())

# Test on impulse image for exact comparison
impulse_freq = frequency_blur_channel_exact(impulse_img, psf_small)
impulse_diff = np.abs(blurred_impulse - impulse_freq)
print("\n=== Impulse Image Test (Should be ~1e-15) ===")
print("Max absolute difference:", impulse_diff.max())
print("Mean absolute difference:", impulse_diff.mean())

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(diff * 1000, cmap="hot")
plt.title("Difference with Gamma (scaled x1000)")
plt.axis("off")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(diff_no_gamma * 1000, cmap="hot")
plt.title("Difference without Gamma (scaled x1000)")
plt.axis("off")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(impulse_diff * 1e15, cmap="hot")
plt.title("Impulse Difference (scaled x1e15)")
plt.axis("off")
plt.colorbar()
plt.show()




# 6. Timing comparison

spatial_times = []
freq_times = []

kernel_sizes = [psf.shape[0] for psf in psfs]  # max(m,n) for each PSF


for psf in psfs:
    # Spatial domain timing
    start = time.time()
    _ = cv2.filter2D(impulse_img, -1, psf, borderType=cv2.BORDER_REFLECT)
    spatial_times.append(time.time() - start)
    
    # Frequency domain timing
    start = time.time()
    _ = frequency_blur_channel_exact(impulse_img, psf)
    freq_times.append(time.time() - start)

# Log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(kernel_sizes, spatial_times, 'o-', label="Spatial Domain (cv2.filter2D)")
plt.loglog(kernel_sizes, freq_times, 's-', label="Frequency Domain (FFT)")
plt.xlabel("Kernel Size max(m,n)")
plt.ylabel("Compute Time T(r) [s]")
plt.title("Compute Time vs Kernel Size (Log-Log)")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
