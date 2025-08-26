import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def disk_kernel(radius):
    r = int(radius)
    diameter = 2 * r + 1
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    mask = (xx**2 + yy**2) <= (radius + 1e-9)**2
    k = mask.astype(np.float32)
    s = k.sum()
    if s == 0:
        raise ValueError("Radius too small produced empty kernel")
    return k / float(s)

# Gamma transform on linear image
def gamma_transform_linear(img, gamma):
    img = np.asarray(img, dtype=np.float32)
    maxval = img.max()
    if maxval <= 0:
        return img.copy()
    norm = img / maxval
    g = np.power(np.clip(norm, 0, None), gamma) # gamma < 1 brightens darks and compresses highlights in normalized domain
    return g # returned as normalized in [0,1]

def try_load_hdr(path):
    try:
        import imageio.v3 as iio3
        img = iio3.imread(path, format_hint='.hdr')
        return np.asarray(img, dtype=np.float32)
    except Exception:
        pass
    try:
        import imageio
        img = imageio.imread(path, format='HDR-FI')
        return np.asarray(img, dtype=np.float32)
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(path, -1)
        if img is not None:
            if img.ndim==3 and img.shape[2]==3:
                img = img[..., ::-1]
            return img.astype(np.float32)
    except Exception:
        pass
    return None

def rescale_to_uint8(arr):
    mn = float(arr.min())
    mx = float(arr.max())
    if mx == mn:
        return np.clip(np.ones_like(arr) * 128, 0, 255).astype(np.uint8)
    scaled = (arr - mn) / (mx - mn) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def convolve_same(image, kernel):
    return fftconvolve(image, kernel, mode='same')

if __name__ == "__main__":
	hdr_path = "rosette.hdr"
	out_dir = "./Q6_OUTPUTS"
	gamma = 0.3
	radius = 8 
	eps = 1e-8 
	os.makedirs(out_dir, exist_ok=True)

	img = try_load_hdr(hdr_path)
	print(f"Loaded HDR file: {hdr_path}")
	img = img.astype(np.float32)
	H, W = img.shape[:2]
	print("Image shape:", img.shape, "min/max:", img.min(), img.max())
	if img.ndim == 3 and img.shape[2] >= 3:
		I = img[..., :3].mean(axis=2)
	else:
		I = img.copy() if img.ndim == 2 else img[..., 0]
	print("Intensity range: min/max:", I.min(), I.max())

	kernel = disk_kernel(radius)
	print("Kernel shape:", kernel.shape, "sum:", kernel.sum())

	I_gamma_norm = gamma_transform_linear(I, gamma)
	A_int = convolve_same(I_gamma_norm, kernel)

	B_blur = convolve_same(I, kernel)
	B_int = gamma_transform_linear(B_blur, gamma) 

	I_safe = I + eps
	orig_max = I.max() if I.max() > 0 else 1.0
	A_int_linear_like = A_int * orig_max
	B_int_linear_like = B_int * (B_blur.max() if B_blur.max() > 0 else orig_max)

	scale_A = (A_int_linear_like / I_safe)[..., None]
	scale_B = (B_int_linear_like / I_safe)[..., None]

	color_A = img * scale_A
	color_B = img * scale_B

	# Prepare displayable uint8 images
	orig_disp = rescale_to_uint8(img / (img.max() if img.max() > 0 else 1.0))
	A_int_disp = rescale_to_uint8(A_int)
	B_int_disp = rescale_to_uint8(B_int)
	color_A_disp = rescale_to_uint8(color_A)
	color_B_disp = rescale_to_uint8(color_B)
	diff_int_disp = rescale_to_uint8(np.abs(A_int - B_int))
	diff_color_disp = rescale_to_uint8(np.abs(color_A_disp.astype(np.int16) - color_B_disp.astype(np.int16)))

	paths = {
		"orig": os.path.join(out_dir, "orig_rescaled.png"),
		"I_gamma_then_blur_gray": os.path.join(out_dir, "I_gamma_then_blur_gray.png"),
		"I_blur_then_gamma_gray": os.path.join(out_dir, "I_blur_then_gamma_gray.png"),
		"color_A": os.path.join(out_dir, "color_gamma_then_blur.png"),
		"color_B": os.path.join(out_dir, "color_blur_then_gamma.png"),
		"diff_int": os.path.join(out_dir, "diff_intensity.png"),
		"diff_color": os.path.join(out_dir, "diff_color.png")
	}

	plt.imsave(paths["orig"], orig_disp)
	plt.imsave(paths["I_gamma_then_blur_gray"], A_int_disp, cmap='gray')
	plt.imsave(paths["I_blur_then_gamma_gray"], B_int_disp, cmap='gray')
	plt.imsave(paths["color_A"], color_A_disp)
	plt.imsave(paths["color_B"], color_B_disp)
	plt.imsave(paths["diff_int"], diff_int_disp, cmap='gray')
	plt.imsave(paths["diff_color"], diff_color_disp)

	fig, axs = plt.subplots(2, 4, figsize=(18, 9))
	axs = axs.ravel()
	axs[0].imshow(orig_disp); axs[0].set_title("Original (rescaled)"); axs[0].axis('off')
	axs[1].imshow(A_int_disp, cmap='gray'); axs[1].set_title("I: Gamma then Blur (gray)"); axs[1].axis('off')
	axs[2].imshow(B_int_disp, cmap='gray'); axs[2].set_title("I: Blur then Gamma (gray)"); axs[2].axis('off')
	axs[3].imshow(diff_int_disp, cmap='gray'); axs[3].set_title("Abs diff (int)"); axs[3].axis('off')
	axs[4].imshow(color_A_disp); axs[4].set_title("Color reconstr: Gamma then Blur"); axs[4].axis('off')
	axs[5].imshow(color_B_disp); axs[5].set_title("Color reconstr: Blur then Gamma"); axs[5].axis('off')
	axs[6].imshow(diff_color_disp, cmap='gray'); axs[6].set_title("Abs diff (color uint8)"); axs[6].axis('off')
	axs[7].axis('off')
	plt.tight_layout()
	plt.show()

	def stats(name, arr):
		print(f"{name}: min={float(arr.min()):.4f}, max={float(arr.max()):.4f}, mean={float(arr.mean()):.4f}, std={float(arr.std()):.4f}")

	stats("I (orig)", I)
	stats("A_int (gamma then blur)", A_int)
	stats("B_int (blur then gamma)", B_int)

	print("\nSaved files:")
	for k, v in paths.items():
		print(f" - {k}: {v}")