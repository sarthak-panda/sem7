from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

def load_image(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def rgb_to_hsi(rgb):
    R = rgb[...,0]
    G = rgb[...,1]
    B = rgb[...,2]
    eps = 1e-8
    I = (R + G + B) / 3.0
    sum_rgb = R + G + B
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.zeros_like(I)
    mask = sum_rgb > eps
    S[mask] = 1 - (3.0 * min_rgb[mask] / (sum_rgb[mask] + eps))
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + eps
    cos_theta = np.clip(num / den, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    H = np.zeros_like(theta)
    H = np.where(B <= G, theta, 2*np.pi - theta)
    H = H / (2*np.pi)
    return np.stack([H, S, I], axis=-1)

def hsi_to_rgb(hsi):
    H = hsi[...,0] * 2 * np.pi
    S = hsi[...,1]
    I = hsi[...,2]
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    eps = 1e-8
    sector1 = (H >= 0) & (H < 2*np.pi/3)
    sector2 = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    sector3 = (H >= 4*np.pi/3) & (H < 2*np.pi)
    h = H
    idx = sector1
    R[idx] = I[idx] * (1 + S[idx] * np.cos(h[idx]) / (np.cos(np.pi/3 - h[idx]) + eps))
    B[idx] = I[idx] * (1 - S[idx])
    G[idx] = 3*I[idx] - (R[idx] + B[idx])
    idx = sector2
    h2 = H[idx] - 2*np.pi/3
    G[idx] = I[idx] * (1 + S[idx] * np.cos(h2) / (np.cos(np.pi/3 - h2) + eps))
    R[idx] = I[idx] * (1 - S[idx])
    B[idx] = 3*I[idx] - (R[idx] + G[idx])
    idx = sector3
    h3 = H[idx] - 4*np.pi/3
    B[idx] = I[idx] * (1 + S[idx] * np.cos(h3) / (np.cos(np.pi/3 - h3) + eps))
    G[idx] = I[idx] * (1 - S[idx])
    R[idx] = 3*I[idx] - (G[idx] + B[idx])
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

def displayHSI():
	img_path = "/mnt/data/1bae8737-699f-42f8-a48e-00b60fc53966.png"
	img = load_image(img_path)
	hsi = rgb_to_hsi(img)
	H = hsi[...,0]# [0,1]
	S = hsi[...,1]
	I = hsi[...,2]
	# Display H, S, I channels as intensity images
	plt.figure(figsize=(12,4))
	plt.subplot(1,3,1); plt.imshow((H*255).astype('uint8'), cmap='gray'); plt.title("Hue (normalized)"); plt.axis('off')
	plt.subplot(1,3,2); plt.imshow((S*255).astype('uint8'), cmap='gray'); plt.title("Saturation"); plt.axis('off')
	plt.subplot(1,3,3); plt.imshow((I*255).astype('uint8'), cmap='gray'); plt.title("Intensity"); plt.axis('off')
	plt.show()

def getMasks():
	img_path = "/mnt/data/1bae8737-699f-42f8-a48e-00b60fc53966.png"
	img = load_image(img_path)
	hsi = rgb_to_hsi(img)
	H = hsi[...,0]# [0,1]
	S = hsi[...,1]
	I = hsi[...,2]
    # Automatically pick a seed pixel:
	# Strategy: find pixel with high saturation and mid-range intensity (not too bright/dark)
	sat_score = S * (1.0 - np.abs(I - 0.5))  # prefer saturated, mid-intensity
	h, w = S.shape
	flat_idx = np.argmax(sat_score)
	ys, xs = divmod(flat_idx, w)
	seed_coord = (ys, xs)
	seed_rgb = img[ys, xs, :]
	seed_hsi = hsi[ys, xs, :]
	print("Chosen seed pixel at (y,x) =", seed_coord)
	print("Seed RGB (0-1):", seed_rgb)
	print("Seed HSI (H in [0,1]):", seed_hsi)
	# Create masks by color slicing using cuboid in RGB and HSI
	def cuboid_mask_rgb(image_rgb, seed_rgb, deltas=(0.15,0.15,0.15)):
		low = seed_rgb - np.array(deltas)
		high = seed_rgb + np.array(deltas)
		low = np.clip(low, 0, 1)
		high = np.clip(high, 0, 1)
		within = np.logical_and(image_rgb >= low, image_rgb <= high)
		mask = np.all(within, axis=-1)
		return mask
	mask_rgb = cuboid_mask_rgb(img, seed_rgb, deltas=(0.15,0.15,0.15))
	def cuboid_mask_hsi(image_hsi, seed_hsi, deltas=(0.06,0.20,0.20)):
		# deltas: (dh, ds, di) for normalized H in [0,1]
		H = image_hsi[...,0]; S = image_hsi[...,1]; I_ = image_hsi[...,2]
		dh, ds, di = deltas
		# hue is circular; compute min circular difference
		diff_h = np.abs(H - seed_hsi[0])
		diff_h = np.minimum(diff_h, 1.0 - diff_h)
		mask_h = diff_h <= dh
		mask_s = np.abs(S - seed_hsi[1]) <= ds
		mask_i = np.abs(I_ - seed_hsi[2]) <= di
		mask = mask_h & mask_s & mask_i
		return mask
	mask_hsi = cuboid_mask_hsi(hsi, seed_hsi, deltas=(0.06,0.20,0.20))
	# Show masks
	plt.figure(figsize=(10,4))
	plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis('off')
	plt.subplot(1,3,2); plt.imshow((mask_rgb*255).astype('uint8'), cmap='gray'); plt.title("Mask (RGB cuboid)"); plt.axis('off')
	plt.subplot(1,3,3); plt.imshow((mask_hsi*255).astype('uint8'), cmap='gray'); plt.title("Mask (HSI cuboid)"); plt.axis('off')
	plt.show()