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

def displayHSI(img_path):
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
	# recheck_rgb = hsi_to_rgb(hsi)
	# plt.imshow(recheck_rgb)
	# plt.axis('off')
	# plt.suptitle("subtitle")
	# plt.show()

def getAndProcessMasks(img_path, automed=True):
	img = load_image(img_path)
	hsi = rgb_to_hsi(img)
	H = hsi[...,0]# [0,1]
	S = hsi[...,1]
	I = hsi[...,2]
	h, w = S.shape
	if automed:
		# Automatically pick a seed pixel:
		# Strategy --> find pixel with high saturation and mid-range intensity (not too bright/dark)
		sat_score = S * (1.0 - np.abs(I - 0.5))  # prefer saturated, mid-intensity
		flat_idx = np.argmax(sat_score)
		ys, xs = divmod(flat_idx, w)
		# seed_coord = (ys, xs)
		seed_coord = (int(ys), int(xs))
	else:
		# interactive seed selection
		plt.figure(figsize=(6, 6))
		plt.imshow(img)
		plt.title("Click one point to choose seed pixel (close window when done)")
		plt.axis('off')
		try:
			pts = plt.ginput(1, timeout=-1) # wait until click
		except Exception as e:
			pts = []
		plt.close()
		if len(pts) == 0:
			sat_score = S * (1.0 - np.abs(I - 0.5))
			flat_idx = np.argmax(sat_score)
			ys, xs = divmod(flat_idx, w)
			# seed_coord = (ys, xs)
			seed_coord = (int(ys), int(xs))
		else:
			x_float, y_float = pts[0]
			xs = int(round(x_float))
			ys = int(round(y_float))
			xs = int(np.clip(xs, 0, w - 1))# clamp into image bounds
			ys = int(np.clip(ys, 0, h - 1))
			seed_coord = (ys, xs)
	# seed_rgb = img[ys, xs, :]
	# seed_hsi = hsi[ys, xs, :]
	seed_rgb = img[seed_coord[0], seed_coord[1], :]
	seed_hsi = hsi[seed_coord[0], seed_coord[1], :]
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

	def transform_hsi(hsi_img, mask, seed_hsi, target_hsi, method="hybrid"):
		# method options:
		# "additive": additive on all components: c + (ct - cs)
		# "hybrid": hue additive (circular), saturation multiplicative, intensity multiplicative
		out = hsi_img.copy()
		H = out[...,0]; S = out[...,1]; I_ = out[...,2]
		delta_H = target_hsi[0] - seed_hsi[0]
		delta_H = (delta_H + 0.5) % 1.0 - 0.5 # wrap to [-0.5,0.5] for minimal shift
		if method == "additive":
			H_new = (H + delta_H) % 1.0
			S_new = np.clip(S + (target_hsi[1] - seed_hsi[1]), 0, 1)
			I_new = np.clip(I_ + (target_hsi[2] - seed_hsi[2]), 0, 1)
		elif method == "hybrid":
			H_new = (H + delta_H) % 1.0
			S_new = S * (target_hsi[1] / (seed_hsi[1] + 1e-8))
			I_new = I_ * (target_hsi[2] / (seed_hsi[2] + 1e-8))
			S_new = np.clip(S_new, 0, 1)
			I_new = np.clip(I_new, 0, 1)
		else:
			raise ValueError("unknown method")
		out[...,0][mask] = H_new[mask]
		out[...,1][mask] = S_new[mask]
		out[...,2][mask] = I_new[mask]
		return out
	
	targets = {
		"target_red": np.array([0.0, 0.9, 0.6]),
		"target_green": np.array([1.0/3.0, 0.9, 0.6]),
		"target_blue": np.array([2.0/3.0, 0.9, 0.5])
	}

	plt.figure(figsize=(12, 8))
	n_targets = len(targets)
	for i, (name, tgt) in enumerate(targets.items()):
		for j, method in enumerate(["additive", "hybrid"]):
			ax_index = i * 2 + j + 1    
			plt.subplot(n_targets, 2, ax_index) 
			modified_hsi = transform_hsi(hsi.copy(), mask_hsi, seed_hsi, tgt, method=method)
			modified_rgb = hsi_to_rgb(modified_hsi)
			plt.imshow(modified_rgb)
			plt.title(f"{name} - {method}")
			plt.axis('off')

	plt.suptitle("Recoloring results (using HSI cuboid mask). Rows = targets, Columns = method", y=0.92)
	plt.tight_layout(rect=[0, 0, 1, 0.93])
	plt.show()

	plt.figure(figsize=(12, 4))
	n_targets = len(targets)
	for i, (name, tgt) in enumerate(targets.items()):
		plt.subplot(1, n_targets, i + 1) 
		modified_hsi = transform_hsi(hsi.copy(), mask_rgb, seed_hsi, tgt, method="hybrid")
		modified_rgb = hsi_to_rgb(modified_hsi)
		plt.imshow(modified_rgb)
		plt.title(f"{name} (RGB mask)")
		plt.axis('off')

	plt.suptitle("Recoloring results using RGB cuboid mask (hybrid transform)", y=0.95)
	plt.tight_layout(rect=[0, 0, 1, 0.93])
	plt.show()

	out_img = hsi_to_rgb(transform_hsi(hsi.copy(), mask_hsi, seed_hsi, targets["target_green"], method="hybrid"))
	out_pil = Image.fromarray((np.clip(out_img,0,1)*255).astype(np.uint8))
	out_path = "./recolored_example_green_hsi_mask_hybrid.png"
	out_pil.save(out_path)
	print("Wrote example recolored image to:", out_path)

if __name__ == "__main__":
	#displayHSI('./FruitBowl.jpg')#part1 
	getAndProcessMasks('./FruitBowl.jpg',automed=False)#part2 and part3