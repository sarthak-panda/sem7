from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

def showStats(img):
	print("type:",type(img))
	print("dtype:", img.dtype)
	print("shape (H, W, C):", img.shape)
	print("number of dimensions (ndim):", img.ndim)
	print("total elements (arr.size):", img.size)   # H * W * C
	print("memory (bytes):", img.nbytes)

def displayColorImage(img,file_name):
	plt.figure(figsize=(8,6))
	plt.imshow(img)           # skimage -> RGB, so show directly
	plt.axis('off')
	plt.title(file_name)
	plt.show()

def displayIntensityImage(img,file_name):
	plt.figure(figsize=(8,6))
	plt.imshow(img, cmap='gray', vmin=0, vmax=255)
	plt.axis('off')
	plt.title(file_name)
	plt.show()

def convertToIntensity(img):
	if img.ndim == 3 and img.shape[2] > 3:
		img_rgb = img[..., :3]	# img[..., :3] is equivalent to img[:, :, :3] for a 3-D array
	else:
		img_rgb = img
	intensity = img_rgb.astype(np.float32).mean(axis=2)	# axis (0=height, 1=width, 2=channels)
	info = np.iinfo(img_rgb.dtype)
	intensity_out = np.clip(np.round(intensity), info.min, info.max).astype(img_rgb.dtype)
	return intensity_out

def collectUserInput(img):
	fig, ax = plt.subplots(figsize=(8,6))
	ax.imshow(img, cmap='gray', vmin=0, vmax=255)
	ax.set_title("Click the 4 page corners")
	ax.axis('off')
	points = plt.ginput(4)
	corners = [[float(x), float(y)] for (x, y) in points]
	plt.close()
	return corners

def order_corners_clockwise(pts):
    pts = np.array(pts, dtype=float)
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
    order = np.argsort(angles)  # ascending angle -> CCW
    pts_sorted = pts[order]
    return [(float(x), float(y)) for x,y in pts_sorted]

def compute_vertical_angle_from_corners(corners):
    # corners: list of 4 (x,y) points in any order
	pts = np.array(corners, dtype=float)
	ys = pts[:,1]
	top_idx = np.argsort(ys)[:2]
	bot_idx = np.argsort(ys)[-2:]
	top_center = pts[top_idx].mean(axis=0)
	bot_center = pts[bot_idx].mean(axis=0)
	# vector pointing roughly down the page
	v = bot_center - top_center
	vx, vy = v[0], v[1]
	theta = math.atan2(vx, vy)
	return theta, top_center, bot_center

def rotation_matrix(theta, cx, cy):
	c = math.cos(theta)
	s = math.sin(theta)
	# Rotation matrix about origin:
	R = np.array([[c, -s, 0],
					[s, c, 0],
					[0, 0, 1]], dtype=float)
	# But we want rotation about (cx,cy):
	T1 = np.array([[1,0,-cx],
					[0,1,-cy],
					[0,0,1]], dtype=float)
	T2 = np.array([[1,0,cx],
					[0,1,cy],
					[0,0,1]], dtype=float)
	M = T2 @ R @ T1
	return M

def bilinear_interpolate(img, x, y):
    h, w = img.shape
    if x < 0 or x >= w-1 or y < 0 or y >= h-1:
        return 0  # background
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = x - x0
    dy = y - y0
    I00 = float(img[y0, x0])
    I10 = float(img[y0, x0+1])
    I01 = float(img[y0+1, x0])
    I11 = float(img[y0+1, x0+1])
    I = (1-dx)*(1-dy)*I00 + dx*(1-dy)*I10 + (1-dx)*dy*I01 + dx*dy*I11
    return I

def nearest_interpolate(img, x, y):
    h, w = img.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return 0
    return float(img[yi, xi])

def bakward_transformation(img, theta, interp='bilinear'):
	h, w = img.shape
	cx, cy = (w-1)/2.0, (h-1)/2.0
	M = rotation_matrix(theta, cx, cy)
	Minv = np.linalg.inv(M)
	corners = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]], dtype=float).T
	rotated_corners = (M @ corners)[:2,:].T
	xs = rotated_corners[:,0]; ys = rotated_corners[:,1]
	min_x, max_x = xs.min(), xs.max()
	min_y, max_y = ys.min(), ys.max()
	out_w = int(math.ceil(max_x - min_x + 1))
	out_h = int(math.ceil(max_y - min_y + 1))
	x_offset = min_x
	y_offset = min_y
	out = np.zeros((out_h, out_w), dtype=np.float32)
	for yo in range(out_h):
		for xo in range(out_w):
			xr = x_offset + xo
			yr = y_offset + yo
			v_out = np.array([xr, yr, 1.0])
			vin = Minv @ v_out
			xi, yi = vin[0], vin[1]
			if interp == 'nearest':
				val = nearest_interpolate(img, xi, yi)
			else:
				val = bilinear_interpolate(img, xi, yi)
			out[yo, xo] = val
	out_dtype = img.dtype
	info = np.iinfo(out_dtype)
	out = np.clip(np.round(out), info.min, info.max).astype(out_dtype)
	return out, M, (x_offset, y_offset)

def rotate_and_crop_document(img, corners, interp='bilinear'):
	theta_align, top_c, bot_c = compute_vertical_angle_from_corners(corners)
	theta_apply = theta_align
	rotated_img, M, (x_off, y_off) = bakward_transformation(img, theta_apply, interp=interp)
	pts = np.array([[x,y,1.0] for (x,y) in corners]).T
	rotated_pts = (M @ pts)[:2,:].T
	min_x = math.floor(rotated_pts[:,0].min())
	max_x = math.ceil(rotated_pts[:,0].max())
	min_y = math.floor(rotated_pts[:,1].min())
	max_y = math.ceil(rotated_pts[:,1].max())
	ox = int(round(min_x - x_off))
	oy = int(round(min_y - y_off))
	ow = int(round(max_x - min_x + 1))
	oh = int(round(max_y - min_y + 1))
	ro_h, ro_w = rotated_img.shape
	x0 = max(0, ox); y0 = max(0, oy)
	x1 = min(ro_w, ox + ow); y1 = min(ro_h, oy + oh)
	cropped = rotated_img[y0:y1, x0:x1].copy()
	return cropped, rotated_img, rotated_pts, theta_apply

def show_results(original, cropped, rotated_full):
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1); plt.title("Original"); plt.imshow(original, cmap='gray', vmin=0, vmax=255); plt.axis('off')
    plt.subplot(1,3,2); plt.title("Rotated Full"); plt.imshow(rotated_full, cmap='gray', vmin=0, vmax=255); plt.axis('off')
    plt.subplot(1,3,3); plt.title("Cropped Document"); plt.imshow(cropped, cmap='gray', vmin=0, vmax=255); plt.axis('off')
    plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Document scanner")
	parser.add_argument("input_file", help="Input image file path")
	parser.add_argument("output_file", help="Output image file path")
	parser.add_argument("interp", choices=["nearest", "bilinear"], help="Interpolation type")
	args = parser.parse_args()
	file_name = args.input_file
	out_name = args.output_file
	interp_type = args.interp

	img = io.imread(file_name) # image stored as (column index, row index) ie (x,y)
	img = convertToIntensity(img)
	corners = collectUserInput(img)
	print(corners)
	corners_ordered = order_corners_clockwise(corners)
	cropped, rotated_full, rotated_pts, theta_used = rotate_and_crop_document(img, corners_ordered, interp=interp_type)
	print("rotation (radians) applied:", theta_used, "-> degrees:", math.degrees(theta_used))
	show_results(img, cropped, rotated_full)
	io.imsave(out_name, cropped)

#python .\p1_q1.py .\testIMG2.jpg t2.jpg bilinear
#python .\p1_q1.py .\testIMG2.jpg t2_n.jpg nearest
#1137
#1217
#4
#23