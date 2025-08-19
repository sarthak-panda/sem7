from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

def displayIntensityImage(img,file_name):
	plt.figure(figsize=(8,6))
	plt.imshow(img, cmap='gray', vmin=0, vmax=255)
	plt.axis('off')
	plt.title(file_name)
	plt.show()

def displayColorImage(img,file_name):
	plt.figure(figsize=(8,6))
	plt.imshow(img)           # skimage -> RGB, so show directly
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

def get_mask_by_threshold(intensity_img):
    if intensity_img.ndim != 2:
        raise ValueError("threshold_image_iterative expects a 2D intensity image")
    h, w = intensity_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    max_i = intensity_img.max()
    for i in range(h):
        inp_row = intensity_img[i]
        out_row = mask[i]
        for j in range(w):
            if inp_row[j] < max_i:
                #add to mask, set mask bit 1
                out_row[j]=1
    return mask

def get_mask_by_threshold_vectorized(intensity_img):
    # Fast vectorized version
    max_i = intensity_img.max()
    mask = (intensity_img < max_i).astype(np.uint8)
    return mask

def applyMask(img,mask):
	m = mask.astype(bool)
	rgb = img[..., :3]
	alpha = img[..., 3:]
	masked_rgb = np.where(m[..., None], rgb, 0)
	return np.concatenate([masked_rgb, alpha], axis=2).astype(img.dtype)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Document scanner")
	parser.add_argument("input_file", help="Input image file path")
	parser.add_argument("output_file", help="Output image file path")

	args = parser.parse_args()
	file_name = args.input_file
	out_name = args.output_file

	img = io.imread(file_name) # image stored as (column index, row index) ie (x,y)
	intensity_img = convertToIntensity(img)
    
	mask = get_mask_by_threshold_vectorized(intensity_img)
	op_img=applyMask(img,mask)
    
	displayColorImage(op_img,out_name)