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

def convertToIntensity(img):
	if img.ndim == 3 and img.shape[2] > 3:
		img_rgb = img[..., :3]	# img[..., :3] is equivalent to img[:, :, :3] for a 3-D array
	else:
		img_rgb = img
	intensity = img_rgb.astype(np.float32).mean(axis=2)	# axis (0=height, 1=width, 2=channels)
	info = np.iinfo(img_rgb.dtype)
	intensity_out = np.clip(np.round(intensity), info.min, info.max).astype(img_rgb.dtype)
	return intensity_out

def threshold_image_iterative(intensity_img, thresh):
    if intensity_img.ndim != 2:
        raise ValueError("threshold_image_iterative expects a 2D intensity image")
    h, w = intensity_img.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        inp_row = intensity_img[i]
        out_row = out[i]
        for j in range(w):
            if inp_row[j] < thresh:
                out_row[j] = 0
            else:
                out_row[j] = 255
    return out

def threshold_image_vectorized(intensity_img, thresh):
    # Fast vectorized version
    out = np.where(intensity_img < thresh, 0, 255).astype(np.uint8)
    return out

def display_side_by_side(original, processed, original_title="Original", processed_title="Processed"):
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(original_title)
    ax1.axis('off')
    
    ax2 = plt.subplot(1,2,2)
    ax2.imshow(processed, cmap='gray', vmin=0, vmax=255)
    ax2.set_title(processed_title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Contrast")
	parser.add_argument("input_file", help="Input image file path")
	parser.add_argument("output_file", help="Output image file path")
	parser.add_argument("contrast_type", choices=["threshold", "linear", "AutoLinear"], help="Contrast type")
	args = parser.parse_args()
	file_name = args.input_file
	out_name = args.output_file
	contrast_type = args.contrast_type

	img = io.imread(file_name) # image stored as (column index, row index) ie (x,y)
	img_n = convertToIntensity(img)
	if contrast_type == "threshold":
		thresh = int(input("Enter threshold (0-255): "))
		bin_img = threshold_image_iterative(img_n, thresh)
		display_side_by_side(img, bin_img, original_title="Original Intensity", processed_title=f"Binarized (thr={thresh})")
		io.imsave(out_name, bin_img)
