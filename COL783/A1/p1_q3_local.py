# from skimage import io
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import argparse

# def displayIntensityImage(img,file_name):
# 	plt.figure(figsize=(8,6))
# 	plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# 	plt.axis('off')
# 	plt.title(file_name)
# 	plt.show()

# def displayColorImage(img,file_name):
# 	plt.figure(figsize=(8,6))
# 	plt.imshow(img)           # skimage -> RGB, so show directly
# 	plt.axis('off')
# 	plt.title(file_name)
# 	plt.show()

# def convertToIntensity(img):
# 	if img.ndim == 3 and img.shape[2] > 3:
# 		img_rgb = img[..., :3]	# img[..., :3] is equivalent to img[:, :, :3] for a 3-D array
# 	else:
# 		img_rgb = img
# 	intensity = img_rgb.astype(np.float32).mean(axis=2)	# axis (0=height, 1=width, 2=channels)
# 	info = np.iinfo(img_rgb.dtype)
# 	intensity_out = np.clip(np.round(intensity), info.min, info.max).astype(img_rgb.dtype)
# 	return intensity_out

# def get_mask_by_threshold(intensity_img):
#     if intensity_img.ndim != 2:
#         raise ValueError("threshold_image_iterative expects a 2D intensity image")
#     h, w = intensity_img.shape
#     mask = np.zeros((h, w), dtype=np.uint8)
#     max_i = intensity_img.max()
#     for i in range(h):
#         inp_row = intensity_img[i]
#         out_row = mask[i]
#         for j in range(w):
#             if inp_row[j] < max_i:
#                 #add to mask, set mask bit 1
#                 out_row[j]=1
#     return mask

# def get_mask_by_threshold_vectorized(intensity_img):
#     # Fast vectorized version
#     max_i = intensity_img.max()
#     mask = (intensity_img < max_i).astype(np.uint8)
#     return mask

# def applyMask(img,mask):
# 	m = mask.astype(bool)
# 	rgb = img[..., :3]
# 	alpha = img[..., 3:]
# 	masked_rgb = np.where(m[..., None], rgb, 0)
# 	return np.concatenate([masked_rgb, alpha], axis=2).astype(img.dtype)

# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser(description="Document scanner")
# 	parser.add_argument("input_file", help="Input image file path")
# 	parser.add_argument("output_file", help="Output image file path")

# 	args = parser.parse_args()
# 	file_name = args.input_file
# 	out_name = args.output_file

# 	img = io.imread(file_name) # image stored as (column index, row index) ie (x,y)
# 	intensity_img = convertToIntensity(img)
    
# 	mask = get_mask_by_threshold_vectorized(intensity_img)
# 	op_img=applyMask(img,mask)
    
# 	displayColorImage(op_img,out_name)

from skimage import io, img_as_ubyte
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def displayColorImage(img, title="image"):
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

def convertToIntensity(img):
    if img.ndim == 3 and img.shape[2] > 3:
        img_rgb = img[..., :3]
    elif img.ndim == 2:
        if img.dtype.kind == 'f':
            return img_as_ubyte(img)
        else:
            return img.astype(np.uint8)
    else:
        img_rgb = img
    if img_rgb.dtype.kind == 'f':
        rgb_u8 = img_as_ubyte(img_rgb)
    else:
        rgb_u8 = img_rgb.astype(np.uint8)
    intensity = rgb_u8.mean(axis=2).astype(np.uint8)
    return intensity

def get_mask_from_intensity(intensity_img, min_size=200, close_radius=3):
    # Otsu threshold
    thresh = threshold_otsu(intensity_img)
    mask = intensity_img < thresh
    # If mask covers most of image, likely inverted — invert it
    h, w = intensity_img.shape
    if mask.sum() > 0.6 * h * w:
        mask = ~mask
    mask = mask.astype(bool)
    return mask

def apply_mask_return_rgba(img, mask):
    if img.ndim == 2:
        rgb = np.stack([img]*3, axis=2)
    else:
        rgb = img[..., :3]
    if rgb.dtype.kind == 'f':
        rgb_u8 = img_as_ubyte(rgb)
    else:
        rgb_u8 = rgb.astype(np.uint8)
    alpha = (mask.astype(np.uint8) * 255).reshape(mask.shape + (1,))
    rgba = np.concatenate([rgb_u8, alpha], axis=2)
    return rgba

def composite_over_white(rgba):
    # rgba: uint8
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = (rgba[..., 3].astype(np.float32) / 255.0)[..., None]
    white = np.ones_like(rgb)
    comp = (rgb * a) + (white * (1.0 - a))
    comp_u8 = np.clip(np.round(comp * 255.0), 0, 255).astype(np.uint8)
    return comp_u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask for IITD logo and output image (PNG preserves transparency).")
    parser.add_argument("input_file", help="Input image file path")
    parser.add_argument("output_file", help="Output image file path (use .png for transparency)")
    parser.add_argument("--min-size", type=int, default=200, help="min size to keep objects in mask (px)")
    parser.add_argument("--close-radius", type=int, default=3, help="radius for closing operation")
    parser.add_argument("--debug", action="store_true", help="Show intermediate images")
    args = parser.parse_args()

    img = io.imread(args.input_file)
    intensity = convertToIntensity(img)
    if args.debug:
        displayColorImage(intensity, title="Intensity (debug)")

    mask = get_mask_from_intensity(intensity, min_size=args.min_size, close_radius=args.close_radius)
    if args.debug:
        displayColorImage((mask.astype(np.uint8)*255), title="Mask (debug)")

    rgba = apply_mask_return_rgba(img, mask)
    if args.debug:
        displayColorImage(rgba, title="RGBA (debug)")

    out_ext = os.path.splitext(args.output_file)[1].lower()
    if out_ext in ['.png', '.tiff', '.tif']:
        io.imsave(args.output_file, rgba)
        print("Saved PNG/TIFF with alpha:", args.output_file)
    elif out_ext in ['.jpg', '.jpeg']:
        comp = composite_over_white(rgba)
        io.imsave(args.output_file, comp)
        print("Saved JPEG (composited over white):", args.output_file)
    else:
        # default to PNG if unknown extension
        io.imsave(args.output_file, rgba)
        print("Saved (defaulted to format preserving alpha):", args.output_file)
