from skimage import io, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.transform import resize
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

def get_mask_from_intensity(intensity_img):
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
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = (rgba[..., 3].astype(np.float32) / 255.0)[..., None]
    white = np.ones_like(rgb)
    comp = (rgb * a) + (white * (1.0 - a))
    comp_u8 = np.clip(np.round(comp * 255.0), 0, 255).astype(np.uint8)
    return comp_u8

def paste_logo_with_50pct(doc_img, logo_rgba, logo_width_frac=0.2, pad=0, interp_order=1):
    if doc_img.ndim == 2:
        doc_rgb = np.stack([doc_img]*3, axis=2)
    else:
        doc_rgb = doc_img[..., :3]
    doc_rgb = img_as_ubyte(doc_rgb) if doc_rgb.dtype.kind == 'f' else doc_rgb.astype(np.uint8)
    H, W = doc_rgb.shape[:2]
    logo_rgb = logo_rgba[..., :3].astype(np.float32)
    logo_alpha = logo_rgba[..., 3].astype(np.float32) / 255.0
    target_w = int(round(W * logo_width_frac))
    if target_w <= 0:
        raise ValueError("increase logo_width_frac or use larger doc")
    orig_h, orig_w = logo_alpha.shape
    scale = target_w / orig_w
    target_h = int(round(orig_h * scale))
    logo_rgb_rs = resize(logo_rgb, (target_h, target_w, 3), order=interp_order, preserve_range=True, anti_aliasing=True)
    logo_alpha_rs = resize(logo_alpha, (target_h, target_w), order=interp_order, preserve_range=True, anti_aliasing=True)
    mask = logo_alpha_rs > 0.01
    y0 = max(0, H - target_h - pad)
    x0 = max(0, W - target_w - pad)
    y1 = y0 + target_h
    x1 = x0 + target_w
    out = doc_rgb.astype(np.float32).copy()
    doc_region = out[y0:y1, x0:x1, :]  # float32 in [0..255]
    logo_rgb_rs = np.clip(logo_rgb_rs, 0, 255).astype(np.float32)
    # Blend: where mask True, out = 0.5*doc + 0.5*logo; else keep doc
    # Expand mask to 3 channels
    mask_3 = np.repeat(mask[..., None], 3, axis=2)
    blended_region = doc_region.copy()
    blended_region[mask_3] = 0.5 * doc_region[mask_3] + 0.5 * logo_rgb_rs[mask_3]
    out[y0:y1, x0:x1, :] = blended_region
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask and output image (PNG preserves transparency).")
    parser.add_argument("input_file", help="Input image file path")
    parser.add_argument("doc_img_file", help="Document image file path")
    parser.add_argument("output_file", help="Output image file path (use .png for transparency)")
    args = parser.parse_args()

    img = io.imread(args.input_file)
    doc_img = io.imread(args.doc_img_file)
    intensity = convertToIntensity(img)
    #displayColorImage(intensity, title="Intensity")
    mask = get_mask_from_intensity(intensity)
    #displayColorImage((mask.astype(np.uint8)*255), title="Mask")
    rgba = apply_mask_return_rgba(img, mask)
    displayColorImage(rgba, title="RGBA")

    folder, fname = os.path.split(args.output_file)
    logo_fname = "logo_" + fname
    logo_file = os.path.join(folder, logo_fname)  
    out_ext = os.path.splitext(logo_file)[1].lower()
    if out_ext in ['.png', '.tiff', '.tif']:
        io.imsave(logo_file, rgba)
        print("Saved PNG/TIFF with alpha:", logo_file)
    elif out_ext in ['.jpg', '.jpeg']:
        comp = composite_over_white(rgba)
        io.imsave(logo_file, comp)
        print("Saved JPEG (composited over white):", logo_file)
    else: # default to PNG if unknown extension
        io.imsave(logo_file, rgba)
        print("Saved (defaulted to format preserving alpha):", logo_file)

    watermarked = paste_logo_with_50pct(doc_img, rgba, logo_width_frac=0.2, pad=0, interp_order=1)
    displayColorImage(watermarked,title="watermarked")
    io.imsave(args.output_file, watermarked)
    print("Saved watermarked image:", args.output_file)

#python .\p1_q3_local.py .\iitlogo-23.jpg .\testIMG1.jpg  .\P3OUTS\p3_t1_tr1.png