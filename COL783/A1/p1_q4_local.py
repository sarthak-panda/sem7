import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import argparse

def read_hdr_image(fname):
    try:
        import imageio
        img = imageio.v2.imread(fname)
        print(img)
        return img.astype(np.float32)
    except Exception:
        try:
            import cv2
            img = cv2.imread(fname, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("cv2.imread returned None")
            # OpenCV loads BGR; convert to RGB if 3 channels
            if img.ndim == 3 and img.shape[2] == 3:
                img = img[..., ::-1]
            return img.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Could not read HDR image with imageio or cv2: {e}")

# returns 1D float array of same shape as values but mapped to [a,b)
def equalize_real_values_flat(values, a=0.0, b=256.0):
    vals = values.ravel()
    N = vals.size
    finite_mask = np.isfinite(vals) # handling NaN/Inf, leave them as is or set to a
    finite_vals = vals[finite_mask]
    if finite_vals.size == 0:
        return np.full_like(vals, a)
    uniq, inverse, counts = np.unique(finite_vals, return_inverse=True, return_counts=True)
    cum_counts = np.cumsum(counts) # (cum_counts[i] = sum_{j<=i} counts[j])
    count_less = cum_counts - counts # number strictly less than uniq[i]
    cdf_mid = (count_less + 0.5 * counts) / float(N) # should we use 0.5 of counts?
    mapped_uniques = a + (b - a) * cdf_mid
    mapped_finite = mapped_uniques[inverse]
    out = np.empty_like(vals, dtype=float)
    out.fill(np.nan)
    out[finite_mask] = mapped_finite
    out[~finite_mask] = a # for any non-finite entries, set to a
    return out.reshape(values.shape)

def hdr_equalize_luminance(rgb_hdr, a=0.0, b=256.0, eps=1e-8):
    assert rgb_hdr.ndim == 3 and rgb_hdr.shape[2] >= 3
    rgb = rgb_hdr[..., :3].astype(np.float32)
    # I found something about linear luminance (Rec.709) or should we use 1/3 weight as usual intensity?
    L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    L_flat = L.ravel()
    L_eq_flat = equalize_real_values_flat(L_flat, a=a, b=b)
    L_eq = L_eq_flat.reshape(L.shape)
    scale = L_eq / (L + eps)
    rgb_out = rgb * scale[..., None]
    rgb_out = np.clip(rgb_out, 0.0, 255.0) # so images as displayable in normal formats
    rgb_u8 = np.round(rgb_out).astype(np.uint8)
    return rgb_u8

def compute_luminance(img):
    if img.ndim == 3 and img.shape[2] >= 3:
        # Rec. 709 / sRGB-like weights (standard)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    else:
        return img.ravel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDR Image Processing.")
    parser.add_argument("input_file", help="Input image file path")
    parser.add_argument("output_file", help="Output image file path")
    args = parser.parse_args()
    input_file = args.input_file
    print("Reading HDR image:", input_file)
    hdr = read_hdr_image(input_file)
    print("Image shape:", hdr.shape, "dtype:", hdr.dtype)

    print("Performing luminance equalization to [0,256)...")
    out = hdr_equalize_luminance(hdr, a=0.0, b=256.0)

    output_file = args.output_file
    plt.imsave(output_file, out)
    print("Saved equalized image to:", output_file)

    orig_vis = hdr.copy()
    lum_hdr = compute_luminance(orig_vis).ravel()
    lum_out = compute_luminance(out).ravel()
    valid_hdr = lum_hdr[np.isfinite(lum_hdr)]
    valid_out = lum_out[np.isfinite(lum_out)]
    combined = np.concatenate([valid_hdr, valid_out])
    bins = np.histogram_bin_edges(combined, bins='auto')
    counts_hdr, edges = np.histogram(valid_hdr, bins=bins)
    counts_out, _ = np.histogram(valid_out, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    preview = orig_vis.copy()
    preview = preview / (preview.max() + 1e-12)
    preview = np.clip(preview, 0, 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].imshow(preview)
    axs[0, 0].set_title("HDR (simple normalized preview)")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(out)
    axs[0, 1].set_title("Histogram-equalized")
    axs[0, 1].axis('off')
    axs[1, 0].plot(bin_centers, counts_hdr, label="orig HDR (luminance)")
    axs[1, 0].plot(bin_centers, counts_out, label="equalized output (luminance)")
    axs[1, 0].set_xlabel("Luminance value")
    axs[1, 0].set_ylabel("Pixel count")
    axs[1, 0].set_title("Luminance histogram (full range)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()