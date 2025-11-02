# import numpy as np
# import sys
# import os
# import matplotlib.pyplot as plt
# import argparse

# def read_hdr_image(fname):
#     try:
#         import imageio
#         img = imageio.v2.imread(fname)
#         print(img)
#         return img.astype(np.float32)
#     except Exception:
#         try:
#             import cv2
#             img = cv2.imread(fname, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
#             if img is None:
#                 raise RuntimeError("cv2.imread returned None")
#             # OpenCV loads BGR; convert to RGB if 3 channels
#             if img.ndim == 3 and img.shape[2] == 3:
#                 img = img[..., ::-1]
#             return img.astype(np.float32)
#         except Exception as e:
#             raise RuntimeError(f"Could not read HDR image with imageio or cv2: {e}")

# # returns 1D float array of same shape as values but mapped to [a,b)
# def equalize_real_values_flat(values, a=0.0, b=256.0):
#     vals = values.ravel()
#     N = vals.size
#     finite_mask = np.isfinite(vals) # handling NaN/Inf, leave them as is or set to a
#     finite_vals = vals[finite_mask]
#     if finite_vals.size == 0:
#         return np.full_like(vals, a)
#     uniq, inverse, counts = np.unique(finite_vals, return_inverse=True, return_counts=True)
#     cum_counts = np.cumsum(counts) # (cum_counts[i] = sum_{j<=i} counts[j])
#     count_less = cum_counts - counts # number strictly less than uniq[i]
#     cdf_mid = (count_less + 0.5 * counts) / float(N) # should we use 0.5 of counts?
#     mapped_uniques = a + (b - a) * cdf_mid
#     mapped_finite = mapped_uniques[inverse]
#     out = np.empty_like(vals, dtype=float)
#     out.fill(np.nan)
#     out[finite_mask] = mapped_finite
#     out[~finite_mask] = a # for any non-finite entries, set to a
#     return out.reshape(values.shape)

# def hdr_equalize_luminance(rgb_hdr, a=0.0, b=256.0, eps=1e-8):
#     assert rgb_hdr.ndim == 3 and rgb_hdr.shape[2] >= 3
#     rgb = rgb_hdr[..., :3].astype(np.float32)
#     # I found something about linear luminance (Rec.709) or should we use 1/3 weight as usual intensity?
#     L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
#     L_flat = L.ravel()
#     L_eq_flat = equalize_real_values_flat(L_flat, a=a, b=b)
#     L_eq = L_eq_flat.reshape(L.shape)
#     scale = L_eq / (L + eps)
#     rgb_out = rgb * scale[..., None]
#     rgb_out = np.clip(rgb_out, 0.0, 255.0) # so images as displayable in normal formats
#     rgb_u8 = np.round(rgb_out).astype(np.uint8)
#     return rgb_u8

# def compute_luminance(img):
#     if img.ndim == 3 and img.shape[2] >= 3:
#         # Rec. 709 / sRGB-like weights (standard)
#         r, g, b = img[..., 0], img[..., 1], img[..., 2]
#         return 0.2126 * r + 0.7152 * g + 0.0722 * b
#     else:
#         return img.ravel()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="HDR Image Processing.")
#     parser.add_argument("input_file", help="Input image file path")
#     parser.add_argument("output_file", help="Output image file path")
#     args = parser.parse_args()
#     input_file = args.input_file
#     print("Reading HDR image:", input_file)
#     hdr = read_hdr_image(input_file)
#     print("Image shape:", hdr.shape, "dtype:", hdr.dtype)

#     print("Performing luminance equalization to [0,256)...")
#     out = hdr_equalize_luminance(hdr, a=0.0, b=256.0)

#     output_file = args.output_file
#     plt.imsave(output_file, out)
#     print("Saved equalized image to:", output_file)

#     orig_vis = hdr.copy()
#     lum_hdr = compute_luminance(orig_vis).ravel()
#     lum_out = compute_luminance(out).ravel()
#     valid_hdr = lum_hdr[np.isfinite(lum_hdr)]
#     valid_out = lum_out[np.isfinite(lum_out)]
#     combined = np.concatenate([valid_hdr, valid_out])
#     bins = np.histogram_bin_edges(combined, bins='auto')
#     counts_hdr, edges = np.histogram(valid_hdr, bins=bins)
#     counts_out, _ = np.histogram(valid_out, bins=bins)
#     bin_centers = 0.5 * (edges[:-1] + edges[1:])
#     preview = orig_vis.copy()
#     preview = preview / (preview.max() + 1e-12)
#     preview = np.clip(preview, 0, 1)

#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     axs[0, 0].imshow(preview)
#     axs[0, 0].set_title("HDR (simple normalized preview)")
#     axs[0, 0].axis('off')
#     axs[0, 1].imshow(out)
#     axs[0, 1].set_title("Histogram-equalized")
#     axs[0, 1].axis('off')
#     axs[1, 0].plot(bin_centers, counts_hdr, label="orig HDR (luminance)")
#     axs[1, 0].plot(bin_centers, counts_out, label="equalized output (luminance)")
#     axs[1, 0].set_xlabel("Luminance value")
#     axs[1, 0].set_ylabel("Pixel count")
#     axs[1, 0].set_title("Luminance histogram (full range)")
#     axs[1, 0].legend()
#     axs[1, 0].grid(True)
#     axs[1, 1].axis('off')
#     plt.tight_layout()
#     plt.show()

# import numpy as np
# import sys
# import os
# import matplotlib.pyplot as plt
# import argparse

# def read_hdr_image(fname):
#     try:
#         import imageio
#         img = imageio.v2.imread(fname)
#         print(img)
#         return img.astype(np.float32)
#     except Exception:
#         try:
#             import cv2
#             img = cv2.imread(fname, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
#             if img is None:
#                 raise RuntimeError("cv2.imread returned None")
#             # OpenCV loads BGR; convert to RGB if 3 channels
#             if img.ndim == 3 and img.shape[2] == 3:
#                 img = img[..., ::-1]
#             return img.astype(np.float32)
#         except Exception as e:
#             raise RuntimeError(f"Could not read HDR image with imageio or cv2: {e}")

# # returns 1D float array of same shape as values but mapped to [a,b)
# def equalize_real_values_flat(values, a=0.0, b=256.0):
#     vals = values.ravel()
#     N = vals.size
#     finite_mask = np.isfinite(vals) # handling NaN/Inf, leave them as is or set to a
#     finite_vals = vals[finite_mask]
#     if finite_vals.size == 0:
#         return np.full_like(vals, a)
#     uniq, inverse, counts = np.unique(finite_vals, return_inverse=True, return_counts=True)
#     cum_counts = np.cumsum(counts) # (cum_counts[i] = sum_{j<=i} counts[j])
#     count_less = cum_counts - counts # number strictly less than uniq[i]
#     cdf_mid = (count_less + 0.5 * counts) / float(N) # should we use 0.5 of counts?
#     mapped_uniques = a + (b - a) * cdf_mid
#     mapped_finite = mapped_uniques[inverse]
#     out = np.empty_like(vals, dtype=float)
#     out.fill(np.nan)
#     out[finite_mask] = mapped_finite
#     out[~finite_mask] = a # for any non-finite entries, set to a
#     return out.reshape(values.shape)

# def hdr_equalize_luminance(rgb_hdr, a=0.0, b=256.0, eps=1e-8):
#     assert rgb_hdr.ndim == 3 and rgb_hdr.shape[2] >= 3
#     rgb = rgb_hdr[..., :3].astype(np.float32)
#     # I found something about linear luminance (Rec.709) or should we use 1/3 weight as usual intensity?
#     L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
#     L_flat = L.ravel()
#     L_eq_flat = equalize_real_values_flat(L_flat, a=a, b=b)
#     L_eq = L_eq_flat.reshape(L.shape)
#     scale = L_eq / (L + eps)
#     rgb_out = rgb * scale[..., None]
#     rgb_out = np.clip(rgb_out, 0.0, 255.0) # so images as displayable in normal formats
#     rgb_u8 = np.round(rgb_out).astype(np.uint8)
#     return rgb_u8

# def compute_luminance(img):
#     if img.ndim == 3 and img.shape[2] >= 3:
#         # Rec. 709 / sRGB-like weights (standard)
#         r, g, b = img[..., 0], img[..., 1], img[..., 2]
#         return 0.2126 * r + 0.7152 * g + 0.0722 * b
#     else:
#         return img.ravel()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="HDR Image Processing.")
#     parser.add_argument("input_file", help="Input image file path")
#     parser.add_argument("output_file", help="Output image file path")
#     args = parser.parse_args()
#     input_file = args.input_file
#     print("Reading HDR image:", input_file)
#     hdr = read_hdr_image(input_file)
#     print("Image shape:", hdr.shape, "dtype:", hdr.dtype)

#     print("Performing luminance equalization to [0,256)...")
#     out = hdr_equalize_luminance(hdr, a=0.0, b=256.0)

#     output_file = args.output_file
#     plt.imsave(output_file, out)
#     print("Saved equalized image to:", output_file)

#     orig_vis = hdr.copy()
#     lum_hdr = compute_luminance(orig_vis).ravel()
#     lum_out = compute_luminance(out).ravel()
#     valid_hdr = lum_hdr[np.isfinite(lum_hdr)]
#     valid_out = lum_out[np.isfinite(lum_out)]
#     combined = np.concatenate([valid_hdr, valid_out])
#     bins = np.histogram_bin_edges(combined, bins='auto')
#     counts_hdr, edges = np.histogram(valid_hdr, bins=bins)
#     counts_out, _ = np.histogram(valid_out, bins=bins)
#     bin_centers = 0.5 * (edges[:-1] + edges[1:])
#     preview = orig_vis.copy()
#     preview = preview / (preview.max() + 1e-12)
#     preview = np.clip(preview, 0, 1)

#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     axs[0, 0].imshow(preview)
#     axs[0, 0].set_title("HDR (simple normalized preview)")
#     axs[0, 0].axis('off')
#     axs[0, 1].imshow(out)
#     axs[0, 1].set_title("Histogram-equalized")
#     axs[0, 1].axis('off')
#     axs[1, 0].plot(bin_centers, counts_hdr, label="orig HDR (luminance)")
#     axs[1, 0].plot(bin_centers, counts_out, label="equalized output (luminance)")
#     axs[1, 0].set_xlabel("Luminance value")
#     axs[1, 0].set_ylabel("Pixel count")
#     axs[1, 0].set_title("Luminance histogram (full range)")
#     axs[1, 0].legend()
#     axs[1, 0].grid(True)
#     axs[1, 1].axis('off')
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import argparse
import imageio

def read_hdr_image(fname):
    try:
        imageio.plugins.freeimage.download()
        img = imageio.imread(fname, format='HDR-FI')
        img = img.astype(np.float32)
        return img
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
            if fname.lower().endswith('.exr'):
                try:
                    import OpenEXR
                    import Imath
                    exr_file = OpenEXR.InputFile(fname)
                    dw = exr_file.header()['dataWindow']
                    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    rgb = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape(size[1], size[0]) for c in ('R', 'G', 'B')]
                    img = np.stack(rgb, axis=-1)
                    return img.astype(np.float32)
                except Exception as exr_e:
                    raise RuntimeError(f"Could not read EXR image with OpenEXR: {exr_e}")
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
    
    print(f"Original luminance range: min={valid_hdr.min():.4f}, max={valid_hdr.max():.4f}")

    num_bins = 512
    eps = 1e-12

    if valid_hdr.size == 0:
        raise ValueError("valid_hdr is empty")
    if valid_out.size == 0:
        raise ValueError("valid_out is empty")

    valid_hdr = valid_hdr[np.isfinite(valid_hdr)]
    valid_out = valid_out[np.isfinite(valid_out)]

    log_hdr = np.log10(valid_hdr + eps)
    log_out = np.log10(valid_out + eps)

    counts_log_hdr, edges_log_hdr = np.histogram(log_hdr, bins=num_bins)
    counts_log_out, edges_log_out = np.histogram(log_out, bins=num_bins)

    bin_centers_log_hdr = 0.5 * (edges_log_hdr[:-1] + edges_log_hdr[1:])
    bin_centers_log_out = 0.5 * (edges_log_out[:-1] + edges_log_out[1:])

    p_lo, p_hi = 0.0, 99.995
    linear_max_hdr = np.percentile(valid_hdr, p_hi)
    linear_min_hdr = max(0.0, np.percentile(valid_hdr, p_lo))
    linear_bins_hdr = np.linspace(linear_min_hdr, linear_max_hdr, num_bins + 1)
    counts_lin_hdr, edges_lin_hdr = np.histogram(valid_hdr, bins=linear_bins_hdr)

    linear_max_out = np.percentile(valid_out, 100)
    linear_min_out = max(0.0, np.percentile(valid_out, 0.0))
    linear_bins_out = np.linspace(linear_min_out, linear_max_out, num_bins + 1)
    counts_lin_out, edges_lin_out = np.histogram(valid_out, bins=linear_bins_out)

    bin_centers_lin_hdr = 0.5 * (edges_lin_hdr[:-1] + edges_lin_hdr[1:])
    bin_centers_lin_out = 0.5 * (edges_lin_out[:-1] + edges_lin_out[1:])

    preview = orig_vis.copy()
    preview = preview / (preview.max() + 1e-12)
    preview = np.clip(preview, 0, 1)

    fig1 = plt.figure(figsize=(7, 5))
    plt.imshow(preview)
    plt.title("HDR (simple normalized preview)")
    plt.axis('off')

    fig2 = plt.figure(figsize=(7, 5))
    plt.imshow(out)
    plt.title("Histogram-equalized")
    plt.axis('off')

    fig3, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(bin_centers_log_hdr, counts_log_hdr, label="Original HDR (log10 L)")
    axs[0, 0].set_xlabel("log10(Luminance)")
    axs[0, 0].set_ylabel("Pixel count (log y)")
    axs[0, 0].set_title("Original HDR Luminance Histogram (log x)")
    axs[0, 0].legend()
    axs[0, 0].set_yscale('log')
    axs[0, 0].grid(True, which="both")

    axs[0, 1].plot(bin_centers_log_out, counts_log_out, label="Equalized output (log10 L)")
    axs[0, 1].set_xlabel("log10(Luminance)")
    axs[0, 1].set_ylabel("Pixel count (log y)")
    axs[0, 1].set_title("Equalized Luminance Histogram (log x)")
    axs[0, 1].legend()
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid(True, which="both")

    axs[1, 0].plot(bin_centers_lin_hdr, counts_lin_hdr, label=f"Original HDR (linear, clipped {p_hi}th pct)")
    axs[1, 0].set_xlabel("Luminance value")
    axs[1, 0].set_ylabel("Pixel count")
    axs[1, 0].set_title("Original HDR Luminance Histogram (linear x, clipped)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(bin_centers_lin_out, counts_lin_out, label=f"Equalized output")
    axs[1, 1].set_xlabel("Luminance value")
    axs[1, 1].set_ylabel("Pixel count")
    axs[1, 1].set_title("Equalized Luminance Histogram (linear x, clipped)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
