import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import math

def convolve(w, f, c=0, zeroPaddingNeeded=False, completeResult=False, kernel_is_column=None):
    """
    Convolve kernel w with image f, add scalar c, return uint8 result.

    Parameters:
      w : array-like (1D or 2D)
      f : array-like (1D or 2D)
      c : scalar
      zeroPaddingNeeded : bool   -> True: zero pad, False: reflect pad
      completeResult : bool     -> False: "same" output (shape == f.shape)
                                 True: "full" output (H+kh-1, W+kw-1)
      kernel_is_column : None/True/False -> only used when w.ndim==1
                        None => treat 1D kernel as row (backwards-compatible)
                        True => treat 1D kernel as column
                        False=> treat 1D kernel as row
    """
    w = np.asarray(w, dtype=np.float64)
    f = np.asarray(f)

    # Interpret 1D kernel according to kernel_is_column
    if w.ndim == 1:
        if kernel_is_column is None or kernel_is_column is False:
            w = w.reshape(1, -1)   # row kernel (1 x N)
        else:
            w = w.reshape(-1, 1)   # column kernel (N x 1)

    # Interpret 1D image as column vector (old behaviour)
    if f.ndim == 1:
        f = f.reshape(-1, 1)       # column (N x 1)

    if f.ndim != 2 or w.ndim != 2:
        raise ValueError("f and w must be 1D or 2D arrays (grayscale kernels/images).")

    H, W = f.shape
    kh, kw = w.shape
    if kh == 0 or kw == 0:
        raise ValueError("Kernel must have non-zero shape.")

    w_flipped = np.flip(np.flip(w, axis=0), axis=1)  # convolution requires flip

    pad_mode = 'constant' if zeroPaddingNeeded else 'reflect'

    if not completeResult:
        # SAME output shape (H x W)
        pad_top = kh // 2
        pad_bottom = kh - pad_top - 1
        pad_left = kw // 2
        pad_right = kw - pad_left - 1

        if pad_mode == 'constant':
            padded = np.pad(f.astype(np.float64),
                            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0)
        else:
            padded = np.pad(f.astype(np.float64),
                            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='reflect')

        out_h, out_w = H, W
        out = np.empty((out_h, out_w), dtype=np.float64)

        for y in range(out_h):
            for x in range(out_w):
                patch = padded[y:y+kh, x:x+kw]
                out[y, x] = np.sum(patch * w_flipped)

    else:
        # FULL output shape (H+kh-1, W+kw-1)
        pad_top = kh - 1
        pad_bottom = kh - 1
        pad_left = kw - 1
        pad_right = kw - 1

        if pad_mode == 'constant':
            padded = np.pad(f.astype(np.float64),
                            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0)
        else:
            padded = np.pad(f.astype(np.float64),
                            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='reflect')

        full_h = H + kh - 1
        full_w = W + kw - 1
        out = np.empty((full_h, full_w), dtype=np.float64)

        for y in range(full_h):
            for x in range(full_w):
                patch = padded[y:y+kh, x:x+kw]
                out[y, x] = np.sum(patch * w_flipped)

    out = out + float(c)
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)

def laplacican(input_image_path):
    """
    Compute Laplacian (∇^2 f) of the image at input_image_path using the
    3x3 kernel and c = 128 by calling the existing `convolve` function.

    Saves output as <original_basename>_laplacian.png and displays both images.
    """
    # load image and convert to grayscale (8-bit)
    img = Image.open(input_image_path).convert('L')
    f = np.array(img, dtype=np.uint8)

    # 3x3 Laplacian kernel (4-neighbour version)
    lap_kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=float)

    # call the convolve function you've defined earlier
    # c = 128 as requested; zeroPaddingNeeded and completeResult left as defaults
    g = convolve(lap_kernel, f, c=128)

    # prepare output path and save result
    base, _ = os.path.splitext(input_image_path)
    out_path = base + "_laplacian.png"
    out_img = Image.fromarray(g)
    out_img.save(out_path)

    # display original and result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(f, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original (grayscale)")
    axes[0].axis('off')

    axes[1].imshow(g, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Laplacian (c=128)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Laplacian saved to: {out_path}")


def convolutionTest():
    np.set_printoptions(formatter={'int': lambda x: f"{x:3d}"})

    # --- Test 1 (same as before) ---
    f1 = np.zeros((5, 5), dtype=np.uint8)
    f1[2, 2] = 1
    w1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=float)

    print("=== Test 1: 5x5 delta image, 3x3 kernel ===")
    for zeroPad in (False, True):
        mode_name = "zero-pad" if zeroPad else "reflect"
        print(f"\nPadding mode: {mode_name}")
        g_same = convolve(w1, f1, c=0, zeroPaddingNeeded=zeroPad, completeResult=False)
        print("completeResult=False (same):")
        print(g_same)
        g_full = convolve(w1, f1, c=0, zeroPaddingNeeded=zeroPad, completeResult=True)
        print("completeResult=True (full):")
        print(g_full)

    # --- Test 2 (previous column kernel-case) ---
    f2 = np.array([1, 1, 1], dtype=np.uint8)   # 1D -> becomes (3,1) column
    w2 = np.array([1, 2, 1], dtype=float)      # 1D -> default treated as row (1x3)
    print("\n=== Test 2: f = column [1,1,1], w = row [1,2,1] (as before) ===")
    for zeroPad in (True, False):
        print(f"\nPadding mode: {'zero-pad' if zeroPad else 'reflect'}")
        print("completeResult=False (same):")
        print(convolve(w2, f2, c=0, zeroPaddingNeeded=zeroPad, completeResult=False))
        print("completeResult=True (full):")
        print(convolve(w2, f2, c=0, zeroPaddingNeeded=zeroPad, completeResult=True))

    # --- Test 3 (NEW): explicitly test a COLUMN kernel ---
    # Here we pass f as a 1x3 row (2D) and w as 1D COLUMN by setting kernel_is_column=True
    f3 = np.array([[1, 1, 1]], dtype=np.uint8)   # shape (1,3) row
    w3 = np.array([1, 2, 1], dtype=float)        # 1D, to be interpreted as column
    print("\n=== Test 3 (NEW): f = row [[1,1,1]] (1x3), w = column [1,2,1] (3x1) ===")
    for zeroPad in (True, False):
        print(f"\nPadding mode: {'zero-pad' if zeroPad else 'reflect'}")
        print("completeResult=False (same):")
        out_same = convolve(w3, f3, c=0, zeroPaddingNeeded=zeroPad, completeResult=False, kernel_is_column=True)
        print(out_same)
        print("completeResult=True (full):")
        out_full = convolve(w3, f3, c=0, zeroPaddingNeeded=zeroPad, completeResult=True, kernel_is_column=True)
        print(out_full)


def gaussian_kernel_1d(sigma):
    """
    Create a 1D Gaussian kernel truncated at |s| > 3*sigma and normalized to sum=1.
    Returns shape (1, m) as a row vector.
    """
    if sigma <= 0:
        sigma = 1e-6
    a = int(math.ceil(3.0 * sigma))
    xs = np.arange(-a, a + 1, dtype=np.float64)  # length m = 2a+1
    g = np.exp(- (xs * xs) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g.reshape(1, -1)  # row kernel (1 x m)

def gaussian_kernel_2d(sigma):
    """
    Create a 2D Gaussian kernel truncated at |s|,|t| > 3*sigma and normalized to sum=1.
    Returns shape (m, m).
    """
    if sigma <= 0:
        sigma = 1e-6
    a = int(math.ceil(3.0 * sigma))
    xs = np.arange(-a, a + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    g = np.exp(- (X*X + Y*Y) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g

# -----------------------
# Filters using convolve
# -----------------------
def gaussianFilter1(imagePath, sigma, out_dir=None):
    """
    Naive 2D Gaussian filter: build full 2D kernel and call convolve once.
    Uses zero padding always (zeroPaddingNeeded=True).
    Returns (output_uint8_HxW, elapsed_seconds, out_path).
    """
    # load grayscale image
    img = Image.open(imagePath).convert('L')
    f = np.array(img, dtype=np.uint8)
    H, W = f.shape

    # Build full 2D Gaussian kernel (assumes gaussian_kernel_2d defined)
    K = gaussian_kernel_2d(sigma)

    t0 = time.perf_counter()
    # use zero padding; single pass => same-size output
    g = convolve(K, f, c=0, zeroPaddingNeeded=True, completeResult=False)
    t1 = time.perf_counter()

    elapsed = t1 - t0

    # save result
    base = os.path.splitext(os.path.basename(imagePath))[0]
    folder = out_dir or f"./GaussianOutputs_{base}"
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"{base}_gauss2d_sigma{sigma:.2f}.png")
    Image.fromarray(g).save(out_path)

    return g, elapsed, out_path

# def gaussianFilter2(imagePath, sigma, out_dir=None):
#     """
#     Separable gaussian: 1D row kernel then 1D column kernel.
#     Returns (output_array_uint8, elapsed_seconds, out_filepath)
#     """
#     img = Image.open(imagePath).convert('L')
#     f = np.array(img, dtype=np.uint8)

#     # Build 1D Gaussian row kernel and its column counterpart
#     g1d_row = gaussian_kernel_1d(sigma)        # shape (1, m)
#     g1d_col = g1d_row.reshape(1, -1)           # still row; we'll pass kernel_is_column=True for second pass

#     # First pass: convolve with row (horizontal smoothing)
#     t0 = time.perf_counter()
#     tmp, t_first, _ = None, 0.0, None
#     # time for first pass
#     tstart = time.perf_counter()
#     out1 = convolve(g1d_row, f, c=0)          # g1d_row is 1xM (row), default interpretation is row
#     tmid = time.perf_counter()
#     # Second pass: convolve with column (vertical smoothing). 
#     # Use kernel_is_column=True so the 1D kernel is interpreted as column (M x 1)
#     out2 = convolve(g1d_row.flatten(), out1, c=0, kernel_is_column=True)
#     tend = time.perf_counter()
#     elapsed = tend - tstart

#     base = os.path.splitext(os.path.basename(imagePath))[0]
#     folder = out_dir or f"./GaussianOutputs_{base}"
#     os.makedirs(folder, exist_ok=True)
#     out_path = os.path.join(folder, f"{base}_gausssep_sigma{sigma:.2f}.png")
#     Image.fromarray(out2).save(out_path)

#     return out2, elapsed, out_path

def gaussianFilter2(imagePath, sigma, out_dir=None):
    """
    Separable Gaussian filter: convolve with 1D row then 1D column kernels.
    Requirements:
      - always use zero padding (zeroPaddingNeeded=True)
      - first pass produces complete "full" intermediate result (completeResult=True)
      - second pass also computed in "full" mode; final full result is cropped
        to original H x W before returning (so both filters produce same-size output).
    Returns (output_uint8_HxW, elapsed_seconds, out_path).
    """
    img = Image.open(imagePath).convert('L')
    f = np.array(img, dtype=np.uint8)
    H, W = f.shape

    # 1D Gaussian kernels (row and column); gaussian_kernel_1d returns row (1 x m)
    g1d_row = gaussian_kernel_1d(sigma)            # shape (1, m)
    m = g1d_row.shape[1]
    pad = m // 2                                   # same 'a' from truncation

    folder_base = out_dir or f"./GaussianOutputs_{os.path.splitext(os.path.basename(imagePath))[0]}"
    os.makedirs(folder_base, exist_ok=True)

    t0 = time.perf_counter()

    # FIRST PASS: horizontal smoothing -> produce COMPLETE (full) intermediate result
    # note: pass the 1D row kernel as-is; request completeResult=True and zero padding
    out1_full = convolve(g1d_row, f, c=0, zeroPaddingNeeded=True, completeResult=True)

    # SECOND PASS: vertical smoothing (treat 1D kernel as column)
    # pass kernel as flattened 1D and signal kernel_is_column=True, compute full result
    out2_full = convolve(g1d_row.flatten(), out1_full, c=0,
                         zeroPaddingNeeded=True, completeResult=True, kernel_is_column=True)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # out2_full shape is (H + m - 1, W + m - 1). Crop center to original H x W:
    start_r = pad
    start_c = pad
    cropped = out2_full[start_r:start_r + H, start_c:start_c + W]

    # save result
    base = os.path.splitext(os.path.basename(imagePath))[0]
    out_path = os.path.join(folder_base, f"{base}_gausssep_sigma{sigma:.2f}.png")
    Image.fromarray(cropped).save(out_path)

    return cropped, elapsed, out_path

# def testGaussian(imagePath, sigma_list=None, out_root="./GaussianOutputs", compare_tol=2):
#     """
#     Run gaussianFilter1 (naive 2D kernel) and gaussianFilter2 (separable) for each sigma
#     in sigma_list. Save outputs into ./GaussianOutputs_<ImageName>/ and plot compute times.

#     Parameters
#     ----------
#     imagePath : str
#     sigma_list : iterable of positive numbers (if None, a default range is used)
#     out_root : base output root folder
#     compare_tol : integer tolerance for max absolute difference when comparing outputs
#                   (due to casting/rounding differences between methods)
#     """
#     if sigma_list is None:
#         sigma_list = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]

#     base = os.path.splitext(os.path.basename(imagePath))[0]
#     out_dir = os.path.join(out_root + "_" + base)
#     os.makedirs(out_dir, exist_ok=True)

#     times_2d = []
#     times_sep = []
#     diffs = []

#     print(f"Running gaussian comparisons on {imagePath}; outputs -> {out_dir}")
#     for sigma in sigma_list:
#         print(f"\n--- sigma = {sigma} ---")

#         g2d, t2d, path2d = gaussianFilter1(imagePath, sigma, out_dir=out_dir)
#         print(f"gaussianFilter1 (2D kernel):   time = {t2d:.4f} s, saved: {os.path.basename(path2d)}")

#         gsep, tsep, pathsep = gaussianFilter2(imagePath, sigma, out_dir=out_dir)
#         print(f"gaussianFilter2 (separable):   time = {tsep:.4f} s, saved: {os.path.basename(pathsep)}")

#         # record times
#         times_2d.append(t2d)
#         times_sep.append(tsep)

#         # compare outputs (uint8). allow small differences due to rounding.
#         # compute max absolute difference
#         max_abs_diff = int(np.max(np.abs(g2d.astype(np.int32) - gsep.astype(np.int32))))
#         diffs.append(max_abs_diff)
#         ok = max_abs_diff <= compare_tol
#         print(f"max absolute difference between outputs: {max_abs_diff} -> {'OK' if ok else 'MISMATCH'}")

#     # Plot times
#     plt.figure(figsize=(8,5))
#     plt.plot(sigma_list, times_2d, marker='o', label='gaussianFilter1 (2D kernel)')
#     plt.plot(sigma_list, times_sep, marker='o', label='gaussianFilter2 (separable)')
#     plt.xlabel('sigma')
#     plt.ylabel('compute time (seconds)')
#     plt.title(f'Gaussian filter compute time (image={base})')
#     plt.grid(True)
#     plt.legend()
#     plot_path = os.path.join(out_dir, f"{base}_gaussian_times.png")
#     plt.savefig(plot_path, dpi=150)
#     plt.show()

#     # summary
#     print("\nSummary:")
#     for s, t1, t2, d in zip(sigma_list, times_2d, times_sep, diffs):
#         print(f" sigma={s:5.2f}  time2D={t1:.4f}s  timeSep={t2:.4f}s  maxDiff={d}")

#     print(f"\nPlot saved to: {plot_path}")
#     return {
#         "sigma_list": sigma_list,
#         "times_2d": times_2d,
#         "times_sep": times_sep,
#         "diffs": diffs,
#         "out_dir": out_dir,
#         "plot_path": plot_path
#     }


def testGaussian(imagePath, sigma_list=None, out_root="./GaussianOutputs", compare_tol=2, compare_tol_sum=100):
    """
    Run gaussianFilter1 (2D kernel) and gaussianFilter2 (separable) for each sigma
    in sigma_list. Save outputs into ./GaussianOutputs_<ImageName>/ and plot compute times.

    Parameters:
      imagePath : str
      sigma_list : list of sigmas (if None a default list is used)
      out_root : base output root folder (folder name will be appended with image base)
      compare_tol : allowed max absolute difference between outputs (int)
      compare_tol_sum : allowed sum of absolute differences between outputs (int)
    Returns:
      dict with recorded times, diffs, paths and plot path.
    """
    if sigma_list is None:
        # defaults that will show time difference for larger kernels
        sigma_list = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]

    base = os.path.splitext(os.path.basename(imagePath))[0]
    out_dir = os.path.join(out_root + "_" + base)
    os.makedirs(out_dir, exist_ok=True)

    times_2d = []
    times_sep = []
    max_diffs = []
    sum_diffs = []
    paths_2d = []
    paths_sep = []

    print(f"Running gaussian comparisons on {imagePath}; outputs -> {out_dir}")
    for sigma in sigma_list:
        print(f"\n--- sigma = {sigma} ---")

        g2d, t2d, path2d = gaussianFilter1(imagePath, sigma, out_dir=out_dir)
        print(f"gaussianFilter1 (2D kernel):   time = {t2d:.4f} s, saved: {os.path.basename(path2d)}")

        gsep, tsep, pathsep = gaussianFilter2(imagePath, sigma, out_dir=out_dir)
        print(f"gaussianFilter2 (separable):   time = {tsep:.4f} s, saved: {os.path.basename(pathsep)}")

        times_2d.append(t2d)
        times_sep.append(tsep)
        paths_2d.append(path2d)
        paths_sep.append(pathsep)

        # Compare outputs (both are uint8 HxW). compute max and sum of absolute differences
        # print(g2d)
        # print(gsep)

        diff = (g2d.astype(np.int32) - gsep.astype(np.int32))
        max_abs_diff = int(np.max(np.abs(diff)))
        sum_abs_diff = int(np.sum(np.abs(diff)))
        max_diffs.append(max_abs_diff)
        sum_diffs.append(sum_abs_diff)

        ok_max = max_abs_diff <= compare_tol
        ok_sum = sum_abs_diff <= compare_tol_sum
        print(f" max_abs_diff = {max_abs_diff} (<= {compare_tol}? {'OK' if ok_max else 'FAIL'})")
        print(f" sum_abs_diff = {sum_abs_diff} (<= {compare_tol_sum}? {'OK' if ok_sum else 'FAIL'})")
        if not (ok_max and ok_sum):
            print("  WARNING: outputs differ beyond tolerances (consider increasing compare_tol_sum or using float-accumulation).")

    # Plot times
    plt.figure(figsize=(8,5))
    plt.plot(sigma_list, times_2d, marker='o', label='gaussianFilter1 (2D kernel)')
    plt.plot(sigma_list, times_sep, marker='o', label='gaussianFilter2 (separable)')
    plt.xlabel('sigma')
    plt.ylabel('compute time (seconds)')
    plt.title(f'Gaussian filter compute time (image={base})')
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(out_dir, f"{base}_gaussian_times.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    # summary
    print("\nSummary:")
    for s, t1, t2, md, sd in zip(sigma_list, times_2d, times_sep, max_diffs, sum_diffs):
        print(f" sigma={s:5.2f}  time2D={t1:.4f}s  timeSep={t2:.4f}s  maxDiff={md}  sumDiff={sd}")

    print(f"\nPlot saved to: {plot_path}")
    return {
        "sigma_list": sigma_list,
        "times_2d": times_2d,
        "times_sep": times_sep,
        "max_diffs": max_diffs,
        "sum_diffs": sum_diffs,
        "paths_2d": paths_2d,
        "paths_sep": paths_sep,
        "out_dir": out_dir,
        "plot_path": plot_path
    }

def kernel_convolve(A, B):
    """
    Linear 2D convolution of kernel arrays A and B (both float ndarray).
    Returns array of shape (hA+hB-1, wA+wB-1) as float64.
    This computes C[p,q] = sum_{i,j} A[i,j] * B[p-i, q-j].
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ha, wa = A.shape
    hb, wb = B.shape
    hc = ha + hb - 1
    wc = wa + wb - 1
    C = np.zeros((hc, wc), dtype=np.float64)
    # For each element in A add scaled copy of B at offset (i,j)
    for i in range(ha):
        for j in range(wa):
            C[i:i+hb, j:j+wb] += A[i, j] * B
    return C

def testLapGauss(imagePath=None, sigma_list=None, out_root="./LapGaussOutputs", compare_tol_max=2, compare_tol_sum=50, loadFromNumpyData=False, numpy_array=None):
    """
    For each sigma in sigma_list compute:
      A = l * (g * f)
      B = g * (l * f)
      C = (l * g) * f
    Use separable Gaussian wherever possible. Record runtimes for A,B,C,
    compare outputs by max absolute diff and sum absolute diff.
    Save visualization images (adds c=128 for display) into out_root_<imagebase>.
    Plot runtime curves (3-methods).
    Returns summary dict.
    """
    if sigma_list is None:
        sigma_list = [1.0, 2.0, 4.0, 8.0, 12.0]

    # load image/array once
    if loadFromNumpyData:
        if numpy_array is None:
            raise ValueError("numpy_array required when loadFromNumpyData=True")
        f = np.asarray(numpy_array, dtype=np.uint8)
        base = "array_input"
    else:
        if imagePath is None:
            raise ValueError("imagePath must be provided when loadFromNumpyData=False")
        img = Image.open(imagePath).convert('L')
        f = np.array(img, dtype=np.uint8)
        base = os.path.splitext(os.path.basename(imagePath))[0]

    H, W = f.shape
    out_dir = os.path.join(out_root + "_" + base)
    os.makedirs(out_dir, exist_ok=True)

    # Laplacian kernel (float)
    l_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)

    times_A = []
    times_B = []
    times_C = []
    max_diffs_AB = []
    sum_diffs_AB = []
    max_diffs_AC = []
    sum_diffs_AC = []
    paths = []

    print(f"Testing Laplacian-of-Gaussian permutations on {base}; outputs -> {out_dir}")

    for sigma in sigma_list:
        print(f"\n=== sigma = {sigma} ===")

        # ---------- METHOD A: l * (g * f) ----------
        t0 = time.perf_counter()
        # use separable gaussian (horizontal full then vertical full -> crop)
        g_row = gaussian_kernel_1d(sigma)   # row 1 x m
        # first pass (horizontal) full
        g_h_full = convolve(g_row, f, c=0, zeroPaddingNeeded=True, completeResult=True)
        # second pass (vertical) full
        g_full = convolve(g_row.flatten(), g_h_full, c=0, zeroPaddingNeeded=True, completeResult=True, kernel_is_column=True)
        # crop to HxW
        pad = g_row.shape[1] // 2
        g_smoothed = g_full[pad:pad+H, pad:pad+W]
        # apply Laplacian (use zero padding)
        A = convolve(l_kernel, g_smoothed, c=0, zeroPaddingNeeded=True, completeResult=False)
        t1 = time.perf_counter()
        timeA = t1 - t0
        times_A.append(timeA)
        print(f"Method A (l * (g * f)) time: {timeA:.4f}s")

        # ---------- METHOD B: g * (l * f) ----------
        t0 = time.perf_counter()
        # first apply Laplacian on original
        lf = convolve(l_kernel, f, c=0, zeroPaddingNeeded=True, completeResult=False)
        # then smooth lf with separable gaussian (same procedure)
        g_row = gaussian_kernel_1d(sigma)
        tmp_h_full = convolve(g_row, lf, c=0, zeroPaddingNeeded=True, completeResult=True)
        tmp_full = convolve(g_row.flatten(), tmp_h_full, c=0, zeroPaddingNeeded=True, completeResult=True, kernel_is_column=True)
        B = tmp_full[pad:pad+H, pad:pad+W]
        t1 = time.perf_counter()
        timeB = t1 - t0
        times_B.append(timeB)
        print(f"Method B (g * (l * f)) time: {timeB:.4f}s")

        # ---------- METHOD C: (l * g) * f ----------
        t0 = time.perf_counter()
        # Get full 2D Gaussian kernel (we will convolve it with laplacian kernel to get combined kernel)
        g2d = gaussian_kernel_2d(sigma)
        # compute kernel L * G (linear convolution of kernel arrays)
        # Note: kernel_convolve computes linear conv: (l * g)
        LG = kernel_convolve(l_kernel, g2d)
        # now apply combined kernel to image (use zero padding)
        C = convolve(LG, f, c=0, zeroPaddingNeeded=True, completeResult=False)
        t1 = time.perf_counter()
        timeC = t1 - t0
        times_C.append(timeC)
        print(f"Method C ((l * g) * f) time: {timeC:.4f}s")

        # ---------- comparisons ----------
        diff_AB = (A.astype(np.int32) - B.astype(np.int32))
        maxAB = int(np.max(np.abs(diff_AB)))
        sumAB = int(np.sum(np.abs(diff_AB)))
        max_diffs_AB.append(maxAB)
        sum_diffs_AB.append(sumAB)

        diff_AC = (A.astype(np.int32) - C.astype(np.int32))
        maxAC = int(np.max(np.abs(diff_AC)))
        sumAC = int(np.sum(np.abs(diff_AC)))
        max_diffs_AC.append(maxAC)
        sum_diffs_AC.append(sumAC)

        print(f" max|A-B| = {maxAB}, sum|A-B| = {sumAB}")
        print(f" max|A-C| = {maxAC}, sum|A-C| = {sumAC}")
        if maxAB > compare_tol_max or sumAB > compare_tol_sum:
            print(" WARNING: A and B differ beyond tolerance")
        if maxAC > compare_tol_max or sumAC > compare_tol_sum:
            print(" WARNING: A and C differ beyond tolerance")

        # Save visualizations (add c=128 to center for display)
        visA = np.clip(A.astype(np.int32) + 128, 0, 255).astype(np.uint8)
        visB = np.clip(B.astype(np.int32) + 128, 0, 255).astype(np.uint8)
        visC = np.clip(C.astype(np.int32) + 128, 0, 255).astype(np.uint8)
        fnameA = os.path.join(out_dir, f"{base}_sigma{sigma:.2f}_A_l_gf.png")
        fnameB = os.path.join(out_dir, f"{base}_sigma{sigma:.2f}_B_g_lf.png")
        fnameC = os.path.join(out_dir, f"{base}_sigma{sigma:.2f}_C_lg_f.png")
        Image.fromarray(visA).save(fnameA)
        Image.fromarray(visB).save(fnameB)
        Image.fromarray(visC).save(fnameC)
        paths.append((fnameA, fnameB, fnameC))

    # Plot runtimes for the three methods
    plt.figure(figsize=(8,5))
    plt.plot(sigma_list, times_A, marker='o', label='l * (g * f)')
    plt.plot(sigma_list, times_B, marker='o', label='g * (l * f)')
    plt.plot(sigma_list, times_C, marker='o', label='(l * g) * f')
    plt.xlabel('sigma')
    plt.ylabel('compute time (s)')
    plt.title(f'Lap(Gauss) permutations runtimes (image={base})')
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(out_dir, f"{base}_lapgauss_times.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    summary = {
        "sigma_list": sigma_list,
        "times_A": times_A,
        "times_B": times_B,
        "times_C": times_C,
        "max_diffs_AB": max_diffs_AB,
        "sum_diffs_AB": sum_diffs_AB,
        "max_diffs_AC": max_diffs_AC,
        "sum_diffs_AC": sum_diffs_AC,
        "image_paths": paths,
        "plot_path": plot_path,
        "out_dir": out_dir
    }
    return summary

if __name__ == "__main__":
    # convolutionTest()
    # laplacican("./testIMG1.jpg")
    testGaussian("./testIMG1.jpg", sigma_list=[1,2,4,8,12,20])
