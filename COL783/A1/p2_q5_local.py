import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import math


def convolve(
    w,
    f,
    c=0,
    zeroPaddingNeeded=False,
    completeResult=False,
    kernel_is_column=None,
    isFloat=False,
):
    w = np.asarray(w, dtype=np.float64)
    if isFloat:
        f = np.asarray(f, dtype=np.float64)
    else:
        f = np.asarray(f)
    if w.ndim == 1:
        if kernel_is_column is None or kernel_is_column is False:
            w = w.reshape(1, -1)
        else:
            w = w.reshape(-1, 1)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    if f.ndim != 2 or w.ndim != 2:
        raise ValueError("f and w must be 1D or 2D arrays (grayscale kernels/images).")
    H, W = f.shape
    kh, kw = w.shape
    if kh == 0 or kw == 0:
        raise ValueError("Kernel must have non-zero shape.")
    w_flipped = np.flip(np.flip(w, axis=0), axis=1)
    pad_mode = "constant" if zeroPaddingNeeded else "reflect"
    if not completeResult:
        pad_top = kh // 2
        pad_bottom = kh - pad_top - 1
        pad_left = kw // 2
        pad_right = kw - pad_left - 1
        const_val = 0.0 if isFloat else 0
        if pad_mode == "constant":
            padded = np.pad(
                f.astype(np.float64),
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=const_val,
            )
        else:
            padded = np.pad(
                f.astype(np.float64),
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="reflect",
            )
        out_h, out_w = H, W
        out = np.empty((out_h, out_w), dtype=np.float64)
        for y in range(out_h):
            for x in range(out_w):
                patch = padded[y : y + kh, x : x + kw]
                out[y, x] = np.sum(patch * w_flipped)
    else:
        pad_top = kh - 1
        pad_bottom = kh - 1
        pad_left = kw - 1
        pad_right = kw - 1
        const_val = 0.0 if isFloat else 0
        if pad_mode == "constant":
            padded = np.pad(
                f.astype(np.float64),
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=const_val,
            )
        else:
            padded = np.pad(
                f.astype(np.float64),
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="reflect",
            )
        full_h = H + kh - 1
        full_w = W + kw - 1
        out = np.empty((full_h, full_w), dtype=np.float64)
        for y in range(full_h):
            for x in range(full_w):
                patch = padded[y : y + kh, x : x + kw]
                out[y, x] = np.sum(patch * w_flipped)
    out = out + float(c)
    if isFloat:
        return out
    else:
        out = np.clip(out, 0.0, 255.0)
        return out.astype(np.uint8)


def laplacican(input_image_path):
    img = Image.open(input_image_path).convert("L")
    f = np.array(img, dtype=np.uint8)
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    g = convolve(lap_kernel, f, c=128)
    base, _ = os.path.splitext(input_image_path)
    out_path = base + "_laplacian.png"
    out_img = Image.fromarray(g)
    out_img.save(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original (grayscale)")
    axes[0].axis("off")
    axes[1].imshow(g, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Laplacian (c=128)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    print(f"Laplacian saved to: {out_path}")


def convolutionTest():
    np.set_printoptions(formatter={"int": lambda x: f"{x:3d}"})
    f1 = np.zeros((5, 5), dtype=np.uint8)
    f1[2, 2] = 1
    w1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
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
    f2 = np.array([1, 1, 1], dtype=np.uint8)
    w2 = np.array([1, 2, 1], dtype=float)
    print("\n=== Test 2: f = column [1,1,1], w = row [1,2,1] (as before) ===")
    for zeroPad in (True, False):
        print(f"\nPadding mode: {'zero-pad' if zeroPad else 'reflect'}")
        print("completeResult=False (same):")
        print(convolve(w2, f2, c=0, zeroPaddingNeeded=zeroPad, completeResult=False))
        print("completeResult=True (full):")
        print(convolve(w2, f2, c=0, zeroPaddingNeeded=zeroPad, completeResult=True))
    f3 = np.array([[1, 1, 1]], dtype=np.uint8)
    w3 = np.array([1, 2, 1], dtype=float)
    print("\n=== Test 3 (NEW): f = row [[1,1,1]] (1x3), w = column [1,2,1] (3x1) ===")
    for zeroPad in (True, False):
        print(f"\nPadding mode: {'zero-pad' if zeroPad else 'reflect'}")
        print("completeResult=False (same):")
        out_same = convolve(
            w3,
            f3,
            c=0,
            zeroPaddingNeeded=zeroPad,
            completeResult=False,
            kernel_is_column=True,
        )
        print(out_same)
        print("completeResult=True (full):")
        out_full = convolve(
            w3,
            f3,
            c=0,
            zeroPaddingNeeded=zeroPad,
            completeResult=True,
            kernel_is_column=True,
        )
        print(out_full)


def gaussian_kernel_1d(sigma):
    if sigma <= 0:
        sigma = 1e-6
    a = int(math.ceil(3.0 * sigma))
    xs = np.arange(-a, a + 1, dtype=np.float64)
    g = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g.reshape(1, -1)


def gaussian_kernel_2d(sigma):
    if sigma <= 0:
        sigma = 1e-6
    a = int(math.ceil(3.0 * sigma))
    xs = np.arange(-a, a + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    g = np.exp(-(X * X + Y * Y) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g


def gaussianFilter1(imagePath, sigma, out_dir=None):
    img = Image.open(imagePath).convert("L")
    f = np.array(img, dtype=np.uint8)
    H, W = f.shape
    K = gaussian_kernel_2d(sigma)
    t0 = time.perf_counter()
    g = convolve(K, f, c=0, zeroPaddingNeeded=True, completeResult=False)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    base = os.path.splitext(os.path.basename(imagePath))[0]
    folder = out_dir or f"./GaussianOutputs_{base}"
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"{base}_gauss2d_sigma{sigma:.2f}.png")
    Image.fromarray(g).save(out_path)
    return g, elapsed, out_path


def gaussianFilter2(imagePath, sigma, out_dir=None):
    img = Image.open(imagePath).convert("L")
    f = np.array(img, dtype=np.uint8)
    H, W = f.shape
    g1d_row = gaussian_kernel_1d(sigma)
    m = g1d_row.shape[1]
    pad = m // 2
    folder_base = (
        out_dir
        or f"./GaussianOutputs_{os.path.splitext(os.path.basename(imagePath))[0]}"
    )
    os.makedirs(folder_base, exist_ok=True)
    t0 = time.perf_counter()
    out1_full = convolve(g1d_row, f, c=0, zeroPaddingNeeded=True, completeResult=True)
    out2_full = convolve(
        g1d_row.flatten(),
        out1_full,
        c=0,
        zeroPaddingNeeded=True,
        completeResult=True,
        kernel_is_column=True,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    start_r = pad
    start_c = pad
    cropped = out2_full[start_r : start_r + H, start_c : start_c + W]
    base = os.path.splitext(os.path.basename(imagePath))[0]
    out_path = os.path.join(folder_base, f"{base}_gausssep_sigma{sigma:.2f}.png")
    Image.fromarray(cropped).save(out_path)
    return cropped, elapsed, out_path


def testGaussian(
    imagePath,
    sigma_list=None,
    out_root="./GaussianOutputs",
    compare_tol=2,
    compare_tol_sum=100,
):
    if sigma_list is None:
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
        print(
            f"gaussianFilter1 (2D kernel):   time = {t2d:.4f} s, saved: {os.path.basename(path2d)}"
        )
        gsep, tsep, pathsep = gaussianFilter2(imagePath, sigma, out_dir=out_dir)
        print(
            f"gaussianFilter2 (separable):   time = {tsep:.4f} s, saved: {os.path.basename(pathsep)}"
        )
        times_2d.append(t2d)
        times_sep.append(tsep)
        paths_2d.append(path2d)
        paths_sep.append(pathsep)
        diff = g2d.astype(np.int32) - gsep.astype(np.int32)
        max_abs_diff = int(np.max(np.abs(diff)))
        sum_abs_diff = int(np.sum(np.abs(diff)))
        max_diffs.append(max_abs_diff)
        sum_diffs.append(sum_abs_diff)
        ok_max = max_abs_diff <= compare_tol
        ok_sum = sum_abs_diff <= compare_tol_sum
        print(
            f" max_abs_diff = {max_abs_diff} (<= {compare_tol}? {'OK' if ok_max else 'FAIL'})"
        )
        print(
            f" sum_abs_diff = {sum_abs_diff} (<= {compare_tol_sum}? {'OK' if ok_sum else 'FAIL'})"
        )
        if not (ok_max and ok_sum):
            print(
                "  WARNING: outputs differ beyond tolerances (consider increasing compare_tol_sum or using float-accumulation)."
            )
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_list, times_2d, marker="o", label="gaussianFilter1 (2D kernel)")
    plt.plot(sigma_list, times_sep, marker="o", label="gaussianFilter2 (separable)")
    plt.xlabel("sigma")
    plt.ylabel("compute time (seconds)")
    plt.title(f"Gaussian filter compute time (image={base})")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(out_dir, f"{base}_gaussian_times.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print("\nSummary:")
    for s, t1, t2, md, sd in zip(sigma_list, times_2d, times_sep, max_diffs, sum_diffs):
        print(
            f" sigma={s:5.2f}  time2D={t1:.4f}s  timeSep={t2:.4f}s  maxDiff={md}  sumDiff={sd}"
        )
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
        "plot_path": plot_path,
    }


def kernel_convolve(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ha, wa = A.shape
    hb, wb = B.shape
    hc = ha + hb - 1
    wc = wa + wb - 1
    C = np.zeros((hc, wc), dtype=np.float64)
    for i in range(ha):
        for j in range(wa):
            C[i : i + hb, j : j + wb] += A[i, j] * B
    return C


def testLapGauss(
    imagePath=None,
    sigma_list=None,
    out_root="./LapGaussOutputs",
    compare_tol_max=2,
    compare_tol_sum=50,
    loadFromNumpyData=False,
    numpy_array=None,
):
    if sigma_list is None:
        sigma_list = [1.0, 2.0, 4.0, 8.0, 12.0]
    if loadFromNumpyData:
        if numpy_array is None:
            raise ValueError("numpy_array required when loadFromNumpyData=True")
        f = np.asarray(numpy_array, dtype=np.uint8)
        base = "array_input"
    else:
        if imagePath is None:
            raise ValueError("imagePath must be provided when loadFromNumpyData=False")
        img = Image.open(imagePath).convert("L")
        f = np.array(img, dtype=np.uint8)
        base = os.path.splitext(os.path.basename(imagePath))[0]
    H, W = f.shape
    out_dir = os.path.join(out_root + "_" + base)
    os.makedirs(out_dir, exist_ok=True)
    l_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
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
        # Method A
        t0 = time.perf_counter()
        g_row = gaussian_kernel_1d(sigma)
        m = g_row.shape[1]
        pad = m // 2
        g_h_full = convolve(
            g_row, f, c=0.0, zeroPaddingNeeded=True, completeResult=True, isFloat=True
        )
        g_full = convolve(
            g_row.flatten(),
            g_h_full,
            c=0.0,
            zeroPaddingNeeded=True,
            completeResult=True,
            kernel_is_column=True,
            isFloat=True,
        )
        g_smoothed = g_full[pad : pad + H, pad : pad + W]
        A = convolve(
            l_kernel,
            g_smoothed,
            c=0.0,
            zeroPaddingNeeded=True,
            completeResult=False,
            isFloat=True,
        )
        t1 = time.perf_counter()
        timeA = t1 - t0
        times_A.append(timeA)
        print(f"Method A (l * (g * f)) time: {timeA:.4f}s")
        # Method B
        t0 = time.perf_counter()
        lf = convolve(
            l_kernel,
            f,
            c=0.0,
            zeroPaddingNeeded=True,
            completeResult=False,
            isFloat=True,
        )
        g_row = gaussian_kernel_1d(sigma)
        m = g_row.shape[1]
        pad = m // 2
        tmp_h_full = convolve(
            g_row, lf, c=0.0, zeroPaddingNeeded=True, completeResult=True, isFloat=True
        )
        tmp_full = convolve(
            g_row.flatten(),
            tmp_h_full,
            c=0.0,
            zeroPaddingNeeded=True,
            completeResult=True,
            kernel_is_column=True,
            isFloat=True,
        )
        B = tmp_full[pad : pad + H, pad : pad + W]
        t1 = time.perf_counter()
        timeB = t1 - t0
        times_B.append(timeB)
        print(f"Method B (g * (l * f)) time: {timeB:.4f}s")
        # Method C
        t0 = time.perf_counter()
        g2d = gaussian_kernel_2d(sigma)
        LG = kernel_convolve(l_kernel, g2d)
        C = convolve(
            LG, f, c=0.0, zeroPaddingNeeded=True, completeResult=False, isFloat=True
        )
        t1 = time.perf_counter()
        timeC = t1 - t0
        times_C.append(timeC)
        print(f"Method C ((l * g) * f) time: {timeC:.4f}s")
        # Stats
        diff_AB = A.astype(np.int32) - B.astype(np.int32)
        maxAB = int(np.max(np.abs(diff_AB)))
        sumAB = int(np.sum(np.abs(diff_AB)))
        max_diffs_AB.append(maxAB)
        sum_diffs_AB.append(sumAB)
        diff_AC = A.astype(np.int32) - C.astype(np.int32)
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
        visA = np.clip((A + 128.0), 0.0, 255.0).astype(np.uint8)
        visB = np.clip((B + 128.0), 0.0, 255.0).astype(np.uint8)
        visC = np.clip((C + 128.0), 0.0, 255.0).astype(np.uint8)
        fnameA = os.path.join(out_dir, f"{base}_sigma{sigma:.2f}_A_l_gf.png")
        fnameB = os.path.join(out_dir, f"{base}_sigma{sigma:.2f}_B_g_lf.png")
        fnameC = os.path.join(out_dir, f"{base}_sigma{sigma:.2f}_C_lg_f.png")
        Image.fromarray(visA).save(fnameA)
        Image.fromarray(visB).save(fnameB)
        Image.fromarray(visC).save(fnameC)
        paths.append((fnameA, fnameB, fnameC))
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_list, times_A, marker="o", label="l * (g * f)")
    plt.plot(sigma_list, times_B, marker="o", label="g * (l * f)")
    plt.plot(sigma_list, times_C, marker="o", label="(l * g) * f")
    plt.xlabel("sigma")
    plt.ylabel("compute time (s)")
    plt.title(f"Lap(Gauss) permutations runtimes (image={base})")
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
        "out_dir": out_dir,
    }
    return summary


if __name__ == "__main__":
    convolutionTest()
    laplacican("./testIMG1.jpg")
    testGaussian("./testIMG1.jpg", sigma_list=[1,2,4,8,12,20])
    testLapGauss(imagePath="./testIMG1.jpg", sigma_list=[1, 2, 4, 8, 12, 20])
