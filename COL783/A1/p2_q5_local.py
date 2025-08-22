import numpy as np

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


def main():
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


if __name__ == "__main__":
    main()
