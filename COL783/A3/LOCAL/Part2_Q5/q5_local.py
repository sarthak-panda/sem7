import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import pickle

def ensure_dirs():
    parts = ["PartA", "PartB", "PartC", "PartD", "PartE", "PartF"]
    if not os.path.exists("./Q5_Output"):
        os.mkdir("./Q5_Output")
    for p in parts:
        part_dir = f"./Q5_Output/{p}"
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)

def read_image_gray(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return arr

def write_image(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def plot_histogram(arr, output_path, title):
    plt.figure()
    plt.hist(arr.flatten(), bins=256, range=(0,255))
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def entropy(arr):
    hist = np.bincount(arr.flatten(), minlength=256)
    p = hist / np.sum(hist)
    H = -np.sum([x*math.log2(x) for x in p if x > 0])
    return H

def standard_deviation(arr):
    return np.std(arr.flatten())

# --- Part A ---
def paeth_predictor(a, b, c):
    # a = left, b = above, c = upper left
    a = int(a); b = int(b); c = int(c)
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc: return a
    elif pb <= pc: return b
    else: return c

def compute_error_image(img):
    H, W = img.shape
    err = np.zeros_like(img, dtype=np.int16)
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                pred = 0
            elif i == 0:
                pred = img[i,j-1]
            elif j == 0:
                pred = img[i-1,j]
            else:
                pred = paeth_predictor(img[i,j-1], img[i-1,j], img[i-1,j-1])
            err[i,j] = (int(img[i,j]) - int(pred)) % 256
    return err

def compute_error_image_updated_for_PartA_B_Piazza(img):
    H, W = img.shape
    err = np.zeros_like(img, dtype=np.int16)
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                pred = 0
            elif i == 0:
                pred = img[i, j-1]
            elif j == 0:
                pred = img[i-1, j]
            else:
                pred = paeth_predictor(img[i, j-1], img[i-1, j], img[i-1, j-1])
            raw_error = int(img[i, j]) - int(pred)
            # Rescale raw_error from [-255, 255] to [0, 255]
            err[i, j] = raw_error + 255 // 2
    return err

def decode_from_error_updated_for_PartA_B_Piazza(err_img):
    H, W = err_img.shape
    out = np.zeros_like(err_img, dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                pred = 0
            elif i == 0:
                pred = out[i, j-1]
            elif j == 0:
                pred = out[i-1, j]
            else:
                pred = paeth_predictor(out[i, j-1], out[i-1, j], out[i-1, j-1])
            # Recover original value
            raw_error = int(err_img[i, j]) - 255 // 2
            out[i, j] = np.clip(int(pred) + raw_error, 0, 255)
    return out


def save_partA(img, err, name):
    write_image(err, f"./Q5_Output/PartA/{name}_error.png")
    hist_path = f"./Q5_Output/PartA/{name}_error_hist.png"
    plot_histogram(err, hist_path, f"Error histogram {name}")
    std = standard_deviation(err)
    ent = entropy(err)
    with open(f"./Q5_Output/PartA/{name}_stats.txt", "w") as f:
        f.write(f"Standard deviation: {std}\n")
        f.write(f"Entropy: {ent}\n")
    return std, ent

def partA(paths):
    print("======Part-A processing...======")
    results = {}
    for fname in paths:
        img = read_image_gray(fname)
        err = compute_error_image(img)
        err_piazza = compute_error_image_updated_for_PartA_B_Piazza(img)
        base = os.path.splitext(os.path.basename(fname))[0]
        std, ent = save_partA(img, err_piazza, base)
        results[base] = (img, err, std, ent, err_piazza)
    return results

# --- Part B ---
def decode_from_error(err_img):
    H, W = err_img.shape
    out = np.zeros_like(err_img, dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                pred = 0
            elif i == 0:
                pred = out[i,j-1]
            elif j == 0:
                pred = out[i-1,j]
            else:
                pred = paeth_predictor(out[i,j-1], out[i-1,j], out[i-1,j-1])
            out[i,j] = (int(pred) + int(err_img[i,j])) % 256
    return out

def save_partB(decoded, img, name):
    write_image(decoded, f"./Q5_Output/PartB/{name}_decoded.png")
    # Compare, output result of np.array_equal
    with open(f"./Q5_Output/PartB/{name}_verify.txt", "w") as f:
        matched = np.array_equal(decoded, img)
        f.write(f"Decoded equal to original: {matched}\n")

def partB(results):
    print("======Part-B processing...======")
    for base, (img, err, _, _) in results.items():
        decoded = decode_from_error_updated_for_PartA_B_Piazza(err)
        save_partB(decoded, img, base)

# --- Part C ---
class LZW:
    @staticmethod
    def encode(data):
        dict_size = 256
        dictionary = {bytes([i]): i for i in range(dict_size)}
        w = b""
        result = []
        for c in data:
            wc = w + bytes([c])
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = bytes([c])
        if w:
            result.append(dictionary[w])
        return result

    @staticmethod
    def decode(codes):
        dict_size = 256
        dictionary = {i: bytes([i]) for i in range(dict_size)}
        result = bytearray()
        w = bytes([codes[0]])
        result += w
        for k in codes[1:]:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[:1]
            else:
                raise ValueError("Bad LZW code")
            result += entry
            dictionary[dict_size] = w + entry[:1]
            dict_size += 1
            w = entry
        return list(result)

def unsigned8_list(arr):
    return arr.flatten().astype(np.uint8).tolist()

def save_partC():
    print("======Part-C processing...======")
    zeros = [0] * 100  # N=100 for example
    codes = LZW.encode(zeros)
    decoded = LZW.decode(codes)
    with open("./Q5_Output/PartC/lzw_test.txt", "w") as f:
        f.write("Encoded length: {}\n".format(len(codes)))
        f.write("Decoded equals input: {}\n".format(decoded == zeros))
    # t=LZW.encode([39,39,126,126,39,39,126,126,39,39,126,126,39,39,126,126])
    # print(t)
    # print(LZW.decode(t))
    # Math proof: For N zeros, output O(sqrt(N)) codes. Each code creates a new dict entry until all possible strings.

# --- Part D ---
import heapq

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other): return self.freq < other.freq

def build_huffman_table(symbols, probs):
    heap = [HuffmanNode(s, p) for s, p in zip(symbols, probs)]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        node = HuffmanNode(None, lo.freq + hi.freq)
        node.left = lo
        node.right = hi
        heapq.heappush(heap, node)
    root = heap[0]
    code_table = {}
    def assign_codes(node, path):
        if node.symbol is not None:
            code_table[node.symbol] = path[:]
        else:
            assign_codes(node.left, path+[0])
            assign_codes(node.right, path+[1])
    assign_codes(root, [])
    return code_table

def huffman_encode(data, table):
    bits = []
    for s in data:
        bits.extend(table[s])
    return bits

def huffman_decode(bits, table):
    reverse_table = {tuple(v): k for k,v in table.items()}
    out = []
    buf = []
    for b in bits:
        buf.append(b)
        t = tuple(buf)
        if t in reverse_table:
            out.append(reverse_table[t])
            buf = []
    return out

def save_partD(lzw_out, name):
    # Frequency count
    symbols, counts = np.unique(lzw_out, return_counts=True)
    probs = counts / np.sum(counts)
    table = build_huffman_table(symbols, probs)
    encoded_bits = huffman_encode(lzw_out, table)
    avg_code_len = np.mean([len(table[s]) for s in lzw_out])#this will have repetation, right
    H = entropy(np.array(lzw_out))
    with open(f"./Q5_Output/PartD/{name}_huffman_stats.txt", "w") as f:
        f.write(f"Entropy: {H}\n")
        f.write(f"Average Huffman code length: {avg_code_len}\n")
        f.write(f"Total encoded bits: {len(encoded_bits)}\n")
    with open(f"./Q5_Output/PartD/{name}_table.pkl","wb") as f:
        pickle.dump(table,f)
    with open(f"./Q5_Output/PartD/{name}_bits.pkl","wb") as f:
        pickle.dump(encoded_bits, f)

def partD(results):
    print("======Part-D processing...======")
    for base, (_, err, _, _) in results.items():
        err_list = unsigned8_list(err)
        lzw_out = LZW.encode(err_list)
        save_partD(lzw_out, base)

# --- Part E ---
def encode_full(img):
    # Step 1: predictive coding
    err_img = compute_error_image(img)
    # Step 2: LZW
    lzw_out = LZW.encode(unsigned8_list(err_img))
    # Step 3: Huffman
    symbols, counts = np.unique(lzw_out, return_counts=True)
    probs = counts / np.sum(counts)
    H = entropy(np.array(unsigned8_list(err_img)))
    table = build_huffman_table(symbols, probs)
    bits = huffman_encode(lzw_out, table)
    return table, bits, err_img, lzw_out, H

def decode_full(M, N, table, bits):
    # Step 1: Huffman decode
    lzw_out = huffman_decode(bits, table)
    # Step 2: LZW decode
    err_list = LZW.decode(lzw_out)
    # Step 3: reshape as error image
    err_img = np.array(err_list, dtype=np.uint8).reshape((M,N))
    # Step 4: predictive decode
    img = decode_from_error(err_img)
    return img

def save_partE(img, M, N, name):
    table, bits, err_img, lzw_out, H = encode_full(img)
    # Save artifacts
    with open(f"./Q5_Output/PartE/{name}_table.pkl","wb") as f:
        pickle.dump(table, f)
    with open(f"./Q5_Output/PartE/{name}_bits.pkl","wb") as f:
        pickle.dump(bits, f)
    decoded = decode_full(M,N,table,bits)

    # --- exact-match check ---
    # if np.array_equal(decoded, img):
    #     print(f"{name}: decoded matches original exactly.")
    # else:
    #     # diagnostics to help debug small differences
    #     diff = decoded.astype(np.int64) - img.astype(np.int64)
    #     n_diff = int(np.count_nonzero(diff))
    #     total = img.size
    #     min_diff = int(diff.min()) if n_diff > 0 else 0
    #     max_diff = int(diff.max()) if n_diff > 0 else 0
    #     print(f"{name}: decoded DOES NOT match original.")
    #     print(f"  differing elements: {n_diff}/{total}")
    #     print(f"  diff range: min={min_diff}, max={max_diff}")
        # If you want to stop execution on mismatch, uncomment the next line:
        # raise AssertionError(f"{name}: decoded image does not match original exactly.")
    # ---------------------------

    write_image(decoded, f"./Q5_Output/PartE/{name}_decoded.png")
    cr_pred = 8/H
    cr_lzw = M*N / len(lzw_out)
    cr_total = 8*M*N / len(bits)
    with open(f"./Q5_Output/PartE/{name}_ratios.txt","w") as f:
        f.write(f"Prediction compression ratio: {cr_pred}\n")
        f.write(f"LZW compression ratio: {cr_lzw}\n")
        f.write(f"Actual compression ratio: {cr_total}\n")
    return cr_pred, cr_lzw, cr_total

def partE(results):
    print("======Part-E processing...======")
    crs = {}
    for base, (img, _, _, _) in results.items():
        M,N = img.shape
        cr_pred, cr_lzw, cr_total = save_partE(img, M, N, base)
        crs[base] = (cr_pred, cr_lzw, cr_total)
    return crs

# --- Part F ---
# def compute_error_all_schemes(img):
#     H,W = img.shape
#     err_rows = []
#     scheme_rows = []
#     for i in range(H):
#         row_errs = []
#         row_schemes = []
#         candidates = []
#         # Generate all 5
#         # scheme 0: none, scheme 1: left, scheme 2: above, scheme 3: upper left (average), scheme 4: Paeth
#         for scheme in range(5):
#             err = np.zeros(W, dtype=np.uint8)
#             for j in range(W):
#                 if scheme == 0: pred = 0
#                 elif scheme == 1: pred = img[i,j-1] if j > 0 else 0
#                 elif scheme == 2: pred = img[i-1,j] if i > 0 else 0
#                 elif scheme == 3: pred = ((int(img[i,j-1]) if j>0 else 0)+(int(img[i-1,j]) if i>0 else 0)//2) if i>0 and j>0 else 0
#                 elif scheme == 4: pred = paeth_predictor(
#                     img[i,j-1] if j>0 else 0,
#                     img[i-1,j] if i>0 else 0,
#                     img[i-1,j-1] if i>0 and j>0 else 0)
#                 err[j] = (int(img[i,j])-int(pred))%256
#             candidates.append((err, scheme))
#         best_err, best_scheme = min(candidates, key=lambda x: entropy(x[0]))
#         err_rows.append(best_err)
#         scheme_rows.append(best_scheme)
#     err_img = np.stack(err_rows,axis=0)
#     return err_img, scheme_rows

# def partF(results):
#     print("======Part-F processing...======")
#     # Scheme selection
#     for base, (img, _, _, _) in results.items():
#         M,N = img.shape
#         err_img, scheme_rows = compute_error_all_schemes(img)
#         # Save artifacts
#         write_image(err_img, f"./Q5_Output/PartF/{base}_adapterror.png")
#         with open(f"./Q5_Output/PartF/{base}_schemes.pkl","wb") as f:
#             pickle.dump(scheme_rows,f)
#         # Encode/decode as in E
#         lzw_out = LZW.encode(unsigned8_list(err_img))
#         symbols, counts = np.unique(lzw_out, return_counts=True)
#         probs = counts/np.sum(counts)
#         table = build_huffman_table(symbols,probs)
#         bits = huffman_encode(lzw_out,table)
#         # Save table and bits
#         with open(f"./Q5_Output/PartF/{base}_table.pkl","wb") as f:
#             pickle.dump(table,f)
#         with open(f"./Q5_Output/PartF/{base}_bits.pkl","wb") as f:
#             pickle.dump(bits,f)
#         # Decode
#         lzw_out_decoded = huffman_decode(bits,table)
#         err_list = LZW.decode(lzw_out_decoded)
#         err_img_dec = np.array(err_list,dtype=np.uint8).reshape((M,N))
#         decoded = decode_from_error(err_img_dec)#problematic? should be adaptive decode per row, right?
#         write_image(decoded, f"./Q5_Output/PartF/{base}_decoded.png")
#         # Run Length Encoding option
#         # RLE encoding for each row:
#         rle_codes = []
#         for row in err_img:
#             # (run length, value)
#             rle_row = []
#             prev = row[0]
#             cnt = 1
#             for v in row[1:]:
#                 if v==prev: cnt+=1
#                 else:
#                     rle_row.append( (cnt, prev) )
#                     prev = v
#                     cnt = 1
#             rle_row.append( (cnt, prev) )
#             rle_codes.extend(rle_row)
#         # Huffman on RLE
#         rle_symbols = [x for x in rle_codes]
#         sym_set, rle_counts = np.unique(rle_symbols,return_counts=True,axis=0)
#         probs = rle_counts / np.sum(rle_counts)
#         # Map sym_set to tuple for hashability
#         sym_list = [tuple(x) for x in sym_set]
#         rle_c_table = build_huffman_table(list(range(len(sym_list))),probs)
#         # Map rle_symbols to sym_list indices
#         rle_encoded_syms = [ sym_list.index(tuple(x)) for x in rle_symbols ]
#         rle_bits = huffman_encode(rle_encoded_syms, rle_c_table)
#         with open(f"./Q5_Output/PartF/{base}_rle_table.pkl","wb") as f:
#             pickle.dump((sym_list,rle_c_table),f)
#         with open(f"./Q5_Output/PartF/{base}_rle_bits.pkl","wb") as f:
#             pickle.dump(rle_bits,f)
#         cr_rle = 8*M*N/len(rle_bits)
#         with open(f"./Q5_Output/PartF/{base}_rle_stats.txt","w") as f:
#             f.write(f"RLE actual compression ratio: {cr_rle}\n")

def compute_error_all_schemes(img):
    """
    Compute error image with adaptive per-row prediction schemes.
    Returns error image of shape H x (W+1), where column 0 contains the scheme index.
    """
    H, W = img.shape
    err_img = np.zeros((H, W+1), dtype=np.uint8)
    
    for i in range(H):
        candidates = []
        # Generate all 5 schemes
        # scheme 0: none, scheme 1: left, scheme 2: above, scheme 3: average, scheme 4: Paeth
        for scheme in range(5):
            err = np.zeros(W, dtype=np.uint8)
            for j in range(W):
                if scheme == 0: 
                    pred = 0
                elif scheme == 1: 
                    pred = img[i, j-1] if j > 0 else 0
                elif scheme == 2: 
                    pred = img[i-1, j] if i > 0 else 0
                elif scheme == 3: 
                    # Average of left and above
                    left = int(img[i, j-1]) if j > 0 else 0
                    above = int(img[i-1, j]) if i > 0 else 0
                    pred = (left + above) // 2
                elif scheme == 4: 
                    pred = paeth_predictor(
                        img[i, j-1] if j > 0 else 0,
                        img[i-1, j] if i > 0 else 0,
                        img[i-1, j-1] if i > 0 and j > 0 else 0)
                err[j] = (int(img[i, j]) - int(pred)) % 256
            candidates.append((err, scheme))
        
        # Choose scheme with minimum entropy for this row
        best_err, best_scheme = min(candidates, key=lambda x: entropy(x[0]))
        err_img[i, 0] = best_scheme  # Store scheme in first column
        err_img[i, 1:] = best_err     # Store error values in remaining columns
    
    return err_img


def decode_from_error_adaptive(err_img):
    """
    Decode error image using per-row adaptive prediction schemes.
    
    Args:
        err_img: Error image (H x W+1), where column 0 contains scheme indices
    
    Returns:
        Decoded image (H x W)
    """
    H, W_plus_1 = err_img.shape
    W = W_plus_1 - 1
    out = np.zeros((H, W), dtype=np.uint8)
    
    for i in range(H):
        scheme = int(err_img[i, 0])
        for j in range(W):
            if scheme == 0:
                pred = 0
            elif scheme == 1:
                pred = out[i, j-1] if j > 0 else 0
            elif scheme == 2:
                pred = out[i-1, j] if i > 0 else 0
            elif scheme == 3:
                # Average of left and above
                left = int(out[i, j-1]) if j > 0 else 0
                above = int(out[i-1, j]) if i > 0 else 0
                pred = (left + above) // 2
            elif scheme == 4:
                pred = paeth_predictor(
                    out[i, j-1] if j > 0 else 0,
                    out[i-1, j] if i > 0 else 0,
                    out[i-1, j-1] if i > 0 and j > 0 else 0)
            
            out[i, j] = (int(pred) + int(err_img[i, j+1])) % 256
    
    return out


class RLE:
    @staticmethod
    def encode(data):
        """
        Run-length encode a list of values.
        Returns list of (run_length, value) tuples.
        """
        if len(data) == 0:
            return []
        
        runs = []
        prev = data[0]
        cnt = 1
        
        for v in data[1:]:
            if v == prev:
                cnt += 1
            else:
                runs.append((cnt, prev))
                prev = v
                cnt = 1
        runs.append((cnt, prev))
        
        return runs
    
    @staticmethod
    def decode(runs):
        """
        Decode run-length encoded data.
        
        Args:
            runs: List of (run_length, value) tuples
        
        Returns:
            Decoded list of values
        """
        result = []
        for cnt, val in runs:
            result.extend([val] * cnt)
        return result


def partF(results):
    print("======Part-F processing...======")
    # Scheme selection
    for base, (img, _, _, _) in results.items():
        M, N = img.shape
        err_img = compute_error_all_schemes(img)  # Shape: M x (N+1)
        
        # Save artifacts
        write_image(err_img, f"./Q5_Output/PartF/{base}_adapterror.png")
        
        # Encode with LZW + Huffman
        err_list = unsigned8_list(err_img)
        lzw_out = LZW.encode(err_list)
        symbols, counts = np.unique(lzw_out, return_counts=True)
        probs = counts / np.sum(counts)
        table = build_huffman_table(symbols, probs)
        bits = huffman_encode(lzw_out, table)
        
        # Save table and bits
        with open(f"./Q5_Output/PartF/{base}_table.pkl", "wb") as f:
            pickle.dump(table, f)
        with open(f"./Q5_Output/PartF/{base}_bits.pkl", "wb") as f:
            pickle.dump(bits, f)
        
        # Decode
        lzw_out_decoded = huffman_decode(bits, table)
        err_list_decoded = LZW.decode(lzw_out_decoded)
        err_img_dec = np.array(err_list_decoded, dtype=np.uint8).reshape((M, N+1))
        
        # Use adaptive decoder
        decoded = decode_from_error_adaptive(err_img_dec)
        write_image(decoded, f"./Q5_Output/PartF/{base}_decoded.png")
        
        # Verify
        with open(f"./Q5_Output/PartF/{base}_verify.txt", "w") as f:
            matched = np.array_equal(decoded, img)
            f.write(f"Adaptive decoded equal to original: {matched}\n")
        
        # Calculate compression ratios
        H_err = entropy(err_img[:, 1:])  # Entropy of error values only
        cr_pred = 8 / H_err
        cr_lzw = (M * N) / len(lzw_out)
        cr_total = 8 * M * N / len(bits)
        
        with open(f"./Q5_Output/PartF/{base}_ratios.txt", "w") as f:
            f.write(f"Adaptive Prediction compression ratio: {cr_pred}\n")
            f.write(f"LZW compression ratio: {cr_lzw}\n")
            f.write(f"Actual compression ratio: {cr_total}\n")
        
        # RLE encoding option
        rle_runs = RLE.encode(err_list)
        
        # Huffman on RLE
        # Flatten runs to a list for unique symbol counting
        rle_symbols = rle_runs
        sym_set_tuples = list(set(tuple(x) for x in rle_symbols))
        sym_counts = {s: 0 for s in sym_set_tuples}
        for run in rle_symbols:
            sym_counts[tuple(run)] += 1
        
        sym_list = list(sym_counts.keys())
        counts_list = [sym_counts[s] for s in sym_list]
        probs_rle = np.array(counts_list) / sum(counts_list)
        
        rle_c_table = build_huffman_table(list(range(len(sym_list))), probs_rle)
        rle_encoded_syms = [sym_list.index(tuple(x)) for x in rle_symbols]
        rle_bits = huffman_encode(rle_encoded_syms, rle_c_table)
        
        with open(f"./Q5_Output/PartF/{base}_rle_table.pkl", "wb") as f:
            pickle.dump((sym_list, rle_c_table), f)
        with open(f"./Q5_Output/PartF/{base}_rle_bits.pkl", "wb") as f:
            pickle.dump(rle_bits, f)
        
        cr_rle = 8 * M * N / len(rle_bits)
        with open(f"./Q5_Output/PartF/{base}_rle_stats.txt", "w") as f:
            f.write(f"RLE actual compression ratio: {cr_rle}\n")
        
        # Decode RLE to verify correctness
        rle_decoded_syms = huffman_decode(rle_bits, rle_c_table)
        rle_decoded_runs = [sym_list[idx] for idx in rle_decoded_syms]
        rle_decoded_list = RLE.decode(rle_decoded_runs)
        rle_err_img_dec = np.array(rle_decoded_list, dtype=np.uint8).reshape((M, N+1))
        rle_decoded_img = decode_from_error_adaptive(rle_err_img_dec)
        write_image(rle_decoded_img, f"./Q5_Output/PartF/{base}_rle_decoded.png")
        
        # Verify RLE decoding
        with open(f"./Q5_Output/PartF/{base}_rle_verify.txt", "w") as f:
            matched_rle = np.array_equal(rle_decoded_img, img)
            f.write(f"RLE decoded equal to original: {matched_rle}\n")


def main():
    ensure_dirs()
    img_files = [
        "../Testcases/q4_q5_f1.jpg", # large region image
        "../Testcases/q4_q5_input_f2.png", # photographic image
    ]
    results = partA(img_files)
    results_b={}
    for k,r in results.items():
        img, err, std, ent,err_piazza=r
        results[k]=(img,err,std,ent)
        results_b[k]=(img,err_piazza,std,ent)
    partB(results_b)
    save_partC()
    partD(results)
    partE(results)
    partF(results)

if __name__ == "__main__":
    main()
