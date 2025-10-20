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
        base = os.path.splitext(os.path.basename(fname))[0]
        std, ent = save_partA(img, err, base)
        results[base] = (img, err, std, ent)
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
        decoded = decode_from_error(err)
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
    avg_code_len = np.mean([len(table[s]) for s in lzw_out])
    H = entropy(np.array(lzw_out, dtype=np.uint16))
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
def compute_error_all_schemes(img):
    H,W = img.shape
    err_rows = []
    scheme_rows = []
    for i in range(H):
        row_errs = []
        row_schemes = []
        candidates = []
        # Generate all 5
        # scheme 0: none, scheme 1: left, scheme 2: above, scheme 3: upper left (average), scheme 4: Paeth
        for scheme in range(5):
            err = np.zeros(W, dtype=np.uint8)
            for j in range(W):
                if scheme == 0: pred = 0
                elif scheme == 1: pred = img[i,j-1] if j > 0 else 0
                elif scheme == 2: pred = img[i-1,j] if i > 0 else 0
                elif scheme == 3: pred = ((img[i,j-1] if j>0 else 0)+(img[i-1,j] if i>0 else 0)//2) if i>0 and j>0 else 0
                elif scheme == 4: pred = paeth_predictor(
                    img[i,j-1] if j>0 else 0,
                    img[i-1,j] if i>0 else 0,
                    img[i-1,j-1] if i>0 and j>0 else 0)
                err[j] = (int(img[i,j])-int(pred))%256
            candidates.append((err, scheme))
        best_err, best_scheme = min(candidates, key=lambda x: entropy(x[0]))
        err_rows.append(best_err)
        scheme_rows.append(best_scheme)
    err_img = np.stack(err_rows,axis=0)
    return err_img, scheme_rows

def partF(results):
    print("======Part-F processing...======")
    # Scheme selection
    for base, (img, _, _, _) in results.items():
        M,N = img.shape
        err_img, scheme_rows = compute_error_all_schemes(img)
        # Save artifacts
        write_image(err_img, f"./Q5_Output/PartF/{base}_adapterror.png")
        with open(f"./Q5_Output/PartF/{base}_schemes.pkl","wb") as f:
            pickle.dump(scheme_rows,f)
        # Encode/decode as in E
        lzw_out = LZW.encode(unsigned8_list(err_img))
        symbols, counts = np.unique(lzw_out, return_counts=True)
        probs = counts/np.sum(counts)
        table = build_huffman_table(symbols,probs)
        bits = huffman_encode(lzw_out,table)
        # Save table and bits
        with open(f"./Q5_Output/PartF/{base}_table.pkl","wb") as f:
            pickle.dump(table,f)
        with open(f"./Q5_Output/PartF/{base}_bits.pkl","wb") as f:
            pickle.dump(bits,f)
        # Decode
        lzw_out_decoded = huffman_decode(bits,table)
        err_list = LZW.decode(lzw_out_decoded)
        err_img_dec = np.array(err_list,dtype=np.uint8).reshape((M,N))
        decoded = decode_from_error(err_img_dec)
        write_image(decoded, f"./Q5_Output/PartF/{base}_decoded.png")
        # Run Length Encoding option
        # RLE encoding for each row:
        rle_codes = []
        for row in err_img:
            # (run length, value)
            rle_row = []
            prev = row[0]
            cnt = 1
            for v in row[1:]:
                if v==prev: cnt+=1
                else:
                    rle_row.append( (cnt, prev) )
                    prev = v
                    cnt = 1
            rle_row.append( (cnt, prev) )
            rle_codes.extend(rle_row)
        # Huffman on RLE
        rle_symbols = [x for x in rle_codes]
        sym_set, rle_counts = np.unique(rle_symbols,return_counts=True,axis=0)
        probs = rle_counts / np.sum(rle_counts)
        # Map sym_set to tuple for hashability
        sym_list = [tuple(x) for x in sym_set]
        rle_c_table = build_huffman_table(list(range(len(sym_list))),probs)
        # Map rle_symbols to sym_list indices
        rle_encoded_syms = [ sym_list.index(tuple(x)) for x in rle_symbols ]
        rle_bits = huffman_encode(rle_encoded_syms, rle_c_table)
        with open(f"./Q5_Output/PartF/{base}_rle_table.pkl","wb") as f:
            pickle.dump((sym_list,rle_c_table),f)
        with open(f"./Q5_Output/PartF/{base}_rle_bits.pkl","wb") as f:
            pickle.dump(rle_bits,f)
        cr_rle = 8*M*N/len(rle_bits)
        with open(f"./Q5_Output/PartF/{base}_rle_stats.txt","w") as f:
            f.write(f"RLE actual compression ratio: {cr_rle}\n")

def main():
    ensure_dirs()
    img_files = [
        "q5_input_1.png", # large region image
        "q5_input_2.png", # photographic image
    ]
    results = partA(img_files)
    partB(results)
    save_partC()
    partD(results)
    partE(results)
    partF(results)

if __name__ == "__main__":
    main()
