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
def paeth_predictor(a, b, c):
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
    with open(f"./Q5_Output/PartB/{name}_verify.txt", "w") as f:
        matched = np.array_equal(decoded, img)
        f.write(f"Decoded equal to original: {matched}\n")
def partB(results):
    print("======Part-B processing...======")
    for base, (img, err, _, _) in results.items():
        decoded = decode_from_error_updated_for_PartA_B_Piazza(err)
        save_partB(decoded, img, base)
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
    zeros = [0] * 100  
    codes = LZW.encode(zeros)
    decoded = LZW.decode(codes)
    with open("./Q5_Output/PartC/lzw_test.txt", "w") as f:
        f.write("Encoded length: {}\n".format(len(codes)))
        f.write("Decoded equals input: {}\n".format(decoded == zeros))
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
    symbols, counts = np.unique(lzw_out, return_counts=True)
    probs = counts / np.sum(counts)
    table = build_huffman_table(symbols, probs)
    encoded_bits = huffman_encode(lzw_out, table)
    avg_code_len = np.mean([len(table[s]) for s in lzw_out])
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
def encode_full(img):
    err_img = compute_error_image(img)
    lzw_out = LZW.encode(unsigned8_list(err_img))
    symbols, counts = np.unique(lzw_out, return_counts=True)
    probs = counts / np.sum(counts)
    H = entropy(np.array(unsigned8_list(err_img)))
    table = build_huffman_table(symbols, probs)
    bits = huffman_encode(lzw_out, table)
    return table, bits, err_img, lzw_out, H
def decode_full(M, N, table, bits):
    lzw_out = huffman_decode(bits, table)
    err_list = LZW.decode(lzw_out)
    err_img = np.array(err_list, dtype=np.uint8).reshape((M,N))
    img = decode_from_error(err_img)
    return img
def save_partE(img, M, N, name):
    table, bits, err_img, lzw_out, H = encode_full(img)
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
def compute_error_all_schemes(img):
    H, W = img.shape
    err_img = np.zeros((H, W+1), dtype=np.uint8)
    for i in range(H):
        candidates = []
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
        best_err, best_scheme = min(candidates, key=lambda x: entropy(x[0]))
        err_img[i, 0] = best_scheme  
        err_img[i, 1:] = best_err     
    return err_img
def decode_from_error_adaptive(err_img):
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
        result = []
        for cnt, val in runs:
            result.extend([val] * cnt)
        return result
def partF(results):
    print("======Part-F processing...======")
    for base, (img, _, _, _) in results.items():
        M, N = img.shape
        err_img = compute_error_all_schemes(img)  
        write_image(err_img, f"./Q5_Output/PartF/{base}_adapterror.png")
        err_list = unsigned8_list(err_img)
        lzw_out = LZW.encode(err_list)
        symbols, counts = np.unique(lzw_out, return_counts=True)
        probs = counts / np.sum(counts)
        table = build_huffman_table(symbols, probs)
        bits = huffman_encode(lzw_out, table)
        with open(f"./Q5_Output/PartF/{base}_table.pkl", "wb") as f:
            pickle.dump(table, f)
        with open(f"./Q5_Output/PartF/{base}_bits.pkl", "wb") as f:
            pickle.dump(bits, f)
        lzw_out_decoded = huffman_decode(bits, table)
        err_list_decoded = LZW.decode(lzw_out_decoded)
        err_img_dec = np.array(err_list_decoded, dtype=np.uint8).reshape((M, N+1))
        decoded = decode_from_error_adaptive(err_img_dec)
        write_image(decoded, f"./Q5_Output/PartF/{base}_decoded.png")
        with open(f"./Q5_Output/PartF/{base}_verify.txt", "w") as f:
            matched = np.array_equal(decoded, img)
            f.write(f"Adaptive decoded equal to original: {matched}\n")
        H_err = entropy(err_img[:, 1:])  
        cr_pred = 8 / H_err
        cr_lzw = (M * N) / len(lzw_out)
        cr_total = 8 * M * N / len(bits)
        with open(f"./Q5_Output/PartF/{base}_ratios.txt", "w") as f:
            f.write(f"Adaptive Prediction compression ratio: {cr_pred}\n")
            f.write(f"LZW compression ratio: {cr_lzw}\n")
            f.write(f"Actual compression ratio: {cr_total}\n")
        rle_runs = RLE.encode(err_list)
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
        rle_decoded_syms = huffman_decode(rle_bits, rle_c_table)
        rle_decoded_runs = [sym_list[idx] for idx in rle_decoded_syms]
        rle_decoded_list = RLE.decode(rle_decoded_runs)
        rle_err_img_dec = np.array(rle_decoded_list, dtype=np.uint8).reshape((M, N+1))
        rle_decoded_img = decode_from_error_adaptive(rle_err_img_dec)
        write_image(rle_decoded_img, f"./Q5_Output/PartF/{base}_rle_decoded.png")
        with open(f"./Q5_Output/PartF/{base}_rle_verify.txt", "w") as f:
            matched_rle = np.array_equal(rle_decoded_img, img)
            f.write(f"RLE decoded equal to original: {matched_rle}\n")
def main():
    ensure_dirs()
    img_files = [
        "../Testcases/q4_q5_f1.jpg", 
        "../Testcases/q4_q5_input_f2.png", 
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
