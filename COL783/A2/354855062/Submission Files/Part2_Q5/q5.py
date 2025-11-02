import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def imread_gray(path):
    im = Image.open(path).convert('L')
    a = np.asarray(im).astype(np.float64)
    return a

# def save_im(arr, path):
#     arr_clamped = np.clip(arr, 0, 255).astype(np.uint8)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     Image.fromarray(arr_clamped).save(path)

def save_im(arr, path):
    arr_clamped = np.clip(arr, 0, 255).astype(np.uint8)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True) 
    Image.fromarray(arr_clamped).save(path)

def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((peak**2) / mse)#TODO: Fix peak for general

def gaussian_psf(shape=(256,256), sigma=3.0, size=25):
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    k /= np.sum(k)
    psf = np.zeros(shape)
    cy, cx = shape[0]//2, shape[1]//2
    sy, sx = size//2, size//2
    psf[cy - sy: cy - sy + size, cx - sx: cx - sx + size] = k
    psf = ifftshift(psf)
    return psf

def freq_indices(shape):
    M, N = shape
    u = np.arange(-M//2, M - M//2)
    v = np.arange(-N//2, N - N//2)
    U, V = np.meshgrid(v, u)
    return U, V

def freq_band_images(img, max_level=6):
    F = fftshift(fft2(img))
    U, V = freq_indices(img.shape)
    levels = []
    bands = []
    mask0 = (U == 0) & (V == 0)
    Li_prev = np.real(ifft2(ifftshift(F * mask0)))
    levels.append(Li_prev)
    for i in range(1, max_level+1):
        bound = 2**(i-1)
        mask = (np.abs(U) <= bound) & (np.abs(V) <= bound)
        Li = np.real(ifft2(ifftshift(F * mask)))
        Bi = Li - Li_prev
        bands.append(Bi + 128.0)
        levels.append(Li)
        Li_prev = Li
    return levels, bands

def inverse_filter(g, h, eps=1e-6):
    M, N = g.shape
    H = fft2(h)
    G = fft2(g)
    denom = H.copy()
    denom_abs = np.abs(denom)
    denom[denom_abs < eps] = eps
    Fhat = G / denom
    fhat = np.real(ifft2(Fhat))
    return fhat, Fhat, H, G

def wiener_filter_constant_K(g, h, K):
    H = fft2(h)
    G = fft2(g)
    H_conj = np.conj(H)
    denom = (np.abs(H)**2 + K)
    Fhat = (H_conj / denom) * G
    fhat = np.real(ifft2(Fhat))
    return fhat, Fhat

# Details Explained in Report
def regularized_deconv(g, h, lam):
    M, N = g.shape
    H = fft2(h)
    G = fft2(g)
    urange = np.arange(0, M)
    vrange = np.arange(0, N)
    U, V = np.meshgrid(vrange, urange)
    omega_u = 2.0 * np.pi * U / M
    omega_v = 2.0 * np.pi * V / N
    P = 4.0 * (np.sin(omega_u/2.0)**2 + np.sin(omega_v/2.0)**2)
    denom = (np.abs(H)**2 + lam * P)
    Fhat = (np.conj(H) / denom) * G
    fhat = np.real(ifft2(Fhat))
    return fhat, Fhat

def main():
    outdir = 'q5_outputs'
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join('..', 'Testcases', 'Q5.png')
    f = imread_gray(fp)
    M, N = f.shape
    print('Loaded', fp, 'shape=', f.shape)
    psf = gaussian_psf(shape=(M, N), sigma=4.0, size=25)
    save_im(np.real(ifft2(fft2(psf))).astype(np.float64), os.path.join(outdir, 'psf_centered.png'))
    F = fft2(f)
    H = fft2(psf)
    conv = np.real(ifft2(H * F))

    def add_noise_with_psnr(signal, target_psnr_db):
        sig_power = np.mean(signal**2)
        sigma2 = sig_power / (10.0**(target_psnr_db / 10.0))
        sigma = np.sqrt(sigma2)
        noise = np.random.normal(loc=0.0, scale=sigma, size=signal.shape)
        return signal + noise, sigma

    np.random.seed(0)
    g20, sigma20 = add_noise_with_psnr(conv, 20.0)
    save_im(g20, os.path.join(outdir, 'g_psnr20.png'))
    print('Added noise PSNR=20dB sigma=', sigma20)

    # Part - A
    print('==========================PART-A PROCESSING...==============================\n')
    max_level = int(np.floor(np.log2(min(M,N)))) - 1
    max_level = min(max_level, 6)
    l_f, b_f = freq_band_images(f, max_level=max_level)
    l_g, b_g = freq_band_images(g20, max_level=max_level)
    for i, Li in enumerate(l_f):
        save_im(Li, os.path.join(outdir, f'PartA/f/lowpass/f_lowpass_l{i}.png'))
    for i, Bi in enumerate(b_f, start=1):
        save_im(Bi, os.path.join(outdir, f'PartA/f/band/f_band_b{i}.png'))
    for i, Li in enumerate(l_g):
        save_im(Li, os.path.join(outdir, f'PartA/g/lowpass/g_lowpass_l{i}.png'))
    for i, Bi in enumerate(b_g, start=1):
        save_im(Bi, os.path.join(outdir, f'PartA/g/band/g_band_b{i}.png'))

    # Part - B
    print('==========================PART-B PROCESSING...==============================\n')
    f_inv, Finv, Hfull, Gfull = inverse_filter(g20, psf, eps=1e-3)
    save_im(f_inv, os.path.join(outdir, 'PartB/f_inverse.png'))
    l_inv, b_inv = freq_band_images(f_inv, max_level=max_level)
    for i, Li in enumerate(l_inv):
        save_im(Li, os.path.join(outdir, f'PartB/inv_lowpass/inv_lowpass_l{i}.png'))
    for i, Bi in enumerate(b_inv, start=1):
        save_im(Bi, os.path.join(outdir, f'PartB/inv_band/inv_band_b{i}.png'))
    print('Inverse PSNR:', psnr(f, f_inv))

    # Part - C
    print('==========================PART-C PROCESSING...==============================\n')
    sigma2 = sigma20**2
    Sn = (M * N) * sigma2
    Sf = np.sum(f.astype(np.float64)**2)
    K_est = Sn / Sf
    print('Estimated S_n =', Sn, 'S_f=', Sf, '=> K_est=', K_est)

    # Part - D
    print('==========================PART-D PROCESSING...==============================\n')
    Ks = np.logspace(np.log10(K_est) - 3, np.log10(K_est) + 3, num=60)
    psnrs = []
    f_wiens = []
    for K in Ks:
        fhat, _ = wiener_filter_constant_K(g20, psf, K)
        p = psnr(f, fhat)
        psnrs.append(p)
        f_wiens.append(fhat)
    psnrs = np.array(psnrs)
    best_idx = np.nanargmax(psnrs)
    bestK = Ks[best_idx]
    bestf = f_wiens[best_idx]
    print('Best Wiener K=', bestK, 'PSNR=', psnrs[best_idx])
    plt.figure(figsize=(6,4))
    plt.semilogx(Ks, psnrs)
    plt.axvline(K_est, color='red', linestyle='--', label=f'K_est={K_est:.2e}')
    plt.scatter([bestK], [psnrs[best_idx]], color='green', label=f'best K={bestK:.2e}\nPSNR={psnrs[best_idx]:.2f}dB')
    plt.xlabel('K (S_n / S_f)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('Wiener PSNR vs K (PSNR target 20 dB)')
    plt.grid(True)
    plt.tight_layout()
    output_dir = os.path.join(outdir, 'PartD')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wiener_psnr_vs_K_psnr20.png'))
    plt.close()
    save_im(bestf, os.path.join(outdir, 'PartD/wiener_best_psnr20.png'))
    l_wien, b_wien = freq_band_images(bestf, max_level=max_level)
    for i, Li in enumerate(l_wien):
        save_im(Li, os.path.join(outdir, f'PartD/wien_lowpass/wien_lowpass_l{i}.png'))
    for i, Bi in enumerate(b_wien, start=1):
        save_im(Bi, os.path.join(outdir, f'PartD/wien_band/wien_band_b{i}.png'))

    # Part - E
    print('==========================PART-E PROCESSING...==============================\n')
    results_noise_levels = {}
    for target_psnr in [30.0, 10.0]:
        np.random.seed(0)
        g, sigma = add_noise_with_psnr(conv, target_psnr)
        save_im(g, os.path.join(outdir, f'PartE/g_psnr{int(target_psnr)}.png'))
        sigma2 = sigma**2
        Sn = (M * N) * sigma2
        Sf = np.sum(f**2)
        K_est_local = Sn / Sf
        Ks_local = np.logspace(np.log10(K_est_local) - 3, np.log10(K_est_local) + 3, num=60)
        psnrs_local = []
        f_wiens_local = []
        for K in Ks_local:
            fhat, _ = wiener_filter_constant_K(g, psf, K)
            psnrs_local.append(psnr(f, fhat))
            f_wiens_local.append(fhat)
        psnrs_local = np.array(psnrs_local)
        idx_best = np.nanargmax(psnrs_local)
        results_noise_levels[target_psnr] = {
            'K_est': K_est_local,
            'Ks': Ks_local,
            'psnrs': psnrs_local,
            'best_idx': idx_best,
            'bestf': f_wiens_local[idx_best]
        }
        plt.figure(figsize=(6,4))
        plt.semilogx(Ks_local, psnrs_local)
        plt.axvline(K_est_local, color='red', linestyle='--', label=f'K_est={K_est_local:.2e}')
        plt.scatter([Ks_local[idx_best]], [psnrs_local[idx_best]], color='green')
        plt.xlabel('K (S_n / S_f)')
        plt.ylabel('PSNR (dB)')
        plt.title(f'Wiener PSNR vs K (PSNR target {int(target_psnr)} dB)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_dir = os.path.join(outdir, 'PartE')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'wiener_psnr_vs_K_psnr{int(target_psnr)}.png'))
        plt.close()
        save_im(results_noise_levels[target_psnr]['bestf'], os.path.join(outdir, f'PartE/wiener_best_psnr{int(target_psnr)}.png'))
        l_w, b_w = freq_band_images(results_noise_levels[target_psnr]['bestf'], max_level=max_level)
        for i, Li in enumerate(l_w):
            save_im(Li, os.path.join(outdir, f'PartE/wien_psnr{int(target_psnr)}_lowpass/wien_psnr{int(target_psnr)}_lowpass_l{i}.png'))
        for i, Bi in enumerate(b_w, start=1):
            save_im(Bi, os.path.join(outdir, f'PartE/wien_psnr{int(target_psnr)}_band/wien_psnr{int(target_psnr)}_band_b{i}.png'))

    # Part- F
    print('==========================PART-F PROCESSING...==============================\n')
    lambdas = np.logspace(-6, 1, num=60)
    psnrs_reg = []
    f_regs = []
    for lam in lambdas:
        fhat, _ = regularized_deconv(g20, psf, lam)
        psnrs_reg.append(psnr(f, fhat))
        f_regs.append(fhat)
    psnrs_reg = np.array(psnrs_reg)
    best_idx_reg = np.nanargmax(psnrs_reg)
    best_lambda = lambdas[best_idx_reg]
    bestf_reg = f_regs[best_idx_reg]
    print('Best regularization lambda=', best_lambda, 'PSNR=', psnrs_reg[best_idx_reg])
    plt.figure(figsize=(6,4))
    plt.semilogx(lambdas, psnrs_reg)
    plt.scatter([best_lambda], [psnrs_reg[best_idx_reg]], color='green', label=f'best lambda={best_lambda:.2e}\nPSNR={psnrs_reg[best_idx_reg]:.2f}dB')
    plt.xlabel('lambda (regularization)')
    plt.ylabel('PSNR (dB)')
    plt.title('Regularized deconvolution PSNR vs lambda')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_dir = os.path.join(outdir, 'PartF')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'reg_psnr_vs_lambda.png'))
    plt.close()
    save_im(bestf_reg, os.path.join(outdir, 'PartF/reg_best.png'))
    l_reg, b_reg = freq_band_images(bestf_reg, max_level=max_level)
    for i, Li in enumerate(l_reg):
        save_im(Li, os.path.join(outdir, f'PartF/reg_lowpass/reg_lowpass_l{i}.png'))
    for i, Bi in enumerate(b_reg, start=1):
        save_im(Bi, os.path.join(outdir, f'PartF/reg_band/reg_band_b{i}.png'))

if __name__ == '__main__':
    main()
