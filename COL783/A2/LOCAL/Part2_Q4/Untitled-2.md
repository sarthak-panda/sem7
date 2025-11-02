Before: Coarse inference for entire tuple
Avg CPU for each tuple 
Perf one issue --fixed
Changes needed:
Granular for each .predict
CPU samples for each inference should not be averaged
rather it would keep them in sereis format

"""
q5_deconvolution.py

Implementation for the deconvolution / Wiener / regularized tasks described by the user.
Assumes input image at: ../Testcases/Q5.png

Outputs saved to ./q5_outputs/ (creates directory).

How to run:
    python q5_deconvolution.py

The script performs:
 - builds a Gaussian blur PSF and degrades the (sharp) input by blur + additive Gaussian noise with specified PSNRs
 - visualizes low-pass and band-pass frequency bands for f, g, inverse-filter result, Wiener result and regularized result
 - computes Wiener filter using constant K sweep and marks estimated K from Parseval/Plancherel
 - computes regularized deconvolution (squared gradient norm) over lambda sweep

Note: the derivations used in the code comments are based on "numpy.fft" normalization where
    sum_x |f|^2 = 1/(MN) sum_u |F|^2  (Parseval)
so if the noise variance per pixel is sigma^2 then under the assumption |N(u,v)| constant,
    S_n = E[|N(u,v)|^2] = MN * sigma^2
and if |F(u,v)| assumed constant,
    S_f = sum_x f^2  (= ||f||^2)
so K_est = S_n / S_f = (MN * sigma^2) / ||f||^2

"""