FLASHATTENTION-2: FASTER ATTENTION WITH
BETTER PARALLELISM AND WORK PARTITIONING

ABSTRACT

Scaling Transformers to longer sequence lengths has been a major problem in the
last several years, promising to improve performance in language modeling and high-
resolution image understanding, as well as to unlock new applications in code, audio,
and video generation.

The attention layer is the main bottleneck in scaling to longer
sequences, as its runtime and memory increase quadratically in the sequence length.

FLASHATTENTION (Dao et al., 2022) exploits the asymmetric GPU memory hier-
archy to bring significant memory saving (linear instead of quadratic) and runtime
speedup (2-4x compared to optimized baselines), with no approximation.

 However,
FLASHATTENTION is still not nearly as fast as optimized matrix-multiply (GEMM)
operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe
that the inefficiency is due to suboptimal work partitioning between different thread
blocks and warps on the GPU, causing either low-occupancy or unnecessary shared
memory reads/writes. 

We propose FLASHATTENTION-2, with better work partition-
ing to address these issues. In particular, we (1) tweak the algorithm to reduce the
number of non-matmul FLOPs (2) parallelize the attention computation, even for a
single head, across different thread blocks to increase occupancy, and (3) within each
thread block, distribute the work between warps to reduce communication through
shared memory. These yield around 2x speedup compared to FLASHATTENTION,
reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close
to the efficiency of GEMM operations.

1 INTRODUCTION

 Dao et al. (2022) proposed to reorder the
attention computation and leverages classical techniques (tiling, recomputation) to significantly speed
it up and reduce memory usage from quadratic to linear in sequence length. This yields 2-4 wall-clock
time speedup over optimized baselines, up to 10-20 memory saving, with no approximation, and as a
result FLASHATTENTION has seen wide adoption in large-scale training and inference of Transformers

while FLASHATTENTION is already
2-4 faster than a standard attention implementation, the forward pass only reaches 30-50% of the theoretical maximum FLOPs/s of the device (Fig. 6), while the backward pass is even more challenging,
reaching only 25-35% of maximum throughput on A100 GPU (Fig. 7). In contrast, optimized GEMM
can reach up to 80-90% of the theoretical maximum device throughput.

Building on FLASHATTENTION, we propose FLASHATTENTION-2 with better parallelism and work
partitioning to address these challenges.
1. In Section 3.1, we tweak the algorithms to reduce the number of non-matmul FLOPs while not
changing the output. While the non-matmul FLOPs only account for a small fraction of the total
FLOPs, they take longer to perform as GPUs have specialized units for matrix multiply, and as
a result the matmul throughput can be up to 16 higher than non-matmul throughput. It is thus
important to reduce non-matmul FLOPs and spend as much time as possible doing matmul FLOPs.

2. We propose to parallelize both the forward pass and backward pass along the sequence length dimen-
sion, in addition to the batch and number of heads dimension. This increases occupancy (utilization
of GPU resources) in the case where the sequences are long (and hence batch size is often small)

3. Even within one block of attention computation, we partition the work between different warps
of a thread block to reduce communication and shared memory reads/writes.

2 BACKGROUND
We provide some background on the performance characteristics and execution model of GPUs. We
also describe the standard implementation of attention, as well as FLASHATTENTION

2.1 HARDWARE CHARACTERISTICS
GPU performance characteristics.
The GPU consists of compute elements (e.g., floating point arith-
metic units) and a memory hierarchy. Most modern GPUs contain specialized units to accelerate matrix
multiply in low-precision (e.g., Tensor Cores on Nvidia GPUs for FP16/BF16 matrix multiply). The
memory hierarchy comprise of high bandwidth memory (HBM), and on-chip SRAM (aka shared mem-
ory). As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth
1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth
estimated around 19TB/s (Jia et al., 2018; Jia and Van Sandt, 2021). As the L2 cache is not directly
controllable by the programmer, we focus on the HBM and SRAM for the purpose of this discussion.

![](2025-08-21-16-18-42.png)

Execution Model. GPUs have a massive number of threads to execute an operation (called a kernel).
Threads are organized into thread blocks, which are scheduled to run on streaming multiprocessors
(SMs). Within each thread blocks, threads are grouped into warps (a group of 32 threads). Threads
within a warp can communicate by fast shuffle instructions or cooperate to perform matrix multiply.
Warps within a thread block can communicate by reading from / writing to shared memory. Each
kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.


2.2 STANDARD ATTENTION IMPLEMENTATION

![](2025-08-21-16-45-35.png)

where softmax is applied row-wise.1 For multi-head attention (MHA), this same computation is
performed in parallel across many heads, and parallel over the batch dimension (number of input
sequences in a batch).

![](2025-08-21-17-47-45.png)
![](2025-08-21-17-49-32.png)

2.3 FLASHATTENTION
To speed up attention on hardware accelerators such as GPU, (Dao et al., 2022) proposes an algorithm
to reduce the memory reads/writes while maintaining the same output (without approximation)

2.3.1 FORWARD PASS
