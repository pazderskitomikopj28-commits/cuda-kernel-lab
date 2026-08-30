# Kernel design notes

## ReduceMean

`reduce_mean_baseline` deliberately uses one thread per row. It is easy to
read, but it serializes the reduction and gives us a useful baseline.
`reduce_mean_warp` distributes the columns across a block, reduces inside each
warp with shuffle instructions, and performs one small shared-memory reduction
between warps. The shared array is only written by lane 0, and the barrier is
placed before warp 0 consumes it.

The launch chooses a multiple of 32 threads, capped at 256. For a real kernel
the best choice still depends on the shape and device; record that choice in a
benchmark rather than assuming 256 is always optimal.

## Branch divergence

The conditional-transform experiment keeps input, selector and output access
contiguous in all variants. Only selector layout and control flow change:

- `grouped` assigns the same selector to each group of 32 consecutive threads,
  so every warp takes one path;
- `divergent` alternates the selector for adjacent threads, forcing each warp
  to execute both paths with complementary active lanes;
- `branchless` uses the alternating selector but evaluates both paths for every
  thread before selecting a result.

Both arithmetic paths contain a runtime-controlled dependent FMA loop and are
kept out of line. This prevents the compiler from reducing the experiment to a
single predicated instruction. The Release `sm_89` binary was also inspected
with `cuobjdump --dump-sass`: the selector in the branch kernel controls a
conditional `BRA`, while the branchless kernel calls both paths and uses a
predicate select.

Branchless code is not assumed to be faster. A divergent warp may serialize
paths, while a branchless implementation deliberately performs extra work for
every lane. The benchmark reports both timings and treats the result as a
workload-dependent tradeoff.

## Tiled transpose

Both tiled kernels load a 32×32 tile into shared memory. The deliberately
conflicted variant uses a `32`-column stride; when a warp reads a column during
the transposed store, its addresses map to the same bank. The padded variant
uses a `33`-column stride, changing that mapping while preserving coalesced
global-memory access. The benchmark reports both timings, and the load and
store are guarded separately so non-multiple dimensions remain correct.

## WMMA

`wmma_gemm` is a deliberately narrow sample for matrix-multiply hardware. It
requires dimensions divisible by 16 and is intended for SM70+ devices. WMMA
and WGMMA are NVIDIA API names; a port to a Biren device should use the
corresponding SUPA matrix/tensor API after checking the target SDK and tile
constraints. Do not treat this sample as evidence that a different GPU exposes
the same instruction set.

`kernel_bench --op wmma --rows 256 --cols 256 --k 256` includes a host-side FP32
reference check and reports the measured TFLOP/s. The tolerance is intentionally
looser than the reduction/transpose checks because the computation uses FP16
inputs and a FP32 accumulator; record the exact tolerance with any published
result.
