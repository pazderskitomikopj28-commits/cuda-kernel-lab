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

## Tiled transpose

The tiled kernel loads a 32×32 tile into shared memory and uses a `33`-column
stride. The extra column changes the bank mapping and avoids the classic
transpose bank conflict. The load and store are guarded separately so
non-multiple dimensions remain correct.

## WMMA

`wmma_gemm` is a deliberately narrow sample for matrix-multiply hardware. It
requires dimensions divisible by 16 and is intended for SM70+ devices. WMMA
and WGMMA are NVIDIA API names; a port to a Biren device should use the
corresponding SUPA matrix/tensor API after checking the target SDK and tile
constraints. Do not treat this sample as evidence that a different GPU exposes
the same instruction set.

`kernel_bench --op wmma --rows 256 --cols 256` includes a host-side FP32
reference check and reports the measured TFLOP/s. The tolerance is intentionally
looser than the reduction/transpose checks because the computation uses FP16
inputs and a FP32 accumulator; record the exact tolerance with any published
result.
