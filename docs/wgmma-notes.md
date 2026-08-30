# WGMMA learning and validation notes

## Why it is not a textual WMMA replacement

WMMA exposes warp-level matrix fragments through a CUDA C++ API. Hopper
WGMMA is a warpgroup-level asynchronous operation: 128 participating threads
cooperate, operands can use shared-memory descriptors, and completion is
managed with fence, commit-group and wait-group operations. Porting a WMMA
kernel therefore requires a new producer/consumer pipeline rather than an API
rename.

## Implementation checklist for an SM90 machine

1. Add a separate `sm_90a` target and keep the current WMMA reference path.
2. Define supported matrix shapes, layouts, data types and accumulator type.
3. Stage operands into shared memory with an explicit, aligned layout; add TMA
   only after the basic pipeline is correct.
4. Ensure all four warps participate in the warpgroup operation and place the
   required fence, commit and wait operations at auditable points.
5. Compare every supported shape against a CPU or cuBLAS reference, then run
   `compute-sanitizer` before collecting performance data.
6. Use Nsight Compute to inspect tensor-pipe utilization, shared-memory
   behavior, stalls and achieved throughput. Record CUDA, driver, GPU and the
   exact `sm_90a` compiler flags.

## Evidence boundary

The repository currently proves a tested WMMA code path and demonstrates the
architectural differences that make WGMMA a separate implementation task. A
WGMMA performance claim should be added only together with SM90 source,
correctness tests, sanitizer output and a reproducible profiler report from a
real Hopper-class device.
