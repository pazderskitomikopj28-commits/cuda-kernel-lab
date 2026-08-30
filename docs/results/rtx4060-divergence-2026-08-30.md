# Controlled branch divergence — RTX 4060 Laptop GPU

Source commit: `ca18e345772725bcbf85e1de959a372500f84de1`.

## Experiment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9;
- driver 591.74; CUDA 12.4.131; MSVC 19.38.33145;
- Release `sm_89` build with the shared CUDA runtime;
- 4,194,304 elements and 64 dependent FMAs in either arithmetic path;
- one unmeasured warm-up and 100 measured iterations per variant;
- five fresh processes; CUDA events measure device execution time.

All variants use contiguous input, selector and output arrays. `grouped` keeps
each 32-thread warp on one path, while `divergent` alternates the selector for
adjacent threads. `branchless` uses the alternating selector, evaluates both
paths and then selects one result.

```powershell
kernel_bench.exe --op divergence --rows 1024 --cols 4096 `
  --work 64 --iters 100
```

## Raw process averages

| Run | Grouped (ms) | Divergent (ms) | Branchless (ms) | Divergence slowdown | Branchless speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6035 | 1.1384 | 1.1333 | 1.8863× | 1.0045× |
| 2 | 0.6036 | 1.1387 | 1.0976 | 1.8863× | 1.0374× |
| 3 | 0.6036 | 1.1386 | 1.1333 | 1.8863× | 1.0046× |
| 4 | 0.6035 | 1.1384 | 1.1333 | 1.8863× | 1.0045× |
| 5 | 0.5799 | 1.1385 | 1.1031 | 1.9632× | 1.0321× |

## Summary across processes

| Metric | Mean | P50 | P95 |
| --- | ---: | ---: | ---: |
| Grouped (ms) | 0.5988 | 0.6035 | 0.6036 |
| Divergent (ms) | 1.1385 | 1.1385 | 1.1387 |
| Branchless (ms) | 1.1201 | 1.1333 | 1.1333 |
| Divergence slowdown | 1.9017× | 1.8863× | 1.9632× |
| Branchless speedup | 1.0166× | 1.0046× | 1.0374× |

The controlled alternating branch is 1.8863× slower at the median than the
warp-uniform selector layout. The branchless variant is effectively tied with
the divergent variant at the median: removing control-flow divergence is not
free because it evaluates both arithmetic paths for every lane.

This conclusion is specific to the selected path length and problem size. On a
small 131,072-element, eight-FMA smoke run, launch and scheduling overhead were
large enough to obscure and invert the relationship. The benchmark therefore
exposes `--work` and the element count instead of presenting one result as a
universal rule.

## Control-flow verification

The Release binary was inspected with:

```powershell
cuobjdump --dump-sass kernel_bench.exe
```

In `select_transform_branch_kernel`, the selector comparison controls a
conditional `BRA` and selects one of two out-of-line path calls. In
`select_transform_branchless_kernel`, both path calls execute before an `FSEL`.
This verifies that the timed kernels preserve the intended control-flow
difference after optimization.

## Correctness and memory safety

- the benchmark evenly sampled 8,192 elements from every variant and observed
  `max_abs_error=0` against a host `fma` reference;
- the unit test checks every element of an odd-sized 1,003-element input for
  both branch and branchless APIs;
- Compute Sanitizer memcheck reported `0 errors`;
- Compute Sanitizer racecheck reported `0 errors, 0 warnings`.

Nsight Compute still does not initialize on this Windows host, so this report
does not claim branch-efficiency or warp-stall counter values. SASS inspection,
CUDA-event timing, full small-input tests and Compute Sanitizer are recorded as
separate checks rather than substitutes for unavailable hardware counters.
