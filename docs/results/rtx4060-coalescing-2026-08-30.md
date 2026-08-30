# Strided global-memory access — RTX 4060 Laptop GPU

Source commit: `25226a705cb6f2a8fae4b01ff2a2ddcaba978d78`.

## Experiment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9;
- driver 591.74; CUDA 12.4.131; MSVC 19.38.33145;
- Release `sm_89` build with the shared CUDA runtime;
- 8,388,608 output elements; one warm-up and 100 measured iterations;
- five fresh processes; CUDA events measure device execution time;
- maximum source allocation: 268,435,457 floats, approximately 1 GiB.

Each thread writes one contiguous output and reads
`input[thread_index * stride]`. The misaligned variant offsets the otherwise
contiguous input pointer by one float. A GPU initialization kernel writes a
deterministic index pattern across the full source allocation.

```powershell
kernel_bench.exe --op coalescing --rows 2048 --cols 4096 --iters 100
```

Effective bandwidth counts eight useful bytes per element: one four-byte input
and one four-byte output. It deliberately excludes unused bytes fetched in
memory sectors and is therefore not physical DRAM bandwidth.

## Raw process averages

| Run | Aligned | Misaligned | Stride 2 | Stride 4 | Stride 8 | Stride 32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 218.3667 | 215.5151 | 144.2066 | 92.6907 | 51.5759 | 51.0071 |
| 2 | 218.4606 | 215.5364 | 144.2289 | 92.6893 | 51.5670 | 51.0008 |
| 3 | 218.4315 | 215.5081 | 144.2889 | 92.6867 | 51.5763 | 50.9972 |
| 4 | 218.2569 | 215.5014 | 144.0352 | 92.6789 | 51.5906 | 51.0115 |
| 5 | 221.5476 | 215.8416 | 145.6000 | 92.6828 | 51.5642 | 51.0040 |

Values are effective GB/s.

## Summary across processes

| Pattern | Mean ms | P50 ms | P95 ms | P50 GB/s | P50 vs aligned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned | 0.3064 | 0.3072 | 0.3075 | 218.4315 | 1.0000× |
| Misaligned by 4 B | 0.3113 | 0.3114 | 0.3114 | 215.5151 | 0.9866× |
| Stride 2 | 0.4645 | 0.4653 | 0.4659 | 144.2289 | 0.6603× |
| Stride 4 | 0.7240 | 0.7240 | 0.7241 | 92.6867 | 0.4243× |
| Stride 8 | 1.3012 | 1.3012 | 1.3015 | 51.5759 | 0.2361× |
| Stride 32 | 1.3158 | 1.3158 | 1.3159 | 51.0040 | 0.2335× |

Four-byte misalignment retained 98.66% of the aligned median on this GPU and
problem size. Increasing stride caused a much larger and monotonic loss until
stride 8, after which stride 32 was nearly identical. The plateau is consistent
with the useful input values already occupying separate memory sectors; larger
address gaps increase the footprint without improving transaction utilization.

This is an observed access-pattern result, not a universal misalignment rule.
Cache state, element size, transaction width and architecture can change the
relationship. Nsight Compute counters would be required to report the actual
sector count and physical bytes transferred, and those counters remain
unavailable on this Windows host.

## Correctness and memory safety

- every timed variant checked 8,192 evenly distributed outputs and observed
  `max_abs_error=0`;
- the unit test checks every output for an odd 1,003-element problem with
  aligned, misaligned, stride 2 and stride 7 access;
- Compute Sanitizer memcheck reported `0 errors`;
- Compute Sanitizer racecheck reported `0 errors, 0 warnings`.
