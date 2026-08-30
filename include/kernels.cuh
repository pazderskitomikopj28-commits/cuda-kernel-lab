#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace kernel_lab {

// Each function launches work on the supplied stream and returns the launch
// status. Inputs and outputs are contiguous row-major arrays.
cudaError_t reduce_mean_baseline(const float* input, float* output,
                                 int rows, int cols,
                                 cudaStream_t stream = nullptr);

cudaError_t reduce_mean_warp(const float* input, float* output,
                             int rows, int cols,
                             cudaStream_t stream = nullptr);

// input: [rows, cols], output: [cols, rows].
cudaError_t transpose_naive(const float* input, float* output,
                            int rows, int cols,
                            cudaStream_t stream = nullptr);

cudaError_t transpose_tiled(const float* input, float* output,
                            int rows, int cols,
                            cudaStream_t stream = nullptr);

// C = A @ B, A: [M, K], B: [K, N], C: [M, N].
// The WMMA sample intentionally requires M, N and K to be multiples of 16;
// this makes boundary behavior explicit and keeps the fast path auditable.
cudaError_t wmma_gemm(const half* a, const half* b, float* c,
                      int m, int n, int k,
                      cudaStream_t stream = nullptr);

}  // namespace kernel_lab
