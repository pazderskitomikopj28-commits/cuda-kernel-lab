#include "kernels.cuh"

#include <mma.h>

#include <algorithm>
#include <cstddef>

namespace {

constexpr int kWarpSize = 32;
constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

__global__ void reduce_mean_baseline_kernel(const float* input, float* output,
                                            int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  float sum = 0.0f;
  const std::size_t offset = static_cast<std::size_t>(row) * cols;
  for (int col = 0; col < cols; ++col) {
    sum += input[offset + col];
  }
  output[row] = sum / static_cast<float>(cols);
}

__global__ void reduce_mean_warp_kernel(const float* input, float* output,
                                        int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  float partial = 0.0f;
  const std::size_t offset = static_cast<std::size_t>(row) * cols;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    partial += input[offset + col];
  }

  for (int delta = kWarpSize / 2; delta > 0; delta /= 2) {
    partial += __shfl_down_sync(0xffffffffu, partial, delta);
  }

  __shared__ float warp_sums[32];
  if (lane == 0) {
    warp_sums[warp] = partial;
  }
  __syncthreads();

  if (warp == 0) {
    partial = lane < (blockDim.x / kWarpSize) ? warp_sums[lane] : 0.0f;
    for (int delta = kWarpSize / 2; delta > 0; delta /= 2) {
      partial += __shfl_down_sync(0xffffffffu, partial, delta);
    }
    if (lane == 0) {
      output[row] = partial / static_cast<float>(cols);
    }
  }
}

__global__ void transpose_naive_kernel(const float* input, float* output,
                                       int rows, int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    output[static_cast<std::size_t>(col) * rows + row] =
        input[static_cast<std::size_t>(row) * cols + col];
  }
}

template <int kPadding>
__global__ void transpose_tiled_kernel(const float* input, float* output,
                                       int rows, int cols) {
  __shared__ float tile[kTileDim][kTileDim + kPadding];

  const int col = blockIdx.x * kTileDim + threadIdx.x;
  const int row = blockIdx.y * kTileDim + threadIdx.y;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (row + j < rows && col < cols) {
      tile[threadIdx.y + j][threadIdx.x] =
          input[static_cast<std::size_t>(row + j) * cols + col];
    }
  }
  __syncthreads();

  const int out_col = blockIdx.y * kTileDim + threadIdx.x;
  const int out_row = blockIdx.x * kTileDim + threadIdx.y;
  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (out_row + j < cols && out_col < rows) {
      output[static_cast<std::size_t>(out_row + j) * rows + out_col] =
          tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void wmma_gemm_kernel(const half* a, const half* b, float* c,
                                 int m, int n, int k) {
  namespace wmma = nvcuda::wmma;
  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * 16;
  if (tile_m >= m || tile_n >= n) {
    return;
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int tile_k = 0; tile_k < k; tile_k += 16) {
    wmma::load_matrix_sync(
        a_frag, a + static_cast<std::size_t>(tile_m) * k + tile_k, k);
    wmma::load_matrix_sync(
        b_frag, b + static_cast<std::size_t>(tile_k) * n + tile_n, n);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(
      c + static_cast<std::size_t>(tile_m) * n + tile_n, c_frag, n,
                           wmma::mem_row_major);
}

cudaError_t check_shape(int rows, int cols) {
  return rows > 0 && cols > 0 ? cudaSuccess : cudaErrorInvalidValue;
}

}  // namespace

namespace kernel_lab {

cudaError_t reduce_mean_baseline(const float* input, float* output, int rows,
                                 int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || check_shape(rows, cols) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  reduce_mean_baseline_kernel<<<rows, 32, 0, stream>>>(input, output, rows, cols);
  return cudaGetLastError();
}

cudaError_t reduce_mean_warp(const float* input, float* output, int rows,
                             int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || check_shape(rows, cols) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  const int threads = std::min(256, std::max(32, ((cols + 31) / 32) * 32));
  reduce_mean_warp_kernel<<<rows, threads, 0, stream>>>(input, output, rows, cols);
  return cudaGetLastError();
}

cudaError_t transpose_naive(const float* input, float* output, int rows,
                            int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || check_shape(rows, cols) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  dim3 block(32, 8);
  dim3 grid((cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y);
  transpose_naive_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
  return cudaGetLastError();
}

cudaError_t transpose_tiled(const float* input, float* output, int rows,
                            int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || check_shape(rows, cols) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  dim3 block(kTileDim, kBlockRows);
  dim3 grid((cols + kTileDim - 1) / kTileDim,
            (rows + kTileDim - 1) / kTileDim);
  transpose_tiled_kernel<1><<<grid, block, 0, stream>>>(input, output, rows,
                                                        cols);
  return cudaGetLastError();
}

cudaError_t transpose_tiled_conflict(const float* input, float* output,
                                     int rows, int cols,
                                     cudaStream_t stream) {
  if (input == nullptr || output == nullptr ||
      check_shape(rows, cols) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  dim3 block(kTileDim, kBlockRows);
  dim3 grid((cols + kTileDim - 1) / kTileDim,
            (rows + kTileDim - 1) / kTileDim);
  transpose_tiled_kernel<0><<<grid, block, 0, stream>>>(input, output, rows,
                                                        cols);
  return cudaGetLastError();
}

cudaError_t wmma_gemm(const half* a, const half* b, float* c, int m, int n,
                      int k, cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
      k <= 0 || (m % 16) != 0 || (n % 16) != 0 || (k % 16) != 0) {
    return cudaErrorInvalidValue;
  }
  dim3 block(32);
  dim3 grid(n / 16, m / 16);
  wmma_gemm_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
  return cudaGetLastError();
}

}  // namespace kernel_lab
