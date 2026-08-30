#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t status = (call);                                         \
    if (status != cudaSuccess) {                                               \
      std::cerr << cudaGetErrorString(status) << '\n';                         \
      return 1;                                                               \
    }                                                                           \
  } while (false)

int main() {
  constexpr int rows = 37;
  constexpr int cols = 113;
  const size_t count = static_cast<size_t>(rows) * cols;
  std::vector<float> input(count);
  for (size_t i = 0; i < count; ++i) input[i] = std::sin(static_cast<float>(i));

  float* d_input = nullptr;
  float* d_reduce = nullptr;
  float* d_transpose = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input),
                       count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_reduce),
                       rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_transpose),
                       count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(kernel_lab::reduce_mean_baseline(d_input, d_reduce, rows, cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> baseline(rows);
  CUDA_CHECK(cudaMemcpy(baseline.data(), d_reduce, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(kernel_lab::reduce_mean_warp(d_input, d_reduce, rows, cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> optimized(rows);
  CUDA_CHECK(cudaMemcpy(optimized.data(), d_reduce, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < rows; ++row) {
    assert(std::abs(baseline[row] - optimized[row]) < 1e-5f);
  }

  CUDA_CHECK(kernel_lab::transpose_tiled(d_input, d_transpose, rows, cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> transposed(count);
  CUDA_CHECK(cudaMemcpy(transposed.data(), d_transpose, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      assert(transposed[static_cast<size_t>(col) * rows + row] ==
             input[static_cast<size_t>(row) * cols + col]);
    }
  }

  cudaFree(d_input);
  cudaFree(d_reduce);
  cudaFree(d_transpose);
  std::cout << "kernel tests passed\n";
  return 0;
}
