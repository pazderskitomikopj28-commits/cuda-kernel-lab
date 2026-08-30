#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t status = (call);                                           \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA failure for " #call ": "                             \
                << cudaGetErrorString(status) << '\n';                          \
      return false;                                                             \
    }                                                                           \
  } while (false)

namespace {

bool expect_close(float actual, float expected, float tolerance,
                  const std::string& context) {
  if (std::abs(actual - expected) <= tolerance) return true;
  std::cerr << context << ": expected " << expected << ", got " << actual
            << '\n';
  return false;
}

bool test_invalid_arguments() {
  if (kernel_lab::reduce_mean_warp(nullptr, nullptr, 1, 1) !=
          cudaErrorInvalidValue ||
      kernel_lab::transpose_tiled(nullptr, nullptr, 1, 1) !=
          cudaErrorInvalidValue ||
      kernel_lab::wmma_gemm(nullptr, nullptr, nullptr, 15, 16, 16) !=
          cudaErrorInvalidValue) {
    std::cerr << "invalid arguments were not rejected\n";
    return false;
  }
  return true;
}

bool test_reduce_and_transpose() {
  constexpr int rows = 37;
  constexpr int cols = 113;
  const std::size_t count = static_cast<std::size_t>(rows) * cols;
  std::vector<float> input(count);
  for (std::size_t i = 0; i < count; ++i) {
    input[i] = std::sin(static_cast<float>(i));
  }

  float* device_input = nullptr;
  float* device_reduce = nullptr;
  float* device_transpose = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input),
                        count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_reduce),
                        rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_transpose),
                        count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<float> result(rows);
  CUDA_CHECK(kernel_lab::reduce_mean_baseline(device_input, device_reduce, rows,
                                               cols));
  CUDA_CHECK(cudaMemcpy(result.data(), device_reduce, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < rows; ++row) {
    float reference = 0.0f;
    for (int col = 0; col < cols; ++col) {
      reference += input[static_cast<std::size_t>(row) * cols + col];
    }
    reference /= static_cast<float>(cols);
    if (!expect_close(result[row], reference, 1e-6f,
                      "baseline reduction row " + std::to_string(row))) {
      return false;
    }
  }

  CUDA_CHECK(kernel_lab::reduce_mean_warp(device_input, device_reduce, rows,
                                           cols));
  CUDA_CHECK(cudaMemcpy(result.data(), device_reduce, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < rows; ++row) {
    float reference = 0.0f;
    for (int col = 0; col < cols; ++col) {
      reference += input[static_cast<std::size_t>(row) * cols + col];
    }
    reference /= static_cast<float>(cols);
    if (!expect_close(result[row], reference, 1e-5f,
                      "warp reduction row " + std::to_string(row))) {
      return false;
    }
  }

  const auto check_transpose = [&](const char* name,
                                   cudaError_t (*launch)(const float*, float*,
                                                         int, int,
                                                         cudaStream_t)) {
    const cudaError_t launch_status =
        launch(device_input, device_transpose, rows, cols, nullptr);
    if (launch_status != cudaSuccess) {
      std::cerr << name << " launch failed: "
                << cudaGetErrorString(launch_status) << '\n';
      return false;
    }
    std::vector<float> transposed(count);
    const cudaError_t copy_status =
        cudaMemcpy(transposed.data(), device_transpose, count * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
      std::cerr << name << " copy failed: " << cudaGetErrorString(copy_status)
                << '\n';
      return false;
    }
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const float actual =
            transposed[static_cast<std::size_t>(col) * rows + row];
        const float expected =
            input[static_cast<std::size_t>(row) * cols + col];
        if (actual != expected) {
          std::cerr << name << " mismatch at (" << row << ", " << col
                    << ")\n";
          return false;
        }
      }
    }
    return true;
  };

  const bool transpose_ok =
      check_transpose("naive transpose", kernel_lab::transpose_naive) &&
      check_transpose("conflicted transpose",
                      kernel_lab::transpose_tiled_conflict) &&
      check_transpose("padded transpose", kernel_lab::transpose_tiled);
  cudaFree(device_input);
  cudaFree(device_reduce);
  cudaFree(device_transpose);
  return transpose_ok;
}

bool test_wmma() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 7) {
    std::cout << "WMMA test skipped: compute capability 7.0+ required\n";
    return true;
  }

  constexpr int dimension = 16;
  constexpr std::size_t count = dimension * dimension;
  std::vector<half> a(count);
  std::vector<half> b(count);
  for (std::size_t i = 0; i < count; ++i) {
    a[i] = __float2half(static_cast<float>(static_cast<int>(i % 7) - 3) /
                        8.0f);
    b[i] = __float2half(static_cast<float>(static_cast<int>(i % 5) - 2) /
                        8.0f);
  }

  half* device_a = nullptr;
  half* device_b = nullptr;
  float* device_c = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a),
                        count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b),
                        count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c),
                        count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, a.data(), count * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, b.data(), count * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(kernel_lab::wmma_gemm(device_a, device_b, device_c, dimension,
                                   dimension, dimension));
  std::vector<float> c(count);
  CUDA_CHECK(cudaMemcpy(c.data(), device_c, count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool valid = true;
  for (int row = 0; row < dimension && valid; ++row) {
    for (int col = 0; col < dimension; ++col) {
      float reference = 0.0f;
      for (int inner = 0; inner < dimension; ++inner) {
        reference +=
            __half2float(a[static_cast<std::size_t>(row) * dimension + inner]) *
            __half2float(b[static_cast<std::size_t>(inner) * dimension + col]);
      }
      if (!expect_close(c[static_cast<std::size_t>(row) * dimension + col],
                        reference, 1e-3f, "WMMA result")) {
        valid = false;
        break;
      }
    }
  }
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  return valid;
}

}  // namespace

int main() {
  if (!test_invalid_arguments() || !test_reduce_and_transpose() ||
      !test_wmma()) {
    return 1;
  }
  std::cout << "kernel tests passed\n";
  return 0;
}
