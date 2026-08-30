#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t status = (call);                                         \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "   \
                << cudaGetErrorString(status) << '\n';                        \
      return 1;                                                               \
    }                                                                           \
  } while (false)

struct Options {
  int rows = 4096;
  int cols = 4096;
  int inner = 0;
  int iterations = 100;
  std::string op = "reduce";
};

Options parse_options(int argc, char** argv) {
  Options options;
  bool rows_set = false;
  bool cols_set = false;
  bool inner_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto read_int = [&](int& target) {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
      const std::string value(argv[++i]);
      std::size_t consumed = 0;
      const int parsed = std::stoi(value, &consumed);
      if (consumed != value.size() || parsed <= 0) {
        throw std::invalid_argument(arg + " requires a positive integer");
      }
      target = parsed;
    };
    if (arg == "--rows") {
      read_int(options.rows);
      rows_set = true;
    } else if (arg == "--cols") {
      read_int(options.cols);
      cols_set = true;
    } else if (arg == "--k") {
      read_int(options.inner);
      inner_set = true;
    } else if (arg == "--iters") {
      read_int(options.iterations);
    } else if (arg == "--op") {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for --op");
      options.op = argv[++i];
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (options.op == "wmma") {
    if (!rows_set) options.rows = 256;
    if (!cols_set) options.cols = 256;
    if (!inner_set) options.inner = 256;
  } else if (!inner_set) {
    options.inner = options.cols;
  }
  return options;
}

cudaError_t elapsed_ms(cudaEvent_t start, cudaEvent_t stop, int iterations,
                       float* result) {
  float total = 0.0f;
  const cudaError_t status = cudaEventElapsedTime(&total, start, stop);
  if (status != cudaSuccess) return status;
  *result = total / static_cast<float>(iterations);
  return cudaSuccess;
}

int benchmark_reduce(const Options& options) {
  const size_t count = static_cast<size_t>(options.rows) * options.cols;
  std::vector<float> host_input(count);
  std::vector<float> host_output(options.rows, 0.0f);
  std::mt19937 generator(7);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (float& value : host_input) value = distribution(generator);

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input),
                       count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output),
                       options.rows * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(kernel_lab::reduce_mean_baseline(device_input, device_output,
                                              options.rows, options.cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) {
    CUDA_CHECK(kernel_lab::reduce_mean_baseline(device_input, device_output,
                                                options.rows, options.cols));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float baseline_ms = 0.0f;
  CUDA_CHECK(elapsed_ms(start, stop, options.iterations, &baseline_ms));

  CUDA_CHECK(kernel_lab::reduce_mean_warp(device_input, device_output,
                                          options.rows, options.cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) {
    CUDA_CHECK(kernel_lab::reduce_mean_warp(device_input, device_output,
                                            options.rows, options.cols));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float optimized_ms = 0.0f;
  CUDA_CHECK(elapsed_ms(start, stop, options.iterations, &optimized_ms));

  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                        options.rows * sizeof(float), cudaMemcpyDeviceToHost));
  float max_abs_error = 0.0f;
  for (int row = 0; row < options.rows; ++row) {
    float reference = 0.0f;
    for (int col = 0; col < options.cols; ++col) {
      reference += host_input[static_cast<size_t>(row) * options.cols + col];
    }
    reference /= static_cast<float>(options.cols);
    max_abs_error = std::max(max_abs_error, std::abs(reference - host_output[row]));
  }

  std::cout << std::fixed << std::setprecision(4)
            << "op=reduce_mean rows=" << options.rows << " cols=" << options.cols
            << " baseline_ms=" << baseline_ms << " warp_ms=" << optimized_ms
            << " speedup=" << baseline_ms / optimized_ms
            << " max_abs_error=" << max_abs_error << '\n';

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_input);
  cudaFree(device_output);
  return max_abs_error < 1e-3f ? 0 : 2;
}

int benchmark_transpose(const Options& options) {
  const size_t count = static_cast<size_t>(options.rows) * options.cols;
  std::vector<float> host_input(count);
  std::vector<float> host_output(count, 0.0f);
  for (size_t i = 0; i < count; ++i) host_input[i] = static_cast<float>(i % 1021);

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input),
                       count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output),
                       count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(kernel_lab::transpose_naive(device_input, device_output,
                                         options.rows, options.cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) {
    CUDA_CHECK(kernel_lab::transpose_naive(device_input, device_output,
                                           options.rows, options.cols));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float naive_ms = 0.0f;
  CUDA_CHECK(elapsed_ms(start, stop, options.iterations, &naive_ms));

  CUDA_CHECK(kernel_lab::transpose_tiled_conflict(
      device_input, device_output, options.rows, options.cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) {
    CUDA_CHECK(kernel_lab::transpose_tiled_conflict(
        device_input, device_output, options.rows, options.cols));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float conflict_ms = 0.0f;
  CUDA_CHECK(elapsed_ms(start, stop, options.iterations, &conflict_ms));

  CUDA_CHECK(kernel_lab::transpose_tiled(device_input, device_output,
                                         options.rows, options.cols));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) {
    CUDA_CHECK(kernel_lab::transpose_tiled(device_input, device_output,
                                           options.rows, options.cols));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float tiled_ms = 0.0f;
  CUDA_CHECK(elapsed_ms(start, stop, options.iterations, &tiled_ms));

  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  float max_abs_error = 0.0f;
  for (int row = 0; row < options.rows; ++row) {
    for (int col = 0; col < options.cols; ++col) {
      const float expected = host_input[static_cast<size_t>(row) * options.cols + col];
      const float actual = host_output[static_cast<size_t>(col) * options.rows + row];
      max_abs_error = std::max(max_abs_error, std::abs(expected - actual));
    }
  }

  std::cout << std::fixed << std::setprecision(4)
            << "op=transpose rows=" << options.rows << " cols=" << options.cols
            << " naive_ms=" << naive_ms << " conflict_ms=" << conflict_ms
            << " padded_ms=" << tiled_ms
            << " padded_vs_conflict_speedup=" << conflict_ms / tiled_ms
            << " padded_vs_naive_speedup=" << naive_ms / tiled_ms
            << " max_abs_error=" << max_abs_error << '\n';
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_input);
  cudaFree(device_output);
  return max_abs_error < 1e-6f ? 0 : 2;
}

int benchmark_wmma(const Options& options) {
  const int m = options.rows;
  const int n = options.cols;
  const int k = options.inner;
  if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0) {
    std::cerr << "wmma requires rows, cols and k to be multiples of 16\n";
    return 2;
  }

  const size_t a_count = static_cast<size_t>(m) * k;
  const size_t b_count = static_cast<size_t>(k) * n;
  const size_t c_count = static_cast<size_t>(m) * n;
  std::vector<half> host_a(a_count);
  std::vector<half> host_b(b_count);
  std::vector<float> host_c(c_count, 0.0f);
  for (size_t i = 0; i < a_count; ++i) {
    host_a[i] = __float2half(
        static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f);
  }
  for (size_t i = 0; i < b_count; ++i) {
    host_b[i] = __float2half(
        static_cast<float>(static_cast<int>(i % 13) - 6) / 16.0f);
  }

  half* device_a = nullptr;
  half* device_b = nullptr;
  float* device_c = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), a_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), b_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), a_count * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), b_count * sizeof(half),
                        cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(kernel_lab::wmma_gemm(device_a, device_b, device_c, m, n, k));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < options.iterations; ++i) {
    CUDA_CHECK(kernel_lab::wmma_gemm(device_a, device_b, device_c, m, n, k));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(elapsed_ms(start, stop, options.iterations, &ms));
  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float reference = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        reference += __half2float(host_a[static_cast<size_t>(row) * k + inner]) *
                     __half2float(host_b[static_cast<size_t>(inner) * n + col]);
      }
      max_abs_error = std::max(
          max_abs_error,
          std::abs(reference - host_c[static_cast<size_t>(row) * n + col]));
    }
  }
  const double tflops = (2.0 * m * n * k) / (static_cast<double>(ms) * 1e9);
  std::cout << std::fixed << std::setprecision(4)
            << "op=wmma_gemm m=" << m << " n=" << n << " k=" << k
            << " ms=" << ms << " TFLOP/s=" << tflops
            << " max_abs_error=" << max_abs_error << '\n';

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  return max_abs_error < 5e-2f ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.op == "reduce") return benchmark_reduce(options);
    if (options.op == "transpose") return benchmark_transpose(options);
    if (options.op == "wmma") return benchmark_wmma(options);
    std::cerr << "unknown --op: " << options.op
              << " (use reduce, transpose or wmma)\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
