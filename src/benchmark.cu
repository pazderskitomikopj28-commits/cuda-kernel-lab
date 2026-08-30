#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
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
  int iterations = 100;
  std::string op = "reduce";
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto read_int = [&](int& target) {
      if (i + 1 < argc) {
        target = std::max(1, std::stoi(argv[++i]));
      }
    };
    if (arg == "--rows") read_int(options.rows);
    else if (arg == "--cols") read_int(options.cols);
    else if (arg == "--iters") read_int(options.iterations);
    else if (arg == "--op" && i + 1 < argc) options.op = argv[++i];
  }
  return options;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop, int iterations) {
  float total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
  return total / static_cast<float>(iterations);
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
  const float baseline_ms = elapsed_ms(start, stop, options.iterations);

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
  const float optimized_ms = elapsed_ms(start, stop, options.iterations);

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
  const float naive_ms = elapsed_ms(start, stop, options.iterations);

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
  const float tiled_ms = elapsed_ms(start, stop, options.iterations);

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
            << " naive_ms=" << naive_ms << " tiled_ms=" << tiled_ms
            << " speedup=" << naive_ms / tiled_ms
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
  const int k = options.cols;
  if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0) {
    std::cerr << "wmma requires rows and cols to be multiples of 16\n";
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
  const float ms = elapsed_ms(start, stop, options.iterations);
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
  const Options options = parse_options(argc, argv);
  if (options.op == "reduce") return benchmark_reduce(options);
  if (options.op == "transpose") return benchmark_transpose(options);
  if (options.op == "wmma") return benchmark_wmma(options);
  std::cerr << "unknown --op: " << options.op
            << " (use reduce, transpose or wmma)\n";
  return 2;
}
