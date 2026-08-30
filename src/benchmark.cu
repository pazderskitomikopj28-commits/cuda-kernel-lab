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

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.op == "reduce") return benchmark_reduce(options);
  if (options.op == "transpose") return benchmark_transpose(options);
  std::cerr << "unknown --op: " << options.op << " (use reduce or transpose)\n";
  return 2;
}
