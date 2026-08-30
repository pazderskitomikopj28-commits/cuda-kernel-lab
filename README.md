# CUDA Kernel Lab

一套可复现的 CUDA 算子优化练习，围绕“正确性 + 性能 + 可解释的优化过程”组织，适合准备 GPU 编程、算子开发和性能工程面试。

## 覆盖的能力

| 问卷能力 | 对应内容 |
| --- | --- |
| Kernel / Grid / Block / Thread | `src/kernels.cu` 中的 reduction 与 transpose 映射 |
| SIMT / Warp / Divergence | warp shuffle reduction、边界分支分析 |
| Shared Memory | 32×32 tiling、`+1` padding、`__syncthreads()` |
| Tensor Core / WMMA | `wmma_gemm`（SM70+，16 的倍数尺寸） |
| Global Memory Coalescing | transpose 的连续加载、转置写回 |
| Nsight 分析 | `scripts/` 与文档中的 Systems/Compute 命令 |
| 工程能力 | CMake、单元测试、benchmark、可读的设计笔记 |

## 构建

需要 CUDA Toolkit 12.x（或兼容版本）、CMake 3.24+ 和支持 CUDA 的 NVIDIA GPU。Windows PowerShell：

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\scripts\run_bench.ps1 -Op reduce -Rows 4096 -Cols 4096
.\scripts\run_bench.ps1 -Op transpose -Rows 4097 -Cols 3073
```

Linux：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
./scripts/run_bench.sh reduce
```

没有 CUDA 环境时可以阅读源码、检查 CMake 和文档，但不要把没有真实运行的性能数值写进简历。当前仓库不包含任何伪造 benchmark 结果。

## 运行 Nsight

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true -o reports/reduce_systems \
  ./build/kernel_bench --op reduce --rows 4096 --cols 4096 --iters 100

ncu --set full --kernel-name-base demangled \
  -o reports/reduce_compute ./build/kernel_bench \
  --op reduce --rows 4096 --cols 4096 --iters 10
```

Systems 用来确认调度、拷贝和 Stream 并发；Compute 用来定位单个 Kernel 的访存、Occupancy 和 Warp Stall。完整设计说明见 [`docs/architecture.md`](docs/architecture.md)。

## 迁移到国产 GPU

本项目把算法与接口边界写清楚，便于迁移到 BIRENSUPA/br_pytorch。WMMA 部分仅是 NVIDIA 参考实现；迁移时需要依据壁仞 SDK 的设备执行组织、矩阵指令、编译器和 profiling 工具重新验证。相关说明见后续的 `gpu-perf-playbook` 项目。

## 许可证

MIT
