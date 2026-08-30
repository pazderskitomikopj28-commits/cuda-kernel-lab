# CUDA Kernel Lab

[![CUDA compile](https://github.com/pazderskitomikopj28-commits/cuda-kernel-lab/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/pazderskitomikopj28-commits/cuda-kernel-lab/actions/workflows/cuda-build.yml)

一套可复现的 CUDA 算子优化实验，围绕“正确性、性能和可解释的优化过程”组织。仓库保留基线实现、优化实现、测试、测量命令和架构笔记，便于重复实验与比较不同 GPU 上的结果。

## 实验组成

| 实验 | 比较对象 | 关注点 |
| --- | --- | --- |
| 行归约 | shared-memory baseline vs warp shuffle | 线程映射、SIMT、warp 内归约和同步边界 |
| 条件变换 | warp-uniform vs alternating divergence vs branchless | 分支发散、数据布局与 predication 取舍 |
| 矩阵转置 | naive vs 32×32 tiled vs `+1` padded tile | 合并访问、shared memory、bank conflict 和 `__syncthreads()` |
| 矩阵乘 | host FP32 reference vs WMMA | Tensor Core、fragment 形状限制和数值正确性 |
| 性能诊断 | CUDA event、Nsight 与 Compute Sanitizer | 重复测量、瓶颈定位和内存/竞争错误检查 |

WGMMA 与 WMMA 不是可互换接口。Hopper warpgroup 异步矩阵指令的实现条件和验证边界单独记录在 [`docs/wgmma-notes.md`](docs/wgmma-notes.md)，不把未在相应硬件上运行的代码作为实验结果。

## 构建

需要 CUDA Toolkit 12.x（或兼容版本）、CMake 3.22+ 和支持 CUDA 的 NVIDIA GPU。Windows PowerShell：

```powershell
.\scripts\build_windows.ps1
.\scripts\run_bench.ps1 -BuildDir D:\DevTools\Builds\cuda-kernel-lab-rtx4060 `
  -Op reduce -Rows 4096 -Cols 4096
.\scripts\run_bench.ps1 -BuildDir D:\DevTools\Builds\cuda-kernel-lab-rtx4060 `
  -Op transpose -Rows 4097 -Cols 3073
.\scripts\run_bench.ps1 -BuildDir D:\DevTools\Builds\cuda-kernel-lab-rtx4060 `
  -Op divergence -Rows 1024 -Cols 4096 -Work 64
.\scripts\run_bench.ps1 -BuildDir D:\DevTools\Builds\cuda-kernel-lab-rtx4060 `
  -Op wmma -Rows 256 -Cols 256 -K 256
```

`build_windows.ps1` 默认使用 `D:\DevTools` 下的 CUDA 12.4、CMake 和
VS2022 Build Tools，锁定 MSVC 14.38 与 `sm_89`。如果源码路径含中文，脚本会在
构建根目录创建英文目录联接，避开 Windows `nvcc` 的路径编码问题；所有路径和
参数都可以覆盖。

Linux：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
./scripts/run_bench.sh reduce
```

GitHub Actions 会在 CUDA 12.4 镜像中验证 SM80 编译；正确性测试和 `compute-sanitizer` 仍需在有 NVIDIA GPU 的机器上执行：

```bash
./scripts/sanitize.sh
```

Windows 真机使用：

```powershell
.\scripts\sanitize.ps1 -BuildDir D:\DevTools\Builds\cuda-kernel-lab-rtx4060
```

没有 CUDA 环境时仍可阅读源码、检查 CMake 和文档，但不应发布未经真实运行验证的性能数值。

## 运行 Nsight

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true -o reports/reduce_systems \
  ./build/kernel_bench --op reduce --rows 4096 --cols 4096 --iters 100

ncu --set basic --page raw --kernel-name-base demangled \
  -o reports/reduce_compute ./build/kernel_bench \
  --op reduce --rows 4096 --cols 4096 --iters 10
```

Windows PowerShell 可用 `scripts/profile.ps1` 同时生成 Systems 报告和 Compute
CSV。Windows 默认关闭需要管理员权限的 CPU sampling/context-switch trace，
只采集分析 CUDA 时间线所需的 `cuda,nvtx`：

```powershell
.\scripts\profile.ps1 -BuildDir D:\DevTools\Builds\cuda-kernel-lab-rtx4060 `
  -ReportDir D:\DevTools\Profiles\kernel-lab -Op reduce
```

Systems 用来确认调度、拷贝和 Stream 并发；Compute 用来定位单个 Kernel 的访存、Occupancy 和 Warp Stall。完整设计说明见 [`docs/architecture.md`](docs/architecture.md)。

当前可执行矩阵路径是 SM70+ WMMA。WGMMA 依赖 SM90+、warpgroup 和异步提交/等待语义，本仓库不把未在 Hopper 真机验证的 PTX 片段包装成“已实现”；学习与验收清单见 [`docs/wgmma-notes.md`](docs/wgmma-notes.md)。

RTX 4060 Laptop GPU 的重复测量、中位数和 Sanitizer 证据见
[`docs/results/rtx4060-laptop-2026-08-30.md`](docs/results/rtx4060-laptop-2026-08-30.md)。

## 迁移到国产 GPU

本项目把算法与接口边界写清楚，便于迁移到 BIRENSUPA/br_pytorch。WMMA 部分仅是 NVIDIA 参考实现；迁移时需要依据壁仞 SDK 的设备执行组织、矩阵指令、编译器和 profiling 工具重新验证。相关说明见后续的 `gpu-perf-playbook` 项目。

## 许可证

MIT
