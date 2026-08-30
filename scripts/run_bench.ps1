param(
  [ValidateSet('reduce', 'transpose', 'divergence', 'coalescing', 'wmma')]
  [string]$Op = 'reduce',
  [int]$Rows = 4096,
  [int]$Cols = 4096,
  [int]$K = 256,
  [int]$Work = 64,
  [int]$Iters = 100,
  [string]$BuildDir = 'build'
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $BuildDir 'kernel_bench.exe'
if (-not (Test-Path $exe)) {
  $releaseExe = Join-Path $BuildDir 'Release\kernel_bench.exe'
  if (Test-Path $releaseExe) { $exe = $releaseExe }
}
if (-not (Test-Path $exe)) {
  throw "Cannot find kernel_bench.exe under $BuildDir. Configure and build with CMake first."
}
& $exe --op $Op --rows $Rows --cols $Cols --k $K --work $Work --iters $Iters
