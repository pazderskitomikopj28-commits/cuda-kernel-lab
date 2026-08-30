param(
  [ValidateSet('reduce', 'transpose')]
  [string]$Op = 'reduce',
  [int]$Rows = 4096,
  [int]$Cols = 4096,
  [int]$Iters = 100,
  [string]$BuildDir = 'build'
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $BuildDir 'kernel_bench.exe'
if (-not (Test-Path $exe)) {
  throw "Cannot find $exe. Configure and build with CMake first."
}
& $exe --op $Op --rows $Rows --cols $Cols --iters $Iters
