[CmdletBinding()]
param(
  [ValidateSet('reduce', 'transpose', 'wmma')]
  [string]$Op = 'reduce',
  [int]$Rows = 4096,
  [int]$Cols = 4096,
  [int]$K = 256,
  [int]$Iters = 10,
  [string]$BuildDir = 'D:\DevTools\Builds\cuda-kernel-lab-rtx4060',
  [string]$ReportDir = 'reports',
  [ValidateSet('basic', 'full')]
  [string]$NcuSet = 'basic'
)

$ErrorActionPreference = 'Stop'
$executable = Join-Path $BuildDir 'Release\kernel_bench.exe'
if (-not (Test-Path $executable)) {
  $executable = Join-Path $BuildDir 'kernel_bench.exe'
}
if (-not (Test-Path $executable)) {
  throw "Cannot find kernel_bench.exe under $BuildDir"
}
foreach ($tool in @('nsys.exe', 'ncu.exe')) {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
    throw "$tool is not available on PATH"
  }
}

if (-not [IO.Path]::IsPathRooted($ReportDir)) {
  $ReportDir = Join-Path (Get-Location) $ReportDir
}
$ReportDir = [IO.Path]::GetFullPath($ReportDir)
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$executable = (Resolve-Path $executable).Path
$arguments = @('--op', $Op, '--rows', $Rows, '--cols', $Cols,
               '--k', $K, '--iters', $Iters)
$trace = if ($env:OS -eq 'Windows_NT') { 'cuda,nvtx' } else { 'cuda,nvtx,osrt' }
$platformOptions = @()
if ($env:OS -eq 'Windows_NT') {
  $platformOptions = @('--sample=none', '--cpuctxsw=none')
}
if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin'))) {
  $env:Path = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:Path
}
Push-Location (Split-Path $executable -Parent)
try {
  nsys profile --force-overwrite=true --trace=$trace --stats=true `
    @platformOptions `
    -o (Join-Path $ReportDir "$Op-systems") $executable @arguments
  if ($LASTEXITCODE -ne 0) { throw "nsys failed: $LASTEXITCODE" }

  ncu --force-overwrite --set $NcuSet --page raw `
    --kernel-name-base demangled --csv `
    --log-file (Join-Path $ReportDir "$Op-compute.csv") `
    $executable @arguments
  if ($LASTEXITCODE -ne 0) { throw "ncu failed: $LASTEXITCODE" }
} finally {
  Pop-Location
}
