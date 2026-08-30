[CmdletBinding()]
param(
  [string]$BuildDir = 'D:\DevTools\Builds\cuda-kernel-lab-rtx4060',
  [ValidateSet('Debug', 'Release')]
  [string]$Configuration = 'Release',
  [string]$ComputeSanitizer = ''
)

$ErrorActionPreference = 'Stop'
if (-not $ComputeSanitizer) {
  $command = Get-Command compute-sanitizer.exe -ErrorAction SilentlyContinue
  if ($command) { $ComputeSanitizer = $command.Source }
}
if (-not $ComputeSanitizer -and $env:CUDA_PATH) {
  $candidate = Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe'
  if (Test-Path $candidate) { $ComputeSanitizer = $candidate }
}
if (-not $ComputeSanitizer -or -not (Test-Path $ComputeSanitizer)) {
  throw 'compute-sanitizer.exe not found. Pass -ComputeSanitizer or set CUDA_PATH.'
}
$cudaRoot = Split-Path (Split-Path $ComputeSanitizer -Parent) -Parent
if (Test-Path (Join-Path $cudaRoot 'bin')) {
  $env:Path = (Join-Path $cudaRoot 'bin') + ';' + $env:Path
}

$executable = Join-Path $BuildDir "$Configuration\kernel_tests.exe"
if (-not (Test-Path $executable)) {
  $executable = Join-Path $BuildDir 'kernel_tests.exe'
}
if (-not (Test-Path $executable)) {
  throw "Cannot find kernel_tests.exe under $BuildDir"
}

foreach ($tool in @('memcheck', 'racecheck')) {
  Write-Host "== $tool =="
  & $ComputeSanitizer --tool $tool --error-exitcode 99 $executable
  if ($LASTEXITCODE -ne 0) {
    throw "$tool failed with exit code $LASTEXITCODE. On WDDM, run the Toolkit's EnableDebuggerInterface.bat as administrator once."
  }
}
