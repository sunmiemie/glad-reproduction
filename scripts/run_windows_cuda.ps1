param(
    [string]$Source = '',
    [string]$Output = '',
    [ValidateRange(0, 2147483647)]
    [int]$MaxFrames = 0,
    [switch]$ShowWindow
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonExe = Join-Path $ProjectRoot '.venv-cuda\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw 'Missing GPU environment. Run scripts/setup_windows_cuda.ps1 first.'
}
if ([string]::IsNullOrWhiteSpace($Source)) {
    $Source = Join-Path $ProjectRoot 'videos\phantom64.mp4'
}
$Source = (Resolve-Path -LiteralPath $Source).Path
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunName = 'rtx4060_' + [IO.Path]::GetFileNameWithoutExtension($Source) + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
    $Output = Join-Path (Join-Path $ProjectRoot 'runs') $RunName
} elseif (-not [IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $ProjectRoot $Output
}

Push-Location $ProjectRoot
try {
    & $PythonExe (Join-Path $PSScriptRoot 'check_environment.py') '--device' 'cuda:0'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA checks failed. CPU fallback is disabled.' }
    $RunArguments = @('-m', 'glad', '--source', $Source, '--output', $Output,
                      '--device', 'cuda:0', '--threads', '4', '--imgsz', '640',
                      '--roi-size', '300', '--max-frames', "$MaxFrames")
    if ($ShowWindow) { $RunArguments += '--display' }
    & $PythonExe @RunArguments
    if ($LASTEXITCODE -ne 0) { throw "Video run failed (exit $LASTEXITCODE)." }
    Write-Host "Finished. Video and summary: $Output"
}
finally {
    Pop-Location
}
