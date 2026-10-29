param(
    [ValidateSet('3.10', '3.11')]
    [string]$PythonVersion = '3.11'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$EnvironmentDir = Join-Path $ProjectRoot '.venv-cuda'
$PythonExe = Join-Path $EnvironmentDir 'Scripts\python.exe'

function Invoke-Checked {
    param([string]$Executable, [string[]]$CommandArguments)
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable $($CommandArguments -join ' ')"
    }
}

if ($env:OS -ne 'Windows_NT') { throw 'This installer is for Windows.' }
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install 64-bit Python 3.11 with the Python launcher, then run this script again.'
}

Push-Location $ProjectRoot
try {
    Write-Host 'Checking Python (64-bit required) ...'
    Invoke-Checked 'py' @("-$PythonVersion", '-c', 'import sys; print(sys.version); assert sys.maxsize > 2**32')
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        if (Test-Path -LiteralPath $EnvironmentDir) {
            throw '.venv-cuda already exists but is incomplete. Move it aside before retrying.'
        }
        Invoke-Checked 'py' @("-$PythonVersion", '-m', 'venv', $EnvironmentDir)
    }
    $VenvConfig = Get-Content -LiteralPath (Join-Path $EnvironmentDir 'pyvenv.cfg')
    if ($VenvConfig -match '^include-system-site-packages\s*=\s*true') {
        throw '.venv-cuda must be isolated. Move the existing environment aside and retry.'
    }
    Write-Host 'Installing GPU dependencies. This step needs Internet access and may download several GB.'
    Invoke-Checked $PythonExe @('-m', 'pip', 'install', '--upgrade', 'pip')
    # Exact CUDA build tags replace a CPU build if this environment was reused.
    Invoke-Checked $PythonExe @('-m', 'pip', 'install', '--upgrade', 'torch==2.8.0+cu126', 'torchvision==0.23.0+cu126', '--index-url', 'https://download.pytorch.org/whl/cu126')
    Invoke-Checked $PythonExe @('-m', 'pip', 'install', '-r', (Join-Path $ProjectRoot 'requirements.txt'))
    Invoke-Checked $PythonExe @('-m', 'pip', 'check')
    Invoke-Checked $PythonExe @((Join-Path $PSScriptRoot 'check_environment.py'), '--device', 'cuda:0')
    Invoke-Checked $PythonExe @((Join-Path $PSScriptRoot 'bootstrap.py'))
    Invoke-Checked $PythonExe @((Join-Path $PSScriptRoot 'check_environment.py'), '--device', 'cuda:0', '--model-smoke-test')
    Write-Host 'CUDA setup and model checks passed. Run scripts/run_windows_cuda.ps1 next.'
}
finally {
    Pop-Location
}
