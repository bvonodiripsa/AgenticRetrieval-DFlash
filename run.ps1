$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"

if (-not (Test-Path $requirementsPath)) {
    throw "requirements.txt not found at $requirementsPath"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvPath ..." -ForegroundColor Cyan
    python -m venv $venvPath
}

Write-Host "Installing dependencies from requirements.txt ..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r $requirementsPath

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
Write-Host "Activating virtual environment ..." -ForegroundColor Cyan
. $activateScript

Write-Host "Virtual environment ready and activated." -ForegroundColor Green
