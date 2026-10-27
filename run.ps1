$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $projectRoot "requirements.txt"

foreach ($requiredPath in @($requirementsPath, (Join-Path $projectRoot "my.yaml"))) {
    if (-not (Test-Path $requiredPath)) {
        throw "Required file not found at $requiredPath"
    }
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvPath ..." -ForegroundColor Cyan
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

Write-Host "Installing dependencies from requirements.txt ..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $venvPython -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
Write-Host "Activating virtual environment ..." -ForegroundColor Cyan
. $activateScript

Write-Host "Virtual environment ready and activated." -ForegroundColor Green

$url = "http://localhost:8080"
Write-Host "Starting the backend and web UI at $url (Ctrl+C to stop) ..." -ForegroundColor Cyan
$server = Start-Process -FilePath $venvPython `
    -ArgumentList @("api.py", "--config", "my.yaml", "--host", "localhost", "--port", "8080") `
    -WorkingDirectory $projectRoot -NoNewWindow -PassThru

try {
    $deadline = (Get-Date).AddMinutes(5)
    while ($true) {
        if ($server.HasExited) {
            throw "The API server exited with code $($server.ExitCode). Check the output above."
        }
        try {
            $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2
            if ($health.status -eq "ok") { break }
        }
        catch [System.Net.WebException], [System.Net.Http.HttpRequestException], [System.Threading.Tasks.TaskCanceledException] {
            Write-Verbose "Waiting for the API server: $($_.Exception.Message)"
        }
        if ((Get-Date) -ge $deadline) {
            throw "The API server did not become ready at $url within 5 minutes."
        }
        Start-Sleep -Seconds 1
    }

    Start-Process $url
    $server.WaitForExit()
    if ($server.ExitCode -ne 0) {
        throw "The API server exited with code $($server.ExitCode)."
    }
}
finally {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id
    }
}
