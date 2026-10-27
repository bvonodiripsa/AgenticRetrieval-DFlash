param([Parameter(Mandatory)][string]$Scenario)
$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$file = Join-Path $root 'run.ps1'
$tokens = $null; $parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ('AST syntax errors: ' + (($parseErrors | ForEach-Object Message) -join '; ')) }
$assignment = $ast.EndBlock.Statements | Where-Object { $_.Extent.Text -match '^?\$url\s*=' } | Select-Object -First 1
if (-not $assignment) { throw 'Could not locate top-level $url assignment.' }
$statements = $ast.EndBlock.Statements | Where-Object { $_.Extent.StartOffset -ge $assignment.Extent.StartOffset }
$launcher = ($statements | ForEach-Object Extent | ForEach-Object Text) -join "`n"
$script:browserUrls = @()
$script:startCalls = @()
$script:stoppedIds = @()
$script:healthCalls = 0
$script:dateCalls = 0
$server = [pscustomobject]@{ HasExited = ($Scenario -eq 'early-exit'); ExitCode = (if ($Scenario -eq 'early-exit') { 7 } else { 0 }); Id = 4242; Scenario=$Scenario }
$server | Add-Member ScriptMethod WaitForExit { if ($this.Scenario -eq 'nonzero') { $this.ExitCode = 3 }; $this.HasExited = $true }
function Start-Process {
    param([string]$FilePath, [object]$ArgumentList, [string]$WorkingDirectory, [switch]$NoNewWindow, [switch]$PassThru)
    if ($PassThru) {
        $script:startCalls += ,([pscustomobject]@{ FilePath=$FilePath; ArgumentList=@($ArgumentList); WorkingDirectory=$WorkingDirectory; NoNewWindow=[bool]$NoNewWindow; PassThru=[bool]$PassThru })
        return $server
    }
    $script:browserUrls += $FilePath
}
function Stop-Process { param([int]$Id); $script:stoppedIds += $Id; $server.HasExited = $true }
function Start-Sleep { param([int]$Seconds) }
function Invoke-RestMethod {
    param([string]$Uri, [int]$TimeoutSec)
    $script:healthCalls++
    if ($Scenario -eq 'transient' -and $script:healthCalls -eq 1) { throw [System.Net.WebException]::new('transient') }
    if ($Scenario -eq 'timeout') { return [pscustomobject]@{ status = 'not-ready' } }
    return [pscustomobject]@{ status = 'ok' }
}
function Get-Date {
    $script:dateCalls++
    if ($Scenario -eq 'timeout' -and $script:dateCalls -gt 1) { return [datetime]'2030-01-01T00:06:00' }
    return [datetime]'2030-01-01T00:00:00'
}
$venvPython = 'C:\fake\.venv\Scripts\python.exe'; $projectRoot = $root
try {
    & ([scriptblock]::Create($launcher))
    if ($Scenario -eq 'early-exit') { throw 'Expected early exit failure.' }
    if ($Scenario -eq 'nonzero') { throw 'Expected nonzero exit failure.' }
    if ($Scenario -eq 'timeout') { throw 'Expected timeout failure.' }
    if ($Scenario -in @('success','transient')) {
        if ($script:browserUrls -ne 'http://localhost:8080') { throw ('Wrong browser URL: ' + ($script:browserUrls -join ',')) }
        if ($script:startCalls.Count -ne 1) { throw 'Expected one server Start-Process call.' }
        $c=$script:startCalls[0]; $expected=@('api.py','--config','my.yaml','--host','localhost','--port','8080')
        if ($c.FilePath -ne $venvPython) { throw 'Wrong python executable.' }
        if ((Compare-Object @($c.ArgumentList) $expected) -or $c.WorkingDirectory -ne $root) { throw 'Wrong arguments or working directory.' }
    }
    if ($Scenario -eq 'transient' -and $script:healthCalls -ne 2) { throw 'Transient retry did not occur.' }
    if ($Scenario -eq 'success' -and $script:healthCalls -ne 1) { throw 'Unexpected health call count.' }
    if ($Scenario -eq 'early-exit' -or $Scenario -eq 'nonzero' -or $Scenario -eq 'timeout') { throw 'Expected exception was not raised.' }
    if ($Scenario -eq 'timeout' -and $script:stoppedIds -ne 4242) { throw 'Owned PID was not cleaned up.' }
    'PASS: ' + $Scenario
    exit 0
} catch {
    $msg=$_.Exception.Message
    if ($Scenario -eq 'early-exit' -and $msg -match 'exited with code 7' -and $script:browserUrls.Count -eq 0) { 'PASS: early-exit'; exit 0 }
    if ($Scenario -eq 'nonzero' -and $msg -match 'exited with code 3') { 'PASS: nonzero'; exit 0 }
    if ($Scenario -eq 'timeout' -and $msg -match 'did not become ready' -and $script:stoppedIds -contains 4242 -and $script:browserUrls.Count -eq 0) { 'PASS: timeout'; exit 0 }
    'FAIL: ' + $Scenario + ': ' + $msg
    exit 1
}


