<#
.SYNOPSIS
    Starts the local web dashboard (reads scanner_data/scanner.db).

.DESCRIPTION
    Runs the FastAPI dashboard with uvicorn and opens it in your default
    browser. The scanner itself (python -m scanner / .\tool\run_scanner.ps1)
    should be running separately — the dashboard only reads what it has
    already written to SQLite; it never calls Binance itself.

.PARAMETER Port
    Port to listen on. Default: 8000.

.PARAMETER NoBrowser
    Do not automatically open a browser tab.

.EXAMPLE
    .\tool\run_dashboard.ps1
#>

param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "No .venv found. Run .\tool\setup.ps1 first." -ForegroundColor Red
    exit 1
}

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 2
        Start-Process $url
    } -ArgumentList "http://127.0.0.1:$Port" | Out-Null
}

Write-Host "Dashboard starting at http://127.0.0.1:$Port (Ctrl+C to stop)" -ForegroundColor Cyan

# uvicorn's own request logs go to stderr; merge into stdout to avoid
# PowerShell rendering routine access logs as red error records.
$ErrorActionPreference = "Continue"
& $venvPython -m uvicorn webapp.main:app --host 127.0.0.1 --port $Port 2>&1 | ForEach-Object { Write-Host $_ }
exit $LASTEXITCODE
