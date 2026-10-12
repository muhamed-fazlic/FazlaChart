<#
.SYNOPSIS
    Runs the Market Scanner locally using the project's virtual environment.

.DESCRIPTION
    Thin wrapper around `python -m scanner` that takes care of:
      - UTF-8 console output (so the tables/arrows render correctly on Windows)
      - using the project's .venv instead of a global Python
      - sensible defaults (top 15 symbols by quote volume, 1h interval, loop forever)

    Any extra arguments you pass are forwarded as-is to `python -m scanner`,
    so you can override the defaults, e.g.:

        .\tool\run_scanner.ps1 -Symbols BTCUSDT,ETHUSDT,INJUSDT -Once

.PARAMETER Symbols
    Comma-separated list of symbols to scan, e.g. BTCUSDT,ETHUSDT.
    If omitted, the scanner picks the top N symbols by quote volume (see -TopN).

.PARAMETER TopN
    Number of top symbols (by 24h quote volume) to scan when -Symbols is not given.
    Default: 15.

.PARAMETER IntervalSeconds
    Seconds to sleep between scan cycles when running continuously.
    Default: 300 (5 minutes).

.PARAMETER Once
    Run a single scan cycle and exit, instead of looping forever.

.EXAMPLE
    .\tool\run_scanner.ps1 -Once

.EXAMPLE
    .\tool\run_scanner.ps1 -Symbols BTCUSDT,ETHUSDT -IntervalSeconds 60
#>

param(
    [string[]]$Symbols,
    [int]$TopN = 15,
    [int]$IntervalSeconds = 300,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

chcp 65001 | Out-Null
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "No .venv found. Run .\tool\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$scannerArgs = @("-m", "scanner", "--interval", "$IntervalSeconds")

if ($Symbols) {
    $scannerArgs += @("--symbols", ($Symbols -join ","))
}
else {
    $scannerArgs += @("--top-n", "$TopN")
}

if ($Once) {
    $scannerArgs += "--once"
}

Write-Host "Running: python $($scannerArgs -join ' ')" -ForegroundColor Cyan

# Python logs INFO/WARNING to stderr; merge it into stdout so PowerShell renders it as
# plain text instead of a (non-terminating, but visually alarming) red error record.
$ErrorActionPreference = "Continue"
& $venvPython @scannerArgs 2>&1 | ForEach-Object { Write-Host $_ }
exit $LASTEXITCODE
