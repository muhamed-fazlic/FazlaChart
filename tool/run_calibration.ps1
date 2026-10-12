<#
.SYNOPSIS
    Runs the label calibration backtest locally using the project's virtual environment.

.DESCRIPTION
    Thin wrapper around `python -m scanner.calibrate`. Answers the core question:
    "If the scanner flags trend X or anomaly Y, how often does that actually pay off
    historically, after trading costs?" No LLM calls, pure historical Binance data.

.PARAMETER Symbols
    Comma-separated list of symbols to backtest, e.g. BTCUSDT,ETHUSDT,INJUSDT.
    Default: BTCUSDT,ETHUSDT,INJUSDT.

.PARAMETER Timeframe
    Kline interval, e.g. 15m, 1h, 4h. Default: 1h.

.PARAMETER Horizons
    Forward horizons in candles, comma-separated. Default: 4,12,24.

.PARAMETER Limit
    Historical candles to fetch (Binance max 1500). Default: 1500.

.EXAMPLE
    .\tool\run_calibration.ps1

.EXAMPLE
    .\tool\run_calibration.ps1 -Symbols SOLUSDT -Timeframe 15m -Horizons 4,12,48
#>

param(
    [string]$Symbols = "BTCUSDT,ETHUSDT,INJUSDT",
    [string]$Timeframe = "1h",
    [string]$Horizons = "4,12,24",
    [int]$Limit = 1500
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

# Relaxed because the calibration CLI may print progress/log lines to stderr,
# which PowerShell would otherwise treat as a terminating error under Stop.
# Merging into stdout avoids a visually alarming (but harmless) red error record.
$ErrorActionPreference = "Continue"
& $venvPython -m scanner.calibrate --symbols $Symbols --timeframe $Timeframe --horizons $Horizons --limit $Limit 2>&1 | ForEach-Object { Write-Host $_ }
exit $LASTEXITCODE
