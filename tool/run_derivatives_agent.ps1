<#
.SYNOPSIS
    Runs the Derivatives/Risk agent locally against a single symbol, using Ollama by default.

.DESCRIPTION
    Thin wrapper around `python -m agents_crypto`. Fetches the live scanner snapshot,
    the cached calibration backtest, computes deterministic ATR-based risk levels, and
    asks a local LLM to INTERPRET those facts (it never invents its own numbers).

    Requires Ollama running locally (https://ollama.com) with the target model pulled:
        ollama pull llama3.1:8b

.PARAMETER Symbol
    Symbol to assess, e.g. INJUSDT. Required.

.PARAMETER Timeframe
    Kline interval, e.g. 15m, 1h, 4h. Default: 1h.

.PARAMETER HorizonCandles
    Calibration horizon (in candles) to match against active labels. Default: 12.

.PARAMETER Provider
    LLM provider. Default: ollama (local, free, no API key).

.PARAMETER Model
    LLM model name. Default: llama3.1:8b.

.EXAMPLE
    .\tool\run_derivatives_agent.ps1 -Symbol INJUSDT

.EXAMPLE
    .\tool\run_derivatives_agent.ps1 -Symbol BTCUSDT -Timeframe 4h -Model llama3.1:8b
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Symbol,
    [string]$Timeframe = "1h",
    [int]$HorizonCandles = 12,
    [string]$Provider = "ollama",
    [string]$Model = "llama3.1:8b"
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

# Relaxed because the agent CLI may print progress/log lines to stderr,
# which PowerShell would otherwise treat as a terminating error under Stop.
$ErrorActionPreference = "Continue"
& $venvPython -m agents_crypto --symbol $Symbol --timeframe $Timeframe --horizon-candles $HorizonCandles --provider $Provider --model $Model 2>&1 | ForEach-Object { Write-Host $_ }
exit $LASTEXITCODE
