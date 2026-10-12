<#
.SYNOPSIS
    One-time (or repeatable) local environment setup for FazlaChart.

.DESCRIPTION
    Creates a virtual environment in .venv (if it does not already exist)
    and installs the project in editable mode with dev dependencies.
    Safe to re-run at any time; it will not recreate an existing .venv.

.EXAMPLE
    .\tool\setup.ps1
#>

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment in .venv ..." -ForegroundColor Cyan
    python -m venv $venvPath
}
else {
    Write-Host "Virtual environment already exists, reusing .venv" -ForegroundColor DarkGray
}

Write-Host "Installing project (editable, with dev extras) ..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[dev]"

if (-not (Test-Path (Join-Path $repoRoot ".env"))) {
    if (Test-Path (Join-Path $repoRoot ".env.example")) {
        Copy-Item (Join-Path $repoRoot ".env.example") (Join-Path $repoRoot ".env")
        Write-Host "Created .env from .env.example (fill in API keys only when you reach the LLM-agent phase)." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Setup complete. Try:" -ForegroundColor Green
Write-Host "  .\tool\run_scanner.ps1"
Write-Host "  .\tool\run_calibration.ps1"
