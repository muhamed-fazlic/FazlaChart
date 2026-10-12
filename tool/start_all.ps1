<#
.SYNOPSIS
    One-click launcher: starts Ollama, the Market Scanner, and the web dashboard,
    then opens the dashboard in Chrome. Designed to be double-clicked (e.g. from
    a desktop shortcut) with zero prompts.

.DESCRIPTION
    - Makes sure the local Ollama server is reachable (it normally auto-starts as a
      background app after installation; this script starts it if it isn't running).
    - Launches the Market Scanner (`scanner`) and the dashboard (`uvicorn`) each in
      their own minimized window so you can glance at their logs, and also mirrors
      their output to tool\.run\*.log.
    - Waits for the dashboard to actually respond before opening the browser, so you
      never see a "can't connect" page.
    - Records each process's PID in tool\.run\*.pid so `tool\stop_all.ps1` can stop
      them cleanly later.
    - Safe to re-run: if the scanner/dashboard are already running (PID file points
      at a live process), it reuses them instead of starting duplicates.

.PARAMETER Symbols
    Top-N symbols for the scanner (see `python -m scanner --help`). Default: 25.

.PARAMETER Interval
    Scanner poll interval in seconds. Default: 300.

.PARAMETER DashboardPort
    Port for the local dashboard. Default: 8000.

.PARAMETER NoBrowser
    Skip opening Chrome automatically (useful if you just want services running).

.EXAMPLE
    .\tool\start_all.ps1
#>

param(
    [int]$TopN = 25,
    [int]$Interval = 300,
    [int]$DashboardPort = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$runDir = Join-Path $repoRoot "tool\.run"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "No .venv found. Run .\tool\setup.ps1 first." -ForegroundColor Red
    exit 1
}

function Test-ProcessAlive($pidFile) {
    if (-not (Test-Path $pidFile)) { return $false }
    $procId = Get-Content $pidFile -ErrorAction SilentlyContinue
    if (-not $procId) { return $false }
    return [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
}

function Wait-ForHttp($url, $timeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) { return $true }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

# --- 1. Ollama -----------------------------------------------------------
# IMPORTANT: on modest hardware (e.g. laptops without a dedicated GPU, <=16GB
# RAM), loading more than one local model at once can exhaust RAM and make
# every request crawl (observed: free RAM dropping below 1GB, calls stalling
# for 10+ minutes). OLLAMA_MAX_LOADED_MODELS=1 forces Ollama to unload the
# previous model before loading a new one, so only one model ever sits in
# memory at a time.
$env:OLLAMA_MAX_LOADED_MODELS = "1"
Write-Host "[1/4] Checking Ollama..." -ForegroundColor Cyan
$ollamaUrl = "http://127.0.0.1:11434/api/tags"
if (Wait-ForHttp $ollamaUrl 2) {
    Write-Host "      Ollama already running." -ForegroundColor Green
} else {
    # Prefer running the server binary directly ("ollama serve") rather than the
    # tray app ("ollama app.exe"): the tray app's own server subprocess has been
    # observed to silently fail to bind its port while "ollama serve" starts
    # reliably in ~5-10s. No tray icon this way, but the API works the same.
    $ollamaServerExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    $ollamaLog = Join-Path $runDir "ollama.log"
    if (Test-Path $ollamaServerExe) {
        Write-Host "      Starting Ollama server..." -ForegroundColor Yellow
        $op = Start-Process -FilePath $ollamaServerExe -ArgumentList "serve" `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $ollamaLog -RedirectStandardError (Join-Path $runDir "ollama.err.log")
        Set-Content -Path (Join-Path $runDir "ollama.pid") -Value $op.Id
    } else {
        Write-Host "      Ollama not found at expected path; assuming it's on PATH as a service." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    }
    if (-not (Wait-ForHttp $ollamaUrl 20)) {
        Write-Host "      WARNING: Ollama did not respond after 20s. The Derivatives/Risk agent" -ForegroundColor Red
        Write-Host "      won't work until it is, but scanner + dashboard will still start." -ForegroundColor Red
    } else {
        Write-Host "      Ollama is up." -ForegroundColor Green
    }
}

# --- 2. Market Scanner -----------------------------------------------------
Write-Host "[2/4] Checking Market Scanner..." -ForegroundColor Cyan
$scannerPidFile = Join-Path $runDir "scanner.pid"
if (Test-ProcessAlive $scannerPidFile) {
    Write-Host "      Already running (PID $(Get-Content $scannerPidFile))." -ForegroundColor Green
} else {
    Write-Host "      Starting scanner (top $TopN symbols, ${Interval}s interval)..." -ForegroundColor Yellow
    $scannerLog = Join-Path $runDir "scanner.log"
    $p = Start-Process -FilePath $venvPython `
        -ArgumentList "-m", "scanner", "--top-n", "$TopN", "--interval", "$Interval" `
        -WindowStyle Minimized -PassThru `
        -RedirectStandardOutput $scannerLog -RedirectStandardError (Join-Path $runDir "scanner.err.log")
    Set-Content -Path $scannerPidFile -Value $p.Id
    Write-Host "      Started (PID $($p.Id)). Logs: $scannerLog" -ForegroundColor Green
}

# --- 3. Dashboard ------------------------------------------------------
Write-Host "[3/4] Checking dashboard..." -ForegroundColor Cyan
$dashboardUrl = "http://127.0.0.1:$DashboardPort"
$dashboardPidFile = Join-Path $runDir "dashboard.pid"
if (Test-ProcessAlive $dashboardPidFile) {
    Write-Host "      Already running (PID $(Get-Content $dashboardPidFile))." -ForegroundColor Green
} else {
    Write-Host "      Starting dashboard on $dashboardUrl..." -ForegroundColor Yellow
    $dashboardLog = Join-Path $runDir "dashboard.log"
    $p = Start-Process -FilePath $venvPython `
        -ArgumentList "-m", "uvicorn", "webapp.main:app", "--host", "127.0.0.1", "--port", "$DashboardPort" `
        -WindowStyle Minimized -PassThru `
        -RedirectStandardOutput $dashboardLog -RedirectStandardError (Join-Path $runDir "dashboard.err.log")
    Set-Content -Path $dashboardPidFile -Value $p.Id
    Write-Host "      Started (PID $($p.Id)). Waiting for it to respond..." -ForegroundColor Yellow
    if (Wait-ForHttp $dashboardUrl 30) {
        Write-Host "      Dashboard is up." -ForegroundColor Green
    } else {
        Write-Host "      WARNING: dashboard didn't respond within 30s; check $dashboardLog" -ForegroundColor Red
    }
}

# --- 4. Open browser -----------------------------------------------------
if (-not $NoBrowser) {
    Write-Host "[4/4] Opening Chrome..." -ForegroundColor Cyan
    $chromePaths = @(
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    $chrome = $chromePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($chrome) {
        Start-Process -FilePath $chrome -ArgumentList $dashboardUrl
    } else {
        Write-Host "      Chrome not found; opening default browser instead." -ForegroundColor Yellow
        Start-Process $dashboardUrl
    }
} else {
    Write-Host "[4/4] Skipped (NoBrowser)." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "All set. Dashboard: $dashboardUrl" -ForegroundColor Green
Write-Host "Run .\tool\stop_all.ps1 to stop the scanner and dashboard when you're done." -ForegroundColor Green
