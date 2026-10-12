<#
.SYNOPSIS
    Stops the Market Scanner and dashboard processes started by tool\start_all.ps1.

.DESCRIPTION
    Reads the PID files in tool\.run\ and stops each process if it's still alive.
    Leaves Ollama running (it's a lightweight background app meant to stay up).

.EXAMPLE
    .\tool\stop_all.ps1
#>

$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot "tool\.run"

function Stop-ByPidFile($name, $pidFile) {
    if (-not (Test-Path $pidFile)) {
        Write-Host "$name not running (no PID file)." -ForegroundColor DarkGray
        return
    }
    $procId = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $procId -Force
        Write-Host "Stopped $name (PID $procId)." -ForegroundColor Green
    } else {
        Write-Host "$name already stopped." -ForegroundColor DarkGray
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

Stop-ByPidFile "Market Scanner" (Join-Path $runDir "scanner.pid")
Stop-ByPidFile "Dashboard" (Join-Path $runDir "dashboard.pid")

Write-Host ""
Write-Host "Done. Ollama was left running (it's a lightweight background service)." -ForegroundColor Cyan
