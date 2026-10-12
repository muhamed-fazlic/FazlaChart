<#
.SYNOPSIS
    Creates a desktop shortcut that runs tool\start_all.ps1 with one double-click.

.DESCRIPTION
    The shortcut launches PowerShell hidden (no console window flashing up) and
    runs start_all.ps1, which starts Ollama + the Market Scanner + the dashboard
    and opens it in Chrome. Re-run this script any time (e.g. after moving the
    repo) to recreate/update the shortcut.

.EXAMPLE
    .\tool\create_desktop_shortcut.ps1
#>

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$startScript = Join-Path $repoRoot "tool\start_all.ps1"
if (-not (Test-Path $startScript)) {
    Write-Host "Could not find $startScript" -ForegroundColor Red
    exit 1
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "FazlaChart.lnk"

$chromePaths = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$iconSource = $chromePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iconSource) { $iconSource = "$env:SystemRoot\System32\shell32.dll" }

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$startScript`""
$shortcut.WorkingDirectory = $repoRoot
$shortcut.IconLocation = "$iconSource,0"
$shortcut.Description = "Start FazlaChart: Ollama + Market Scanner + Dashboard, opens in Chrome"
$shortcut.Save()

Write-Host "Desktop shortcut created: $shortcutPath" -ForegroundColor Green
Write-Host "Double-click it any time to start everything and open the dashboard." -ForegroundColor Green
