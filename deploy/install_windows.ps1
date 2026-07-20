param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path,
    [string]$Python = "py -3.11",
    [string]$Wheelhouse = ""
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Invoke-Expression "$Python -m venv .venv"
}

$pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if ($Wheelhouse -ne "") {
    if (-not (Test-Path $Wheelhouse)) { throw "Wheelhouse not found: $Wheelhouse" }
    & $pip install --no-index --find-links $Wheelhouse -r requirements.txt
} else {
    Write-Warning "No wheelhouse supplied. This branch requires package access to install dependencies."
    & $pip install -r requirements.txt
}

if (-not (Test-Path "configs\monitor_settings.env")) {
    Copy-Item "configs\monitor_settings.env.example" "configs\monitor_settings.env"
    Write-Warning "Created configs\monitor_settings.env from the example. Edit RTSP, model, ROI, SMTP, and recipients before running."
}

Write-Host "Installation complete. Test with:"
Write-Host "  & $pythonExe scripts\check_rtsp.py --seconds 15"
Write-Host "  & $pythonExe scripts\test_email.py"
Write-Host "  & $pythonExe scripts\run_monitor.py"
