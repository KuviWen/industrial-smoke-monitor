$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
Set-Location $ProjectRoot
& (Join-Path $ProjectRoot ".venv\Scripts\python.exe") (Join-Path $ProjectRoot "scripts\run_monitor.py") --settings (Join-Path $ProjectRoot "configs\monitor_settings.env")
