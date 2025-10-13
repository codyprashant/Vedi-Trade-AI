Param(
  [int]$Port = 8001,
  [string]$Bind = "0.0.0.0",
  [switch]$Reload
)

$ErrorActionPreference = "Stop"

# Resolve project root (parent of scripts directory)
$ROOT = (Resolve-Path "$PSScriptRoot\..").Path
$VENV = Join-Path $ROOT ".venv"
$PY_VENV = Join-Path $VENV "Scripts\python.exe"

Write-Host "Project root: $ROOT"

if (-not (Test-Path $VENV)) {
  Write-Host "Creating virtual environment at $VENV"
  python -m venv $VENV
}

if (-not (Test-Path $PY_VENV)) {
  Write-Error "Venv python not found at $PY_VENV"
  exit 1
}

# Install dependencies
$REQS = Join-Path $ROOT "requirements.txt"
if (-not (Test-Path $REQS)) {
  Write-Error "requirements.txt not found at $REQS"
  exit 1
}
Write-Host "Installing dependencies from $REQS"
& $PY_VENV -m pip install -r $REQS

# Start FastAPI via uvicorn
$reloadFlag = if ($Reload) { "--reload" } else { "" }
Write-Host "Starting Yahoo backend: python -m uvicorn app.yahoo_server:app --host $Bind --port $Port $reloadFlag"
& $PY_VENV -m uvicorn app.yahoo_server:app --host $Bind --port $Port $reloadFlag