Param(
  [int]$Port = 8000,
  [string]$Bind = "0.0.0.0",
  [switch]$Reload
)

$ErrorActionPreference = "Stop"

# Resolve project root (parent of scripts directory)
$ROOT = (Resolve-Path "$PSScriptRoot\..").Path
$VENV = Join-Path $ROOT ".venv"
$PY_VENV = Join-Path $VENV "Scripts\python.exe"
$PIP_VENV = "$PY_VENV" + " -m pip"

Write-Host "Project root: $ROOT"

# Ensure Python is available
try {
  $pyVer = python --version
  Write-Host "Python detected: $pyVer"
} catch {
  Write-Error "Python is not available in PATH. Please install Python 3.10+ and retry."
  exit 1
}

# Create venv if missing
if (-not (Test-Path $VENV)) {
  Write-Host "Creating virtual environment at $VENV"
  python -m venv $VENV
}

# Use venv’s python/pip explicitly (no activation needed)
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
Write-Host "Starting backend: python -m uvicorn app.server:app --host $Bind --port $Port $reloadFlag"
& $PY_VENV -m uvicorn app.server:app --host $Bind --port $Port $reloadFlag